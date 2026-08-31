"""Conversion engines.

An engine turns one analysis window of audio into the same number of output
samples.  The realtime loop does not care what happens inside -- passthrough,
a fake fixed cost, or a real RVC forward pass -- only that ``convert()``
returns ``len(window)`` samples and that its cost is measured.

Measurement lives in :class:`BaseEngine` so every engine is timed the same way
and a fake engine is a truthful stand-in for a real one during latency work.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from .dsp import Resampler

__all__ = ["BaseEngine", "PassthroughEngine", "FixedCostEngine",
           "CallableEngine", "RVCTorchEngine", "build_engine", "ENGINE_CHOICES"]


class BaseEngine:
    """Timing and warmup scaffolding shared by every engine.

    Subclasses implement :meth:`_convert`.  ``convert()`` wraps it with a
    perf-counter measurement folded into an exponential moving average, plus a
    running peak -- the peak is what actually causes dropouts, so it is tracked
    separately rather than being smoothed away.
    """

    name = "base"
    #: Sample rate the engine wants its input at; ``None`` means "whatever the
    #: stream runs at", which lets the caller skip resampling entirely.
    native_sr: Optional[int] = None

    def __init__(self, ema_alpha: float = 0.1) -> None:
        self.ema_alpha = float(ema_alpha)
        self.infer_ms_ema = 0.0
        self.infer_ms_max = 0.0
        self.infer_ms_last = 0.0
        self.calls = 0
        self._warm = False

    # -- subclass hook ------------------------------------------------------
    def _convert(self, window: np.ndarray, sr: int) -> np.ndarray:
        raise NotImplementedError

    # -- public API ---------------------------------------------------------
    def convert(self, window: np.ndarray, sr: int) -> np.ndarray:
        t0 = time.perf_counter()
        out = self._convert(window, sr)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        self.infer_ms_last = dt_ms
        self.calls += 1
        if self.calls == 1:
            self.infer_ms_ema = dt_ms
        else:
            self.infer_ms_ema += self.ema_alpha * (dt_ms - self.infer_ms_ema)
        if dt_ms > self.infer_ms_max:
            self.infer_ms_max = dt_ms

        out = np.asarray(out, dtype=np.float32).reshape(-1)
        if out.size != window.size:
            raise ValueError(
                f"{self.name}: engine returned {out.size} samples for a "
                f"{window.size}-sample window; engines must be length preserving"
            )
        return out

    def warmup(self, window_samples: int, sr: int, iters: int = 8) -> None:
        """Run ``iters`` conversions at the real window length before going live.

        Skipping this is not free: lazy CUDA context creation, kernel
        autotuning and the first cuDNN algorithm search all land on the first
        real utterance, and the user hears their first word arrive a second
        late.  Warm up at the *production* window length -- a different length
        re-triggers the same one-off costs.
        """
        dummy = np.zeros(int(window_samples), dtype=np.float32)
        for _ in range(max(0, int(iters))):
            self._convert(dummy, sr)
        self._warm = True
        # Warmup timings are not representative; do not pollute the EMA.
        self.reset_stats()

    def reset_stats(self) -> None:
        self.infer_ms_ema = 0.0
        self.infer_ms_max = 0.0
        self.infer_ms_last = 0.0
        self.calls = 0

    @property
    def is_warm(self) -> bool:
        return self._warm

    def rtf(self, block_ms: float) -> float:
        """Real-time factor: inference cost divided by the audio it covers."""
        return self.infer_ms_ema / block_ms if block_ms > 0 else 0.0

    def close(self) -> None:
        pass


class PassthroughEngine(BaseEngine):
    """Copies the window through unchanged.

    The baseline for every latency measurement: whatever TOTAL this run
    reports is the floor imposed by devices, buffering and the crossfade, with
    no model in the path.
    """

    name = "passthrough"

    def _convert(self, window: np.ndarray, sr: int) -> np.ndarray:
        return np.asarray(window, dtype=np.float32).copy()


class FixedCostEngine(BaseEngine):
    """Burns a fixed wall-clock budget per window, then passes audio through.

    Used to answer "how heavy a model can this configuration carry?" before the
    model exists.  Set ``cost_ms`` to a candidate inference time; if under/over
    stay at zero, the pipeline tolerates a model of that cost.
    """

    name = "fixed"

    def __init__(self, cost_ms: float = 25.0, ema_alpha: float = 0.1,
                 busy_wait: bool = False) -> None:
        super().__init__(ema_alpha=ema_alpha)
        self.cost_ms = float(cost_ms)
        self.busy_wait = bool(busy_wait)

    def _convert(self, window: np.ndarray, sr: int) -> np.ndarray:
        target = self.cost_ms / 1000.0
        if self.busy_wait:
            end = time.perf_counter() + target
            while time.perf_counter() < end:
                pass
        else:
            time.sleep(target)
        return np.asarray(window, dtype=np.float32).copy()


class CallableEngine(BaseEngine):
    """Wraps any ``fn(window, sr) -> np.ndarray`` as an engine.

    Handy for tests and for dropping in an experimental converter without
    writing a class.
    """

    name = "callable"

    def __init__(self, fn: Callable[[np.ndarray, int], np.ndarray],
                 name: str = "callable", ema_alpha: float = 0.1) -> None:
        super().__init__(ema_alpha=ema_alpha)
        self.fn = fn
        self.name = name

    def _convert(self, window: np.ndarray, sr: int) -> np.ndarray:
        return self.fn(window, sr)


class RVCTorchEngine(BaseEngine):
    """RVC adapter.

    Deliberately does **not** import RVC, torch or fairseq.  It takes an
    injected callable::

        infer_fn(wav16k: np.ndarray) -> np.ndarray   # returns audio at model_sr

    and owns only the parts that belong to the realtime path: resampling into
    the model's 16 kHz input domain, resampling the result back to the stream
    rate, and enforcing the length contract.  That keeps this file importable
    with no ML stack installed, and keeps the RVC dependency mess confined to
    whatever module builds ``infer_fn``.

    Wire it to ``tools/rvc_for_realtime.py``'s ``RVC`` class, not to
    ``infer/modules/vc/pipeline.py``.  The batch pipeline recomputes f0 for
    every call with no cache and will not hit realtime.
    """

    name = "rvc"
    native_sr = 16000

    def __init__(
        self,
        infer_fn: Callable[[np.ndarray], np.ndarray],
        stream_sr: int,
        model_sr: int = 40000,
        input_sr: int = 16000,
        ema_alpha: float = 0.1,
        resample_quality: str = "HQ",
    ) -> None:
        super().__init__(ema_alpha=ema_alpha)
        if not callable(infer_fn):
            raise TypeError("infer_fn must be callable")
        self.infer_fn = infer_fn
        self.stream_sr = int(stream_sr)
        self.model_sr = int(model_sr)
        self.input_sr = int(input_sr)
        self._to_model = Resampler(self.stream_sr, self.input_sr, quality=resample_quality)
        self._from_model = Resampler(self.model_sr, self.stream_sr, quality=resample_quality)

    def _convert(self, window: np.ndarray, sr: int) -> np.ndarray:
        n = int(np.asarray(window).size)
        wav16k = self._to_model.process(window)
        out = np.asarray(self.infer_fn(wav16k), dtype=np.float32).reshape(-1)
        out = self._from_model.process(out)
        # Resampling ratios rarely land on an exact integer; trim or pad the
        # last sample or two so the caller's length contract still holds.
        if out.size > n:
            out = out[:n]
        elif out.size < n:
            pad = np.full(n - out.size, out[-1] if out.size else 0.0, dtype=np.float32)
            out = np.concatenate((out, pad))
        return out


ENGINE_CHOICES = ("passthrough", "fixed", "rvc")


def build_engine(kind: str, sr: int, fixed_cost_ms: float = 25.0,
                 infer_fn: Optional[Callable] = None, **kwargs) -> BaseEngine:
    kind = kind.lower()
    if kind == "passthrough":
        return PassthroughEngine()
    if kind == "fixed":
        return FixedCostEngine(cost_ms=fixed_cost_ms, **kwargs)
    if kind == "rvc":
        if infer_fn is None:
            raise ValueError(
                "engine 'rvc' needs an infer_fn; see docs/RVC_INTEGRATION.md"
            )
        return RVCTorchEngine(infer_fn, stream_sr=sr, **kwargs)
    raise ValueError(f"unknown engine {kind!r}; choose from {ENGINE_CHOICES}")
