"""Conversion engines.

An engine turns one analysis window of audio into audio.  The realtime loop
does not care what happens inside -- passthrough, a fake fixed cost, or a real
RVC forward pass -- only that ``convert()`` returns the agreed number of
samples and that its cost is measured.

Measurement lives in :class:`BaseEngine` so every engine is timed the same way
and a fake engine is a truthful stand-in for a real one during latency work.

**The tail contract.**  The loop only ever keeps the last ``X + B`` samples of
the conversion, so an engine that can restrict its own synthesis to that range
should.  For a neural vocoder that is the difference between synthesising a
140 ms window and a 40 ms tail -- measured at roughly 3.4x on RVC.  An engine
advertises this by setting ``supports_tail = True``; ``convert(..., tail=n)``
then expects exactly ``n`` samples back.  Engines that do not support it keep
the simple length-preserving contract and the caller slices as before.
"""

from __future__ import annotations

import inspect
import time
import warnings
from typing import Callable, Optional

import numpy as np

from .dsp import Resampler

__all__ = ["BaseEngine", "PassthroughEngine", "FixedCostEngine",
           "CallableEngine", "RVCTorchEngine", "build_engine", "ENGINE_CHOICES",
           "RVC_GRID_MS", "check_rvc_grid"]

#: RVC's realtime path counts in ``zc = sr // 100`` frames, i.e. 10 ms units,
#: and additionally advances its pitch cache by ``block_frame_16k // 160``.
#: Both are integer divisions, so a window size that is not a whole number of
#: 10 ms units silently shifts f0 by a fraction of a frame every block.
RVC_GRID_MS = 10.0


def check_rvc_grid(**ms_values: float) -> None:
    """Reject window sizes RVC's realtime path cannot represent exactly.

    Fail at startup with the offending numbers rather than at runtime with
    audio that drifts in pitch for reasons nobody can see.
    """
    bad = {name: value for name, value in ms_values.items()
           if abs(round(value / RVC_GRID_MS) * RVC_GRID_MS - value) > 1e-9}
    if bad:
        detail = ", ".join(f"{k}={v}ms" for k, v in sorted(bad.items()))
        raise ValueError(
            f"RVC needs every window size on a {RVC_GRID_MS:.0f}ms grid; got {detail}. "
            "Use e.g. block 30 / crossfade 10 / extra 100 (B+X is still 40ms)."
        )


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
    #: True when ``_convert`` honours ``tail`` and returns only that many
    #: samples.  See the module docstring for why this matters.
    supports_tail: bool = False

    def __init__(self, ema_alpha: float = 0.1) -> None:
        self.ema_alpha = float(ema_alpha)
        self.infer_ms_ema = 0.0
        self.infer_ms_max = 0.0
        self.infer_ms_last = 0.0
        self.calls = 0
        self._warm = False

    # -- subclass hook ------------------------------------------------------
    def _convert(self, window: np.ndarray, sr: int,
                 tail: Optional[int] = None) -> np.ndarray:
        raise NotImplementedError

    # -- public API ---------------------------------------------------------
    def convert(self, window: np.ndarray, sr: int,
                tail: Optional[int] = None) -> np.ndarray:
        """Convert one window.

        ``tail`` is how many trailing samples the caller actually needs.  It is
        a hint: an engine that advertises ``supports_tail`` returns exactly that
        many, and one that does not returns the whole window for the caller to
        slice.  Either way the caller gets what it asked for.
        """
        use_tail = tail if (tail and self.supports_tail) else None
        t0 = time.perf_counter()
        out = self._convert(window, sr, use_tail) if use_tail else self._convert(window, sr)
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
        expected = use_tail if use_tail else window.size
        if out.size != expected:
            what = ("the requested tail" if use_tail
                    else f"a {window.size}-sample window")
            raise ValueError(
                f"{self.name}: engine returned {out.size} samples for {what} "
                f"(expected {expected})"
            )
        return out

    def warmup(self, window_samples: int, sr: int, iters: int = 8,
               tail: Optional[int] = None) -> None:
        """Run ``iters`` conversions at the real window length before going live.

        Skipping this is not free: lazy CUDA context creation, kernel
        autotuning and the first cuDNN algorithm search all land on the first
        real utterance, and the user hears their first word arrive a second
        late.  Warm up at the *production* window length -- a different length
        re-triggers the same one-off costs -- and so does a different ``tail``,
        so pass the same one the live loop will use.

        On Windows this is only meaningful while the scheduler tick is raised
        (see :mod:`rtvc.timing`).  At the default ~15.6 ms tick the measured
        cost of a 30 ms warmup lands anywhere between 33 and 37 ms, and a
        pre-roll sized from that number is far too large.
        """
        dummy = np.zeros(int(window_samples), dtype=np.float32)
        use_tail = tail if (tail and self.supports_tail) else None
        for _ in range(max(0, int(iters))):
            if use_tail:
                self._convert(dummy, sr, use_tail)
            else:
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
    supports_tail = True

    def _convert(self, window: np.ndarray, sr: int,
                 tail: Optional[int] = None) -> np.ndarray:
        out = np.asarray(window, dtype=np.float32)
        return out[-tail:].copy() if tail else out.copy()


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

    supports_tail = True

    def _convert(self, window: np.ndarray, sr: int,
                 tail: Optional[int] = None) -> np.ndarray:
        target = self.cost_ms / 1000.0
        if self.busy_wait:
            end = time.perf_counter() + target
            while time.perf_counter() < end:
                pass
        else:
            time.sleep(target)
        out = np.asarray(window, dtype=np.float32)
        return out[-tail:].copy() if tail else out.copy()


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

    def _convert(self, window: np.ndarray, sr: int,
                 tail: Optional[int] = None) -> np.ndarray:
        return self.fn(window, sr)


class RVCTorchEngine(BaseEngine):
    """RVC adapter.

    Deliberately does **not** import RVC or torch.  It takes an injected
    callable and owns only the parts that belong to the realtime path:
    resampling into the model's 16 kHz input domain, resampling the result back
    to the stream rate, enforcing the length contract, and refusing window
    sizes RVC cannot represent.  Everything version-specific about RVC lives in
    whatever module builds ``infer_fn`` (see ``rvc_backend.py``), so this file
    stays importable with no ML stack installed.

    **Two shapes of ``infer_fn`` are accepted**, picked automatically:

    * ``infer_fn(wav16k) -> audio at model_sr`` -- simple; the engine slices
      the tail itself, so the model synthesises the whole window.
    * ``infer_fn(wav16k, skip_head, return_length) -> audio at model_sr`` --
      preferred; the model is told to synthesise only the tail the loop keeps,
      which is where the ~3.4x saving comes from.  ``skip_head`` and
      ``return_length`` are passed in **16 kHz samples**; converting them to
      whatever units the RVC build wants is the backend's job.

    Connect it to ``infer/rtrvc.py``'s ``RVC`` class -- not
    ``infer/modules/vc/pipeline.py``, whose f0 cache does not apply and which
    will not reach realtime.
    """

    name = "rvc"
    native_sr = 16000
    supports_tail = True

    def __init__(
        self,
        infer_fn: Callable,
        stream_sr: int,
        model_sr: int = 40000,
        input_sr: int = 16000,
        ema_alpha: float = 0.1,
        resample_quality: str = "HQ",
        block_ms: Optional[float] = None,
        crossfade_ms: Optional[float] = None,
        extra_ms: Optional[float] = None,
    ) -> None:
        super().__init__(ema_alpha=ema_alpha)
        if not callable(infer_fn):
            raise TypeError("infer_fn must be callable")

        grid = {name: value for name, value in (
            ("block_ms", block_ms), ("crossfade_ms", crossfade_ms),
            ("extra_ms", extra_ms)) if value is not None}
        if grid:
            check_rvc_grid(**grid)

        self.infer_fn = infer_fn
        self.stream_sr = int(stream_sr)
        self.model_sr = int(model_sr)
        self.input_sr = int(input_sr)
        self._tail_aware = self._accepts_tail_arguments(infer_fn)
        self._to_model = Resampler(self.stream_sr, self.input_sr, quality=resample_quality)
        self._from_model = Resampler(self.model_sr, self.stream_sr, quality=resample_quality)
        if not self._from_model.is_integer_ratio:
            warnings.warn(
                f"model rate {self.model_sr} is not a whole factor of the stream rate "
                f"{self.stream_sr}; resampling back costs about 20 dB of extra noise "
                "(measured -35 dB vs -56 dB for an integer ratio). A 48k or 24k model "
                "avoids this entirely.",
                RuntimeWarning, stacklevel=2,
            )

    @staticmethod
    def _accepts_tail_arguments(fn: Callable) -> bool:
        """True when ``fn`` takes (wav16k, skip_head, return_length)."""
        try:
            params = list(inspect.signature(fn).parameters.values())
        except (TypeError, ValueError):
            return False
        if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
            return True
        positional = [p for p in params
                      if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                    inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        return len(positional) >= 3

    @property
    def tail_aware(self) -> bool:
        """True when the model is told to synthesise only the kept tail."""
        return self._tail_aware

    def _convert(self, window: np.ndarray, sr: int,
                 tail: Optional[int] = None) -> np.ndarray:
        window = np.asarray(window, dtype=np.float32).reshape(-1)
        want = int(tail) if tail else window.size
        wav16k = self._to_model.process(window)

        if self._tail_aware:
            # Express the request in the model's own 16 kHz domain.  With no
            # tail requested this degenerates to "synthesise everything".
            return_16k = min(wav16k.size,
                             int(round(want * self.input_sr / self.stream_sr)))
            skip_16k = max(0, wav16k.size - return_16k)
            raw = self.infer_fn(wav16k, skip_16k, return_16k)
        else:
            raw = self.infer_fn(wav16k)

        out = self._from_model.process(np.asarray(raw, dtype=np.float32).reshape(-1))
        if not self._tail_aware and tail and out.size > want:
            out = out[-want:]
        return self._fit(out, want)

    @staticmethod
    def _fit(out: np.ndarray, want: int) -> np.ndarray:
        """Resampling ratios rarely land on an exact integer; absorb ±1 sample."""
        if out.size > want:
            return out[:want]
        if out.size < want:
            pad = np.full(want - out.size, out[-1] if out.size else 0.0, dtype=np.float32)
            return np.concatenate((out, pad))
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
