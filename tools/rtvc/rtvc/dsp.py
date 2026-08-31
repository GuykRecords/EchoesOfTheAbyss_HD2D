"""DSP building blocks for the realtime voice-conversion path.

All classes here are *streaming* objects: they keep internal state between
calls so that consecutive blocks of a continuous signal are processed without
discontinuities at the block boundaries.  Every ``process()`` takes and returns
a 1-D float32 array of PCM in the range [-1, 1].

Nothing in this module imports torch or sounddevice, so it can be exercised on
a machine with no audio hardware and no GPU.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

try:  # optional, strongly preferred for quality + speed
    import soxr as _soxr
except ImportError:  # pragma: no cover - exercised only on machines without soxr
    _soxr = None


__all__ = [
    "HighPass",
    "NoiseGate",
    "SoftLimiter",
    "Resampler",
    "StreamResampler",
    "equal_power_windows",
    "db_to_amp",
    "amp_to_db",
    "rms",
    "have_soxr",
]

_EPS = 1e-12


def have_soxr() -> bool:
    """True when the soxr backend is importable."""
    return _soxr is not None


def db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def amp_to_db(amp: float) -> float:
    return float(20.0 * math.log10(max(float(amp), _EPS)))


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def _as_f32(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 1:
        a = a.reshape(-1)
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    return a


# ---------------------------------------------------------------------------
# High pass
# ---------------------------------------------------------------------------


class HighPass:
    """Butterworth high-pass filter with state carried across blocks.

    Removes rumble, desk thumps and DC offset before anything else touches the
    signal.  ``sosfilt`` state (``zi``) is kept on the instance, which is what
    makes block-by-block filtering identical to filtering the whole stream at
    once -- filter it statelessly per block and you get a click every block.
    """

    def __init__(self, sr: int, cutoff: float = 80.0, order: int = 2) -> None:
        if not 0.0 < cutoff < sr * 0.5:
            raise ValueError(f"cutoff {cutoff} out of range for sr {sr}")
        self.sr = int(sr)
        self.cutoff = float(cutoff)
        self.order = int(order)
        self.sos = butter(self.order, self.cutoff / (self.sr * 0.5),
                          btype="highpass", output="sos").astype(np.float64)
        self._zi = np.zeros((self.sos.shape[0], 2), dtype=np.float64)

    def reset(self) -> None:
        self._zi[:] = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = _as_f32(x)
        if x.size == 0:
            return x
        y, self._zi = sosfilt(self.sos, x.astype(np.float64), zi=self._zi)
        return y.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Noise gate
# ---------------------------------------------------------------------------


class NoiseGate:
    """RMS noise gate with hysteresis, hold time and one-pole gain smoothing.

    * opens when the frame RMS rises above ``open_db``
    * closes only after the RMS has stayed below ``close_db`` for ``hold_ms``
    * the 0/1 decision is never applied directly; it drives a one-pole IIR so
      the gain ramps instead of stepping (a step is an audible click)

    The signal is scanned in short frames (``frame`` samples) so the hold timer
    has finer resolution than the audio block size.  Within one frame the gain
    target is constant, so the one-pole response has a closed form
    ``g[n] = target + (g_prev - target) * a**(n+1)`` and can be vectorised
    instead of looped per sample.
    """

    def __init__(
        self,
        sr: int,
        open_db: float = -42.0,
        close_db: float = -50.0,
        hold_ms: float = 180.0,
        attack_ms: float = 5.0,
        release_ms: float = 60.0,
        frame: int = 128,
    ) -> None:
        if close_db > open_db:
            raise ValueError("close_db must be <= open_db (hysteresis)")
        self.sr = int(sr)
        self.open_amp = db_to_amp(open_db)
        self.close_amp = db_to_amp(close_db)
        self.open_db = float(open_db)
        self.close_db = float(close_db)
        self.hold_samples = int(round(sr * hold_ms / 1000.0))
        self.frame = max(1, int(frame))
        self._a_attack = self._pole(attack_ms)
        self._a_release = self._pole(release_ms)
        self._gain = 0.0
        self._open = False
        self._below = 0  # samples spent under close_db while open

    def _pole(self, ms: float) -> float:
        # time constant -> per-sample one-pole coefficient
        n = max(1.0, self.sr * ms / 1000.0)
        return float(math.exp(-1.0 / n))

    def reset(self) -> None:
        self._gain = 0.0
        self._open = False
        self._below = 0

    @property
    def gain(self) -> float:
        return self._gain

    @property
    def is_open(self) -> bool:
        return self._open

    def process(self, x: np.ndarray) -> np.ndarray:
        x = _as_f32(x)
        if x.size == 0:
            return x
        out = np.empty_like(x)
        for start in range(0, x.size, self.frame):
            seg = x[start:start + self.frame]
            level = rms(seg)
            if self._open:
                if level < self.close_amp:
                    self._below += seg.size
                    if self._below >= self.hold_samples:
                        self._open = False
                else:
                    self._below = 0
            else:
                if level > self.open_amp:
                    self._open = True
                    self._below = 0

            target = 1.0 if self._open else 0.0
            a = self._a_attack if target > self._gain else self._a_release
            ramp = a ** np.arange(1, seg.size + 1, dtype=np.float64)
            g = target + (self._gain - target) * ramp
            out[start:start + seg.size] = (seg.astype(np.float64) * g).astype(np.float32)
            self._gain = float(g[-1])
        return out


# ---------------------------------------------------------------------------
# Crossfade windows
# ---------------------------------------------------------------------------


def equal_power_windows(n: int, dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(fade_out, fade_in)`` of length ``n`` with ``fo**2 + fi**2 == 1``.

    Equal *power* (cos/sin), not equal amplitude.  Two uncorrelated signals
    crossfaded with these keep a constant loudness through the transition.
    Two *identical* signals gain up to +3 dB in the middle -- that is the
    arithmetic working as intended, not a bug, and it is why a passthrough run
    is slightly louder inside the crossfade region.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        empty = np.zeros(0, dtype=dtype)
        return empty, empty.copy()
    t = (np.arange(n, dtype=np.float64) + 0.5) / n
    fade_out = np.cos(t * (np.pi / 2.0)).astype(dtype)
    fade_in = np.sin(t * (np.pi / 2.0)).astype(dtype)
    return fade_out, fade_in


# ---------------------------------------------------------------------------
# Limiter
# ---------------------------------------------------------------------------


class SoftLimiter:
    """``tanh`` soft limiter with a ceiling (default -1 dBFS).

    ``y = c * tanh(x / c)``: unity slope at the origin, asymptotic at ``c``.
    Stateless, so it is safe to call on any block in any order.  It is the last
    stage before the output ring so a hot conversion cannot clip the device.
    """

    def __init__(self, ceiling_db: float = -1.0) -> None:
        self.ceiling_db = float(ceiling_db)
        self.ceiling = db_to_amp(ceiling_db)

    def process(self, x: np.ndarray) -> np.ndarray:
        x = _as_f32(x)
        if x.size == 0:
            return x
        return (self.ceiling * np.tanh(x.astype(np.float64) / self.ceiling)).astype(np.float32)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


class Resampler:
    """One-shot sample-rate conversion.  soxr when available, scipy otherwise.

    Use this for whole utterances / warmup buffers.  For a continuous stream
    use :class:`StreamResampler`, which does not restart the filter at every
    block boundary.
    """

    def __init__(self, sr_in: int, sr_out: int, quality: str = "VHQ",
                 backend: str = "auto") -> None:
        if sr_in <= 0 or sr_out <= 0:
            raise ValueError("sample rates must be positive")
        self.sr_in = int(sr_in)
        self.sr_out = int(sr_out)
        self.quality = quality
        if backend not in ("auto", "soxr", "scipy"):
            raise ValueError(f"unknown backend {backend!r}")
        if backend == "soxr" and _soxr is None:
            raise RuntimeError("backend='soxr' requested but soxr is not installed")
        self.backend = ("soxr" if (backend == "auto" and _soxr is not None)
                        else ("soxr" if backend == "soxr" else "scipy"))
        frac = Fraction(self.sr_out, self.sr_in).limit_denominator(10_000)
        self._up, self._down = frac.numerator, frac.denominator

    @property
    def ratio(self) -> float:
        return self.sr_out / self.sr_in

    @property
    def is_identity(self) -> bool:
        return self.sr_in == self.sr_out

    @property
    def is_integer_ratio(self) -> bool:
        """True when one rate is a whole multiple of the other.

        Worth checking before choosing a model's sample rate.  Measured on
        this pipeline: an integer ratio (48k <-> 16k) reconstructs at about
        -56 dB rms error, while 48k <-> 44.1k only reaches about -35 dB.  That
        is 20 dB of avoidable noise bought by picking the wrong model rate.
        """
        hi, lo = max(self.sr_in, self.sr_out), min(self.sr_in, self.sr_out)
        return hi % lo == 0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = _as_f32(x)
        if self.is_identity or x.size == 0:
            return x
        if self.backend == "soxr":
            return _as_f32(_soxr.resample(x, self.sr_in, self.sr_out, quality=self.quality))
        return _as_f32(resample_poly(x, self._up, self._down))


class StreamResampler:
    """Block-wise resampling without boundary clicks (overlap-trim).

    Resampling filters need signal on *both* sides of every output sample.
    Feed a resampler isolated blocks and each block is filtered against
    implicit silence at both edges, which puts a discontinuity at every block
    boundary -- a periodic click at the block rate.

    So each call resamples ``[left glue | region to emit | right glue]`` and
    keeps only the middle.  The glue (*のりしろ*) is real audio, so the filter
    never sees a false edge inside the region we keep.

    * **left glue** is the tail of the previous input -- free, it is past audio.
    * **right glue** is the tail of the *current* input, which means the region
      emitted lags the input by ``pad`` samples.  That is real latency and it
      is reported by :attr:`latency_ms`.  Pass ``lookahead=False`` to spend
      nothing and accept a small taper at each seam instead.

    Output length tracks a running input/output sample counter, so an integer
    number of blocks in yields the expected number of samples out with no slow
    drift over a long session.
    """

    def __init__(self, sr_in: int, sr_out: int, pad_ms: float = 10.0,
                 lookahead: bool = True, quality: str = "VHQ",
                 backend: str = "auto") -> None:
        self._rs = Resampler(sr_in, sr_out, quality=quality, backend=backend)
        self.sr_in = self._rs.sr_in
        self.sr_out = self._rs.sr_out
        self.pad = max(0, int(round(self.sr_in * pad_ms / 1000.0)))
        self.lookahead = bool(lookahead) and self.pad > 0
        self.reset()

    @property
    def ratio(self) -> float:
        return self._rs.ratio

    @property
    def is_identity(self) -> bool:
        return self._rs.is_identity

    @property
    def backend(self) -> str:
        return self._rs.backend

    @property
    def latency_samples(self) -> int:
        """Added delay in input samples (0 unless ``lookahead`` is on)."""
        return self.pad if self.lookahead else 0

    @property
    def latency_ms(self) -> float:
        return self.latency_samples * 1000.0 / self.sr_in

    def reset(self) -> None:
        self._left = np.zeros(self.pad, dtype=np.float32)
        self._carry = np.zeros(self.latency_samples, dtype=np.float32)
        self._in_total = 0
        self._out_total = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = _as_f32(x)
        if self.is_identity or x.size == 0:
            return x

        region = np.concatenate((self._carry, x)) if self._carry.size else x
        emit_in = region[:x.size]        # what this call is responsible for
        right = region[x.size:]          # held-back lookahead, becomes next carry

        buf = np.concatenate((self._left, emit_in, right)) if self.pad else emit_in
        y = self._rs.process(buf)

        drop = int(round(self._left.size * self.ratio))
        if drop:
            y = y[drop:]

        self._in_total += x.size
        want = max(0, int(round(self._in_total * self.ratio)) - self._out_total)
        if y.size >= want:
            y = y[:want]
        elif y.size:
            # Rounding at the very first block; hold the last sample rather
            # than injecting a zero-valued step.
            y = np.concatenate((y, np.full(want - y.size, y[-1], dtype=np.float32)))
        else:
            y = np.zeros(want, dtype=np.float32)
        self._out_total += y.size

        if self.pad:
            merged = np.concatenate((self._left, emit_in))
            self._left = merged[-self.pad:].copy()
        self._carry = right.copy()
        return y
