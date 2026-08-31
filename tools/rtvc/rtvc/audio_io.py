"""Device I/O: a lock-protected ring buffer and a single duplex stream.

``sounddevice`` (and therefore PortAudio) is imported lazily so that the rest
of the package -- DSP, engines, the offline harness -- stays importable on a
machine with no audio stack at all (CI containers, headless servers).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["RingBuffer", "AudioIO", "IOStats", "require_sounddevice",
           "list_devices", "format_device_table", "resolve_device"]


def require_sounddevice():
    """Import sounddevice on demand, with an actionable error if it is missing."""
    try:
        import sounddevice as sd  # noqa: WPS433 (deliberate lazy import)
    except OSError as exc:  # PortAudio shared library missing
        raise RuntimeError(
            "sounddevice imported but PortAudio could not be loaded "
            f"({exc}). Install the PortAudio runtime, or use --offline."
        ) from exc
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice is not installed. `pip install sounddevice`, "
            "or run with --offline to use the device-free harness."
        ) from exc
    return sd


# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------


class RingBuffer:
    """Fixed-capacity mono/multi-channel FIFO, safe for producer/consumer use.

    On overflow the *oldest* samples are dropped, never the newest.  For a
    live monitoring path that is the right trade: dropping new audio would
    stall the stream behind a backlog that only ever grows, while dropping old
    audio costs a glitch now and returns immediately to realtime.
    """

    def __init__(self, capacity: int, channels: int = 1, dtype=np.float32) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = int(capacity)
        self.channels = int(channels)
        self.dtype = np.dtype(dtype)
        shape = (self.capacity,) if channels == 1 else (self.capacity, channels)
        self._buf = np.zeros(shape, dtype=self.dtype)
        self._lock = threading.Lock()
        self._write = 0
        self._count = 0
        self.dropped = 0          # samples discarded because we overflowed
        self.max_fill = 0         # high-water mark, useful for buffer sizing

    def __len__(self) -> int:
        with self._lock:
            return self._count

    @property
    def available(self) -> int:
        with self._lock:
            return self._count

    def clear(self) -> None:
        with self._lock:
            self._write = 0
            self._count = 0

    def write(self, data: np.ndarray) -> int:
        """Append ``data``; return the number of samples dropped (0 normally)."""
        data = np.asarray(data, dtype=self.dtype)
        if self.channels == 1:
            data = data.reshape(-1)
        n = data.shape[0]
        if n == 0:
            return 0
        dropped = 0
        with self._lock:
            if n >= self.capacity:
                # Keep only the newest `capacity` samples.
                dropped = self._count + n - self.capacity
                self._buf[:] = data[-self.capacity:]
                self._write = 0
                self._count = self.capacity
            else:
                end = self._write + n
                if end <= self.capacity:
                    self._buf[self._write:end] = data
                else:
                    first = self.capacity - self._write
                    self._buf[self._write:] = data[:first]
                    self._buf[:end - self.capacity] = data[first:]
                self._write = end % self.capacity
                new_count = self._count + n
                if new_count > self.capacity:
                    dropped = new_count - self.capacity
                    new_count = self.capacity
                self._count = new_count
            if self._count > self.max_fill:
                self.max_fill = self._count
        self.dropped += dropped
        return dropped

    def read(self, n: int) -> Optional[np.ndarray]:
        """Pop exactly ``n`` samples, or ``None`` if fewer are buffered."""
        if n <= 0:
            return self._empty(0)
        with self._lock:
            if self._count < n:
                return None
            start = (self._write - self._count) % self.capacity
            end = start + n
            if end <= self.capacity:
                out = self._buf[start:end].copy()
            else:
                out = np.concatenate((self._buf[start:], self._buf[:end - self.capacity]))
            self._count -= n
            return out

    def read_available(self, n: int) -> np.ndarray:
        """Pop up to ``n`` samples; zero-pads nothing, may return fewer."""
        with self._lock:
            take = min(n, self._count)
        if take == 0:
            return self._empty(0)
        got = self.read(take)
        return got if got is not None else self._empty(0)

    def _empty(self, n: int) -> np.ndarray:
        shape = (n,) if self.channels == 1 else (n, self.channels)
        return np.zeros(shape, dtype=self.dtype)


# ---------------------------------------------------------------------------
# Duplex stream
# ---------------------------------------------------------------------------


@dataclass
class IOStats:
    underflow: int = 0      # callback had no converted audio ready
    overflow: int = 0       # PortAudio reported input overflow
    dropped_in: int = 0     # input samples discarded by the input ring
    dropped_out: int = 0    # output samples discarded by the output ring
    callbacks: int = 0
    status_flags: List[str] = field(default_factory=list)


class AudioIO:
    """One duplex ``sd.Stream`` carrying both capture and playback.

    Two design rules are load-bearing and should not be "simplified" away:

    1. **One stream, not two.**  Separate input and output streams run on
       independent clocks; the drift between them has to be absorbed by an
       ever-growing buffer, which is latency you cannot get back.
    2. **The callback only moves bytes.**  It writes the captured block into
       the input ring and reads a block out of the output ring.  No inference,
       no allocation-heavy work, no locks held longer than a memcpy.  Anything
       slower than realtime inside the callback is an instant dropout.
    """

    def __init__(
        self,
        sr: int,
        blocksize: int,
        in_device: Optional[Any] = None,
        out_device: Optional[Any] = None,
        channels: int = 1,
        latency: Any = "low",
        exclusive: bool = False,
        ring_seconds: float = 2.0,
        dtype: str = "float32",
    ) -> None:
        self.sr = int(sr)
        self.blocksize = int(blocksize)
        self.channels = int(channels)
        self.in_device = in_device
        self.out_device = out_device
        self.latency = latency
        self.exclusive = bool(exclusive)
        self.dtype = dtype

        cap = max(int(sr * ring_seconds), self.blocksize * 8)
        self.in_ring = RingBuffer(cap, channels=1)
        self.out_ring = RingBuffer(cap, channels=1)
        self.stats = IOStats()

        self._stream = None
        self._sd = None
        #: Under/overrun counting starts only once the worker has produced
        #: its first converted block.  Before that the output ring is
        #: legitimately empty, and counting those would make `under` a
        #: number that can never be zero -- useless as a pass/fail signal.
        self.counting = False
        #: Set by the callback after every captured block so the worker
        #: thread can wake on data instead of polling on a timer.
        self.input_event = threading.Event()
        self._out_min_fill = None  # minimum out-ring occupancy since last poll

    # -- device latency reported by PortAudio, in milliseconds ---------------
    @property
    def input_latency_ms(self) -> float:
        if self._stream is None:
            return 0.0
        lat = self._stream.latency
        value = lat[0] if isinstance(lat, (tuple, list)) else lat
        return float(value) * 1000.0

    @property
    def output_latency_ms(self) -> float:
        if self._stream is None:
            return 0.0
        lat = self._stream.latency
        value = lat[1] if isinstance(lat, (tuple, list)) else lat
        return float(value) * 1000.0

    def _extra_settings(self):
        if not self.exclusive:
            return None
        sd = self._sd
        wasapi = getattr(sd, "WasapiSettings", None)
        if wasapi is None:
            raise RuntimeError(
                "--exclusive requires the WASAPI host API (Windows). "
                "This platform's sounddevice build has no WasapiSettings."
            )
        # sd.Stream takes a (input, output) pair when duplex.
        return (wasapi(exclusive=True), wasapi(exclusive=True))

    def _callback(self, indata, outdata, frames, time_info, status):
        self.stats.callbacks += 1
        if status:
            text = str(status)
            if self.counting:
                if status.input_overflow:
                    self.stats.overflow += 1
                if status.output_underflow:
                    self.stats.underflow += 1
            if text and text not in self.stats.status_flags:
                self.stats.status_flags.append(text)

        # capture -> ring (mono: take channel 0)
        mono_in = indata[:, 0] if indata.ndim > 1 else indata
        self.stats.dropped_in += self.in_ring.write(mono_in)
        self.input_event.set()

        # ring -> playback
        chunk = self.out_ring.read(frames)
        if chunk is None:
            # Nothing converted yet: emit silence rather than stale audio.
            outdata.fill(0)
            if self.counting:
                self.stats.underflow += 1
        else:
            if outdata.ndim > 1:
                outdata[:] = chunk.reshape(-1, 1)
            else:
                outdata[:] = chunk

        fill = self.out_ring.available
        if self._out_min_fill is None or fill < self._out_min_fill:
            self._out_min_fill = fill

    def take_out_min_fill(self) -> int:
        """Minimum output-ring occupancy observed since the previous call.

        Reported as ``out-buf`` in the latency table.  The *minimum* is the
        honest number: the mean would double-count the block we are already
        accounting for under ``block``.
        """
        value = self._out_min_fill
        self._out_min_fill = None
        return int(value) if value is not None else 0

    def prefill(self, samples: int) -> None:
        """Seed the output ring with silence.

        Buys the worker a head start against jitter, at the cost of exactly
        that much extra output latency -- which then shows up honestly in
        the ``out-buf`` column rather than hiding somewhere.  Default is 0
        so a measurement run reports the true floor.
        """
        if samples > 0:
            self.out_ring.write(np.zeros(int(samples), dtype=np.float32))

    def start(self) -> None:
        if self._stream is not None:
            return
        sd = self._sd = require_sounddevice()
        self._stream = sd.Stream(
            samplerate=self.sr,
            blocksize=self.blocksize,
            device=(self.in_device, self.out_device),
            channels=(self.channels, self.channels),
            dtype=self.dtype,
            latency=self.latency,
            callback=self._callback,
            extra_settings=self._extra_settings(),
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
        finally:
            self._stream.close()
            self._stream = None

    def __enter__(self) -> "AudioIO":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Device discovery helpers
# ---------------------------------------------------------------------------


def list_devices() -> Sequence[dict]:
    sd = require_sounddevice()
    return sd.query_devices()


def format_device_table() -> str:
    sd = require_sounddevice()
    devices = sd.query_devices()
    apis = sd.query_hostapis()
    lines = [f"{'idx':>4}  {'in':>3} {'out':>3}  {'default sr':>10}  host API / name"]
    lines.append("-" * 78)
    for idx, dev in enumerate(devices):
        api = apis[dev["hostapi"]]["name"]
        lines.append(
            f"{idx:>4}  {dev['max_input_channels']:>3} {dev['max_output_channels']:>3}  "
            f"{dev['default_samplerate']:>10.0f}  [{api}] {dev['name']}"
        )
    return "\n".join(lines)


def resolve_device(spec: Optional[str], kind: str) -> Optional[Any]:
    """Accept an index (``23``), a name substring, or ``None`` for the default."""
    if spec is None or spec == "":
        return None
    try:
        return int(spec)
    except (TypeError, ValueError):
        pass
    sd = require_sounddevice()
    devices = sd.query_devices()
    needle = str(spec).lower()
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    matches = [i for i, d in enumerate(devices)
               if needle in d["name"].lower() and d[key] > 0]
    if not matches:
        raise ValueError(f"no {kind} device matching {spec!r}; try --list-devices")
    if len(matches) > 1:
        names = ", ".join(f"{i}:{devices[i]['name']}" for i in matches[:6])
        raise ValueError(f"{spec!r} matches several {kind} devices ({names}); use the index")
    return matches[0]
