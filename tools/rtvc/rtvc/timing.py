"""Scheduler timer resolution.

On Windows the default scheduler tick is ~15.6 ms.  Everything that waits --
``time.sleep``, ``Event.wait``, and PortAudio's own internal waits -- rounds up
to that tick, which is half a dozen audio blocks.  PortAudio raises the
resolution itself while a stream is open, but only while a stream is open, so
work done *before* the stream starts (notably warmup) is measured against the
coarse clock and comes out both wrong and unstable.

Holding ``timeBeginPeriod(1)`` for the whole run makes warmup timings match
live timings, which is what the pre-roll is sized from.  Everywhere except
Windows this is a no-op.
"""

from __future__ import annotations

import sys

__all__ = ["HighResolutionTimer"]


class HighResolutionTimer:
    """Request a 1 ms scheduler period for as long as this object is active.

    ``timeBeginPeriod`` is reference counted by Windows and must be paired with
    ``timeEndPeriod``; leaving it raised costs battery life system-wide.  Use it
    as a context manager so the pairing survives an exception.
    """

    def __init__(self, period_ms: int = 1) -> None:
        self.period_ms = int(period_ms)
        self.active = False
        self.available = sys.platform == "win32"
        self._winmm = None

    def start(self) -> bool:
        if self.active or not self.available:
            return self.active
        try:
            import ctypes

            self._winmm = ctypes.WinDLL("winmm")
            if self._winmm.timeBeginPeriod(self.period_ms) == 0:  # TIMERR_NOERROR
                self.active = True
        except Exception:
            # Not fatal: measurements get noisier, nothing breaks.
            self.available = False
        return self.active

    def stop(self) -> None:
        if not self.active:
            return
        try:
            self._winmm.timeEndPeriod(self.period_ms)
        except Exception:
            pass
        finally:
            self.active = False

    def describe(self) -> str:
        if not self.available:
            return "scheduler timer: not applicable on this platform"
        if self.active:
            return f"scheduler timer: {self.period_ms}ms (raised)"
        return "scheduler timer: DEFAULT (~15.6ms) -- warmup timings will be unreliable"

    def __enter__(self) -> "HighResolutionTimer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
