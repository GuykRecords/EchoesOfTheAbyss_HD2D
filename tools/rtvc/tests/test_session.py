"""Live-path checks that do not need an audio device.

``AudioIO`` is driven by calling its real PortAudio callback from a feeder
thread, so the ring handoff, the underrun accounting and the worker loop are
all exercised as written -- only the device itself is replaced.
"""

import threading
import time

import numpy as np
import pytest

from rtvc.audio_io import AudioIO
from rtvc.engines import FixedCostEngine, PassthroughEngine
from rtvc.realtime import RealtimeSession, WindowProcessor

SR = 48000
B = 1536      # 32 ms
X = 384       # 8 ms
EXTRA = 4800  # 100 ms, keeps the test quick


class FakeStatus:
    """Stand-in for sounddevice's CallbackFlags."""

    def __init__(self, input_overflow=False, output_underflow=False):
        self.input_overflow = input_overflow
        self.output_underflow = output_underflow

    def __bool__(self):
        return self.input_overflow or self.output_underflow

    def __str__(self):
        return "input overflow" if self.input_overflow else "output underflow"


class FakeIO(AudioIO):
    """AudioIO with the PortAudio stream replaced by a timed feeder thread."""

    def __init__(self, io_block=128, **kwargs):
        super().__init__(sr=SR, blocksize=io_block, **kwargs)
        self._feeder = None
        self._feeding = threading.Event()
        self.captured = []

    @property
    def input_latency_ms(self):
        return 10.0

    @property
    def output_latency_ms(self):
        return 12.0

    def start(self):
        self._feeding.set()
        self._feeder = threading.Thread(target=self._feed, daemon=True)
        self._feeder.start()

    def stop(self):
        self._feeding.clear()
        if self._feeder is not None:
            self._feeder.join(timeout=2.0)
            self._feeder = None

    def _feed(self):
        period = self.blocksize / self.sr
        rng = np.random.default_rng(0)
        next_at = time.perf_counter()
        while self._feeding.is_set():
            indata = (rng.standard_normal(self.blocksize) * 0.05).astype(np.float32)
            outdata = np.zeros(self.blocksize, dtype=np.float32)
            self._callback(indata, outdata, self.blocksize, None, FakeStatus())
            self.captured.append(outdata.copy())
            next_at += period
            delay = next_at - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_at = time.perf_counter()


def make_session(engine=None, prefill_samples=0, io_block=128):
    proc = WindowProcessor(engine or PassthroughEngine(), SR, B, X, EXTRA,
                           highpass=True, gate=False, limiter=True)
    io = FakeIO(io_block=io_block)
    return RealtimeSession(proc, io, report_sec=0.25, prefill_samples=prefill_samples)


def test_session_runs_end_to_end_without_underruns():
    """The whole live loop, minus the sound card: audio in, audio out, clean."""
    session = make_session()
    session.run(duration=1.5)

    stats = session.io.stats
    assert stats.callbacks > 100, "the feeder never ran"
    assert session.proc.engine.calls > 20, "the worker produced almost nothing"
    assert stats.underflow == 0, f"{stats.underflow} underruns on a passthrough run"
    assert stats.dropped_in == 0 and stats.dropped_out == 0

    played = np.concatenate(session.io.captured)
    assert np.all(np.isfinite(played))
    assert np.max(np.abs(played)) > 0.0, "nothing was ever played"


def test_startup_silence_is_not_counted_as_an_underrun():
    """`under` has to be able to reach 0, or it is useless as a pass/fail signal.

    Before the worker has converted anything the output ring is legitimately
    empty; counting those callbacks would pin `under` above zero forever.
    """
    io = FakeIO()
    assert io.counting is False
    out = np.zeros(io.blocksize, dtype=np.float32)
    for _ in range(50):
        io._callback(np.zeros(io.blocksize, dtype=np.float32), out, io.blocksize,
                     None, FakeStatus(output_underflow=True))
    assert io.stats.underflow == 0

    io.counting = True
    io._callback(np.zeros(io.blocksize, dtype=np.float32), out, io.blocksize,
                 None, FakeStatus(output_underflow=True))
    assert io.stats.underflow > 0


def test_latency_report_adds_up_and_matches_the_configuration():
    session = make_session()
    session.run(duration=1.0)

    reports = []
    while not session.reports.empty():
        reports.append(session.reports.get())
    assert reports, "no periodic report was emitted"

    r = reports[-1]
    assert r.block_ms == pytest.approx(32.0)
    assert r.xfade_ms == pytest.approx(8.0)
    assert r.in_dev_ms == pytest.approx(10.0)
    assert r.out_dev_ms == pytest.approx(12.0)
    expected = (r.in_dev_ms + r.block_ms + r.infer_ms + r.xfade_ms
                + r.out_buf_ms + r.out_dev_ms)
    assert r.total_ms == pytest.approx(expected)
    assert "TOTAL" in r.line() and "RTF" in r.line()


def test_prefill_shows_up_as_output_buffer_latency_not_as_free_headroom():
    """Pre-roll is real latency; it must be visible in out-buf, not hidden."""
    session = make_session(prefill_samples=int(SR * 0.020))  # 20 ms
    session.run(duration=1.0)
    reports = []
    while not session.reports.empty():
        reports.append(session.reports.get())
    assert reports
    assert max(r.out_buf_ms for r in reports) >= 15.0


def test_a_too_slow_engine_shows_up_as_underruns_not_as_silence():
    """An engine slower than realtime must be visible in the counters."""
    session = make_session(engine=FixedCostEngine(cost_ms=80.0))  # RTF 2.5
    session.run(duration=1.5)
    stats = session.io.stats
    assert stats.underflow > 0 or stats.dropped_in > 0, (
        "an engine at RTF 2.5 produced no under/drop; the counters are not wired up"
    )


def test_worker_stops_promptly_when_the_session_stops():
    session = make_session()
    session.run(duration=0.5)
    assert session._worker is None
    time.sleep(0.05)
    calls = session.proc.engine.calls
    time.sleep(0.15)
    assert session.proc.engine.calls == calls, "worker kept running after stop()"


# --------------------------------------------------------------------------
# A stream that opens and then does nothing
# --------------------------------------------------------------------------


class DeadIO(FakeIO):
    """Opens successfully and never calls back -- how exclusive mode fails."""

    def _feed(self):
        while self._feeding.is_set():
            time.sleep(0.01)


class StallingIO(FakeIO):
    """Runs briefly, then stops calling back with no error."""

    def __init__(self, stall_after=40, **kwargs):
        super().__init__(**kwargs)
        self.stall_after = stall_after

    def _feed(self):
        period = self.blocksize / self.sr
        rng = np.random.default_rng(0)
        while self._feeding.is_set():
            if self.stats.callbacks < self.stall_after:
                indata = (rng.standard_normal(self.blocksize) * 0.05).astype(np.float32)
                outdata = np.zeros(self.blocksize, dtype=np.float32)
                self._callback(indata, outdata, self.blocksize, None, FakeStatus())
            time.sleep(period)


def test_a_silent_stream_is_an_error_not_a_perfect_score():
    """The most dangerous failure: every counter reads 0, which looks flawless."""
    from rtvc.realtime import StreamDead

    proc = WindowProcessor(PassthroughEngine(), SR, B, X, EXTRA)
    session = RealtimeSession(proc, DeadIO(), report_sec=0.25)
    with pytest.raises(StreamDead, match="no audio"):
        session.run(duration=3.0)
    assert session.io.stats.callbacks == 0
    assert session.io.stats.underflow == 0, (
        "counters stayed at zero -- exactly why the callback count has to be checked"
    )


def test_a_stream_that_stops_midway_is_caught():
    from rtvc.realtime import StreamDead

    proc = WindowProcessor(PassthroughEngine(), SR, B, X, EXTRA)
    session = RealtimeSession(proc, StallingIO(stall_after=40), report_sec=0.3)
    with pytest.raises(StreamDead, match="stopped calling back"):
        session.run(duration=5.0)


def test_high_resolution_timer_is_safe_to_use_anywhere():
    from rtvc.timing import HighResolutionTimer

    with HighResolutionTimer() as clock:
        described = clock.describe()
    assert not clock.active, "timeEndPeriod must be paired with timeBeginPeriod"
    assert "scheduler timer" in described
    clock.stop()  # idempotent
