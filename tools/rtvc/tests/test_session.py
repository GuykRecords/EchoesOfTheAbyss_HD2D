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
    """The whole live loop, minus the sound card: audio in, audio out, clean.

    Run with a cushion, because that is the configuration this is meant to
    prove.  With no pre-roll the output ring legitimately reaches empty every
    cycle, so a single scheduler hiccup produces an underrun -- that is the
    documented cost of --prefill-ms 0, not a defect to assert against.
    """
    duration = 1.5
    session = make_session(prefill_samples=int(SR * 0.008))
    session.run(duration=duration)

    stats = session.io.stats
    assert stats.callbacks > 100, "the feeder never ran"
    # Blocks are 32ms, so a 1.5s run is ~46 of them. Allow generous slack for a
    # loaded machine, but not so much that a feeder running at a fraction of
    # realtime would pass.
    expected = duration * 1000.0 / session.proc.block_ms
    assert session.proc.engine.calls > expected * 0.6, (
        f"the worker produced {session.proc.engine.calls} blocks, expected "
        f"around {expected:.0f}"
    )
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


# --------------------------------------------------------------------------
# Device pairing: catch the mistake before PortAudio does
# --------------------------------------------------------------------------


class FakeSd:
    """Just enough of sounddevice to answer device queries."""

    def __init__(self, devices, apis):
        self._devices = devices
        self._apis = apis

    def query_devices(self, index):
        return self._devices[index]

    def query_hostapis(self, index):
        return self._apis[index]


def _device(name, hostapi, ins, outs, samplerate=48000.0):
    return {"name": name, "hostapi": hostapi,
            "max_input_channels": ins, "max_output_channels": outs,
            "default_samplerate": samplerate}


def _paired_io(in_device, out_device, channels=1):
    io = AudioIO(sr=SR, blocksize=128, in_device=in_device,
                 out_device=out_device, channels=channels)
    io._sd = FakeSd(
        devices={
            0: _device("Realtek mic", 0, 2, 0),        # WASAPI capture
            1: _device("CABLE Input", 0, 0, 2),        # WASAPI playback
            2: _device("Realtek mic (KS)", 1, 2, 0),   # a different host API
            3: _device("mono headset mic", 0, 1, 0),
        },
        apis={0: {"name": "Windows WASAPI"}, 1: {"name": "Windows WDM-KS"}},
    )
    return io


def test_devices_on_different_host_apis_are_rejected_with_the_reason():
    """PortAudio calls this 'Illegal combination of I/O devices'. Say what it is."""
    io = _paired_io(2, 1)
    with pytest.raises(RuntimeError, match="different host APIs"):
        io._check_pairing()


def test_a_playback_device_used_as_input_is_named_as_such():
    io = _paired_io(1, 1)
    with pytest.raises(RuntimeError, match="no input channels"):
        io._check_pairing()


def test_a_capture_device_used_as_output_is_named_as_such():
    io = _paired_io(0, 0)
    with pytest.raises(RuntimeError, match="no output channels"):
        io._check_pairing()


def test_asking_for_more_channels_than_the_device_has_is_rejected():
    io = _paired_io(3, 1, channels=2)
    with pytest.raises(RuntimeError, match="input channel"):
        io._check_pairing()


def test_a_valid_pairing_passes():
    _paired_io(0, 1)._check_pairing()


def test_describe_devices_names_both_endpoints_and_their_apis():
    text = _paired_io(0, 1).describe_devices()
    assert "Realtek mic" in text and "CABLE Input" in text
    assert "Windows WASAPI" in text


# --------------------------------------------------------------------------
# Naming a device instead of numbering it
# --------------------------------------------------------------------------


class FakeSdWithList(FakeSd):
    def __init__(self, devices, apis):
        super().__init__({i: d for i, d in enumerate(devices)}, apis)
        self._list = devices

    def query_devices(self, index=None):
        return self._list if index is None else self._list[index]

    def query_hostapis(self, index=None):
        if index is None:
            return [self._apis[i] for i in sorted(self._apis)]
        return self._apis[index]


@pytest.fixture
def stub_devices(monkeypatch):
    """A machine that looks like the real one: same names across host APIs."""
    devices = [
        _device("マイク (Realtek(R) Audio)", 0, 2, 0),          # 0 MME
        _device("CABLE Input (VB-Audio Virtual Cable)", 0, 0, 16),  # 1 MME
        _device("マイク (Realtek(R) Audio)", 1, 2, 0),          # 2 WASAPI
        _device("CABLE In 16ch (VB-Audio Virtual Cable)", 1, 0, 2),  # 3 WASAPI
        _device("CABLE Input (VB-Audio Virtual Cable)", 1, 0, 2),    # 4 WASAPI
        _device("Output (VB-Audio Point)", 2, 0, 16),           # 5 WDM-KS
    ]
    apis = {0: {"name": "MME"}, 1: {"name": "Windows WASAPI"},
            2: {"name": "Windows WDM-KS"}}
    fake = FakeSdWithList(devices, apis)
    monkeypatch.setattr("rtvc.audio_io.require_sounddevice", lambda: fake)
    return fake


def test_a_name_plus_host_api_picks_exactly_one_device(stub_devices):
    """The stable way to name a device: substring + host API."""
    from rtvc.audio_io import resolve_device

    assert resolve_device("Realtek", "input", "WASAPI") == 2
    assert resolve_device("CABLE Input", "output", "WASAPI") == 4


def test_the_same_name_without_a_host_api_is_ambiguous_and_says_so(stub_devices):
    from rtvc.audio_io import resolve_device

    with pytest.raises(ValueError, match="matches several"):
        resolve_device("Realtek", "input")


def test_an_unknown_name_lists_what_is_actually_available(stub_devices):
    from rtvc.audio_io import resolve_device

    with pytest.raises(ValueError) as excinfo:
        resolve_device("Shure", "input", "WASAPI")
    assert "Realtek" in str(excinfo.value), "the error should show the real candidates"


def test_an_index_pointing_at_the_wrong_host_api_is_caught(stub_devices):
    """Exactly the failure a sleeping wireless headset causes: indices shift."""
    from rtvc.audio_io import resolve_device

    with pytest.raises(ValueError, match="not on host API"):
        resolve_device("5", "output", "WASAPI")


def test_an_index_pointing_at_a_playback_device_used_as_input_is_caught(stub_devices):
    from rtvc.audio_io import resolve_device

    with pytest.raises(ValueError, match="no input channels"):
        resolve_device("4", "input")


def test_an_out_of_range_index_is_reported_plainly(stub_devices):
    from rtvc.audio_io import resolve_device

    with pytest.raises(ValueError, match="does not exist"):
        resolve_device("99", "input")


def test_device_table_can_be_narrowed_to_one_host_api(stub_devices):
    from rtvc.audio_io import format_device_table

    text = format_device_table("WASAPI")
    assert "Windows WASAPI" in text
    assert "Windows WDM-KS" not in text
    assert "Prefer names" in text


def test_out_buf_is_not_measured_before_the_cushion_is_down():
    """The first report used to read 0.00 because it spanned the startup gap."""
    io = FakeIO()
    out = np.zeros(io.blocksize, dtype=np.float32)
    for _ in range(20):
        io._callback(np.zeros(io.blocksize, dtype=np.float32), out, io.blocksize,
                     None, FakeStatus())
    assert io.take_out_min_fill() == 0, "nothing measured yet is reported as 0"

    io.counting = True
    io.out_ring.write(np.ones(2000, dtype=np.float32))
    io._callback(np.zeros(io.blocksize, dtype=np.float32), out, io.blocksize,
                 None, FakeStatus())
    assert io.take_out_min_fill() > 0, "steady-state occupancy must be reported"


def test_a_live_take_can_be_saved_and_replayed_offline(tmp_path):
    """Comparing settings by ear needs the same performance every time."""
    from scipy.io import wavfile

    from rtvc.realtime import run_offline

    wav = tmp_path / "take.wav"
    proc = WindowProcessor(PassthroughEngine(), SR, B, X, EXTRA)
    session = RealtimeSession(proc, FakeIO(), report_sec=0.5,
                              prefill_samples=int(SR * 0.008),
                              record_in=str(wav))
    session.run(duration=1.0)

    assert wav.exists()
    sr, captured = wavfile.read(wav)
    assert sr == SR
    assert captured.ndim == 1
    assert captured.size >= B * 10, "most of the take should have been kept"

    # The saved take drives an offline run of the same length.
    replay = WindowProcessor(PassthroughEngine(), SR, B, X, EXTRA)
    out = run_offline(replay, captured.astype(np.float32), verbose=False)
    assert out.size == (captured.size // B) * B


def test_nothing_is_recorded_unless_asked(tmp_path):
    proc = WindowProcessor(PassthroughEngine(), SR, B, X, EXTRA)
    session = RealtimeSession(proc, FakeIO(), report_sec=0.5)
    session.run(duration=0.6)
    assert session.save_recording() is None
    assert not list(tmp_path.iterdir())
