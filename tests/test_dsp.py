"""Acceptance checks for the DSP primitives.

Each test states a property that must hold for the realtime path to be sound,
so a failure here names the defect rather than "audio sounds wrong".
"""

import numpy as np
import pytest

from rtvc.dsp import (
    HighPass,
    NoiseGate,
    Resampler,
    SoftLimiter,
    StreamResampler,
    amp_to_db,
    db_to_amp,
    equal_power_windows,
    rms,
)

SR = 48000


def sine(freq, seconds=1.0, sr=SR, amp=0.5):
    t = np.arange(int(sr * seconds), dtype=np.float64) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --------------------------------------------------------------------------
# HighPass
# --------------------------------------------------------------------------


def test_highpass_block_processing_equals_whole_stream():
    """State must carry across blocks, or every boundary is a click."""
    x = sine(220.0, 0.5) + sine(30.0, 0.5)
    whole = HighPass(SR).process(x)

    streamed = HighPass(SR)
    blocks = [streamed.process(x[i:i + 1536]) for i in range(0, x.size, 1536)]
    assert np.allclose(whole, np.concatenate(blocks), atol=1e-6)


def test_highpass_attenuates_rumble_and_passes_voice():
    hp_low = HighPass(SR)
    hp_high = HighPass(SR)
    low = hp_low.process(sine(30.0, 0.5))
    high = hp_high.process(sine(1000.0, 0.5))
    # ignore the settling transient at the very start
    assert amp_to_db(rms(low[SR // 10:])) < amp_to_db(rms(sine(30.0, 0.5))) - 15.0
    assert abs(amp_to_db(rms(high[SR // 10:])) - amp_to_db(rms(sine(1000.0, 0.5)))) < 0.5


def test_highpass_reset_clears_state():
    hp = HighPass(SR)
    hp.process(sine(1000.0, 0.1))
    hp.reset()
    a = hp.process(sine(1000.0, 0.1))
    hp.reset()
    b = hp.process(sine(1000.0, 0.1))
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# NoiseGate
# --------------------------------------------------------------------------


def test_gate_closes_on_noise_floor():
    gate = NoiseGate(SR)
    quiet = (1e-4 * np.random.default_rng(0).standard_normal(SR)).astype(np.float32)
    out = gate.process(quiet)
    assert not gate.is_open
    assert rms(out[SR // 2:]) < rms(quiet) * 0.05


def test_gate_opens_on_speech_level():
    gate = NoiseGate(SR)
    out = gate.process(sine(200.0, 0.5, amp=0.2))
    assert gate.is_open
    # after the attack ramp the signal passes essentially untouched
    assert rms(out[SR // 4:]) > rms(sine(200.0, 0.5, amp=0.2)) * 0.9


def test_gate_hold_keeps_it_open_through_a_short_pause():
    """A 180 ms hold must survive a 100 ms gap and close after a 400 ms one."""
    gate = NoiseGate(SR, hold_ms=180.0)
    gate.process(sine(200.0, 0.3, amp=0.2))
    assert gate.is_open
    gate.process(np.zeros(int(SR * 0.10), dtype=np.float32))
    assert gate.is_open, "gate closed inside the hold window"
    gate.process(np.zeros(int(SR * 0.40), dtype=np.float32))
    assert not gate.is_open, "gate never closed after the hold expired"


def test_gate_gain_is_smoothed_not_stepped():
    """The 0/1 decision must reach the audio through a ramp, never a step."""
    gate = NoiseGate(SR)
    loud = sine(200.0, 0.2, amp=0.2)
    signal = np.concatenate((np.zeros(int(SR * 0.2), dtype=np.float32), loud))
    out = gate.process(signal)
    # No sample-to-sample jump larger than the source itself can produce.
    assert np.max(np.abs(np.diff(out))) <= np.max(np.abs(np.diff(signal))) + 1e-4


def test_gate_hysteresis_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        NoiseGate(SR, open_db=-50.0, close_db=-42.0)


# --------------------------------------------------------------------------
# Crossfade windows
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 8, 384, 1536])
def test_equal_power_windows_sum_of_squares_is_one(n):
    fo, fi = equal_power_windows(n)
    assert fo.shape == fi.shape == (n,)
    assert np.allclose(fo.astype(np.float64) ** 2 + fi.astype(np.float64) ** 2, 1.0, atol=1e-6)


def test_equal_power_windows_are_monotonic_and_mirrored():
    fo, fi = equal_power_windows(256)
    assert np.all(np.diff(fo) < 0)
    assert np.all(np.diff(fi) > 0)
    assert np.allclose(fo, fi[::-1], atol=1e-6)


def test_equal_power_windows_empty():
    fo, fi = equal_power_windows(0)
    assert fo.size == 0 and fi.size == 0


# --------------------------------------------------------------------------
# SoftLimiter
# --------------------------------------------------------------------------


def test_soft_limiter_never_exceeds_ceiling():
    lim = SoftLimiter(-1.0)
    hot = (np.linspace(-8.0, 8.0, 4096)).astype(np.float32)
    out = lim.process(hot)
    # tanh is asymptotic at the ceiling, so equality is possible in float32;
    # what must never happen is going past it (or past full scale).
    assert np.max(np.abs(out)) <= db_to_amp(-1.0) * (1.0 + 1e-6)
    assert np.max(np.abs(out)) < 1.0


def test_soft_limiter_is_monotonic():
    lim = SoftLimiter(-1.0)
    out = lim.process(np.linspace(-4.0, 4.0, 2048).astype(np.float32))
    assert np.all(np.diff(out) > 0), "a limiter that folds back would distort badly"


def test_soft_limiter_is_near_transparent_at_low_level():
    lim = SoftLimiter(-1.0)
    quiet = sine(440.0, 0.1, amp=0.02)
    out = lim.process(quiet)
    assert np.max(np.abs(out - quiet)) < 1e-3


# --------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["soxr", "scipy"])
@pytest.mark.parametrize("sr_in,sr_out", [(48000, 16000), (16000, 48000), (48000, 40000)])
def test_resampler_length_and_content(backend, sr_in, sr_out):
    rs = Resampler(sr_in, sr_out, backend=backend)
    x = sine(440.0, 0.5, sr=sr_in)
    y = rs.process(x)
    assert abs(y.size - x.size * sr_out / sr_in) <= 2
    # energy is preserved to within a fraction of a dB
    assert abs(amp_to_db(rms(y)) - amp_to_db(rms(x))) < 0.5


def test_resampler_identity_is_a_noop():
    rs = Resampler(48000, 48000)
    x = sine(440.0, 0.1)
    assert rs.is_identity
    assert np.array_equal(rs.process(x), x)


def test_stream_resampler_output_length_does_not_drift():
    """Block-wise output must track the ideal rate exactly over many blocks."""
    srs = StreamResampler(48000, 16000)
    block = 1536
    total = 0
    for _ in range(300):
        total += srs.process(np.zeros(block, dtype=np.float32)).size
    expected = 300 * block * 16000 / 48000
    assert abs(total - expected) <= 2, f"drifted to {total}, expected {expected}"


def _seam_roughness(y, skip=3000):
    """Largest sample-to-sample step, normalised by a clean tone's own step.

    A boundary click shows up here and nowhere else: the tone itself has a
    known maximum slope, so a ratio near 1.0 means the seams are invisible.
    """
    ideal_step = 0.5 * 2 * np.pi * 440.0 / 16000.0
    return float(np.max(np.abs(np.diff(y[skip:]))) / ideal_step)


def test_stream_resampler_lookahead_removes_block_boundary_clicks():
    """With glue on both sides, the seams must be as smooth as the tone itself."""
    x = sine(440.0, 1.0)
    block = 1536

    srs = StreamResampler(48000, 16000, pad_ms=10.0, lookahead=True)
    streamed = np.concatenate([srs.process(x[i:i + block])
                               for i in range(0, x.size, block)])

    naive_rs = Resampler(48000, 16000)
    naive = np.concatenate([naive_rs.process(x[i:i + block])
                            for i in range(0, x.size, block)])

    assert _seam_roughness(streamed) < 1.05, "block boundaries are still audible"
    assert _seam_roughness(naive) > 2.0, (
        "the naive per-block resample was expected to be clearly worse; "
        "if it is not, this test no longer proves anything"
    )


def test_stream_resampler_without_lookahead_is_still_better_than_naive():
    """Zero added latency mode: seams taper slightly but nothing like a click."""
    x = sine(440.0, 1.0)
    block = 1536
    srs = StreamResampler(48000, 16000, pad_ms=10.0, lookahead=False)
    streamed = np.concatenate([srs.process(x[i:i + block])
                               for i in range(0, x.size, block)])
    assert srs.latency_ms == 0.0
    assert _seam_roughness(streamed) < 1.5


def test_stream_resampler_reports_its_true_latency():
    """The advertised latency must equal the measured delay, to the sample."""
    sr, block = 48000, 1536
    impulse = np.zeros(sr, dtype=np.float32)
    impulse[sr // 2] = 1.0
    ideal_out_index = (sr // 2) * 16000 // sr

    for lookahead in (True, False):
        srs = StreamResampler(48000, 16000, pad_ms=10.0, lookahead=lookahead)
        y = np.concatenate([srs.process(impulse[i:i + block])
                            for i in range(0, impulse.size, block)])
        measured_ms = (int(np.argmax(np.abs(y))) - ideal_out_index) * 1000.0 / 16000.0
        assert abs(measured_ms - srs.latency_ms) < 0.2, (
            f"lookahead={lookahead}: claims {srs.latency_ms}ms, measures {measured_ms}ms"
        )


def test_stream_resampler_identity_passes_through():
    srs = StreamResampler(48000, 48000)
    x = sine(440.0, 0.05)
    assert np.array_equal(srs.process(x), x)


def test_db_amp_roundtrip():
    for db in (-60.0, -42.0, -1.0, 0.0):
        assert abs(amp_to_db(db_to_amp(db)) - db) < 1e-6
