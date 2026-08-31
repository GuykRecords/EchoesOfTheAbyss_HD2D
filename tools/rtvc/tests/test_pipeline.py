"""Acceptance checks for the ring buffer, the engines and the window loop.

The centrepiece is :func:`test_passthrough_output_matches_the_exact_oracle`.
With a passthrough engine the whole window/crossfade machine has a closed-form
answer, so the loop is checked against arithmetic rather than against "it
sounded fine".
"""

import threading

import numpy as np
import pytest

from rtvc.audio_io import RingBuffer, require_sounddevice
from rtvc.dsp import equal_power_windows
from rtvc.engines import (
    BaseEngine,
    CallableEngine,
    FixedCostEngine,
    PassthroughEngine,
    RVCTorchEngine,
    build_engine,
)
from rtvc.realtime import WindowProcessor, build_parser, main, run_offline

SR = 48000
B = 1536      # 32 ms
X = 384       # 8 ms
EXTRA = 24000  # 500 ms


# --------------------------------------------------------------------------
# RingBuffer
# --------------------------------------------------------------------------


def test_ring_buffer_is_fifo():
    rb = RingBuffer(16)
    rb.write(np.arange(10, dtype=np.float32))
    assert np.array_equal(rb.read(4), np.arange(4, dtype=np.float32))
    assert np.array_equal(rb.read(6), np.arange(4, 10, dtype=np.float32))
    assert rb.available == 0


def test_ring_buffer_read_returns_none_when_short():
    rb = RingBuffer(16)
    rb.write(np.arange(3, dtype=np.float32))
    assert rb.read(4) is None
    assert rb.available == 3, "a failed read must not consume anything"


def test_ring_buffer_drops_oldest_on_overflow():
    """Newest audio wins: dropping new samples would stall behind a backlog."""
    rb = RingBuffer(8)
    rb.write(np.arange(8, dtype=np.float32))
    dropped = rb.write(np.arange(100, 104, dtype=np.float32))
    assert dropped == 4
    assert rb.dropped == 4
    assert np.array_equal(rb.read(8),
                          np.array([4, 5, 6, 7, 100, 101, 102, 103], dtype=np.float32))


def test_ring_buffer_write_larger_than_capacity_keeps_the_tail():
    rb = RingBuffer(4)
    dropped = rb.write(np.arange(10, dtype=np.float32))
    assert dropped == 6
    assert np.array_equal(rb.read(4), np.array([6, 7, 8, 9], dtype=np.float32))


def test_ring_buffer_wraps_correctly():
    rb = RingBuffer(8)
    for i in range(20):
        rb.write(np.array([i], dtype=np.float32))
        got = rb.read(1)
        assert got is not None and got[0] == i


def test_ring_buffer_survives_concurrent_producer_and_consumer():
    """No torn reads, no reordering, no duplicated samples under contention."""
    rb = RingBuffer(4096)
    total = 120_000
    produced = np.arange(total, dtype=np.float32)
    consumed = []
    errors = []
    done = threading.Event()

    def produce():
        try:
            for i in range(0, total, 128):
                rb.write(produced[i:i + 128])
        except Exception as exc:  # pragma: no cover
            errors.append(exc)
        finally:
            done.set()

    def consume():
        try:
            while not (done.is_set() and rb.available < 64):
                chunk = rb.read(64)
                if chunk is not None:
                    consumed.append(chunk)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=produce), threading.Thread(target=consume)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads), "producer/consumer deadlocked"

    assert not errors
    flat = np.concatenate(consumed)
    # Samples may be missing (dropped on overflow) but never out of order.
    assert np.all(np.diff(flat) > 0), "ring buffer reordered or duplicated samples"


# --------------------------------------------------------------------------
# Engines
# --------------------------------------------------------------------------


def test_passthrough_engine_is_transparent():
    eng = PassthroughEngine()
    x = np.random.default_rng(0).standard_normal(1024).astype(np.float32)
    assert np.array_equal(eng.convert(x, SR), x)


def test_fixed_cost_engine_costs_what_it_claims():
    eng = FixedCostEngine(cost_ms=10.0)
    x = np.zeros(1024, dtype=np.float32)
    for _ in range(6):
        eng.convert(x, SR)
    assert 9.0 <= eng.infer_ms_ema <= 25.0, f"measured {eng.infer_ms_ema}ms for a 10ms budget"


def test_engine_tracks_peak_separately_from_the_average():
    """Dropouts are caused by the peak, so it must not be smoothed away."""
    costs = iter([0.0, 0.0, 0.03, 0.0, 0.0])

    def fn(window, sr):
        import time
        time.sleep(next(costs))
        return window

    eng = CallableEngine(fn)
    for _ in range(5):
        eng.convert(np.zeros(64, dtype=np.float32), SR)
    assert eng.infer_ms_max > eng.infer_ms_ema * 2


def test_engine_rejects_a_length_changing_conversion():
    eng = CallableEngine(lambda w, sr: w[:-1])
    with pytest.raises(ValueError, match="length preserving"):
        eng.convert(np.zeros(128, dtype=np.float32), SR)


def test_warmup_marks_the_engine_and_discards_its_own_timings():
    eng = FixedCostEngine(cost_ms=1.0)
    eng.warmup(1024, SR, iters=3)
    assert eng.is_warm
    assert eng.calls == 0, "warmup timings must not pollute the reported average"
    assert eng.infer_ms_ema == 0.0


def test_rtf_is_inference_over_block_duration():
    eng = CallableEngine(lambda w, sr: w)
    eng.infer_ms_ema = 16.0
    assert eng.rtf(32.0) == pytest.approx(0.5)


def test_build_engine_rejects_unknown_and_unconfigured():
    with pytest.raises(ValueError):
        build_engine("nope", SR)
    with pytest.raises(ValueError, match="infer_fn"):
        build_engine("rvc", SR)


def test_rvc_engine_keeps_the_length_contract_across_sample_rates():
    """The adapter owns 48k -> 16k in and 40k -> 48k out; length must survive."""
    seen = {}

    def fake_infer(wav16k):
        seen["in_len"] = wav16k.size
        # a real model returns model_sr audio of the equivalent duration
        return np.zeros(int(round(wav16k.size * 40000 / 16000)), dtype=np.float32)

    eng = RVCTorchEngine(fake_infer, stream_sr=48000, model_sr=40000)
    window = np.zeros(EXTRA + X + B, dtype=np.float32)
    out = eng.convert(window, 48000)
    assert out.size == window.size
    assert abs(seen["in_len"] - window.size / 3) <= 2


def test_rvc_engine_requires_a_callable():
    with pytest.raises(TypeError):
        RVCTorchEngine(None, stream_sr=48000)


def test_base_engine_is_abstract():
    with pytest.raises(NotImplementedError):
        BaseEngine().convert(np.zeros(8, dtype=np.float32), SR)


# --------------------------------------------------------------------------
# WindowProcessor
# --------------------------------------------------------------------------


def make_processor(**kwargs):
    params = dict(highpass=False, gate=False, limiter=False)
    params.update(kwargs)
    return WindowProcessor(PassthroughEngine(), SR, B, X, EXTRA, **params)


def test_algorithmic_latency_is_block_plus_crossfade_only():
    """EXTRA must never appear in the latency figure -- it is past audio."""
    proc = make_processor()
    assert proc.algorithmic_latency_ms == pytest.approx(40.0)
    big_context = WindowProcessor(PassthroughEngine(), SR, B, X, EXTRA * 4,
                                  highpass=False, gate=False, limiter=False)
    assert big_context.algorithmic_latency_ms == proc.algorithmic_latency_ms
    assert big_context.window_len > proc.window_len


def test_each_block_in_yields_exactly_one_block_out():
    proc = make_processor()
    rng = np.random.default_rng(1)
    for _ in range(20):
        out = proc.process_block(rng.standard_normal(B).astype(np.float32) * 0.1)
        assert out.shape == (B,)
        assert np.all(np.isfinite(out))


def test_process_block_rejects_a_wrong_sized_block():
    proc = make_processor()
    with pytest.raises(ValueError):
        proc.process_block(np.zeros(B - 1, dtype=np.float32))


def test_passthrough_output_matches_the_exact_oracle():
    """Closed-form check of the window + overlap + crossfade arithmetic.

    With a passthrough engine the loop must reproduce the input delayed by X,
    with the equal-power pair applied to the X samples straddling each block
    boundary.  Any error in the window layout, the tail slice or the overlap
    bookkeeping breaks this identity.
    """
    proc = make_processor()
    rng = np.random.default_rng(7)
    n_blocks = 12
    signal = rng.standard_normal(n_blocks * B).astype(np.float32) * 0.2

    out = np.concatenate([proc.process_block(signal[k * B:(k + 1) * B])
                          for k in range(n_blocks)])

    fade_out, fade_in = equal_power_windows(X)
    gain = np.ones(B, dtype=np.float32)
    gain[:X] = fade_out + fade_in

    expected = np.empty_like(out)
    expected[:X] = 0.0                       # first block has no glue behind it
    expected[X:] = signal[:-X]
    expected *= np.tile(gain, n_blocks)

    # Skip block 0: it is the documented startup transient (no previous glue).
    assert np.allclose(out[B:], expected[B:], atol=1e-6)


def test_zero_crossfade_is_a_sample_exact_passthrough():
    proc = WindowProcessor(PassthroughEngine(), SR, B, 0, EXTRA,
                           highpass=False, gate=False, limiter=False)
    rng = np.random.default_rng(3)
    for _ in range(5):
        block = rng.standard_normal(B).astype(np.float32) * 0.2
        assert np.array_equal(proc.process_block(block), block)


def test_crossfade_longer_than_block_is_rejected():
    with pytest.raises(ValueError):
        WindowProcessor(PassthroughEngine(), SR, B, B + 1, EXTRA)


def test_reset_returns_the_processor_to_its_initial_state():
    proc = make_processor()
    rng = np.random.default_rng(11)
    blocks = [rng.standard_normal(B).astype(np.float32) * 0.2 for _ in range(4)]
    first = [proc.process_block(b) for b in blocks]
    proc.reset()
    second = [proc.process_block(b) for b in blocks]
    for a, b in zip(first, second):
        assert np.array_equal(a, b)


def test_limiter_stage_bounds_a_hot_signal():
    proc = WindowProcessor(PassthroughEngine(), SR, B, X, EXTRA,
                           highpass=False, gate=False, limiter=True)
    hot = np.full(B, 4.0, dtype=np.float32)
    for _ in range(3):
        out = proc.process_block(hot)
    assert np.max(np.abs(out)) < 1.0


def test_full_dsp_chain_gates_silence_and_stays_finite():
    proc = WindowProcessor(PassthroughEngine(), SR, B, X, EXTRA,
                           highpass=True, gate=True, limiter=True)
    quiet = (1e-4 * np.random.default_rng(0).standard_normal(B * 20)).astype(np.float32)
    out = np.concatenate([proc.process_block(quiet[k * B:(k + 1) * B]) for k in range(20)])
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out[B * 5:])) < 1e-5, "the gate should have shut on a noise floor"


# --------------------------------------------------------------------------
# Offline harness and CLI
# --------------------------------------------------------------------------


def test_offline_run_produces_finite_audio_of_the_right_length():
    proc = WindowProcessor(PassthroughEngine(), SR, B, X, EXTRA)
    signal = np.zeros(B * 10, dtype=np.float32)
    out = run_offline(proc, signal, verbose=False)
    assert out.size == B * 10
    assert np.all(np.isfinite(out))


def test_cli_offline_end_to_end(tmp_path, capsys):
    wav = tmp_path / "out.wav"
    code = main([
        "--offline", "--offline-seconds", "1.0", "--engine", "passthrough",
        "--sr", "48000", "--block-ms", "32", "--crossfade-ms", "8",
        "--warmup", "1", "--offline-out", str(wav),
    ])
    assert code == 0
    assert wav.exists()
    text = capsys.readouterr().out
    assert "algorithmic latency (B+X) 40.00ms" in text


def test_cli_rejects_crossfade_longer_than_block(capsys):
    code = main(["--offline", "--block-ms", "10", "--crossfade-ms", "20"])
    assert code == 2
    assert "crossfade-ms" in capsys.readouterr().err


def test_cli_parser_defaults_match_the_documented_baseline():
    args = build_parser().parse_args([])
    assert (args.sr, args.io_block, args.block_ms, args.crossfade_ms) == (48000, 128, 32.0, 8.0)
    assert args.extra_ms == 500.0
    assert args.engine == "passthrough"


def test_sounddevice_absence_is_reported_clearly():
    """The DSP half of the package must stay usable with no audio stack."""
    try:
        require_sounddevice()
    except RuntimeError as exc:
        assert "--offline" in str(exc)
