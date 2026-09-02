"""The RVC glue, exercised without torch, RVC or a GPU.

Only ``RVCBackend.__init__`` needs the ML stack, and it imports it lazily, so
everything around it -- unit conversion, model discovery, the CLI's failure
path -- is testable in CI.
"""

import numpy as np
import pytest

from rtvc.engines import RVCTorchEngine
from rtvc.rvc_backend import (
    ZC_SAMPLES_16K,
    build_rvc_engine,
    find_first_model,
    to_zc_units,
)


def test_the_module_imports_without_torch_or_rvc():
    """CI has neither; the import must still be clean."""
    import rtvc.rvc_backend as backend

    assert backend.ZC_SAMPLES_16K == 160


@pytest.mark.parametrize("samples,expected", [(160, 1), (480, 3), (1600, 10), (0, 0)])
def test_16k_samples_convert_to_rvc_10ms_units(samples, expected):
    """The engine speaks samples, RVC speaks 10ms units; the seam is here."""
    assert to_zc_units(samples, "x") == expected


def test_a_length_off_the_10ms_grid_is_refused_rather_than_rounded():
    """Rounding here is what makes f0 drift a fraction of a frame every block."""
    with pytest.raises(ValueError, match="割り切れない"):
        to_zc_units(512, "return_length")


def test_the_documented_window_lands_on_whole_units():
    # B=30 X=10 EXTRA=100 at 48k -> tail 40ms, skip 100ms
    tail_16k = int(round(0.040 * 16000))
    skip_16k = int(round(0.100 * 16000))
    assert tail_16k % ZC_SAMPLES_16K == 0 and skip_16k % ZC_SAMPLES_16K == 0
    assert to_zc_units(tail_16k, "return_length") == 4
    assert to_zc_units(skip_16k, "skip_head") == 10


# --------------------------------------------------------------------------
# Model discovery
# --------------------------------------------------------------------------


def test_no_weights_directory_is_not_an_error_just_no_model(tmp_path):
    assert find_first_model(str(tmp_path)) == (None, "")


def test_the_matching_index_is_found_next_to_the_model(tmp_path):
    weights = tmp_path / "assets" / "weights"
    indices = tmp_path / "assets" / "indices"
    weights.mkdir(parents=True)
    indices.mkdir(parents=True)
    (weights / "myvoice.pth").write_bytes(b"")
    (indices / "added_myvoice_v2.index").write_bytes(b"")
    (indices / "added_someone_else.index").write_bytes(b"")

    pth, index = find_first_model(str(tmp_path))
    assert pth.endswith("myvoice.pth")
    assert index.endswith("added_myvoice_v2.index")


def test_a_model_without_an_index_is_still_usable(tmp_path):
    weights = tmp_path / "assets" / "weights"
    weights.mkdir(parents=True)
    (weights / "myvoice.pth").write_bytes(b"")
    pth, index = find_first_model(str(tmp_path))
    assert pth.endswith("myvoice.pth") and index == ""


class Args:
    rvc_root = None
    rvc_model = None
    rvc_index = None
    rvc_index_rate = 0.0
    rvc_key = 0
    rvc_formant = 0.0
    f0_method = "rmvpe"


def test_a_missing_speaker_model_says_what_to_do(tmp_path):
    """The expected state today: assets/weights is empty until a voice is trained."""
    args = Args()
    args.rvc_root = str(tmp_path)
    with pytest.raises(FileNotFoundError, match="自分の声で学習"):
        build_rvc_engine(args, 48000, 1440, 480, 4800)


# --------------------------------------------------------------------------
# The tail must keep its end
# --------------------------------------------------------------------------


@pytest.mark.parametrize("extra_samples", [-2, -1, 0, 1, 2])
def test_an_off_by_a_sample_conversion_keeps_the_end_of_the_tail(extra_samples):
    """A tail's last sample is 'now'.

    Trimming or padding the far end instead slides the audio by a sample or
    two every block, which is a click at the block rate rather than an
    off-by-one nobody hears.
    """
    tail = 1920  # 40 ms at 48k

    def infer(wav16k, skip_head, return_length):
        # Return the ramp the caller expects, off by a sample or two.
        n = int(round(return_length * 48000 / 16000)) + extra_samples
        return np.arange(n, dtype=np.float32)

    eng = RVCTorchEngine(infer, stream_sr=48000, model_sr=48000,
                         block_ms=30.0, crossfade_ms=10.0, extra_ms=100.0)
    out = eng.convert(np.zeros(6720, dtype=np.float32), 48000, tail=tail)

    assert out.size == tail
    if extra_samples >= 0:
        # the final sample of what the model produced must survive
        assert out[-1] == pytest.approx(tail + extra_samples - 1)
    else:
        assert out[-1] == pytest.approx(tail + extra_samples - 1)


# --------------------------------------------------------------------------
# RVC's Config parses sys.argv, and would eat ours
# --------------------------------------------------------------------------


def test_third_party_argument_parsing_cannot_see_our_command_line():
    """RVC's Config() runs its own argparse over sys.argv.

    As a standalone app that is correct; called as a library it read
    `--engine rvc --host-api WASAPI ...`, found them unrecognised, and killed
    the process before a single model was loaded.
    """
    import argparse
    import sys

    from rtvc.rvc_backend import _own_argv_only

    original = ["realtime.py", "--engine", "rvc", "--host-api", "WASAPI"]
    sys.argv = list(original)
    try:
        with _own_argv_only():
            # Stand-in for RVC's Config.__init__
            parser = argparse.ArgumentParser()
            parser.add_argument("--port", type=int, default=7865)
            parser.add_argument("--dml", action="store_true")
            args = parser.parse_args()      # would SystemExit without the guard
            assert args.port == 7865
        assert sys.argv == original, "argv must be put back"
    finally:
        sys.argv = ["pytest"]


def test_argv_is_restored_even_when_the_body_raises():
    import sys

    from rtvc.rvc_backend import _own_argv_only

    original = ["realtime.py", "--engine", "rvc"]
    sys.argv = list(original)
    try:
        with pytest.raises(ValueError):
            with _own_argv_only():
                raise ValueError("boom")
        assert sys.argv == original
    finally:
        sys.argv = ["pytest"]
