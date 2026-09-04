"""Turning an existing recording into training material.

The tool only ever creates files; the source is never touched. That, and the
segmentation rules, are what these tests pin down.
"""

from pathlib import Path

import check_dataset as cd
import numpy as np
import prepare_dataset as pd
import pytest
from scipy.io import wavfile

SR = 44100


def speech(seconds=60.0, sr=SR, utterance=5.0, gap=0.8, seed=0):
    """Alternating speech and silence, the shape a long recording actually has."""
    n = int(sr * seconds)
    t = np.arange(n) / sr
    f0 = 115 + 22 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    x = 0.5 * np.sin(phase) + 0.15 * np.sin(2 * phase)

    env = np.zeros(n)
    pos = 0.3
    while pos + utterance < seconds:
        env[int(pos * sr):int((pos + utterance) * sr)] = 1.0
        pos += utterance + gap
    rng = np.random.default_rng(seed)
    return (x * env + 2e-5 * rng.standard_normal(n)).astype(np.float32)


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "src"
    folder.mkdir()
    wavfile.write(folder / "interview.wav", SR, speech())
    return folder


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------


def test_speech_is_split_on_the_pauses(source):
    sr, x = pd.decode(source / "interview.wav")
    segments = pd.segment(x, sr)
    assert 8 <= len(segments) <= 12, f"got {len(segments)} segments"
    lengths = [s.length / sr for s in segments]
    assert all(2.0 <= L <= 15.0 for L in lengths), lengths


def test_each_segment_keeps_the_silence_the_checker_needs(source):
    """The checker estimates room tone from the pause; trimming it breaks that."""
    sr, x = pd.decode(source / "interview.wav")
    segments = pd.segment(x, sr, pad=0.3)
    seg = segments[1]
    head = x[seg.start:seg.start + int(0.2 * sr)]
    assert float(np.max(np.abs(head))) < 0.01, "the segment starts mid-word"


def test_a_long_unbroken_stretch_is_split_rather_than_dropped(tmp_path):
    sr = 48000
    x = speech(seconds=40.0, sr=sr, utterance=38.0, gap=0.5)
    segments = pd.segment(x, sr, max_seconds=15.0)
    assert len(segments) >= 3
    assert all(s.length / sr <= 15.1 for s in segments)


def test_segments_shorter_than_the_minimum_are_dropped(tmp_path):
    sr = 48000
    x = speech(seconds=20.0, sr=sr, utterance=0.8, gap=1.0)
    assert pd.segment(x, sr, min_seconds=2.0) == []


def test_a_quietly_recorded_source_is_not_treated_as_silence(tmp_path):
    """The threshold is relative to the source, not an absolute level."""
    sr = 48000
    quiet = (speech(seconds=30.0, sr=sr) * 0.02).astype(np.float32)
    assert len(pd.segment(quiet, sr)) >= 3


def test_silence_only_input_yields_nothing(tmp_path):
    assert pd.segment(np.zeros(48000 * 5, dtype=np.float32), 48000) == []


# --------------------------------------------------------------------------
# Whole-file processing
# --------------------------------------------------------------------------


def test_output_is_48k_mono_and_normalised(source, tmp_path):
    out = tmp_path / "raw"
    segments, sr_in, notes = pd.prepare_file(source / "interview.wav", out)
    assert sr_in == SR
    assert any("アップサンプル" in n for n in notes), "upsampling should be called out"

    written = sorted(out.glob("*.wav"))
    assert len(written) == len(segments)
    for path in written:
        sr, data = wavfile.read(path)
        assert sr == 48000
        assert data.ndim == 1
        assert -7.0 < 20 * np.log10(np.max(np.abs(data))) < -5.0


def test_the_source_file_is_never_modified(source, tmp_path):
    before = (source / "interview.wav").read_bytes()
    pd.prepare_file(source / "interview.wav", tmp_path / "raw")
    assert (source / "interview.wav").read_bytes() == before


def test_dry_run_writes_nothing(source, tmp_path):
    out = tmp_path / "raw"
    segments, _, _ = pd.prepare_file(source / "interview.wav", out, dry_run=True)
    assert segments
    assert not out.exists()


def test_clipping_in_the_source_is_reported_because_it_cannot_be_undone(tmp_path):
    folder = tmp_path / "src"
    folder.mkdir()
    wavfile.write(folder / "hot.wav", 48000,
                  np.clip(speech(seconds=20.0, sr=48000) * 4, -1.0, 1.0).astype(np.float32))
    _, _, notes = pd.prepare_file(folder / "hot.wav", tmp_path / "raw", dry_run=True)
    assert any("クリップ" in n for n in notes)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_refuses_to_write_into_a_folder_that_already_has_files(source, tmp_path,
                                                                  capsys):
    out = tmp_path / "raw"
    out.mkdir()
    (out / "existing.wav").write_bytes(b"")
    assert pd.main([str(source), "--out", str(out)]) == 2
    assert "空ではありません" in capsys.readouterr().err
    assert pd.main([str(source), "--out", str(out), "--force"]) == 0


def test_cli_reports_a_missing_source(tmp_path, capsys):
    assert pd.main([str(tmp_path / "nope"), "--out", str(tmp_path / "raw")]) == 2
    assert "見つかりません" in capsys.readouterr().err


def test_cli_reports_a_folder_with_no_audio(tmp_path, capsys):
    folder = tmp_path / "src"
    folder.mkdir()
    (folder / "notes.txt").write_text("x")
    assert pd.main([str(folder), "--out", str(tmp_path / "raw")]) == 2
    assert "音声ファイルがありません" in capsys.readouterr().err


def test_prepared_material_passes_the_checker(tmp_path):
    """The two tools have to agree, or the pipeline has a gap in the middle."""
    folder = tmp_path / "src"
    folder.mkdir()
    for i in range(4):
        wavfile.write(folder / f"part{i}.wav", SR, speech(seconds=200.0, seed=i))

    out = tmp_path / "raw"
    assert pd.main([str(folder), "--out", str(out)]) == 0

    report = cd.analyse_dataset(out)
    assert report.ok, [p for f in report.files for p in f.problems] + report.problems
    assert report.total_seconds / 60.0 >= 10.0


@pytest.mark.parametrize("suffix", [".mp3", ".flac"])
def test_compressed_sources_are_decoded_when_a_decoder_is_available(tmp_path, suffix):
    sf = pytest.importorskip("soundfile", reason="no decoder installed")
    path = tmp_path / f"src{suffix}"
    sf.write(str(path), speech(seconds=30.0), SR)
    sr, x = pd.decode(path)
    assert sr == SR
    assert x.ndim == 1 and x.size > SR * 25
