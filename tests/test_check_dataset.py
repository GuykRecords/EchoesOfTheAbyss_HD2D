"""The recording checker decides whether a re-record is needed.

Re-recording is the most expensive kind of rework in this project, so the
thresholds and their edge cases are pinned down rather than trusted.
"""

from pathlib import Path

import check_dataset as cd
import numpy as np
import pytest
from scipy.io import wavfile

SR = 48000


def utterance(seconds=6.0, sr=SR, amp=0.35, with_silence=True, seed=0):
    """Speech-shaped audio with the pauses real speech has."""
    n = int(sr * seconds)
    t = np.arange(n) / sr
    f0 = 120 + 25 * np.sin(2 * np.pi * 0.8 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    x = amp * np.sin(phase) + amp * 0.3 * np.sin(2 * phase)
    if with_silence:
        env = np.ones(n)
        lead = int(0.3 * sr)
        env[:lead] = 0.0
        env[-lead:] = 0.0
        for start in np.arange(0.5, seconds - 0.5, 1.5):
            a = int(start * sr)
            env[a:a + int(0.3 * sr)] = 0.0
        x *= env
    rng = np.random.default_rng(seed)
    return (x + 3e-5 * rng.standard_normal(n)).astype(np.float32)


def write(folder: Path, name: str, data, sr=SR):
    folder.mkdir(parents=True, exist_ok=True)
    wavfile.write(folder / name, sr, data)
    return folder / name


@pytest.fixture
def good_dataset(tmp_path):
    folder = tmp_path / "raw"
    for i in range(110):
        write(folder, f"take{i:03d}.wav", utterance(seed=i))
    return folder


def test_a_clean_dataset_passes(good_dataset):
    report = cd.analyse_dataset(good_dataset)
    assert report.ok, [p for f in report.files for p in f.problems] + report.problems
    assert report.total_seconds / 60.0 >= 10.0


def test_the_check_never_modifies_anything(good_dataset):
    before = {p: p.read_bytes() for p in good_dataset.glob("*.wav")}
    cd.analyse_dataset(good_dataset)
    after = {p: p.read_bytes() for p in good_dataset.glob("*.wav")}
    assert before == after


def test_clipping_is_caught(tmp_path):
    path = write(tmp_path, "hot.wav", np.clip(utterance() * 4, -1.0, 1.0))
    report = cd.analyse_file(path)
    assert report.clipped_samples > 0
    assert any("クリップ" in p for p in report.problems)


def test_a_low_sample_rate_is_caught(tmp_path):
    path = write(tmp_path, "low.wav", utterance(sr=16000), sr=16000)
    assert any("サンプルレート" in p for p in cd.analyse_file(path).problems)


@pytest.mark.parametrize("seconds,word", [(1.0, "短すぎる"), (20.0, "長すぎる")])
def test_length_limits(tmp_path, seconds, word):
    path = write(tmp_path, "len.wav", utterance(seconds=seconds))
    assert any(word in p for p in cd.analyse_file(path).problems)


def test_a_noisy_room_is_caught(tmp_path):
    rng = np.random.default_rng(0)
    noisy = utterance() + 0.02 * rng.standard_normal(int(SR * 6.0)).astype(np.float32)
    path = write(tmp_path, "noisy.wav", noisy.astype(np.float32))
    report = cd.analyse_file(path)
    assert not report.noise_floor_unknown
    assert any("環境音が大きい" in p for p in report.problems)


def test_a_file_with_no_pauses_is_reported_as_unmeasurable_not_as_noisy(tmp_path):
    """Without silence the quietest frames are quiet *speech*, not the room.

    Calling that a noisy room would fail clean recordings for no reason.
    """
    path = write(tmp_path, "nonstop.wav", utterance(with_silence=False))
    report = cd.analyse_file(path)
    assert report.noise_floor_unknown
    assert any("判定できない" in p for p in report.problems)
    assert not any("環境音が大きい" in p for p in report.problems)


def test_natural_pauses_are_not_treated_as_dead_air(tmp_path):
    """Real speech is roughly a third silence; that is not a defect."""
    path = write(tmp_path, "normal.wav", utterance())
    report = cd.analyse_file(path)
    assert 0.15 < report.silence_ratio < 0.6
    assert not any("無音" in p for p in report.problems)


def test_dc_offset_is_caught(tmp_path):
    path = write(tmp_path, "dc.wav", (utterance() + 0.02).astype(np.float32))
    assert any("直流" in p for p in cd.analyse_file(path).problems)


def test_too_little_material_is_a_dataset_level_problem(tmp_path):
    folder = tmp_path / "raw"
    for i in range(5):
        write(folder, f"t{i}.wav", utterance(seed=i))
    report = cd.analyse_dataset(folder)
    assert not report.ok
    assert any("総時間が足りない" in p for p in report.problems)


def test_mixed_sample_rates_are_a_dataset_level_problem(tmp_path):
    folder = tmp_path / "raw"
    write(folder, "a.wav", utterance())
    write(folder, "b.wav", utterance(sr=44100), sr=44100)
    assert any("混在" in p for p in cd.analyse_dataset(folder).problems)


def test_an_empty_folder_says_so(tmp_path):
    folder = tmp_path / "raw"
    folder.mkdir()
    assert any(".wav がありません" in p for p in cd.analyse_dataset(folder).problems)


def test_cli_exit_codes(good_dataset, tmp_path, capsys):
    assert cd.main([str(good_dataset)]) == 0
    assert "合格" in capsys.readouterr().out
    assert cd.main([str(tmp_path / "missing")]) == 2


def test_cli_fails_on_a_bad_dataset(tmp_path, capsys):
    folder = tmp_path / "raw"
    write(folder, "hot.wav", np.clip(utterance() * 4, -1.0, 1.0))
    assert cd.main([str(folder)]) == 1
    assert "不合格" in capsys.readouterr().out


def test_findings_are_grouped_rather_than_listed_once_per_file(tmp_path, capsys):
    """100 files failing for one reason must not print 100 blocks."""
    folder = tmp_path / "raw"
    for i in range(40):
        write(folder, f"long{i:02d}.wav", utterance(seconds=20.0, seed=i))
    cd.main([str(folder)])
    out = capsys.readouterr().out
    assert out.count("長すぎる") == 1
    assert "[  40 本]" in out
