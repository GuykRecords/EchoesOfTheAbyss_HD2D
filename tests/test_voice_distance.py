"""The similarity measure, checked against signals whose answer we know."""

import numpy as np
import pytest
from scipy.io import wavfile

from voice_distance import (
    ANALYSIS_SR,
    N_MELS,
    hz_to_mel,
    mel_filterbank,
    mel_to_hz,
    print_distance,
    reference_print,
    voice_print,
)


def _voiced(sr: int, seconds: float, f0: float, formants, seed: int = 0,
            jitter: float = 0.03, noise: float = 0.02) -> np.ndarray:
    """A crude vowel: a buzz at f0, shaped by resonances at `formants`.

    The jitter and breath noise are not decoration. A perfectly steady comb
    is not speech, and a measure tuned against one would be tuned against a
    signal it will never see.
    """
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    t = np.arange(n) / sr
    wobble = 1.0 + jitter * np.sin(2 * np.pi * 5.0 * t)
    wobble += jitter * 0.5 * rng.standard_normal(n).cumsum() / np.sqrt(n)
    phase = 2 * np.pi * np.cumsum(f0 * wobble) / sr

    x = np.zeros(n)
    for h in range(1, 40):
        if h * f0 >= sr / 2:
            break
        gain = 1.0 / h
        for f in formants:
            gain += 0.6 / (1.0 + ((h * f0 - f) / 120.0) ** 2)
        x += gain * np.sin(h * phase + rng.uniform(0, 2 * np.pi))
    x = x / np.abs(x).max()
    if noise:
        x = x + noise * rng.standard_normal(n)
    return (x / np.abs(x).max()).astype(np.float32)


def test_the_mel_scale_round_trips():
    hz = np.array([0.0, 100.0, 1000.0, 8000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(hz)), hz)


def test_every_mel_filter_is_non_negative_and_carries_weight():
    """Peaks land between FFT bins, so a filter can top out below 1 -- but
    never above it, and never at nothing."""
    bank = mel_filterbank()
    assert bank.shape[0] == N_MELS
    assert (bank >= 0).all()
    peaks = bank.max(axis=1)
    assert (peaks <= 1.0 + 1e-9).all()
    assert (peaks > 0.4).all()


def test_a_signal_is_at_zero_distance_from_itself():
    x = _voiced(ANALYSIS_SR, 1.0, 120.0, (700, 1200, 2600))
    a, _ = voice_print(x, ANALYSIS_SR)
    b, _ = voice_print(x, ANALYSIS_SR)
    assert print_distance(a, b) == pytest.approx(0.0, abs=1e-9)


def test_loudness_alone_does_not_move_the_distance():
    """The mean is removed, so a quieter take of the same voice still matches."""
    x = _voiced(ANALYSIS_SR, 1.0, 120.0, (700, 1200, 2600))
    a, _ = voice_print(x, ANALYSIS_SR)
    b, _ = voice_print(x * 0.1, ANALYSIS_SR)
    assert print_distance(a, b) < 0.05


def test_the_vocal_tract_moves_the_distance_more_than_the_note_does():
    """What the measure is for: it should track the tract, not the pitch.

    It is not pitch-blind -- a 25ms window only smooths the harmonics, it
    does not erase them -- so this asserts the ordering, not independence.
    """
    ref, _ = voice_print(_voiced(ANALYSIS_SR, 1.5, 120.0, (700, 1200, 2600), 0), ANALYSIS_SR)
    higher_note = _voiced(ANALYSIS_SR, 1.5, 140.0, (700, 1200, 2600), 1)
    other_tract = _voiced(ANALYSIS_SR, 1.5, 120.0, (450, 900, 2100), 2)

    pitch_only = print_distance(voice_print(higher_note, ANALYSIS_SR)[0], ref)
    tract = print_distance(voice_print(other_tract, ANALYSIS_SR)[0], ref)
    assert tract > pitch_only


def test_silence_is_kept_out_of_the_average():
    """A file that is half room tone must still print as the voice.

    Compared against what the room alone prints as: if the quiet half were
    being averaged in, the padded take would drift a good part of that way.
    """
    speech = _voiced(ANALYSIS_SR, 1.5, 120.0, (700, 1200, 2600))
    rng = np.random.default_rng(1)
    hiss = (rng.standard_normal(int(ANALYSIS_SR * 1.5)) * 1e-3).astype(np.float32)

    clean, _ = voice_print(speech, ANALYSIS_SR)
    padded, _ = voice_print(np.concatenate([speech, hiss]), ANALYSIS_SR)
    room, _ = voice_print(hiss, ANALYSIS_SR)

    drift = print_distance(clean, padded)
    all_the_way = print_distance(clean, room)
    assert drift < all_the_way / 10.0


def test_a_rate_other_than_the_analysis_rate_is_resampled_not_refused():
    """The reference material is 48k. Every real call takes this branch, and
    a synthetic test at 16k never would."""
    voice = _voiced(48000, 1.5, 120.0, (700, 1200, 2600), 0)
    at_48k, frames = voice_print(voice, 48000)
    assert frames > 0

    at_16k, _ = voice_print(_voiced(ANALYSIS_SR, 1.5, 120.0, (700, 1200, 2600), 0),
                            ANALYSIS_SR)
    stranger, _ = voice_print(_voiced(ANALYSIS_SR, 1.5, 120.0, (450, 900, 2100), 0),
                              ANALYSIS_SR)
    # Same vowel at a different rate must land nearer than a different vowel.
    assert print_distance(at_48k, at_16k) < print_distance(at_48k, stranger)


def _write(path, x, sr=ANALYSIS_SR):
    wavfile.write(path, sr, (np.clip(x, -1, 1) * 32767).astype(np.int16))


def test_the_floor_comes_from_splitting_the_reference_in_half(tmp_path):
    """Same speaker, split alternately, is the closest anything can measure."""
    folder = tmp_path / "ref"
    folder.mkdir()
    for i in range(6):
        _write(folder / f"{i:02d}.wav",
               _voiced(ANALYSIS_SR, 1.2, 118.0 + i, (700, 1200, 2600), seed=i))

    ref, floor, count = reference_print(folder)
    assert count == 6
    assert 0.0 <= floor < 2.0

    stranger, _ = voice_print(
        _voiced(ANALYSIS_SR, 1.2, 118.0, (450, 900, 2100), seed=9), ANALYSIS_SR)
    assert print_distance(stranger, ref) > 2 * floor


def test_a_reference_with_too_few_files_reports_no_floor_rather_than_a_wrong_one(tmp_path):
    folder = tmp_path / "ref"
    folder.mkdir()
    for i in range(2):
        _write(folder / f"{i}.wav", _voiced(ANALYSIS_SR, 1.2, 120.0, (700, 1200, 2600), seed=i))
    _, floor, count = reference_print(folder)
    assert count == 2
    assert np.isnan(floor)


def test_an_empty_reference_folder_says_so(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    with pytest.raises(FileNotFoundError):
        reference_print(folder)
