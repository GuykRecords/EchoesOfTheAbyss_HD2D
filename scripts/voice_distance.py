"""変換結果が、目標の声にどれだけ近いかを測る。

**耳の代わりではない。** 測っているのは「長時間平均した音色の傾き」であって、
「同じ人に聞こえるか」ではない。**候補に順位を付けるためだけに使う。**
気に入った音を数字が否定してきたら、耳を採る。

なぜ長時間平均なのか: 候補と目標は**喋っている内容が違う**。
個々のフレームを比べても意味がないが、20 秒ぶんも均せば、話者の声道の形
（フォルマントの位置）が残り、内容の差は薄まる。

**声の高さが違う相手との比較には向かない。** 25 ms 窓で倍音をならしてあるので
影響は小さいが、消えてはいない。同じ `--rvc-key` の候補どうしを並べるのが
正しい使い方で、変換前の入力との比較は「桁の目安」程度に読むこと。

使い方::

    python scripts\\voice_distance.py --reference D:\\Claude\\Project\\voice\\ena ^
        k12_e150.wav fm1_e150.wav fm15_e150.wav fm2_e150.wav

出力の読み方は 3 行で決まる::

    下限 (素材を二分割して比べた値)   1.31 dB   ← これ以上は近づけない
    元の入力 (変換前)                 6.02 dB   ← 何もしなければこの距離
    候補                              ...       ← この 2 つの間のどこにいるか

**下限と元の入力が出ていないと、候補の数字は読めない。** 3.5 という値が
良いのか悪いのかは、その 2 つと並べて初めて分かる。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_dataset import AUDIO_SUFFIXES, decode, resolve_path  # noqa: E402

from rtvc.dsp import Resampler  # noqa: E402

__all__ = ["hz_to_mel", "mel_to_hz", "mel_filterbank", "voice_print",
           "print_distance", "reference_print", "main", "DYNAMIC_RANGE_DB"]

#: 声道の形が出る帯域。16 kHz に落として測るので上限はナイキストの下に置く。
ANALYSIS_SR = 16000
#: 25 ms 窓。話者の音色を測る道具はどれもこの長さを使う。
#: **長くすると倍音が 1 本ずつ分解されてしまい、声道の形ではなく声の高さを
#: 測り始める。** 64 ms 窓で試したところ、ピッチを変えただけの信号が、
#: 声道を変えた信号より遠くに出た。25 ms ではそれが逆転する。
N_FFT = 400
HOP = 100
N_MELS = 40
FMIN = 80.0
FMAX = 7600.0

#: フレームを残す下限。いちばん大きいフレームからこれだけ下まで。
#: 無音を平均に混ぜると、部屋の暗騒音の色が話者の色として効いてしまう。
LOUD_FLOOR_DB = 35.0

#: 1 フレームの中で見るダイナミックレンジ。これより下は床に潰す。
#: 声のフォルマントはこの範囲に収まる。下を残しても数字が暴れるだけ。
DYNAMIC_RANGE_DB = 60.0


def hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=float) / 2595.0) - 1.0)


def mel_filterbank(sr: int = ANALYSIS_SR, n_fft: int = N_FFT, n_mels: int = N_MELS,
                   fmin: float = FMIN, fmax: float = FMAX) -> np.ndarray:
    """(n_mels, n_fft//2+1) の三角フィルタ。各フィルタの頂点は 1。"""
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    bank = np.zeros((n_mels, freqs.size), dtype=np.float64)
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        rising = (freqs - lo) / max(mid - lo, 1e-9)
        falling = (hi - freqs) / max(hi - mid, 1e-9)
        bank[i] = np.clip(np.minimum(rising, falling), 0.0, None)
    return bank


def _frames(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """(frames, n_fft) の窓掛け済みフレーム。足りなければ空を返す。"""
    if x.size < n_fft:
        return np.zeros((0, n_fft), dtype=np.float64)
    count = 1 + (x.size - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(count)[:, None]
    return x[idx] * np.hanning(n_fft)[None, :]


def voice_print(x: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    """(log-mel の平均, 使ったフレーム数)。

    平均から自身の平均を引いてある。**音量の差は距離に効かない。**
    """
    if sr != ANALYSIS_SR:
        x = Resampler(sr, ANALYSIS_SR).process(np.asarray(x, dtype=np.float32))
    x = np.asarray(x, dtype=np.float64)
    frames = _frames(x, N_FFT, HOP)
    if frames.shape[0] == 0:
        return np.zeros(N_MELS), 0

    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    mel = power @ mel_filterbank(n_fft=N_FFT).T                      # (frames, n_mels)
    # 床は絶対値ではなく、そのフレームの最大から DYNAMIC_RANGE_DB 下に置く。
    # 絶対値で切ると、同じ声を小さく録っただけで下の帯域が床に潰れ、
    # 「音量が違う」ことを「音色が違う」と読み違える。
    peak = mel.max(axis=1, keepdims=True)
    mel = np.maximum(mel, peak * 10.0 ** (-DYNAMIC_RANGE_DB / 10.0))
    log_mel = 10.0 * np.log10(np.maximum(mel, 1e-30))

    energy = 10.0 * np.log10(np.maximum(power.sum(axis=1), 1e-12))
    loud = energy >= energy.max() - LOUD_FLOOR_DB
    if not loud.any():
        loud = np.ones_like(energy, dtype=bool)

    mean = log_mel[loud].mean(axis=0)
    return mean - mean.mean(), int(loud.sum())


def print_distance(a: np.ndarray, b: np.ndarray) -> float:
    """帯域ごとの差の二乗平均平方根。単位は dB。0 は完全一致。"""
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _audio_files(target: Path) -> List[Path]:
    if target.is_dir():
        return sorted(p for p in target.iterdir()
                      if p.suffix.lower() in AUDIO_SUFFIXES)
    return [target]


def _pooled(paths: Sequence[Path]) -> np.ndarray:
    """複数ファイルを、フレーム数で重み付けして 1 本にまとめる。"""
    total = np.zeros(N_MELS)
    weight = 0
    for path in paths:
        sr, data = decode(path)
        vec, frames = voice_print(data, sr)
        if frames:
            total += vec * frames
            weight += frames
    if weight == 0:
        raise ValueError("音のあるフレームが 1 つも無い")
    pooled = total / weight
    return pooled - pooled.mean()


def reference_print(target: Path) -> Tuple[np.ndarray, float, int]:
    """(基準ベクトル, 下限, ファイル数)。

    下限は、素材を 1 本おきに二分割して互いに比べた値。**同じ話者の同じ収録**
    どうしでもこれだけは離れる、という測定系そのものの誤差。候補がこれに
    近づいたら、それ以上は測っても意味がない。
    """
    paths = _audio_files(target)
    if not paths:
        raise FileNotFoundError(f"音声ファイルが無い: {target}")
    pooled = _pooled(paths)
    floor = (print_distance(_pooled(paths[0::2]), _pooled(paths[1::2]))
             if len(paths) >= 4 else float("nan"))
    return pooled, floor, len(paths)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="変換結果が目標の声にどれだけ近いかを測る（順位付け専用）")
    p.add_argument("candidates", nargs="+", help="比べたい WAV")
    p.add_argument("--reference", required=True,
                   help="目標の声。学習素材のフォルダを渡すのが本筋")
    p.add_argument("--source", default=None,
                   help="変換前の入力。上限の目安として一緒に測る")
    args = p.parse_args(argv)

    try:
        ref, floor, count = reference_print(resolve_path(Path(args.reference)))
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2

    print(f"基準 {count} 本  |  数字は小さいほど近い（dB）")
    print("-" * 58)
    if np.isfinite(floor):
        print(f"  {'下限（素材を二分割して比較）':<34} {floor:6.2f}")

    if args.source:
        try:
            sr, data = decode(resolve_path(Path(args.source)))
            vec, _ = voice_print(data, sr)
            print(f"  {'元の入力（変換前）':<38} {print_distance(vec, ref):6.2f}")
        except (FileNotFoundError, ValueError) as exc:
            print(f"  元の入力: 読めない ({exc})")
    print()

    scored = []
    for name in args.candidates:
        path = resolve_path(Path(name))
        try:
            sr, data = decode(path)
            vec, frames = voice_print(data, sr)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  {path.name}: 読めない ({exc})")
            continue
        if frames == 0:
            print(f"  {path.name}: 音のあるフレームが無い")
            continue
        scored.append((print_distance(vec, ref), path.name))

    for i, (dist, name) in enumerate(sorted(scored)):
        mark = "←" if i == 0 else " "
        print(f"  {name:<38} {dist:6.2f} {mark}")

    print()
    print("順位付けのための数字であって、良し悪しの判定ではない。")
    print("気に入った音を数字が否定してきたら、耳を採ること。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
