#!/usr/bin/env python3
"""学習用に録った音声が使い物になるかを判定する。読み取り専用。

録り直しは一番高くつくやり直しなので、**学習を回す前に**機械で判定する。
「たぶん大丈夫」ではなく、合否と理由を数字で出す。

    python scripts/check_dataset.py D:\\voice\\raw
    python scripts/check_dataset.py D:\\voice\\raw --min-minutes 20

終了コード 0 = 合格 / 1 = 直すべき問題あり。
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

__all__ = ["FileReport", "DatasetReport", "analyse_file", "analyse_dataset",
           "load_wav", "main", "THRESHOLDS"]

#: 合否の基準。ここを人間が先に固定するから自動判定が意味を持つ。
THRESHOLDS = {
    "min_total_minutes": 10.0,   # これ未満は声質が安定しない
    "good_total_minutes": 20.0,  # ここまであれば十分。増やしすぎても伸びない
    "min_seconds": 2.0,          # 短すぎるとピッチ推定が当たらない
    "max_seconds": 15.0,         # 長すぎると学習が不安定になる
    "min_sr": 40000,             # RVC v2 の下限
    "peak_dbfs_max": -1.0,       # これを超えていたらクリップ寸前か既にしている
    "noise_floor_dbfs_max": -50.0,  # 静かな区間がこれより大きければ環境が悪い
    "silence_ratio_max": 0.60,   # 半分以上が無音なら中身が薄い。間の取り方は自然でよい
    "dc_offset_max": 0.003,      # 直流が乗っていると前処理が狂う
}


def db(x: float) -> float:
    return 20.0 * math.log10(max(float(x), 1e-12))


def load_wav(path: Path) -> Tuple[int, np.ndarray, int]:
    """WAV を (sr, float32 モノラル) で読む。整数形式は正規化する。"""
    from scipy.io import wavfile

    sr, data = wavfile.read(path)
    data = np.asarray(data)
    channels = 1 if data.ndim == 1 else data.shape[1]
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        data = data.astype(np.float32) / float(max(abs(info.min), info.max))
    else:
        data = data.astype(np.float32)
    return int(sr), data, channels


@dataclass
class FileReport:
    path: Path
    sr: int
    channels: int
    seconds: float
    peak_dbfs: float
    rms_dbfs: float
    noise_floor_dbfs: float
    #: 無音区間が見つからず環境音を推定できなかった場合 True。
    #: そのときの noise_floor_dbfs は「一番静かな喋り」であって環境音ではない。
    noise_floor_unknown: bool
    silence_ratio: float
    dc_offset: float
    clipped_samples: int
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def analyse_file(path: Path, thresholds: dict = THRESHOLDS) -> FileReport:
    sr, x, channels = load_wav(path)
    problems: List[str] = []

    seconds = x.size / sr if sr else 0.0
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0
    dc = float(np.mean(x)) if x.size else 0.0
    clipped = int(np.count_nonzero(np.abs(x) >= 0.999))

    # 20ms ごとの RMS。静かな方から 10% を「その録音の環境音」とみなす。
    frame = max(1, int(sr * 0.02))
    usable = (x.size // frame) * frame
    if usable:
        frames = x[:usable].reshape(-1, frame)
        frame_rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
        quiet = np.sort(frame_rms)[:max(1, len(frame_rms) // 10)]
        noise_floor = float(np.mean(quiet))
        silence_ratio = float(np.mean(frame_rms < rms * 0.05)) if rms else 1.0
    else:
        noise_floor, silence_ratio = 0.0, 1.0

    # 「一番静かな 10%」が環境音と言えるのは、そこが喋りより十分下にあるとき
    # だけ。切れ目なく喋り続けたファイルではこれは「一番小さい声」であって
    # 環境音ではないので、静かだとも うるさいとも判定してはいけない。
    noise_floor_unknown = bool(rms) and noise_floor > rms * 0.1  # 20 dB 以内

    if sr < thresholds["min_sr"]:
        problems.append(f"サンプルレートが低い ({sr}Hz < {thresholds['min_sr']}Hz)")
    if channels != 1:
        problems.append(f"{channels}ch。モノラルに変換すること")
    if seconds < thresholds["min_seconds"]:
        problems.append(f"短すぎる ({seconds:.1f}s < {thresholds['min_seconds']}s)")
    if seconds > thresholds["max_seconds"]:
        problems.append(f"長すぎる ({seconds:.1f}s > {thresholds['max_seconds']}s)。分割する")
    if clipped:
        problems.append(f"クリップしている ({clipped} サンプル)。録り直し")
    elif db(peak) > thresholds["peak_dbfs_max"]:
        problems.append(f"ピークが高すぎる ({db(peak):.1f} dBFS)。入力ゲインを下げる")
    if noise_floor_unknown:
        problems.append(
            "無音区間が無く環境音を判定できない。前後に 0.3 秒ほど無音を残して録ること"
        )
    elif db(noise_floor) > thresholds["noise_floor_dbfs_max"]:
        problems.append(f"環境音が大きい ({db(noise_floor):.1f} dBFS)。静かな場所で録り直し")
    if silence_ratio > thresholds["silence_ratio_max"]:
        problems.append(f"ほとんど無音 ({silence_ratio * 100:.0f}%)。中身がある区間を切り出す")
    if abs(dc) > thresholds["dc_offset_max"]:
        problems.append(f"直流が乗っている (DC {dc:+.4f})")

    return FileReport(path, sr, channels, seconds, db(peak), db(rms),
                      db(noise_floor), noise_floor_unknown, silence_ratio, dc,
                      clipped, problems)


@dataclass
class DatasetReport:
    files: List[FileReport]
    problems: List[str] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return sum(f.seconds for f in self.files)

    @property
    def ok(self) -> bool:
        return not self.problems and all(f.ok for f in self.files)


def analyse_dataset(folder: Path, thresholds: dict = THRESHOLDS,
                    min_minutes: Optional[float] = None) -> DatasetReport:
    paths = sorted(p for p in folder.rglob("*.wav") if p.is_file())
    files = [analyse_file(p, thresholds) for p in paths]
    report = DatasetReport(files)

    floor = min_minutes if min_minutes is not None else thresholds["min_total_minutes"]
    minutes = report.total_seconds / 60.0
    if not files:
        report.problems.append(f"{folder} に .wav がありません")
    elif minutes < floor:
        report.problems.append(
            f"総時間が足りない ({minutes:.1f} 分 < {floor:.0f} 分)。"
            f"{thresholds['good_total_minutes']:.0f} 分あれば十分"
        )

    rates = {f.sr for f in files}
    if len(rates) > 1:
        report.problems.append(f"サンプルレートが混在している: {sorted(rates)}")
    return report


def print_report(report: DatasetReport, verbose: bool = False) -> None:
    minutes = report.total_seconds / 60.0
    print(f"ファイル {len(report.files)} 本 / 合計 {minutes:.1f} 分")
    if report.files:
        lengths = sorted(f.seconds for f in report.files)
        mid = lengths[len(lengths) // 2]
        print(f"長さ  最短 {lengths[0]:.1f}s / 中央 {mid:.1f}s / 最長 {lengths[-1]:.1f}s")
        print(f"レート {sorted({f.sr for f in report.files})}")
        peaks = [f.peak_dbfs for f in report.files]
        floors = [f.noise_floor_dbfs for f in report.files if not f.noise_floor_unknown]
        line = f"ピーク {min(peaks):.1f} 〜 {max(peaks):.1f} dBFS"
        if floors:
            line += f"  |  環境音 {min(floors):.1f} 〜 {max(floors):.1f} dBFS"
        else:
            line += "  |  環境音: 判定不能（無音区間なし）"
        print(line)

    bad = [f for f in report.files if not f.ok]
    if verbose:
        for f in report.files:
            mark = "  " if f.ok else "NG"
            floor = ("      ?" if f.noise_floor_unknown
                     else f"{f.noise_floor_dbfs:7.1f}")
            print(f"{mark} {f.path.name:<40} {f.seconds:6.2f}s "
                  f"peak {f.peak_dbfs:7.1f}  floor {floor}")
    if bad:
        # 100 本が同じ理由で落ちたときに 100 行読ませない。
        # 種類ごとにまとめて、例を数本だけ挙げる。
        by_kind: dict = {}
        for f in bad:
            for problem in f.problems:
                kind = problem.split("(")[0].split("。")[0].strip()
                by_kind.setdefault(kind, []).append(f.path.name)
        print(f"\n--- 直すファイル {len(bad)} 本 / 指摘 {len(by_kind)} 種 ---")
        for kind, names in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
            print(f"  [{len(names):>4} 本] {kind}")
            print(f"           例: {', '.join(names[:3])}"
                  + (" ..." if len(names) > 3 else ""))
    for problem in report.problems:
        print(f"\n[全体] {problem}")

    print()
    if report.ok:
        print("合格。この素材で学習に進んでよい。")
    else:
        print("不合格。上の指摘を直してから学習すること。")
        print("（録り直しは一番高くつくやり直しなので、ここで止めるのが安い）")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="学習用音声が使い物になるかを判定する（読み取り専用）")
    ap.add_argument("folder", type=Path, help=".wav が入っているフォルダ")
    ap.add_argument("--min-minutes", type=float, default=None,
                    help=f"必要な総時間 (既定: {THRESHOLDS['min_total_minutes']:.0f} 分)")
    ap.add_argument("-v", "--verbose", action="store_true", help="全ファイルを表示")
    args = ap.parse_args(argv)

    if not args.folder.is_dir():
        print(f"error: {args.folder} が見つかりません", file=sys.stderr)
        return 2

    report = analyse_dataset(args.folder, min_minutes=args.min_minutes)
    print_report(report, verbose=args.verbose)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
