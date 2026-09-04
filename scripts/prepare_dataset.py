#!/usr/bin/env python3
"""既存の音源（MP3 など）を学習用の素材に整える。**新規作成のみ。入力は変更しない。**

MP3 は 1 本の長いファイルで、学習にはそのまま使えない。必要なのは

    48kHz / モノラル / 2〜15 秒 / 前後に無音 / クリップ無し の WAV が多数

なので、この工程でそこまで持っていく。

    python scripts/prepare_dataset.py D:\\voice\\source.mp3 --out D:\\voice\\raw
    python scripts/prepare_dataset.py D:\\voice\\src --out D:\\voice\\raw --dry-run

出力したら `check_dataset.py` に通すこと。合格して初めて訓練に進む。
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rtvc.dsp import Resampler  # noqa: E402  (path set above)

__all__ = ["decode", "to_mono", "segment", "prepare_file", "main",
           "Segment", "AUDIO_SUFFIXES"]

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus",
                  ".wma", ".aiff", ".aif"}


def db(x: float) -> float:
    return 20.0 * math.log10(max(float(x), 1e-12))


def amp(db_value: float) -> float:
    return float(10.0 ** (db_value / 20.0))


# ---------------------------------------------------------------------------
# 読み込み
# ---------------------------------------------------------------------------


def to_mono(data: np.ndarray) -> np.ndarray:
    return data if data.ndim == 1 else data.mean(axis=1)


def decode(path: Path) -> Tuple[int, np.ndarray]:
    """(sr, float32 モノラル) を返す。

    WAV は scipy で読む（追加の依存が要らない）。それ以外は soundfile、
    駄目なら ffmpeg。どちらも無ければ、**何を入れればよいかを名指しで**言う。
    """
    if path.suffix.lower() == ".wav":
        from scipy.io import wavfile

        sr, data = wavfile.read(path)
        data = np.asarray(data)
        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            data = data.astype(np.float32) / float(max(abs(info.min), info.max))
        return int(sr), to_mono(data.astype(np.float32))

    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        return int(sr), to_mono(np.asarray(data, dtype=np.float32))
    except ImportError:
        pass
    except Exception as exc:
        raise RuntimeError(f"{path.name} を soundfile で読めない: {exc}") from exc

    return _decode_with_ffmpeg(path)


def _decode_with_ffmpeg(path: Path) -> Tuple[int, np.ndarray]:
    import shutil

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            f"{path.suffix} を読むには soundfile か ffmpeg が要る。\n"
            "  pip install soundfile   … 手軽。mp3/flac/ogg を読める\n"
            "  または ffmpeg を PATH に通す … 対応形式が最も広い"
        )
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-f", "f32le",
         "-acodec", "pcm_f32le", "-ac", "1", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg が {path.name} を読めない: "
                           f"{proc.stderr.decode(errors='replace')[:300]}")
    probe = subprocess.run(
        [ffmpeg.replace("ffmpeg", "ffprobe"), "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate", "-of", "csv=p=0", str(path)],
        capture_output=True,
    )
    sr = int(probe.stdout.decode().strip() or 48000)
    return sr, np.frombuffer(proc.stdout, dtype=np.float32).copy()


# ---------------------------------------------------------------------------
# 無音で切る
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def _frame_rms(x: np.ndarray, frame: int) -> np.ndarray:
    usable = (x.size // frame) * frame
    if usable == 0:
        return np.zeros(0)
    frames = x[:usable].reshape(-1, frame)
    return np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))


def segment(
    x: np.ndarray,
    sr: int,
    silence_dbfs: float = -45.0,
    min_silence: float = 0.30,
    min_seconds: float = 2.0,
    max_seconds: float = 15.0,
    pad: float = 0.30,
) -> List[Segment]:
    """喋っている区間を切り出す。

    しきい値は「絶対値」と「その音源の中で一番大きいフレームから 35dB 下」の
    **大きい方**。録音レベルは音源ごとに違うので、絶対値だけだと小さく録れた
    音源が丸ごと無音扱いになる。
    """
    frame = max(1, int(sr * 0.02))
    rms = _frame_rms(x, frame)
    if rms.size == 0:
        return []

    threshold = max(amp(silence_dbfs), float(np.max(rms)) * amp(-35.0))
    voiced = rms > threshold
    if not voiced.any():
        return []

    gap_frames = max(1, int(min_silence / 0.02))
    runs: List[Segment] = []
    start = None
    silence = 0
    for i, is_voice in enumerate(voiced):
        if is_voice:
            if start is None:
                start = i
            silence = 0
        elif start is not None:
            silence += 1
            if silence >= gap_frames:
                runs.append(Segment(start * frame, (i - silence + 1) * frame))
                start = None
    if start is not None:
        runs.append(Segment(start * frame, len(voiced) * frame))

    pad_samples = int(pad * sr)
    out: List[Segment] = []
    for run in runs:
        run = Segment(max(0, run.start - pad_samples),
                      min(x.size, run.end + pad_samples))
        out.extend(_split_long(x, run, sr, max_seconds, frame))

    keep = int(min_seconds * sr)
    return [s for s in out if s.length >= keep]


def _split_long(x, seg: Segment, sr: int, max_seconds: float, frame: int) -> List[Segment]:
    """長すぎる区間を、内側の一番静かなところで割る。"""
    limit = int(max_seconds * sr)
    if seg.length <= limit:
        return [seg]
    inner = x[seg.start:seg.end]
    rms = _frame_rms(inner, frame)
    if rms.size < 3:
        mid = seg.start + seg.length // 2
    else:
        lo, hi = len(rms) // 4, len(rms) * 3 // 4
        mid = seg.start + (lo + int(np.argmin(rms[lo:hi]))) * frame
    left = Segment(seg.start, mid)
    right = Segment(mid, seg.end)
    return _split_long(x, left, sr, max_seconds, frame) + \
        _split_long(x, right, sr, max_seconds, frame)


# ---------------------------------------------------------------------------
# 一本ぶんの処理
# ---------------------------------------------------------------------------


def prepare_file(
    path: Path,
    out_dir: Path,
    target_sr: int = 48000,
    peak_dbfs: float = -6.0,
    dry_run: bool = False,
    **segment_kwargs,
) -> Tuple[List[Segment], int, List[str]]:
    """1 ファイルを読み、48k モノラルに直し、切って書き出す。

    戻り値は (区間, 元の sr, 注意書き)。
    """
    notes: List[str] = []
    sr, x = decode(path)

    if sr != target_sr:
        if sr < target_sr:
            notes.append(
                f"{sr}Hz → {target_sr}Hz にアップサンプルする。"
                "元に無い高域が増えるわけではないので、音質の上限は元のまま"
            )
        x = Resampler(sr, target_sr).process(x)

    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak >= 0.999:
        notes.append("元の時点でクリップしている。歪みは取り除けない")
    if peak > 0:
        x = (x * (amp(peak_dbfs) / peak)).astype(np.float32)

    segments = segment(x, target_sr, **segment_kwargs)
    if not segments:
        notes.append("喋っている区間が見つからない。無音か、しきい値が合っていない")

    if not dry_run and segments:
        from scipy.io import wavfile

        out_dir.mkdir(parents=True, exist_ok=True)
        stem = path.stem[:40].replace(" ", "_")
        for i, seg in enumerate(segments):
            wavfile.write(out_dir / f"{stem}_{i:04d}.wav", target_sr,
                          x[seg.start:seg.end].astype(np.float32))
    return segments, sr, notes


def _inputs(target: Path) -> List[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.rglob("*")
                  if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="既存の音源を学習用の 48k モノラル WAV に切り分ける（入力は変更しない）")
    ap.add_argument("source", type=Path, help="音声ファイル、またはそれが入ったフォルダ")
    ap.add_argument("--out", type=Path, required=True, help="出力先フォルダ")
    ap.add_argument("--target-sr", type=int, default=48000)
    ap.add_argument("--peak-dbfs", type=float, default=-6.0, help="正規化後のピーク")
    ap.add_argument("--min-seconds", type=float, default=2.0)
    ap.add_argument("--max-seconds", type=float, default=15.0)
    ap.add_argument("--silence-dbfs", type=float, default=-45.0)
    ap.add_argument("--min-silence", type=float, default=0.30,
                    help="これ以上続く静けさを区切りとみなす")
    ap.add_argument("--pad", type=float, default=0.30,
                    help="各区間の前後に残す無音。判定ツールが環境音の推定に使う")
    ap.add_argument("--dry-run", action="store_true", help="書き出さずに結果だけ見る")
    ap.add_argument("--force", action="store_true", help="出力先が空でなくても実行")
    args = ap.parse_args(argv)

    if not args.source.exists():
        print(f"error: {args.source} が見つかりません", file=sys.stderr)
        return 2
    sources = _inputs(args.source)
    if not sources:
        print(f"error: {args.source} に音声ファイルがありません", file=sys.stderr)
        return 2
    if (not args.dry_run and args.out.is_dir()
            and any(args.out.iterdir()) and not args.force):
        print(f"error: {args.out} が空ではありません。別の場所を指定するか --force",
              file=sys.stderr)
        return 2

    print(f"入力 {len(sources)} 本 → {args.out}"
          + ("  (dry-run: 書き出しません)" if args.dry_run else ""))
    total_seconds = 0.0
    total_clips = 0
    failures = 0

    for path in sources:
        try:
            segments, sr, notes = prepare_file(
                path, args.out, target_sr=args.target_sr, peak_dbfs=args.peak_dbfs,
                dry_run=args.dry_run, min_seconds=args.min_seconds,
                max_seconds=args.max_seconds, silence_dbfs=args.silence_dbfs,
                min_silence=args.min_silence, pad=args.pad,
            )
        except RuntimeError as exc:
            print(f"  {path.name}: {exc}")
            failures += 1
            continue
        seconds = sum(s.length for s in segments) / args.target_sr
        total_seconds += seconds
        total_clips += len(segments)
        print(f"  {path.name}  {sr}Hz → {len(segments)} 本 / {seconds / 60:.1f} 分")
        for note in notes:
            print(f"      ! {note}")

    print(f"\n合計 {total_clips} 本 / {total_seconds / 60:.1f} 分")
    if failures:
        print(f"読めなかったファイル: {failures} 本")
    if total_clips and not args.dry_run:
        print(f"\n次: python scripts\\check_dataset.py {args.out}")
        print("  合格が出てから訓練に進むこと。")
    return 0 if total_clips and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
