"""RVC 側のグルー。:class:`~rtvc.engines.RVCTorchEngine` に渡す infer_fn を組み立てる。

**このモジュールだけが RVC 本体と torch に依存する。** import は
:class:`RVCBackend` の生成時まで遅らせてあるので、torch の無い環境でも
``import rtvc.rvc_backend`` 自体は通る（CI のテストがそれを確かめている）。

実行は ``.venv-rvc`` で行うこと。計測用 ``.venv`` は torch 2.11 で、
RVC が要求する torch<2.8 と非互換。``.venv-rvc`` には sounddevice / scipy / soxr も
入っているので、``realtime.py`` ごと ``.venv-rvc`` で走らせればプロセスは 1 つで済む。
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import numpy as np

from .engines import RVCTorchEngine

__all__ = ["RVCBackend", "find_first_model", "build_rvc_engine",
           "DEFAULT_RVC_ROOT", "ZC_SAMPLES_16K", "to_zc_units"]

DEFAULT_RVC_ROOT = r"D:\Claude\Project\RVC"

#: RVC のリアルタイム経路は ``zc = sr // 100`` すなわち 10 ms を 1 単位として数える。
#: 16 kHz では 160 サンプル。
ZC_SAMPLES_16K = 160


def to_zc_units(samples_16k: int, what: str) -> int:
    """16 kHz サンプル数を RVC の 10 ms 単位に直す。

    エンジン側の契約はサンプル数（単位が自明で、テストしやすい）。
    RVC が欲しいのは 10 ms 単位。その換算はここでしか起きない。
    """
    if samples_16k % ZC_SAMPLES_16K != 0:
        raise ValueError(
            f"{what}={samples_16k} 16kHz サンプルは {ZC_SAMPLES_16K} で割り切れない。"
            "窓のどれかが 10 ms グリッドから外れている。"
        )
    return samples_16k // ZC_SAMPLES_16K


class RVCBackend:
    """``infer/rtrvc.py`` の ``RVC`` を保持し、numpy in / numpy out の infer_fn を提供する。"""

    def __init__(
        self,
        pth_path: str,
        block_ms: float,
        index_path: str = "",
        index_rate: float = 0.0,
        key: int = 0,
        formant: float = 0.0,
        f0method: str = "rmvpe",
        rvc_root: str = DEFAULT_RVC_ROOT,
    ) -> None:
        if not os.path.isfile(pth_path):
            raise FileNotFoundError(f"話者モデルが見つからない: {pth_path}")
        if index_rate and not os.path.isfile(index_path):
            raise FileNotFoundError(f"index_rate>0 なのに index が無い: {index_path}")
        if f0method not in ("rmvpe", "fcpe"):
            raise ValueError(
                f"f0method は rmvpe か fcpe のみ（harvest/crepe はリアルタイム不可）: {f0method}"
            )

        # RVC はカレントディレクトリ相対で assets/ を読む箇所がある（rmvpe.pt など）。
        self.rvc_root = os.path.abspath(rvc_root)
        if not os.path.isdir(self.rvc_root):
            raise FileNotFoundError(f"RVC が見つからない: {self.rvc_root}")
        if self.rvc_root not in sys.path:
            sys.path.insert(0, self.rvc_root)
        self._prev_cwd = os.getcwd()
        os.chdir(self.rvc_root)

        try:
            import torch

            from configs.config import Config
            from infer.rtrvc import RVC
        except ImportError as exc:
            os.chdir(self._prev_cwd)
            raise RuntimeError(
                f"RVC / torch を import できない（{exc}）。"
                ".venv-rvc で実行しているか確認すること。"
            ) from exc

        self.torch = torch
        self.config = Config()
        self.device = str(self.config.device)
        self.f0method = f0method

        self.rvc = RVC(key, formant, pth_path, index_path, index_rate, self.config)
        if getattr(self.rvc, "net_g", None) is None:
            os.chdir(self._prev_cwd)
            raise RuntimeError(
                "RVC の初期化に失敗した。rtrvc.RVC.__init__ は例外を握り潰して "
                "traceback を print するだけなので、上の出力を確認すること。"
            )

        self.tgt_sr = int(self.rvc.tgt_sr)
        self.is_half = bool(self.config.is_half)
        # RVC は毎回このぶんだけピッチキャッシュを前進させる。非整数だと f0 がずれる。
        self.block_frame_16k = int(round(block_ms * 16))
        if self.block_frame_16k % ZC_SAMPLES_16K != 0:
            os.chdir(self._prev_cwd)
            raise ValueError(
                f"block_ms={block_ms} は 10 ms の倍数でなければならない "
                f"(block_frame_16k={self.block_frame_16k} が {ZC_SAMPLES_16K} で"
                "割り切れず f0 キャッシュがずれる)"
            )

    def infer_fn(self, wav16k: np.ndarray, skip_head: int, return_length: int) -> np.ndarray:
        """16 kHz モノラル窓 -> ``tgt_sr`` の変換結果。

        ``skip_head`` / ``return_length`` は**16 kHz サンプル数**で受け取り、
        ここで RVC の 10 ms 単位へ直す。
        """
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(wav16k, dtype=np.float32)).to(self.device)
        y = self.rvc.infer(
            x,
            self.block_frame_16k,
            to_zc_units(int(skip_head), "skip_head"),
            to_zc_units(int(return_length), "return_length"),
            self.f0method,
        )
        # CUDA は非同期。ここで同期しないと infer 計測が実際より短く出る。
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        return y.detach().float().cpu().numpy().reshape(-1)

    def close(self) -> None:
        try:
            os.chdir(self._prev_cwd)
        except OSError:
            pass


def find_first_model(rvc_root: str = DEFAULT_RVC_ROOT) -> Tuple[Optional[str], str]:
    """``assets/weights`` の .pth と、同名の ``assets/indices/*.index`` を探す。"""
    weights = os.path.join(rvc_root, "assets", "weights")
    indices = os.path.join(rvc_root, "assets", "indices")
    if not os.path.isdir(weights):
        return None, ""
    pths = sorted(f for f in os.listdir(weights) if f.endswith(".pth"))
    if not pths:
        return None, ""
    pth = os.path.join(weights, pths[0])
    stem = os.path.splitext(pths[0])[0]
    for name in (sorted(os.listdir(indices)) if os.path.isdir(indices) else []):
        if name.endswith(".index") and stem in name:
            return pth, os.path.join(indices, name)
    return pth, ""


def build_rvc_engine(args, sr: int, block: int, crossfade: int, extra: int) -> RVCTorchEngine:
    """CLI の引数から RVC エンジンを組み立てる。失敗は例外で返す（CLI が exit=4 にする）。"""
    root = os.path.abspath(args.rvc_root or DEFAULT_RVC_ROOT)
    pth = os.path.abspath(args.rvc_model) if args.rvc_model else None
    index = os.path.abspath(args.rvc_index) if args.rvc_index else ""

    if pth is None:
        pth, auto_index = find_first_model(root)
        if pth is None:
            raise FileNotFoundError(
                f"{os.path.join(root, 'assets', 'weights')} に .pth が無い。\n"
                "  自分の声で学習した話者モデルを置くこと（--rvc-model で明示も可）。"
            )
        index = index or auto_index

    block_ms = block * 1000.0 / sr
    backend = RVCBackend(
        pth_path=pth, block_ms=block_ms, index_path=index,
        index_rate=args.rvc_index_rate, key=args.rvc_key,
        formant=getattr(args, "rvc_formant", 0.0),
        f0method=args.f0_method, rvc_root=root,
    )

    print(f"rvc model   : {os.path.basename(pth)} | tgt_sr {backend.tgt_sr} "
          f"| fp16 {backend.is_half} | {backend.device}")
    print(f"rvc params  : f0={args.f0_method} | key={args.rvc_key} "
          f"| index_rate={args.rvc_index_rate} "
          f"| index={os.path.basename(index) if index else '(なし)'}")

    engine = RVCTorchEngine(
        backend.infer_fn, stream_sr=sr, model_sr=backend.tgt_sr,
        block_ms=block_ms,
        crossfade_ms=crossfade * 1000.0 / sr,
        extra_ms=extra * 1000.0 / sr,
    )
    engine.backend = backend          # keep it alive, and reachable for close()
    engine.close = backend.close      # type: ignore[method-assign]
    return engine
