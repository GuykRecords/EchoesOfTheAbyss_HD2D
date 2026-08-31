"""RVC 側のグルー。engines.RVCTorchEngine に渡す infer_fn を組み立てる。

このモジュールだけが RVC 本体と torch に依存する。**.venv-rvc で実行すること**
（.venv は torch 2.11 で、RVC が要求する torch<2.8 と非互換）。
.venv-rvc には sounddevice / scipy / soxr も入っているので、
realtime.py ごと .venv-rvc で走らせればプロセスは 1 つで済む。

    from engines import RVCTorchEngine
    from rvc_backend import RVCBackend

    be = RVCBackend(pth_path=r"D:\\...\\assets\\weights\\me.pth", block_ms=30)
    engine = RVCTorchEngine(be.infer_fn, model_sr=be.tgt_sr, io_sr=48000,
                            block_ms=30, xfade_ms=10, extra_ms=100)
"""
from __future__ import annotations

import os
import sys

import numpy as np

DEFAULT_RVC_ROOT = r"D:\Claude\Project\RVC"


class RVCBackend:
    """infer/rtrvc.py の RVC を保持し、numpy in / numpy out の infer_fn を提供する。"""

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
    ):
        if not os.path.isfile(pth_path):
            raise FileNotFoundError(f"話者モデルが見つからない: {pth_path}")
        if index_rate and not os.path.isfile(index_path):
            raise FileNotFoundError(f"index_rate>0 なのに index が無い: {index_path}")
        if f0method not in ("rmvpe", "fcpe"):
            raise ValueError(f"f0method は rmvpe か fcpe のみ（harvest/crepe はリアルタイム不可）: {f0method}")

        # RVC はカレントディレクトリ相対で assets/ を読む箇所がある（rmvpe.pt など）
        self.rvc_root = os.path.abspath(rvc_root)
        if self.rvc_root not in sys.path:
            sys.path.insert(0, self.rvc_root)
        self._prev_cwd = os.getcwd()
        os.chdir(self.rvc_root)

        import torch

        from configs.config import Config
        from infer.rtrvc import RVC

        self.torch = torch
        self.config = Config()
        self.device = self.config.device
        self.f0method = f0method

        self.rvc = RVC(key, formant, pth_path, index_path, index_rate, self.config)
        if getattr(self.rvc, "net_g", None) is None:
            raise RuntimeError(
                "RVC の初期化に失敗した（rtrvc.RVC.__init__ は例外を握り潰して "
                "traceback を print するだけなので、上の出力を確認すること）"
            )

        self.tgt_sr = int(self.rvc.tgt_sr)
        self.is_half = bool(self.config.is_half)
        # RVC は毎回このぶんだけピッチキャッシュを前進させる。非整数だと f0 がずれる。
        self.block_frame_16k = int(round(block_ms * 16))
        if self.block_frame_16k % 160 != 0:
            raise ValueError(
                f"block_ms={block_ms} は 10 ms の倍数でなければならない "
                f"(block_frame_16k={self.block_frame_16k} が 160 で割り切れず f0 キャッシュがずれる)"
            )

    def infer_fn(self, wav16k: np.ndarray, skip_head: int, return_length: int) -> np.ndarray:
        """16 kHz モノラル窓 -> tgt_sr の変換結果。skip_head/return_length は 10 ms 単位。"""
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(wav16k, dtype=np.float32)).to(self.device)
        y = self.rvc.infer(x, self.block_frame_16k, int(skip_head), int(return_length), self.f0method)
        # CUDA は非同期。ここで同期しないと infer 計測が実際より短く出る。
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        return y.detach().float().cpu().numpy().reshape(-1)

    def close(self) -> None:
        try:
            os.chdir(self._prev_cwd)
        except OSError:
            pass


def find_first_model(rvc_root: str = DEFAULT_RVC_ROOT):
    """assets/weights の .pth と、同名の assets/indices/*.index を探す。"""
    weights = os.path.join(rvc_root, "assets", "weights")
    indices = os.path.join(rvc_root, "assets", "indices")
    if not os.path.isdir(weights):
        return None, ""
    pths = sorted(f for f in os.listdir(weights) if f.endswith(".pth"))
    if not pths:
        return None, ""
    pth = os.path.join(weights, pths[0])
    stem = os.path.splitext(pths[0])[0]
    idx = ""
    if os.path.isdir(indices):
        for f in sorted(os.listdir(indices)):
            if f.endswith(".index") and stem in f:
                idx = os.path.join(indices, f)
                break
    return pth, idx


if __name__ == "__main__":
    # 話者モデルが置かれているかの確認だけ行う（推論は realtime.py 側から）
    pth, idx = find_first_model()
    if pth is None:
        print("assets/weights に .pth がまだ無い。自分の声で学習したモデルを置くこと。")
        raise SystemExit(1)
    print(f"model : {pth}")
    print(f"index : {idx or '(なし)'}")
    be = RVCBackend(pth_path=pth, block_ms=30, index_path=idx, index_rate=0.0)
    print(f"tgt_sr={be.tgt_sr} | is_half={be.is_half} | device={be.device} "
          f"| block_frame_16k={be.block_frame_16k}")
