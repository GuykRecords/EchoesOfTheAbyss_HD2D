"""変換エンジン。realtime.py からは engine.convert(x, sr) だけを呼ぶ。

BaseEngine が推論時間の計測（EMA / peak）と warmup を担当し、
サブクラスは _convert() だけ実装する。
"""
from __future__ import annotations

import inspect
import time

import numpy as np

from dsp import Resampler


class BaseEngine:
    name = "base"

    def __init__(self, ema_alpha: float = 0.1):
        self.ema_alpha = float(ema_alpha)
        self.infer_ms_ema = 0.0
        self.infer_ms_last = 0.0
        self.infer_ms_peak = 0.0
        self.calls = 0
        self.warmed = False
        self.warmup_ms_total = 0.0
        self.warmup_ms_mean = 0.0
        self.warmup_ms_peak = 0.0
        self.warmup_ms_steady = 0.0

    # --- サブクラスが実装する ---
    def _convert(self, x: np.ndarray, sr: int, tail: int) -> np.ndarray:
        raise NotImplementedError

    # --- 共通の計測ラッパ ---
    def convert(self, x: np.ndarray, sr: int, tail: int | None = None) -> np.ndarray:
        """窓 x を変換する。tail を渡すと末尾 tail サンプルだけ返してよい。

        呼び出し側は末尾 B+X しか使わないので、tail を尊重できるエンジンは
        そのぶん合成量を減らせる（窓 136 ms に対し B+X 40 ms なら約 1/3.4）。
        """
        want = int(tail) if tail else int(x.size)
        t0 = time.perf_counter()
        y = self._convert(x, sr, want)
        dt = (time.perf_counter() - t0) * 1000.0

        self.calls += 1
        self.infer_ms_last = dt
        if self.calls == 1:
            self.infer_ms_ema = dt
        else:
            a = self.ema_alpha
            self.infer_ms_ema = a * dt + (1.0 - a) * self.infer_ms_ema
        if dt > self.infer_ms_peak:
            self.infer_ms_peak = dt

        y = np.asarray(y, dtype=np.float32)
        if y.size != want:  # 長さは呼び出し側の前提なので必ず合わせる
            if y.size > want:
                y = y[-want:]
            else:
                y = np.concatenate([np.zeros(want - y.size, dtype=np.float32), y])
        return y

    def warmup(self, sr: int, n_samples: int, iters: int = 3, tail: int | None = None) -> float:
        """本番と同じ窓長でから回しし、初回のもたつきを計測から追い出す。

        戻り値は「後半イテレーションの最大 ms」= 定常推定。呼び出し側は出力リングの
        プリフィル量を決めるのにこれを使う（初回出力は B + infer 後になるため）。
        """
        dummy = np.zeros(n_samples, dtype=np.float32)
        want = int(tail) if tail else n_samples
        each = []
        for _ in range(max(1, iters)):
            t0 = time.perf_counter()
            self._convert(dummy, sr, want)
            each.append((time.perf_counter() - t0) * 1000.0)
        self.warmup_ms_total = float(sum(each))
        self.warmup_ms_mean = float(sum(each) / len(each))
        self.warmup_ms_peak = float(max(each))
        # 前半（コールドスタート）を捨てた定常推定。プリフィルはこれで決める。
        tail = each[len(each) // 2:] or each
        self.warmup_ms_steady = float(max(tail))
        # warmup 分は本計測の統計に入れない
        self.infer_ms_ema = 0.0
        self.infer_ms_last = 0.0
        self.infer_ms_peak = 0.0
        self.calls = 0
        self.warmed = True
        return self.warmup_ms_steady

    def reset_stats(self) -> None:
        self.infer_ms_ema = 0.0
        self.infer_ms_last = 0.0
        self.infer_ms_peak = 0.0
        self.calls = 0


class PassthroughEngine(BaseEngine):
    """素通し。I/O とクロスフェード経路そのものの遅延・音質を測るための基準。"""

    name = "passthrough"

    def _convert(self, x: np.ndarray, sr: int, tail: int) -> np.ndarray:
        return x[-tail:]


class FixedCostEngine(BaseEngine):
    """指定 ms だけ必ず食う擬似エンジン。実エンジンを載せる前の余裕度テスト用。

    Windows の time.sleep は分解能が粗いので、粗いスリープ + スピン待ちで詰める。
    """

    name = "fixed"

    def __init__(self, cost_ms: float = 20.0, spin_ms: float = 2.0, **kw):
        super().__init__(**kw)
        self.cost_ms = float(cost_ms)
        self.spin_ms = float(spin_ms)

    def _convert(self, x: np.ndarray, sr: int, tail: int) -> np.ndarray:
        deadline = time.perf_counter() + self.cost_ms / 1000.0
        coarse = deadline - self.spin_ms / 1000.0
        now = time.perf_counter()
        if coarse > now:
            time.sleep(coarse - now)
        while time.perf_counter() < deadline:
            pass
        return x[-tail:]


class RVCTorchEngine(BaseEngine):
    """RVC (infer/rtrvc.py の RVC クラス) を rtvc の窓設計に接続する。

    engines.py を torch / RVC 非依存に保つため、推論そのものは呼び出し側から
    callable として注入する。RVC 側の venv (.venv-rvc) に依存するのは注入元だけ。

    注入する callable:
        infer_fn(wav16k: np.ndarray, skip_head: int, return_length: int)
            -> np.ndarray  (model_sr, 長さ return_length * model_sr / 100)

        skip_head / return_length はどちらも **10 ms 単位** (RVC の zc = sr//100)。
        引数 1 個の infer_fn(wav16k) も受け付ける（その場合は末尾を切り出して使う）が、
        vocoder が窓全体を合成することになるので推奨しない。

    10 ms グリッド制約:
        RVC のリアルタイム経路は 10 ms 量子化されている。
        - skip_head = extra_frame // zc、return_length = (...) // zc がどちらも整数
        - ピッチキャッシュの前進量 shift = block_frame_16k // 160 も整数でないと
          f0 が毎ブロックずれていく
        よって block-ms / crossfade-ms / extra-ms はすべて 10 の倍数でなければならない。
        B=30 / X=10 / EXTRA=100 ならアルゴリズム遅延は B+X=40 ms で現行構成と同じ。
    """

    name = "rvc"
    ZC_MS = 10.0

    def __init__(self, infer_fn, model_sr: int, io_sr: int,
                 block_ms: float, xfade_ms: float, extra_ms: float, **kw):
        super().__init__(**kw)
        for label, v in (("block-ms", block_ms), ("crossfade-ms", xfade_ms), ("extra-ms", extra_ms)):
            if abs(v / self.ZC_MS - round(v / self.ZC_MS)) > 1e-6:
                raise ValueError(
                    f"{label}={v} は 10 ms の倍数でなければならない。"
                    f"RVC のリアルタイム経路は zc=sr//100 の 10 ms グリッドで、"
                    f"端数があると skip_head/return_length が丸められ f0 キャッシュがずれる。"
                )
        self.infer_fn = infer_fn
        self.model_sr = int(model_sr)
        self.io_sr = int(io_sr)
        self.block_ms = float(block_ms)
        self.xfade_ms = float(xfade_ms)
        self.extra_ms = float(extra_ms)

        self.skip_head_zc = int(round(extra_ms / self.ZC_MS))
        self.block_frame_16k = int(round(block_ms * 16))  # 16 kHz サンプル数

        self._rs = Resampler(quality="HQ")
        try:  # 引数 1 個の簡易 infer_fn も許容する
            self._takes_head = len(inspect.signature(infer_fn).parameters) >= 3
        except (TypeError, ValueError):
            self._takes_head = True

    def _convert(self, x: np.ndarray, sr: int, tail: int) -> np.ndarray:
        if sr != self.io_sr:
            raise ValueError(f"io_sr mismatch: engine={self.io_sr} caller={sr}")

        wav16k = self._rs(x, self.io_sr, 16000)
        return_length_zc = int(round(tail / self.io_sr * 1000.0 / self.ZC_MS))

        if self._takes_head:
            y = self.infer_fn(wav16k, self.skip_head_zc, return_length_zc)
        else:
            y = self.infer_fn(wav16k)

        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if self.model_sr != self.io_sr:
            y = self._rs(y, self.model_sr, self.io_sr)
        return y  # 長さ合わせは BaseEngine.convert が行う


ENGINES = {
    "passthrough": PassthroughEngine,
    "fixed": FixedCostEngine,
}


def build_engine(kind: str, **kw) -> BaseEngine:
    if kind not in ENGINES:
        raise ValueError(f"unknown engine: {kind} (choices: {', '.join(ENGINES)})")
    cls = ENGINES[kind]
    if cls is FixedCostEngine:
        return cls(cost_ms=kw.get("cost_ms", 20.0))
    return cls()
