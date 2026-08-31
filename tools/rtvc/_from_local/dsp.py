"""リアルタイム音声処理用の DSP プリミティブ。

すべてモノラル float32 の 1 次元配列を前提とし、ブロック境界をまたいで
状態を保持する（ブロック単位で呼んでもクリックが出ないこと）。
"""
from __future__ import annotations

import numpy as np
from scipy import signal

try:  # soxr は任意。無ければ scipy にフォールバックする
    import soxr as _soxr
except Exception:  # pragma: no cover
    _soxr = None


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def lin_to_db(x: float, eps: float = 1e-12) -> float:
    return float(20.0 * np.log10(max(float(x), eps)))


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def rms_db(x: np.ndarray) -> float:
    return lin_to_db(rms(x))


class HighPass:
    """butter + sosfilt のハイパス。zi をインスタンスに持ってブロック間で継続する。"""

    def __init__(self, sr: int, cutoff_hz: float = 80.0, order: int = 2):
        self.sr = sr
        self.cutoff_hz = cutoff_hz
        self.sos = signal.butter(order, cutoff_hz, btype="highpass", fs=sr, output="sos")
        # sosfilt_zi は定常入力用の初期値なので、無音スタート想定でゼロから始める
        self.zi = np.zeros((self.sos.shape[0], 2), dtype=np.float64)

    def reset(self) -> None:
        self.zi[:] = 0.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x.astype(np.float32, copy=False)
        y, self.zi = signal.sosfilt(self.sos, x.astype(np.float64, copy=False), zi=self.zi)
        return y.astype(np.float32, copy=False)


class NoiseGate:
    """RMS 判定 + ヒステリシス + ホールドのノイズゲート。

    ゲインは 1 次 IIR (one-pole) でサンプル単位に滑らかに追従させるので、
    開閉のたびにブチッと鳴らない。
    """

    def __init__(
        self,
        sr: int,
        open_db: float = -42.0,
        close_db: float = -50.0,
        hold_ms: float = 180.0,
        attack_ms: float = 5.0,
        release_ms: float = 60.0,
        floor_gain: float = 0.0,
    ):
        self.sr = sr
        self.open_db = open_db
        self.close_db = close_db
        self.hold_samples = int(round(sr * hold_ms / 1000.0))
        self.attack_coef = float(np.exp(-1.0 / max(1.0, sr * attack_ms / 1000.0)))
        self.release_coef = float(np.exp(-1.0 / max(1.0, sr * release_ms / 1000.0)))
        self.floor_gain = float(floor_gain)
        self._gain = self.floor_gain
        self._hold = 0
        self.is_open = False
        self.last_level_db = -120.0

    def reset(self) -> None:
        self._gain = self.floor_gain
        self._hold = 0
        self.is_open = False

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, bool]:
        if x.size == 0:
            return x.astype(np.float32, copy=False), self.is_open

        level = rms_db(x)
        self.last_level_db = level

        if level >= self.open_db:
            self.is_open = True
            self._hold = self.hold_samples
        elif level < self.close_db:
            if self._hold > 0:
                self._hold -= x.size
            else:
                self.is_open = False
        # close_db <= level < open_db はヒステリシス帯。状態を維持する。

        target = 1.0 if self.is_open else self.floor_gain
        coef = self.attack_coef if target > self._gain else self.release_coef
        # g[n] = target + (g[-1] - target) * coef^(n+1)  → 1次IIRの解析解
        n = np.arange(1, x.size + 1, dtype=np.float64)
        env = target + (self._gain - target) * np.power(coef, n)
        self._gain = float(env[-1])
        return (x.astype(np.float32, copy=False) * env.astype(np.float32)), self.is_open


def equal_power_windows(n: int) -> tuple[np.ndarray, np.ndarray]:
    """等パワークロスフェード窓 (fade_out, fade_in) を返す。fade_out^2 + fade_in^2 == 1。"""
    if n <= 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty.copy()
    t = (np.arange(n, dtype=np.float64) + 0.5) / n
    fade_out = np.cos(0.5 * np.pi * t).astype(np.float32)
    fade_in = np.sin(0.5 * np.pi * t).astype(np.float32)
    return fade_out, fade_in


class SoftLimiter:
    """tanh のソフトリミッタ。小信号はほぼ線形、上限は ceiling_db に漸近する。"""

    def __init__(self, ceiling_db: float = -1.0):
        self.ceiling_db = ceiling_db
        self.ceiling = db_to_lin(ceiling_db)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        c = self.ceiling
        return (c * np.tanh(x.astype(np.float32, copy=False) / c)).astype(np.float32, copy=False)


class Resampler:
    """ワンショットのリサンプラ。soxr があれば優先、無ければ scipy.resample_poly。"""

    def __init__(self, quality: str = "HQ"):
        self.quality = quality
        self.backend = "soxr" if _soxr is not None else "scipy"

    def __call__(self, x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        if sr_in == sr_out or x.size == 0:
            return x
        if _soxr is not None:
            return _soxr.resample(x, sr_in, sr_out, quality=self.quality).astype(np.float32, copy=False)
        g = int(np.gcd(int(sr_in), int(sr_out)))
        return signal.resample_poly(x, sr_out // g, sr_in // g).astype(np.float32, copy=False)


class StreamResampler:
    """ブロック逐次リサンプラ (overlap-trim)。

    直前ブロックの末尾 margin サンプルを「のりしろ」として先頭に付けてから
    リサンプルし、のりしろ相当の出力を捨てる。これでフィルタの立ち上がりが
    ブロック境界に出ないためクリックが乗らない。出力長は累積比で決めるので
    長時間流してもドリフトしない。

    注意: 48k<->16k/24k のような整数比では一括リサンプルとほぼ一致する（実測 rms -56 dB）。
    48k->44.1k のような非整数比はブロックごとに出力グリッドの位相が端数でずれるため
    誤差が増える（実測 rms -35 dB）。モデル側のレートは 48k の整数分の 1 を選ぶこと。
    """

    def __init__(self, sr_in: int, sr_out: int, margin_ms: float = 10.0, quality: str = "HQ"):
        self.sr_in = int(sr_in)
        self.sr_out = int(sr_out)
        self.ratio = self.sr_out / self.sr_in
        self.margin = max(1, int(round(sr_in * margin_ms / 1000.0)))
        self.prev = np.zeros(self.margin, dtype=np.float32)
        self._acc = 0.0
        self._r = Resampler(quality=quality)
        self.backend = self._r.backend

    def reset(self) -> None:
        self.prev[:] = 0.0
        self._acc = 0.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        if self.sr_in == self.sr_out or x.size == 0:
            return x

        buf = np.concatenate([self.prev, x])
        y = self._r(buf, self.sr_in, self.sr_out)

        # 次回ののりしろ（今回入力の末尾 margin サンプル）
        if x.size >= self.margin:
            self.prev = x[-self.margin:].copy()
        else:
            self.prev = np.concatenate([self.prev[x.size:], x])

        self._acc += x.size * self.ratio
        want = int(np.floor(self._acc))
        self._acc -= want

        drop = int(round(self.margin * self.ratio))
        seg = y[drop:drop + want]
        if seg.size < want:  # 端数で足りない場合だけ末尾を複製して埋める
            pad = np.full(want - seg.size, seg[-1] if seg.size else 0.0, dtype=np.float32)
            seg = np.concatenate([seg, pad])
        return seg.astype(np.float32, copy=False)
