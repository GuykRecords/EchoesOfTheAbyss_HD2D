"""デバイス I/O 層。

sd.Stream の duplex を 1 本だけ張り、コールバックではリングバッファの
読み書きしかしない（推論・確保・print は一切しない）。重い処理は別スレッド。
"""
from __future__ import annotations

import threading
import time

import numpy as np
import sounddevice as sd


def format_devices() -> str:
    return str(sd.query_devices())


def print_devices() -> None:
    print(format_devices())


class RingBuffer:
    """lock 付き単一リングバッファ。溢れたら古い方を捨てる（遅延を溜めない）。"""

    def __init__(self, capacity: int, dtype=np.float32):
        self.cap = int(capacity)
        self.buf = np.zeros(self.cap, dtype=dtype)
        self.dtype = dtype
        self._w = 0
        self._r = 0
        self._count = 0
        self._lock = threading.Lock()
        self.overflow_samples = 0   # 古い方を捨てた総サンプル数
        self.overflow_events = 0
        self.underflow_samples = 0  # 読み出しで足りずゼロ埋めした総サンプル数
        self.underflow_events = 0
        self._min_level = self.cap  # read() 直後の最小占有量（=真のクッション量）

    def __len__(self) -> int:
        return self.available()

    def available(self) -> int:
        with self._lock:
            return self._count

    def clear(self) -> None:
        with self._lock:
            self._w = self._r = self._count = 0

    def write(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=self.dtype)
        n = x.size
        if n == 0:
            return
        if n > self.cap:  # 1回の書き込みが容量超え。新しい方だけ残す
            self.overflow_samples += n - self.cap
            self.overflow_events += 1
            x = x[-self.cap:]
            n = self.cap
        with self._lock:
            free = self.cap - self._count
            if n > free:  # 古い方を捨てて場所を空ける
                drop = n - free
                self._r = (self._r + drop) % self.cap
                self._count -= drop
                self.overflow_samples += drop
                self.overflow_events += 1
            end = self._w + n
            if end <= self.cap:
                self.buf[self._w:end] = x
            else:
                k = self.cap - self._w
                self.buf[self._w:] = x[:k]
                self.buf[:n - k] = x[k:]
            self._w = end % self.cap
            self._count += n

    def read(self, n: int, out: np.ndarray | None = None) -> np.ndarray:
        """n サンプル読む。足りない分は 0 埋めしてアンダーフローを計上する。"""
        if out is None:
            out = np.zeros(n, dtype=self.dtype)
        else:
            out[:] = 0.0
        with self._lock:
            m = min(n, self._count)
            if m < n:
                self.underflow_samples += n - m
                self.underflow_events += 1
            if m > 0:
                end = self._r + m
                if end <= self.cap:
                    out[:m] = self.buf[self._r:end]
                else:
                    k = self.cap - self._r
                    out[:k] = self.buf[self._r:]
                    out[k:m] = self.buf[:m - k]
                self._r = end % self.cap
                self._count -= m
            if self._count < self._min_level:
                self._min_level = self._count
        return out

    def take_min_level(self) -> int:
        """前回呼び出し以降の最小占有量を返してリセットする。"""
        with self._lock:
            m = min(self._min_level, self._count)
            self._min_level = self._count
        return m

    def discard(self, n: int) -> int:
        """先頭 n サンプルを捨てる。実際に捨てた数を返す。"""
        with self._lock:
            m = min(n, self._count)
            self._r = (self._r + m) % self.cap
            self._count -= m
        return m


class AudioIO:
    """duplex ストリーム 1 本 + 入出力リングバッファ。"""

    def __init__(
        self,
        in_device,
        out_device,
        samplerate: int = 48000,
        blocksize: int = 128,
        ring_seconds: float = 2.0,
        input_channel: int = 0,
        exclusive: bool = False,
        latency: str | float = "low",
    ):
        self.sr = int(samplerate)
        self.blocksize = int(blocksize)
        self.input_channel = int(input_channel)
        self.exclusive = bool(exclusive)

        in_info = sd.query_devices(in_device, "input")
        out_info = sd.query_devices(out_device, "output")
        self.in_name = in_info["name"]
        self.out_name = out_info["name"]
        self.in_channels = max(1, min(2, int(in_info["max_input_channels"])))
        self.out_channels = max(1, min(2, int(out_info["max_output_channels"])))
        if self.input_channel >= self.in_channels:
            self.input_channel = 0

        cap = int(self.sr * ring_seconds)
        self.in_ring = RingBuffer(cap)
        self.out_ring = RingBuffer(cap)

        self.block_event = threading.Event()
        self.callback_count = 0
        self.status_input_overflow = 0
        self.status_output_underflow = 0
        self.callback_errors = 0
        self._out_scratch = np.zeros(self.blocksize, dtype=np.float32)

        extra = None
        if self.exclusive:
            try:
                extra = (sd.WasapiSettings(exclusive=True), sd.WasapiSettings(exclusive=True))
            except Exception as e:  # WASAPI 以外のホストAPIなら黙って共有モード
                print(f"[audio_io] WASAPI exclusive is unavailable ({e}); falling back to shared mode")
                extra = None

        self.stream = sd.Stream(
            device=(in_device, out_device),
            samplerate=self.sr,
            blocksize=self.blocksize,
            dtype="float32",
            channels=(self.in_channels, self.out_channels),
            latency=latency,
            callback=self._callback,
            extra_settings=extra,
        )

    # --- リアルタイムスレッド（ここで重い処理をしてはいけない） ---
    def _callback(self, indata, outdata, frames, time_info, status):
        if status:
            if status.input_overflow:
                self.status_input_overflow += 1
            if status.output_underflow:
                self.status_output_underflow += 1
        try:
            self.callback_count += 1
            self.in_ring.write(indata[:, self.input_channel])
            if frames != self._out_scratch.size:
                self._out_scratch = np.zeros(frames, dtype=np.float32)
            y = self.out_ring.read(frames, out=self._out_scratch)
            for ch in range(outdata.shape[1]):
                outdata[:, ch] = y
            self.block_event.set()
        except Exception:
            self.callback_errors += 1
            outdata.fill(0.0)

    # --- 制御 ---
    def start(self) -> None:
        self.stream.start()

    def stop(self) -> None:
        try:
            self.stream.stop()
        finally:
            self.stream.close()

    def wait_for_callbacks(self, timeout: float = 2.0, need: int = 2) -> bool:
        """ストリームが実際に回り始めたか確認する。

        WASAPI 排他 duplex のように「open も start も成功するのにコールバックが
        1 回も来ない」構成があるため、無音を計測して成功と誤認しないよう明示的に見る。
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.callback_count >= need:
                return True
            time.sleep(0.02)
        return self.callback_count >= need

    def wait_block(self, timeout: float = 0.5) -> bool:
        ok = self.block_event.wait(timeout)
        self.block_event.clear()
        return ok

    @property
    def device_latency_ms(self) -> tuple[float, float]:
        lat = self.stream.latency
        return (float(lat[0]) * 1000.0, float(lat[1]) * 1000.0)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False
