"""リアルタイム音声変換の I/O 計測ツール。

    [ 過去コンテキスト EXTRA | クロスフェード X | ブロック B ]  <- engine.convert に渡す窓
                              |<------ 末尾 B+X を取り出す ----->|

取り出した B+X のうち、先頭 X を前回の「のりしろ」と等パワークロスフェードして
B サンプルを出力し、末尾 X を次回ののりしろとして保持する。
アルゴリズム遅延は B + X（ブロック待ち + のりしろ分の先送り）。

デバイス I/O はコールバック 1 本（duplex）で、コールバックはリング操作のみ。
推論とクロスフェードはワーカースレッドで行う。
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import threading
import time

import numpy as np

from audio_io import AudioIO, format_devices
from dsp import HighPass, NoiseGate, SoftLimiter, equal_power_windows
from engines import build_engine


class Worker(threading.Thread):
    """ブロック単位で 窓組み -> engine.convert -> クロスフェード -> 出力 を回す。"""

    def __init__(self, io: AudioIO, engine, block: int, xfade: int, extra: int,
                 use_gate: bool = True, max_backlog_blocks: int = 4):
        super().__init__(name="rtvc-worker", daemon=True)
        self.io = io
        self.engine = engine
        self.sr = io.sr
        self.B = block
        self.X = xfade
        self.EXTRA = extra
        self.W = extra + xfade + block

        self.hp = HighPass(self.sr, cutoff_hz=80.0, order=2)
        self.gate = NoiseGate(self.sr, open_db=-42.0, close_db=-50.0, hold_ms=180.0) if use_gate else None
        self.limiter = SoftLimiter(ceiling_db=-1.0)
        self.fade_out, self.fade_in = equal_power_windows(self.X)

        self.hist = np.zeros(self.W, dtype=np.float32)       # 直近 W サンプルの入力
        self.prev_tail = np.zeros(self.X, dtype=np.float32)  # 前回ののりしろ

        self.max_backlog = self.B * max_backlog_blocks
        self._running = threading.Event()
        self._running.set()

        # 統計
        self.blocks = 0
        self.dropped_blocks = 0
        self.loop_ms_ema = 0.0
        self.loop_ms_peak = 0.0
        self.gate_open = False

    def stop(self) -> None:
        self._running.clear()

    def run(self) -> None:
        B, X = self.B, self.X
        while self._running.is_set():
            if self.io.in_ring.available() < B:
                self.io.wait_block(0.2)
                continue

            # 追いつけていない分は捨てて遅延を溜めない
            while self.io.in_ring.available() > self.max_backlog:
                self.io.in_ring.discard(B)
                self.dropped_blocks += 1

            blk = self.io.in_ring.read(B)
            t0 = time.perf_counter()

            blk = self.hp(blk)
            if self.gate is not None:
                blk, self.gate_open = self.gate(blk)

            # 窓を左にずらして新ブロックを末尾に足す
            self.hist = np.concatenate([self.hist[B:], blk])

            converted = self.engine.convert(self.hist, self.sr, tail=B + X)
            seg = converted[-(B + X):]  # 末尾 B+X

            out = seg[:B].copy()
            if X > 0:
                out[:X] = self.prev_tail * self.fade_out + seg[:X] * self.fade_in
                self.prev_tail = seg[B:B + X].copy()

            self.io.out_ring.write(self.limiter(out))

            dt = (time.perf_counter() - t0) * 1000.0
            self.blocks += 1
            self.loop_ms_ema = dt if self.blocks == 1 else 0.1 * dt + 0.9 * self.loop_ms_ema
            self.loop_ms_peak = max(self.loop_ms_peak, dt)


@contextlib.contextmanager
def windows_timer_resolution(ms: int = 1):
    """Windows のシステムタイマ分解能を上げる。

    PortAudio はストリーム開始時にこれを上げるが、warmup とワーカーの sleep は
    その前後にも走るので自前で握っておく。上げないと warmup 実測が 10 ms 単位で
    ぶれ、プリフィルが過大になる。
    """
    if sys.platform != "win32":
        yield
        return
    import ctypes
    winmm = ctypes.WinDLL("winmm")
    ok = winmm.timeBeginPeriod(ms) == 0
    try:
        yield ok
    finally:
        if ok:
            winmm.timeEndPeriod(ms)


def parse_device(v):
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="realtime voice-conversion I/O latency meter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in-device", default=None, help="入力デバイス (index or name)")
    p.add_argument("--out-device", default=None, help="出力デバイス (index or name)")
    p.add_argument("--sr", type=int, default=48000, help="サンプルレート")
    p.add_argument("--io-block", type=int, default=128, help="デバイスの blocksize [frames]")
    p.add_argument("--block-ms", type=float, default=None, help="処理ブロック B [ms] (既定: 32、engine=rvc なら 30)")
    p.add_argument("--crossfade-ms", type=float, default=None, help="クロスフェード X [ms] (既定: 8、engine=rvc なら 10)")
    p.add_argument("--extra-ms", type=float, default=None, help="過去コンテキスト EXTRA [ms] (既定: 96、engine=rvc なら 100)")
    p.add_argument("--engine", choices=["passthrough", "fixed", "rvc"], default="passthrough")
    p.add_argument("--fixed-cost-ms", type=float, default=20.0, help="engine=fixed のときの擬似推論時間")
    p.add_argument("--rvc-model", default=None, help="engine=rvc: 話者モデル .pth (省略時は assets/weights の先頭)")
    p.add_argument("--rvc-index", default=None, help="engine=rvc: .index (省略時は同名を自動探索)")
    p.add_argument("--rvc-index-rate", type=float, default=0.0, help="faiss 検索の混合率。0.0〜0.3 推奨")
    p.add_argument("--rvc-key", type=int, default=0, help="ピッチシフト [半音]")
    p.add_argument("--f0-method", choices=["rmvpe", "fcpe"], default="rmvpe",
                   help="rmvpe が既定。速度不足なら fcpe（harvest/crepe はリアルタイム不可）")
    p.add_argument("--rvc-root", default=r"D:\Claude\Project\RVC", help="RVC 本体の場所")
    p.add_argument("--prefill-ms", type=float, default=16.0,
                   help="出力リングの定常クッション目標 [ms]（実プリフィル = B + これ）")
    p.add_argument("--exclusive", action="store_true", help="WASAPI 排他モード")
    p.add_argument("--no-gate", action="store_true", help="ノイズゲートを無効化")
    p.add_argument("--warmup-iters", type=int, default=None,
                   help="warmup の回数 (既定: 3、engine=rvc なら 8)")
    p.add_argument("--list-devices", action="store_true", help="デバイス一覧を表示して終了")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_devices:
        print(format_devices())
        return 0

    with windows_timer_resolution(1):
        return _run(args)


def _run(args) -> int:
    is_rvc = args.engine == "rvc"
    # engine=rvc は 10 ms グリッド必須なので既定値もそれに合わせる（B+X=40 ms は据え置き）
    if args.block_ms is None:
        args.block_ms = 30.0 if is_rvc else 32.0
    if args.crossfade_ms is None:
        args.crossfade_ms = 10.0 if is_rvc else 8.0
    if args.extra_ms is None:
        args.extra_ms = 100.0 if is_rvc else 96.0
    if args.warmup_iters is None:
        # RVC は CUDA Graph の確保とカーネル autotune で初回が重い
        args.warmup_iters = 8 if is_rvc else 3

    sr = args.sr
    B = max(1, int(round(sr * args.block_ms / 1000.0)))
    X = max(0, int(round(sr * args.crossfade_ms / 1000.0)))
    EXTRA = max(0, int(round(sr * args.extra_ms / 1000.0)))
    if X > B:
        print(f"error: crossfade-ms ({args.crossfade_ms}) must be <= block-ms ({args.block_ms})", file=sys.stderr)
        return 2

    backend = None
    if is_rvc:
        from engines import RVCTorchEngine
        from rvc_backend import RVCBackend, find_first_model

        pth, idx = args.rvc_model, args.rvc_index
        if pth is None:
            pth, auto_idx = find_first_model(args.rvc_root)
            if pth is None:
                print("error: %s に .pth が無い。"
                      % os.path.join(args.rvc_root, "assets", "weights"), file=sys.stderr)
                print("       自分の声で学習した話者モデルを置くこと（--rvc-model で明示も可）。", file=sys.stderr)
                return 4
            if idx is None:
                idx = auto_idx
        backend = RVCBackend(
            pth_path=pth, block_ms=args.block_ms, index_path=idx or "",
            index_rate=args.rvc_index_rate, key=args.rvc_key,
            f0method=args.f0_method, rvc_root=args.rvc_root,
        )
        engine = RVCTorchEngine(
            backend.infer_fn, model_sr=backend.tgt_sr, io_sr=sr,
            block_ms=args.block_ms, xfade_ms=args.crossfade_ms, extra_ms=args.extra_ms,
        )
    else:
        engine = build_engine(args.engine, cost_ms=args.fixed_cost_ms)

    io = AudioIO(
        in_device=parse_device(args.in_device),
        out_device=parse_device(args.out_device),
        samplerate=sr,
        blocksize=args.io_block,
        exclusive=args.exclusive,
    )

    W = EXTRA + X + B
    warm_est_ms = engine.warmup(sr, W, iters=args.warmup_iters, tail=B + X)

    block_ms = B / sr * 1000.0
    xfade_ms = X / sr * 1000.0
    in_dev_ms, out_dev_ms = io.device_latency_ms

    bar = "=" * 92
    print(bar)
    print(f"engine      : {engine.name}" + (f" (cost {args.fixed_cost_ms:.1f} ms)" if args.engine == "fixed" else ""))
    if backend is not None:
        print(f"rvc model   : {os.path.basename(pth)} | tgt_sr {backend.tgt_sr} | fp16 {backend.is_half} "
              f"| device {backend.device}")
        print(f"rvc params  : f0={args.f0_method} | key={args.rvc_key} | index_rate={args.rvc_index_rate} "
              f"| index={os.path.basename(idx) if idx else '(なし)'}")
    print(f"in  device  : [{args.in_device}] {io.in_name}  ({io.in_channels}ch, use ch{io.input_channel})")
    print(f"out device  : [{args.out_device}] {io.out_name}  ({io.out_channels}ch)")
    print(f"samplerate  : {sr} Hz | io-block {args.io_block} frames ({args.io_block / sr * 1000.0:.2f} ms)"
          f" | exclusive={args.exclusive}")
    print(f"window      : EXTRA {EXTRA} + X {X} + B {B} = {W} samples ({W / sr * 1000.0:.1f} ms)")
    print(f"algo delay  : B + X = {block_ms:.1f} + {xfade_ms:.1f} = {block_ms + xfade_ms:.1f} ms")
    print(f"gate        : {'off' if args.no_gate else 'on (-42/-50 dB, hold 180 ms)'} | HPF 80 Hz | limiter -1 dBFS")
    print(f"warmup      : {args.warmup_iters} iters | mean {engine.warmup_ms_mean:.2f} ms "
          f"| peak {engine.warmup_ms_peak:.2f} ms | steady {warm_est_ms:.2f} ms")
    print(f"out prefill : B + infer + cushion = {block_ms:.1f} + {warm_est_ms:.1f} + {args.prefill_ms:.1f} "
          f"= {block_ms + warm_est_ms + args.prefill_ms:.1f} ms")
    print(bar)

    worker = Worker(io, engine, B, X, EXTRA, use_gate=not args.no_gate)

    # ワーカーの初回出力は「B ぶん溜まってから infer 時間の後」なので、その両方を
    # 上乗せしないと起動時アンダーランでクッションを食い潰し、定常クッション 0 で回る。
    prefill = B + int(round(sr * (warm_est_ms + args.prefill_ms) / 1000.0))
    io.out_ring.write(np.zeros(prefill, dtype=np.float32))

    worker.start()
    io.start()

    if not io.wait_for_callbacks(timeout=2.0):
        worker.stop()
        worker.join(timeout=1.0)
        io.stop()
        print("error: ストリームは開いたがオーディオコールバックが 1 回も呼ばれなかった。", file=sys.stderr)
        print("       計測値は無意味なので中断する。", file=sys.stderr)
        if args.exclusive:
            print("       原因はほぼ確実に WASAPI 排他 duplex。PortAudio はこのデバイス対で", file=sys.stderr)
            print("       入力側を排他にした duplex を開始できない（例外は出ないまま無反応）。", file=sys.stderr)
            print("       --exclusive を外すか、入出力を別ストリームに分ける必要がある。", file=sys.stderr)
        return 3

    t_start = time.perf_counter()
    t_next = t_start + 2.0
    io.out_ring.take_min_level()
    last_cb = io.callback_count
    try:
        while True:
            time.sleep(0.05)
            now = time.perf_counter()
            if now < t_next:
                continue
            t_next += 2.0

            infer_ms = engine.infer_ms_ema
            # 出力リングの最小占有量 = 実測クッション。平均を使うと block(B) と
            # ドレイン分が二重計上になるので最小を採る。
            out_buf_ms = io.out_ring.take_min_level() / sr * 1000.0
            total = in_dev_ms + block_ms + infer_ms + xfade_ms + out_buf_ms + out_dev_ms
            rtf = worker.loop_ms_ema / block_ms if block_ms > 0 else 0.0
            under = io.out_ring.underflow_events + io.status_output_underflow
            over = io.in_ring.overflow_events + io.status_input_overflow

            cb = io.callback_count
            if cb == last_cb:
                print(f"[{now - t_start:6.1f}s] STREAM STALLED: このウィンドウでコールバックが 0 回。"
                      f"以降の計測値は信用できない。", flush=True)
            last_cb = cb

            print(
                f"[{now - t_start:6.1f}s] "
                f"in-dev {in_dev_ms:5.2f} | block {block_ms:6.2f} | infer {infer_ms:6.2f} | "
                f"xfade {xfade_ms:5.2f} | out-buf {out_buf_ms:5.2f} | out-dev {out_dev_ms:5.2f} | "
                f"TOTAL {total:7.2f} ms | RTF {rtf:5.3f} | "
                f"under={under} over={over} drop={worker.dropped_blocks}",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\n-- stopping --", flush=True)
    finally:
        worker.stop()
        worker.join(timeout=1.0)
        io.stop()
        if backend is not None:
            backend.close()

    elapsed = time.perf_counter() - t_start
    print(bar)
    print(f"elapsed        : {elapsed:.2f} s")
    print(f"blocks         : {worker.blocks} processed / {worker.dropped_blocks} dropped")
    print(f"callbacks      : {io.callback_count} (errors {io.callback_errors})")
    print(f"infer          : ema {engine.infer_ms_ema:.3f} ms | peak {engine.infer_ms_peak:.3f} ms "
          f"| calls {engine.calls}")
    print(f"worker loop    : ema {worker.loop_ms_ema:.3f} ms | peak {worker.loop_ms_peak:.3f} ms "
          f"| RTF(peak) {worker.loop_ms_peak / block_ms:.3f}")
    print(f"out underflow  : {io.out_ring.underflow_events} ring events "
          f"({io.out_ring.underflow_samples} samples) | device {io.status_output_underflow}")
    print(f"in  overflow   : {io.in_ring.overflow_events} ring events "
          f"({io.in_ring.overflow_samples} samples) | device {io.status_input_overflow}")
    print(bar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
