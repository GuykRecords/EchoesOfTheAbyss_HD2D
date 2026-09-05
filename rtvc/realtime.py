"""Realtime conversion loop and latency instrumentation.

The window
---------

Every iteration builds one analysis window::

    [ EXTRA | X | B ]
      past    xf  new

* ``B``     -- the new block of captured audio (``--block-ms``)
* ``X``     -- crossfade length (``--crossfade-ms``)
* ``EXTRA`` -- past context the model gets for free (``--extra-ms``)

``EXTRA`` is the asymmetry the whole design rests on: it costs compute but
**no latency**, because it is audio that already went past.  So:

* need less latency  -> shrink ``B`` and ``X``
* audio sounds broken -> grow ``EXTRA`` (0.5s -> 1.0s)

Those two knobs are not interchangeable.  Confusing them is the usual way a
realtime VC setup ends up both laggy and bad-sounding.

Algorithmic latency is ``B + X``.  Nothing else in this file adds to it.

The overlap
-----------

The engine converts the whole window but only the last ``X + B`` samples are
kept.  Its first ``X`` samples cover the same instants as the previous
iteration's final ``X``, so the two are equal-power crossfaded; the tail's own
last ``X`` samples are held back as the *のりしろ* (glue) for the next round.
Each iteration therefore emits exactly ``B`` samples and every seam is
smoothed across audio the model saw twice.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .dsp import HighPass, NoiseGate, SoftLimiter, equal_power_windows
from .engines import RVC_GRID_MS, BaseEngine, build_engine, check_rvc_grid
from .timing import HighResolutionTimer

__all__ = ["WindowProcessor", "RealtimeSession", "LatencyReport", "StreamDead",
           "run_offline", "build_parser", "main"]


class StreamDead(RuntimeError):
    """The stream opened but is not delivering audio.

    WASAPI exclusive mode can fail this way: ``start()`` returns, no exception
    is raised, and the callback is simply never invoked.  Every counter then
    stays at zero, which reads as a perfect run -- the most dangerous possible
    way to be broken.
    """


# ---------------------------------------------------------------------------
# Device-independent core
# ---------------------------------------------------------------------------


def sola_offset(candidate: np.ndarray, reference: np.ndarray) -> int:
    """Find where ``reference`` best lines up inside ``candidate``.

    Normalised cross-correlation, the standard SOLA search.  Normalising by
    the candidate's local energy matters: without it the loudest position
    wins rather than the best-matching one, and the join drifts towards
    whatever is loudest instead of whatever fits.
    """
    n = reference.size
    span = candidate.size - n
    if n == 0 or span <= 0:
        return max(0, span)
    correlation = np.correlate(candidate, reference, mode="valid")
    energy = np.sqrt(
        np.convolve(np.square(candidate, dtype=np.float64), np.ones(n), "valid")
    ) + 1e-9
    return int(np.argmax(correlation / energy))


class WindowProcessor:
    """Turns a stream of ``B``-sample blocks into a stream of ``B``-sample blocks.

    Knows nothing about audio devices or threads, which is what makes it
    testable offline: the realtime worker and the ``--offline`` harness drive
    exactly the same object.
    """

    def __init__(
        self,
        engine: BaseEngine,
        sr: int,
        block: int,
        crossfade: int,
        extra: int,
        highpass: bool = True,
        gate: bool = True,
        limiter: bool = True,
        hp_cutoff: float = 80.0,
        sola_search: int = 0,
    ) -> None:
        if block <= 0:
            raise ValueError("block must be > 0")
        if crossfade < 0 or crossfade > block:
            raise ValueError("crossfade must satisfy 0 <= X <= B")
        if extra < 0:
            raise ValueError("extra must be >= 0")
        if sola_search < 0:
            raise ValueError("sola_search must be >= 0")
        if sola_search and crossfade == 0:
            raise ValueError("SOLA needs a crossfade to align against (X > 0)")

        self.engine = engine
        self.sr = int(sr)
        self.block = int(block)
        self.crossfade = int(crossfade)
        self.extra = int(extra)
        #: How far back the join may slide to find alignment. Costs exactly
        #: this much latency, and without it a neural vocoder's blocks cannot
        #: be joined at all -- successive windows are not phase-coherent.
        self.sola_search = int(sola_search)
        self.window_len = self.extra + self.crossfade + self.block
        #: What the engine is asked for: the kept region plus the search span.
        self.tail_len = self.crossfade + self.block + self.sola_search

        self.hp = HighPass(sr, cutoff=hp_cutoff) if highpass else None
        self.gate = NoiseGate(sr) if gate else None
        self.limiter = SoftLimiter(-1.0) if limiter else None
        self.fade_out, self.fade_in = equal_power_windows(self.crossfade)

        self._hist = np.zeros(self.extra + self.crossfade, dtype=np.float32)
        self._overlap: Optional[np.ndarray] = None
        #: Where the last join landed, for reporting. `sola_search` means the
        #: newest possible position; 0 means it slid the whole way back.
        self.last_sola_offset = 0

    # -- latency bookkeeping ------------------------------------------------
    @property
    def block_ms(self) -> float:
        return self.block * 1000.0 / self.sr

    @property
    def crossfade_ms(self) -> float:
        return self.crossfade * 1000.0 / self.sr

    @property
    def extra_ms(self) -> float:
        return self.extra * 1000.0 / self.sr

    @property
    def sola_ms(self) -> float:
        return self.sola_search * 1000.0 / self.sr

    @property
    def algorithmic_latency_ms(self) -> float:
        """``B + X + S``.  EXTRA is deliberately absent -- it is past audio.

        The SOLA span is here because the join may slide that far back, so the
        output can be that far behind the newest input.
        """
        return self.block_ms + self.crossfade_ms + self.sola_ms

    def reset(self) -> None:
        self._hist[:] = 0.0
        self._overlap = None
        self.last_sola_offset = 0
        if self.hp:
            self.hp.reset()
        if self.gate:
            self.gate.reset()

    def warmup(self, iters: int = 8) -> float:
        # Same window length *and* same tail as the live loop, or the
        # one-off costs warmup exists to absorb simply happen again.
        return self.engine.warmup(self.window_len, self.sr, iters=iters,
                                  tail=self.tail_len)

    # -- the loop body ------------------------------------------------------
    def process_block(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if block.size != self.block:
            raise ValueError(f"expected {self.block} samples, got {block.size}")

        pre = block
        if self.hp is not None:
            pre = self.hp.process(pre)
        if self.gate is not None:
            pre = self.gate.process(pre)

        window = np.concatenate((self._hist, pre)) if self._hist.size else pre
        if self._hist.size:
            self._hist = window[-self._hist.size:].copy()

        keep = self.crossfade + self.block
        # Only the tail is ever used, so tell the engine that: a vocoder that
        # honours it synthesises 40ms instead of 140ms.
        converted = self.engine.convert(window, self.sr, tail=self.tail_len)
        tail = converted[-self.tail_len:]

        # Slide the join to where it actually fits. Passthrough is already
        # phase-coherent and always lands at the newest position; a vocoder is
        # not, and this is the difference between speech and mush.
        if self.sola_search and self._overlap is not None:
            offset = sola_offset(tail[:self.crossfade + self.sola_search],
                                 self._overlap)
        else:
            offset = self.sola_search
        self.last_sola_offset = offset
        seg = tail[offset:offset + keep]

        if self.crossfade == 0:
            out = seg.copy()
        elif self._overlap is None:
            # First block: no previous glue to blend against.
            out = seg[:self.block].copy()
        else:
            blended = self._overlap * self.fade_out + seg[:self.crossfade] * self.fade_in
            out = np.concatenate((blended, seg[self.crossfade:keep - self.crossfade]))

        if self.crossfade:
            self._overlap = seg[-self.crossfade:].copy()

        if self.limiter is not None:
            out = self.limiter.process(out)
        return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Latency accounting
# ---------------------------------------------------------------------------


@dataclass
class LatencyReport:
    in_dev_ms: float
    block_ms: float
    infer_ms: float
    xfade_ms: float
    sola_ms: float
    out_buf_ms: float
    out_dev_ms: float
    rtf: float
    underflow: int
    overflow: int
    dropped: int
    loop_ms: float
    elapsed_s: float

    @property
    def total_ms(self) -> float:
        return (self.in_dev_ms + self.block_ms + self.infer_ms + self.xfade_ms
                + self.sola_ms + self.out_buf_ms + self.out_dev_ms)

    def line(self) -> str:
        return (
            f"[{self.elapsed_s:6.1f}s] "
            f"in-dev {self.in_dev_ms:6.2f} | block {self.block_ms:6.2f} | "
            f"infer {self.infer_ms:6.2f} | xfade {self.xfade_ms:5.2f} | "
            f"sola {self.sola_ms:5.2f} | "
            f"out-buf {self.out_buf_ms:5.2f} | out-dev {self.out_dev_ms:6.2f} | "
            f"TOTAL {self.total_ms:7.2f} ms | RTF {self.rtf:5.3f} | "
            f"under {self.underflow} over {self.overflow} drop {self.dropped} | "
            f"loop {self.loop_ms:5.3f}ms"
        )


# ---------------------------------------------------------------------------
# Live session
# ---------------------------------------------------------------------------


class RealtimeSession:
    """Owns the duplex stream, the worker thread and the reporting timer."""

    def __init__(self, processor: WindowProcessor, io, report_sec: float = 2.0,
                 prefill_samples: int = 0, record_in: Optional[str] = None) -> None:
        self.proc = processor
        self.io = io
        self.prefill_samples = int(prefill_samples)
        #: Where to save the raw captured audio, so the same take can be run
        #: through different settings offline instead of being re-performed.
        self.record_in = record_in
        self._recorded: list = []
        self.report_sec = float(report_sec)
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._loop_ms_ema = 0.0
        self._loop_calls = 0
        self.reports: "queue.Queue[LatencyReport]" = queue.Queue()

    # -- worker -------------------------------------------------------------
    def _run_worker(self) -> None:
        block = self.proc.block
        timeout = max(0.005, block / self.proc.sr)
        while not self._stop.is_set():
            if self.io.in_ring.available < block:
                # Clear *before* the final check, otherwise a block that lands
                # between the check and the clear is swallowed and the worker
                # sleeps through audio that is already waiting.
                self.io.input_event.clear()
                if self.io.in_ring.available < block:
                    self.io.input_event.wait(timeout)
                continue
            chunk = self.io.in_ring.read(block)
            if chunk is None:
                continue
            if self.record_in is not None:
                # Raw, before the high-pass and gate, so an offline replay
                # applies exactly the same processing rather than doing it twice.
                self._recorded.append(chunk.copy())
            t0 = time.perf_counter()
            out = self.proc.process_block(chunk)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if self._loop_calls == 0:
                # Lay the cushion down here, not before the stream starts: the
                # output ring drains during the first block's capture, so a
                # pre-roll written earlier would already be gone by now.
                self.io.prefill(self.prefill_samples)
            self.io.stats.dropped_out += self.io.out_ring.write(out)
            # From here on the output ring should never be empty; only now do
            # under/overruns mean something.
            self.io.counting = True

            self._loop_calls += 1
            if self._loop_calls == 1:
                self._loop_ms_ema = dt_ms
            else:
                self._loop_ms_ema += 0.1 * (dt_ms - self._loop_ms_ema)

    # -- reporting ----------------------------------------------------------
    def snapshot(self, elapsed_s: float) -> LatencyReport:
        stats = self.io.stats
        out_buf_samples = self.io.take_out_min_fill()
        return LatencyReport(
            in_dev_ms=self.io.input_latency_ms,
            block_ms=self.proc.block_ms,
            infer_ms=self.proc.engine.infer_ms_ema,
            xfade_ms=self.proc.crossfade_ms,
            sola_ms=self.proc.sola_ms,
            out_buf_ms=out_buf_samples * 1000.0 / self.proc.sr,
            out_dev_ms=self.io.output_latency_ms,
            rtf=self.proc.engine.rtf(self.proc.block_ms),
            underflow=stats.underflow,
            overflow=stats.overflow,
            dropped=stats.dropped_in + stats.dropped_out,
            loop_ms=self._loop_ms_ema,
            elapsed_s=elapsed_s,
        )

    # -- lifecycle ----------------------------------------------------------
    def _assert_alive(self, deadline_s: float = 1.5) -> None:
        """Fail loudly if the device never calls back.

        Checked before anything is reported, because a silent stream produces
        a table full of zeros that looks like a flawless run.
        """
        end = time.perf_counter() + deadline_s
        while time.perf_counter() < end:
            if self.io.stats.callbacks > 0:
                return
            time.sleep(0.02)
        raise StreamDead(
            f"the stream opened but delivered no audio in {deadline_s:.1f}s "
            "(callback count is still 0). The device accepted the format and "
            "then did nothing."
        )

    def run(self, duration: float = 0.0) -> None:
        self.io.counting = False
        self.io.start()
        self._worker = threading.Thread(target=self._run_worker, name="rtvc-worker",
                                        daemon=True)
        self._worker.start()
        started = time.perf_counter()
        next_report = started + self.report_sec
        last_callbacks = 0
        try:
            self._assert_alive()
            while not self._stop.is_set():
                now = time.perf_counter()
                if now >= next_report:
                    seen = self.io.stats.callbacks
                    if seen == last_callbacks:
                        raise StreamDead(
                            f"the stream stopped calling back after {seen} callbacks "
                            f"({now - started:.1f}s in); the numbers below this point "
                            "would be meaningless"
                        )
                    last_callbacks = seen
                    report = self.snapshot(now - started)
                    print(report.line(), flush=True)
                    self.reports.put(report)
                    next_report += self.report_sec
                if duration and (now - started) >= duration:
                    break
                time.sleep(0.02)
        except KeyboardInterrupt:
            print("\nstopping...", file=sys.stderr, flush=True)
        finally:
            self.stop()
            saved = self.save_recording()
            self._print_summary(time.perf_counter() - started)
            if saved:
                seconds = sum(c.size for c in self._recorded) / self.proc.sr
                print(f"recorded {seconds:.1f}s of input to {saved}")
                print("  replay it through other settings without performing again:")
                print(f"  python realtime.py --engine rvc --offline "
                      f"--offline-input {saved} --offline-out out.wav ...")

    def save_recording(self) -> Optional[str]:
        if self.record_in is None or not self._recorded:
            return None
        _write_wav(self.record_in, self.proc.sr, np.concatenate(self._recorded))
        return self.record_in

    def stop(self) -> None:
        self._stop.set()
        self.io.input_event.set()
        if self._worker is not None:
            self._worker.join(timeout=1.0)
            self._worker = None
        self.io.stop()
        self.proc.engine.close()

    def _print_summary(self, elapsed: float) -> None:
        stats = self.io.stats
        engine = self.proc.engine
        print("-" * 78)
        print(f"ran {elapsed:.1f}s | engine {engine.name} | calls {engine.calls}")
        print(f"infer ema {engine.infer_ms_ema:.3f}ms  peak {engine.infer_ms_max:.3f}ms  "
              f"RTF {engine.rtf(self.proc.block_ms):.3f}")
        print(f"algorithmic latency (B+X+S) {self.proc.algorithmic_latency_ms:.2f}ms  "
              f"window {self.proc.window_len} samples "
              f"({self.proc.window_len * 1000.0 / self.proc.sr:.1f}ms)")
        print(f"under {stats.underflow}  over {stats.overflow}  "
              f"drop-in {stats.dropped_in}  drop-out {stats.dropped_out}")
        if stats.status_flags:
            print("portaudio status: " + "; ".join(stats.status_flags))


# ---------------------------------------------------------------------------
# Offline harness (no audio hardware required)
# ---------------------------------------------------------------------------


def _synthetic_signal(sr: int, seconds: float) -> np.ndarray:
    """Speech-ish test signal: swept tone + harmonic + gated silence + noise."""
    n = int(sr * seconds)
    t = np.arange(n, dtype=np.float64) / sr
    f0 = 110.0 + 40.0 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    sig = 0.35 * np.sin(phase) + 0.15 * np.sin(2 * phase) + 0.07 * np.sin(3 * phase)
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 0.4 * t))  # syllable-rate amplitude
    sig *= envelope
    rng = np.random.default_rng(0)
    sig += 0.0005 * rng.standard_normal(n)                 # noise floor under the gate
    return sig.astype(np.float32)


def run_offline(
    processor: WindowProcessor,
    signal: np.ndarray,
    report_sec: float = 2.0,
    verbose: bool = True,
) -> np.ndarray:
    """Push ``signal`` through the real pipeline with no device in the path.

    Exercises exactly the code the live path uses, so a regression in the
    windowing, crossfade or DSP is caught on a machine with no sound card.
    """
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    block = processor.block
    n_blocks = signal.size // block
    out = np.empty(n_blocks * block, dtype=np.float32)

    started = time.perf_counter()
    next_report = report_sec
    for i in range(n_blocks):
        chunk = signal[i * block:(i + 1) * block]
        out[i * block:(i + 1) * block] = processor.process_block(chunk)
        audio_s = (i + 1) * block / processor.sr
        if verbose and report_sec and audio_s >= next_report:
            eng = processor.engine
            print(
                f"[{audio_s:6.1f}s audio] infer {eng.infer_ms_ema:6.3f}ms "
                f"peak {eng.infer_ms_max:6.3f}ms | block {processor.block_ms:.2f} "
                f"| xfade {processor.crossfade_ms:.2f} "
                f"| algo-latency {processor.algorithmic_latency_ms:.2f}ms "
                f"| RTF {eng.rtf(processor.block_ms):.3f}",
                flush=True,
            )
            next_report += report_sec
    wall = time.perf_counter() - started

    if verbose:
        audio_s = n_blocks * block / processor.sr
        print("-" * 78)
        print(f"offline: {n_blocks} blocks / {audio_s:.2f}s audio in {wall:.2f}s wall "
              f"(x{audio_s / wall if wall else float('inf'):.1f} realtime)")
        print(f"engine {processor.engine.name}: ema {processor.engine.infer_ms_ema:.3f}ms "
              f"peak {processor.engine.infer_ms_max:.3f}ms "
              f"RTF {processor.engine.rtf(processor.block_ms):.3f}")
        print(f"algorithmic latency (B+X+S) {processor.algorithmic_latency_ms:.2f}ms")
        print(f"out peak {float(np.max(np.abs(out))) if out.size else 0.0:.4f}  "
              f"non-finite {int(np.count_nonzero(~np.isfinite(out)))}")
    return out


def _read_wav(path: str, sr: int) -> np.ndarray:
    from scipy.io import wavfile

    file_sr, data = wavfile.read(path)
    data = np.asarray(data)
    if data.ndim > 1:
        data = data[:, 0]
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    else:
        data = data.astype(np.float32)
    if file_sr != sr:
        from .dsp import Resampler

        data = Resampler(file_sr, sr).process(data)
    return data


def _write_wav(path: str, sr: int, data: np.ndarray) -> None:
    from scipy.io import wavfile

    wavfile.write(path, sr, np.clip(data, -1.0, 1.0).astype(np.float32))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class _Help(argparse.ArgumentDefaultsHelpFormatter):
    """Show defaults, except the ones that are resolved later from --engine."""

    def _get_help_string(self, action):
        if action.default is None:
            return action.help
        return super()._get_help_string(action)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="realtime.py",
        description="Realtime voice-conversion I/O and latency measurement tool.",
        formatter_class=_Help,
    )
    p.add_argument("--list-devices", action="store_true",
                   help="print the audio device table and exit")

    dev = p.add_argument_group("devices")
    dev.add_argument("--in-device", default=None,
                     help="input device: a name substring (recommended) or an index")
    dev.add_argument("--out-device", default=None,
                     help="output device: a name substring (recommended) or an index")
    dev.add_argument("--host-api", default=None,
                     help="restrict device names to one host API, e.g. WASAPI. "
                          "Indices move when Windows re-enumerates; a name plus "
                          "--host-api does not")
    dev.add_argument("--sr", type=int, default=48000, help="stream sample rate")
    dev.add_argument("--io-block", type=int, default=128,
                     help="PortAudio blocksize in samples (device-side granularity)")
    dev.add_argument("--channels", type=int, default=1, help="channels per direction")
    dev.add_argument("--latency", default="low",
                     help="PortAudio latency hint: low, high, or seconds as a float")
    dev.add_argument("--exclusive", action="store_true",
                     help="request WASAPI exclusive mode (Windows only)")

    # Left as None so "the user typed the default value" is distinguishable
    # from "the user typed nothing" -- the RVC window depends on that.
    win = p.add_argument_group("window")
    win.add_argument("--block-ms", type=float, default=None,
                     help="B: new audio per window. Controls how clear the "
                          "conversion is (default: 32, or 130 for --engine rvc)")
    win.add_argument("--crossfade-ms", type=float, default=None,
                     help="X: crossfade length. Controls how smooth the joins "
                          "are (default: 8, or 50 for --engine rvc)")
    win.add_argument("--extra-ms", type=float, default=None,
                     help="EXTRA: past context; costs compute, not latency "
                          "(default: 500, or 2500 for --engine rvc)")
    win.add_argument("--sola-search-ms", type=float, default=None,
                     help="S: how far the join may slide to find alignment. "
                          "Costs this much latency. A neural vocoder needs it -- "
                          "its blocks are not phase-coherent (default: 0, or 10 "
                          "for --engine rvc)")

    eng = p.add_argument_group("engine")
    eng.add_argument("--engine", default="passthrough",
                     choices=("passthrough", "fixed", "rvc"),
                     help="conversion engine")
    eng.add_argument("--fixed-cost-ms", type=float, default=25.0,
                     help="simulated inference cost for --engine fixed")
    eng.add_argument("--warmup", type=int, default=8, help="warmup iterations before going live")

    rvc = p.add_argument_group("rvc (only with --engine rvc)")
    rvc.add_argument("--rvc-root", default=None, help="path to the RVC checkout")
    rvc.add_argument("--rvc-model", default=None, help="speaker .pth (default: the only one found)")
    rvc.add_argument("--rvc-index", default=None, help="faiss .index file")
    rvc.add_argument("--rvc-index-rate", type=float, default=0.0,
                     help="faiss blend; searching every block is expensive, start at 0")
    rvc.add_argument("--rvc-index-nprobe", type=int, default=16,
                     help="how many IVF cells the index search looks at. faiss "
                          "defaults to 1, which returns fewer than the 8 neighbours "
                          "RVC asks for; RVC then silently drops the block's index "
                          "blend and blames the index file. 0 leaves faiss alone")
    rvc.add_argument("--rvc-key", type=int, default=0, help="pitch shift in semitones")
    rvc.add_argument("--rvc-formant", type=float, default=0.0, help="formant shift")
    rvc.add_argument("--f0-method", default="rmvpe", choices=("rmvpe", "fcpe"),
                     help="harvest/crepe are not realtime-capable and are not offered")

    dsp = p.add_argument_group("dsp")
    dsp.add_argument("--no-gate", action="store_true", help="disable the noise gate")
    dsp.add_argument("--gate", action="store_true",
                     help="force the noise gate on. It is off by default for "
                          "--engine rvc: measured on real speech, it eats the "
                          "quiet ends of words before the model ever sees them")
    dsp.add_argument("--no-highpass", action="store_true", help="disable the 80 Hz high-pass")
    dsp.add_argument("--no-limiter", action="store_true", help="disable the output limiter")

    run = p.add_argument_group("run")
    run.add_argument("--report-sec", type=float, default=2.0, help="stats interval")
    run.add_argument("--duration", type=float, default=0.0,
                     help="stop after N seconds (0 = run until Ctrl+C)")
    run.add_argument("--ring-seconds", type=float, default=2.0, help="ring buffer capacity")
    run.add_argument("--prefill-ms", default=None,
                     help="output cushion in ms, or 'auto' to size it from the measured "
                          "warmup cost. Costs exactly that much latency and shows up in "
                          "out-buf. Default: 0, or 'auto' for --engine rvc")

    off = p.add_argument_group("offline (no audio hardware)")
    off.add_argument("--offline", action="store_true",
                     help="run the pipeline on a file or synthetic signal instead of devices")
    off.add_argument("--offline-input", default=None, help="WAV file to process")
    off.add_argument("--offline-seconds", type=float, default=10.0,
                     help="length of the synthetic signal when no input file is given")
    off.add_argument("--offline-out", default=None, help="write the processed audio here")
    off.add_argument("--record-in", default=None, metavar="PATH",
                     help="save the captured input to a WAV during a live run, so the "
                          "same take can be replayed through other settings offline")
    return p


def _parse_latency(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_devices:
        from .audio_io import format_device_table

        try:
            print(format_device_table(args.host_api))
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 3
        return 0

    sr = args.sr
    block_ms, crossfade_ms, extra_ms, sola_ms = resolve_window(args)
    is_rvc = args.engine == "rvc"

    if is_rvc:
        try:
            # The tail asked of RVC is X + B + S, so the search span has to sit
            # on the same 10ms grid as everything else.
            check_rvc_grid(block_ms=block_ms, crossfade_ms=crossfade_ms,
                           extra_ms=extra_ms, sola_search_ms=sola_ms)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    block = max(1, int(round(sr * block_ms / 1000.0)))
    crossfade = int(round(sr * crossfade_ms / 1000.0))
    extra = int(round(sr * extra_ms / 1000.0))
    sola_search = int(round(sr * sola_ms / 1000.0))
    if crossfade > block:
        print(f"error: --crossfade-ms ({crossfade_ms}) must not exceed "
              f"--block-ms ({block_ms})", file=sys.stderr)
        return 2

    # Raised for the whole run, warmup included: at Windows' default ~15.6ms
    # tick a 30ms warmup measures 33-37ms, and the pre-roll sized from it is
    # far too large. See rtvc/timing.py.
    clock = HighResolutionTimer()
    clock.start()
    try:
        return _run(args, sr, block, crossfade, extra, sola_search, clock)
    finally:
        clock.stop()


#: Paths the user typed, which must be pinned down before anything can move
#: the working directory out from under them.
_USER_PATHS = ("offline_input", "offline_out", "record_in")


def resolve_window(args) -> tuple:
    """Fill in the window sizes the user did not give, and settle the gate.

    RVC gets its own defaults, and they are the ones listening tests settled
    on rather than the ones that keep the latency figure tidy: 30/10/100 with
    a gate produced audio that was not speech. Also stores ``args.gate_on``.
    """
    is_rvc = args.engine == "rvc"
    fallback = (130.0, 50.0, 2500.0, 10.0) if is_rvc else (32.0, 8.0, 500.0, 0.0)
    block_ms = fallback[0] if args.block_ms is None else args.block_ms
    crossfade_ms = fallback[1] if args.crossfade_ms is None else args.crossfade_ms
    extra_ms = fallback[2] if args.extra_ms is None else args.extra_ms
    sola_ms = fallback[3] if args.sola_search_ms is None else args.sola_search_ms
    # A gate in front of a voice converter trades the ends of words for a
    # little less room noise, which is a bad trade: the model needs those.
    args.gate_on = bool(args.gate or (not args.no_gate and not is_rvc))
    return block_ms, crossfade_ms, extra_ms, sola_ms


def absolutise_user_paths(args) -> None:
    """Resolve the user's file arguments against the directory they typed them in.

    The RVC backend has to ``chdir`` into the RVC checkout -- RVC reads its
    assets by relative path, during the first inference as well as at load.
    That silently re-bases every relative path the user gave us, so a
    ``--offline-out out.wav`` lands inside the RVC checkout and a
    ``--offline-input take.wav`` is not found at all.
    """
    import os as _os

    for name in _USER_PATHS:
        value = getattr(args, name, None)
        if value:
            setattr(args, name, _os.path.abspath(value))


def _run(args, sr, block, crossfade, extra, sola_search, clock) -> int:
    absolutise_user_paths(args)
    is_rvc = args.engine == "rvc"
    if is_rvc:
        try:
            from .rvc_backend import build_rvc_engine
        except ImportError:
            print(
                "error: --engine rvc needs rtvc/rvc_backend.py, which is not in this "
                "checkout.\n"
                "  It must expose:\n"
                "      build_rvc_engine(args, sr, block, crossfade, extra) -> RVCTorchEngine\n"
                "  and hand RVCTorchEngine a callable shaped either\n"
                "      infer_fn(wav16k)                              -> audio at model_sr\n"
                "      infer_fn(wav16k, skip_head, return_length)    -> audio at model_sr\n"
                "  with skip_head/return_length in 16 kHz SAMPLES (divide by 160 for\n"
                "  RVC's zc units), and torch.cuda.synchronize() before returning.\n"
                "  See docs/RVC_INTEGRATION.md.",
                file=sys.stderr)
            return 4
        try:
            engine = build_rvc_engine(args, sr, block, crossfade, extra)
        except (RuntimeError, FileNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 4
    else:
        engine = build_engine(args.engine, sr, fixed_cost_ms=args.fixed_cost_ms)
    try:
        proc = WindowProcessor(
            engine, sr, block, crossfade, extra,
            highpass=not args.no_highpass,
            gate=args.gate_on,
            limiter=not args.no_limiter,
            sola_search=sola_search,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"rtvc | sr {sr} | B {block} ({proc.block_ms:.2f}ms) "
          f"X {crossfade} ({proc.crossfade_ms:.2f}ms) "
          f"S {sola_search} ({proc.sola_ms:.2f}ms) "
          f"EXTRA {extra} ({proc.extra_ms:.2f}ms) "
          f"window {proc.window_len} ({proc.window_len * 1000.0 / sr:.1f}ms)")
    print(f"engine {engine.name} | algorithmic latency B+X+S = "
          f"{proc.algorithmic_latency_ms:.2f}ms | gate "
          f"{'on' if args.gate_on else 'off'} | hp "
          f"{'off' if args.no_highpass else 'on'}")
    print(clock.describe())

    warm_ms = proc.warmup(args.warmup)

    # A cushion sized by hand is a guess; sized from the warmup it is a
    # measurement. Only meaningful once the scheduler tick is raised, which is
    # why the timer is held around warmup too.
    spec = args.prefill_ms if args.prefill_ms is not None else ("auto" if is_rvc else "0")
    if str(spec).lower() == "auto":
        cushion_ms = warm_ms
        origin = f"auto (steady warmup {warm_ms:.2f}ms over {args.warmup} iterations)"
    else:
        try:
            cushion_ms = float(spec)
        except ValueError:
            print(f"error: --prefill-ms must be a number or 'auto', got {spec!r}",
                  file=sys.stderr)
            return 2
        origin = "given"
    print(f"warmup      : steady {warm_ms:.2f}ms  peak {proc.engine.warmup_ms_peak:.2f}ms")
    print(f"out cushion : {cushion_ms:.2f}ms  [{origin}]")

    if args.offline:
        signal = (_read_wav(args.offline_input, sr) if args.offline_input
                  else _synthetic_signal(sr, args.offline_seconds))
        out = run_offline(proc, signal, report_sec=args.report_sec)
        if args.offline_out:
            _write_wav(args.offline_out, sr, out)
            print(f"wrote {args.offline_out} ({out.size} samples)")
        return 0

    from .audio_io import AudioIO, resolve_device

    try:
        in_device = resolve_device(args.in_device, "input", args.host_api)
        out_device = resolve_device(args.out_device, "output", args.host_api)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    io = AudioIO(
        sr=sr,
        blocksize=args.io_block,
        in_device=in_device,
        out_device=out_device,
        channels=args.channels,
        latency=_parse_latency(args.latency),
        exclusive=args.exclusive,
        ring_seconds=args.ring_seconds,
    )
    session = RealtimeSession(
        proc, io, report_sec=args.report_sec,
        prefill_samples=int(round(sr * cushion_ms / 1000.0)),
        record_in=args.record_in,
    )
    try:
        print(io.describe_devices())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    print("running -- Ctrl+C to stop")
    try:
        session.run(duration=args.duration)
    except StreamDead as exc:
        print(f"error: {exc}", file=sys.stderr)
        if args.exclusive:
            print("hint: exclusive mode fails silently on some devices -- the stream "
                  "opens and never calls back. Open one side at a time to find which "
                  "device refuses it.", file=sys.stderr)
        return 3
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
