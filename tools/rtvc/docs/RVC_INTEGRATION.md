# RVC 本体の導入と接続

> **⚠ 初期の引き継ぎメモとの差分が大きい。** 実機で確認した結果、旧メモの
> 「`tools/rvc_for_realtime.py` を使う」「fairseq が要る」「`protect` / `rms_mix_rate`
> を設定する」は**いずれも現在の RVC には当てはまらない**。以下が実測に基づく現状。

---

## 0. 環境

| 項目 | 値 | 備考 |
|---|---|---|
| venv | `D:\Claude\Project\.venv-rvc` | Python 3.10.9 |
| torch | **2.7.1+cu128** | upstream が「2.8 未満」で固定している。上げない |
| numpy | 1.26.4 | |
| transformers | 4.49.0 | HuBERT のロードに使う |
| faiss-cpu | 1.15.0 | index 検索 |
| torchfcpe | 0.0.4 | `--f0-method fcpe` 用 |
| RVC | `D:\Claude\Project\RVC` HEAD 81eed5e | |

**`.venv-rvc` で計測ツールも動く**ことは検証済み。プロセスは 1 つで済む。

upstream の `requirements.txt` は Python 3.12 用しか無いが、
**リアルタイム推論に必要な部分集合は 3.10 で全部入る**（検証済み）。

> パスは ASCII のみ。日本語やスペースを含むパスに置くと落ちる既知の問題がある。

---

## 1. 旧メモとの差分（重要な 5 点）

### ① `tools/rvc_for_realtime.py` は存在しない

現在は **`infer/rtrvc.py`** の `RVC` クラス。バッチ用の
`infer/modules/vc/pipeline.py` は使わない（f0 キャッシュが効かず realtime に届かない）。

### ② fairseq は依存から消えた

HuBERT は **transformers** で `assets/hubert_base` からローカル読み込みする。
「numpy 1.23.5 + fairseq で壊れる」という旧メモの懸念はもう当てはまらない。
（それでも計測用 `.venv` と RVC 用 `.venv-rvc` は分けたまま運用している）

### ③ torch は 2.7.1+cu128 固定

upstream が "must stay below 2.8" と明示している。計測用 `.venv` の torch 2.11 とは別物。

### ④ `protect` / `rms_mix_rate` はリアルタイム経路に存在しない

`infer/cli.py` とバッチ側だけの引数。旧メモの「`protect = 0.5` / `rms_mix_rate = 0.25`」は
**リアルタイムでは設定する場所が無い**。

### ⑤ 10ms グリッド制約

`skip_head` / `return_length` は `zc = sr // 100` の **10ms 単位**。
さらにピッチキャッシュの前進量 `block_frame_16k // 160` も整数でないと
**f0 が毎ブロックずれる**。

→ **`--block-ms` / `--crossfade-ms` / `--extra-ms` は必ず 10 の倍数。**

`B=30 / X=10 / EXTRA=100` ならアルゴリズム遅延 `B+X = 40ms` で従来と同値。
`--engine rvc` を指定して窓を明示しなければ、この 30/10/100 に自動で切り替わる。
10 の倍数でない値を渡すと `ValueError` で**起動時に**弾かれる。

```
RVC needs every window size on a 10ms grid; got block_ms=32.0ms, crossfade_ms=8.0ms.
Use e.g. block 30 / crossfade 10 / extra 100 (B+X is still 40ms).
```

実行時に静かにピッチがずれるより、起動時に止まる方がはるかにましなので、ここは厳格にしてある。

---

## 2. 接続インタフェース

役割分担はこうなっている。

| モジュール | 依存 | 役割 |
|---|---|---|
| `rtvc/engines.py` | numpy / scipy のみ | 窓・tail・長さの契約・計測 |
| `rtvc/rvc_backend.py` | **torch と RVC** | `infer/rtrvc.py` の `RVC` を握る。単位換算もここ |

`rvc_backend.py` は torch を**遅延 import** するので、torch の無い環境でも
`import rtvc.rvc_backend` は通る（CI がそれを確かめている）。

### エンジンに注入する callable

```python
# ① 単純形（エンジン側が末尾を切り出す）
infer_fn(wav16k) -> np.ndarray            # model_sr のオーディオ

# ② tail 対応形（推奨。rvc_backend.py はこちら）
infer_fn(wav16k, skip_head, return_length) -> np.ndarray
```

**`skip_head` / `return_length` は 16kHz サンプル数で渡す。**
RVC が要求する 10ms 単位（`zc = sr // 100`、16kHz では 160 サンプル）への換算は
`rvc_backend.to_zc_units()` が担当し、割り切れなければ**例外を投げる**。
黙って丸めると f0 が毎ブロック少しずつずれていくため。

### 実際の RVC 側 API（実機で確認済み）

```python
from configs.config import Config
from infer.rtrvc import RVC

rvc = RVC(key, formant, pth_path, index_path, index_rate, config)
y = rvc.infer(x_tensor_16k, block_frame_16k, skip_head_zc, return_length_zc, f0method)
```

- `skip_head` / `return_length` は **10ms 単位**
- `block_frame_16k` は **16kHz サンプル数**（`block_ms * 16`）
- `RVC.__init__` は**例外を握り潰して traceback を print するだけ**なので、
  戻ったあと `net_g is None` を確認しないと壊れたまま進む
- **`Config()` は自前の argparse で `sys.argv` を読む。**
  RVC は単独アプリなのでそれで正しいが、ライブラリとして呼ぶと
  `--engine` などを自分の引数として解釈し、`unrecognized arguments` で
  **プロセスごと落とす**。`rvc_backend.py` は構築の間だけ argv を退避する
- RVC はカレントディレクトリ相対で `assets/` を読む箇所がある（`rmvpe.pt` など）ため、
  `rvc_backend.py` は `os.chdir(rvc_root)` してから import する（`close()` で戻す）

### なぜ ② が推奨か — 約 3.4 倍

ループが実際に使うのは末尾 `X + B` だけ。
`return_length` を渡せば vocoder の合成量を **窓 140ms → 末尾 40ms** に削れる。
`infer < 32ms` を狙うなら実質必須。

`--engine rvc` では自動的に ② が使われ、warmup も**本番と同じ窓長・同じ tail**で 8 回走る。

### 出力クッションは warmup から自動で決まる

`--engine rvc` では `--prefill-ms` の既定が **`auto`** になる。
warmup の**後半イテレーションの最大値**（コールドスタートを除いた定常推定）を
そのままクッションに使う。

```
warmup      : steady 18.42ms  peak 512.30ms
out cushion : 18.42ms  [auto (steady warmup 18.42ms over 8 iterations)]
```

初回出力は `B + infer` 後にしか出ないので、ここを見誤ると起動直後の
アンダーランでクッションを食い潰し、余裕ゼロのまま走ることになる。
数値を明示すれば手動指定もできる（`--prefill-ms 12`）。

---

## 3. 計測上の注意 — `torch.cuda.synchronize()`

**CUDA は非同期。** `infer_fn` の中で最後に `torch.cuda.synchronize()` を呼ばないと、
GPU の処理完了を待たずに戻るため **infer 時間が実際より小さく出る**。

同じ理由で warmup も synchronize 込みで回すこと。

## 3.1 Windows のタイマ分解能

Windows の既定スケジューラ刻みは ~15.6ms。PortAudio はストリームを開いている間だけ
これを上げるので、**ストリーム開始前に走る warmup は粗いクロックで測られる**。
実測で 30ms の warmup が 33〜37ms とぶれ、そこから算出したプリフィルが過大になっていた。

ツール側は `rtvc/timing.py` の `HighResolutionTimer` で
`timeBeginPeriod(1)` を**実行全体にわたって**自前で握る。起動時に状態を表示する。

---

## 4. モデルのサンプルレートは 48k か 24k を選ぶ

`StreamResampler` の実測:

| 比 | rms 誤差 |
|---|---|
| 整数比（48k ↔ 16k, 48k ↔ 24k） | **−56 dB** |
| 非整数比（48k ↔ 44.1k, 48k ↔ 40k） | **−35 dB** |

**20dB の差。** モデル側レートは 48k の整数分の 1（16k / 24k）か、48k そのものを選ぶ。
40k モデルを使うと `RVCTorchEngine` が起動時に `RuntimeWarning` を出す。

---

## 5. 推論パラメータ

| パラメータ | 初期値 | 備考 |
|---|---|---|
| `f0_method` | `rmvpe` | 速度不足なら `fcpe`。**`harvest` / `crepe` はリアルタイム不可**なので CLI にも無い |
| `index_rate` | 0.0〜0.3 | faiss 検索はブロック毎だとコストが高い。既定 0.0 |
| `key` | 0 | 半音単位のピッチシフト |
| `is_half` (fp16) | `True` | Ada なので有効 |
| ~~`protect`~~ | — | **リアルタイム経路に存在しない** |
| ~~`rms_mix_rate`~~ | — | **リアルタイム経路に存在しない** |

---

## 6. CLI

```powershell
python realtime.py --engine rvc --host-api WASAPI `
  --in-device "INZONE Buds - Chat" --out-device "CABLE Input" `
  --rvc-root D:\Claude\Project\RVC
```

- 窓を明示しなければ **30/10/100** に自動設定される
- `--rvc-model` を省くと `assets/weights` の先頭の `.pth` と、同名の `.index` を自動で探す
- `--prefill-ms` を省くと **`auto`**（warmup の定常推定）
- **デバイスは番号ではなく名前で指定する。** index は Windows が再列挙するたびにずれる

その他: `--rvc-index-rate` (既定 0.0) / `--rvc-key` (半音) / `--rvc-formant` /
`--f0-method {rmvpe,fcpe}`

| 終了コード | 意味 |
|---|---|
| 0 | 正常終了 |
| 2 | 窓の指定が不正（10ms グリッド違反、X > B など） |
| 3 | ストリームが開いたのに音が流れない／途中で止まった |
| 4 | RVC の初期化に失敗（`.pth` が無い、RVC が見つからない など） |

---

## 7. 接続前の動作確認（GPU 不要）

形が合っているかは偽の推論関数で先に確かめられる。

```python
import numpy as np
from rtvc.engines import RVCTorchEngine

def fake_infer(wav16k, skip_head, return_length):
    return np.zeros(int(return_length * 48000 / 16000), dtype=np.float32)

eng = RVCTorchEngine(fake_infer, stream_sr=48000, model_sr=48000,
                     block_ms=30, crossfade_ms=10, extra_ms=100)
assert eng.tail_aware
out = eng.convert(np.zeros(6720, dtype=np.float32), 48000, tail=1920)
assert out.size == 1920
```

同じ内容が `tests/test_pipeline.py` の
`test_rvc_engine_detects_a_tail_aware_backend_and_passes_16k_units` として自動テストに入っている。

---

## 8. 速度が出なかったときの順番

1. `return_length`（tail）が効いているか確認する — `eng.tail_aware` が `True` か
2. `f0_method` を `rmvpe` → `fcpe` に落とす
3. `--rvc-index-rate` を 0.0 にする（faiss を切る）
4. `--extra-ms` を 100 → 減らす（**遅延ではなく計算量が減る**。10 の倍数を維持）
5. `--block-ms` を 30 → 40 に**増やす**（推論回数が減り RTF が下がる。ただし遅延は増える）

**`--crossfade-ms` を削って速度を稼ごうとしないこと。** X は計算量にほぼ効かず、
削ると継ぎ目が鳴るだけで損しかしない。
