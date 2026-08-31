# リアルタイムVC 進捗（2026-08-26 時点）

## 1. 完了していること

### rtvc 計測ツール（`D:\Claude\Project\rtvc`）
`dsp.py` / `audio_io.py` / `engines.py` / `realtime.py` / `rvc_backend.py`。
venv は `D:\Claude\Project\.venv`（torch 2.11）だが、**`.venv-rvc` でも全部動く**ので
RVC を使うときはプロセス 1 つで済む（検証済み）。

オフライン検証済み:
- 線形クロスフェードに差し替えた passthrough の再構成誤差 **6e-8** → 窓組みと `B+X` のアライメントは厳密に正しい
- HPF のブロック分割 vs 一括 = 誤差 1.5e-8、ゲートは出力の最大差分が入力と一致（クリック無し）
- 等パワー窓が passthrough で +3dB 盛るのは原理どおり（実測 0.207 = 0.5×(√2−1) と解析値一致）
- `StreamResampler` は整数比なら rms −56dB、48k→44.1k など非整数比は −35dB まで悪化 →
  **モデル側レートは 48k の整数分の1（16k/24k）を選ぶこと**

### 実測ベースライン（in=23 INZONE / out=22 CABLE Input / WASAPI共有）
| 構成 | TOTAL | RTF | under/over/drop |
|---|---|---|---|
| passthrough B32/X8 | 94.67 ms | 0.01 | 0/0/0 |
| fixed 25ms B32/X8 | 126.00 ms | 0.79 | 0/0/0 |
| fixed 25ms B30/X10/EXTRA100（RVC互換） | 122.70 ms | 0.85 | 0/0/0 |

内訳は `in-dev 22.00 + block + infer + xfade + out-buf + out-dev 24.67`。

### 途中で潰したバグ（すべて修正済み）
1. **out-buf を平均占有量で計上していた** → 平均は B/2 のドレイン分を含み `block` と二重計上。最小占有量に変更（実測が16ms過大だった）
2. **プリフィルが推論時間を見ていなかった** → 初回出力は B+infer 後なので、`B + infer + cushion` に変更。これが無いと起動時アンダーランでクッションを食い潰し定常0で走る
3. **warmup がストリーム開始前で Windows タイマ分解能が上がっていなかった** → `timeBeginPeriod(1)` を自前で握る。warmup 実測が 33〜37ms とぶれてプリフィルが過大になっていた
4. **排他モードのサイレント失敗** → コールバック0回でも under=0 と表示されていた。起動確認と途中ストール検出を追加、exit=3 で停止

### WASAPI 排他の結論（検証完了・再試行不要）
| 構成 | 結果 |
|---|---|
| in=23 排他 単独 | 開くが **実効 51079 Hz（+6.42%）で破綻** |
| out=22 排他 単独 | lat **4.50ms** / 48000.9 Hz 正常 |
| in=23 を含む duplex 排他 | **DEAD**（cb=0、例外なし） |
| in=24 排他 duplex | OK（duplex排他自体は可能） |
| duplex排他が成立する構成の報告値 | 42.67/45.33ms = **共有より悪化** |
| 共有 duplex + 明示 latency 0.001〜0.01 | 全て 22.00/24.67ms から動かない |

**犯人は INZONE Buds（ワイヤレス）の排他入力単独。** PortAudio の制限でもデバイス22でもない。
Realtek Mic の WDM-KS は +0.01% / lat 11.00ms と正常だった。

→ **duplex 1本を維持する限りデバイス側 46.67ms は動かせない。**
ユーザー判断で「46.67ms を受け入れて進む」に決定済み。

### RVC 本体（`D:\Claude\Project\RVC`, HEAD 81eed5e / 2026-08-04）
`.venv-rvc`（Python 3.10.9）に torch **2.7.1+cu128** / torchaudio 2.7.1。
RTX 4070 / sm_89 / `is_half=True` 確認。numpy 1.26.4 / transformers 4.49.0 /
faiss-cpu 1.15.0 / torchfcpe 0.0.4 / praat-parselmouth 0.4.7。WebUI・訓練UI・pymss は未導入。

import 実測で `rtrvc.RVC` / `infer.hubert` / `SynthesizerTrnMs768NSFsid` / `RMVPE` / `Config` 全通過。
**HuBERT を fp16/cuda:0 で、RMVPE も実ロード成功（VRAM 368MB）。**

ダウンロード済み資産（合計約2.8GB、HuggingFace `lj1995/VoiceConversionWebUI`）:
`assets/hubert_base` 181M / `assets/rmvpe` 173M / `assets/pretrained` 1.1G /
`assets/pretrained_v2` 1.3G / `logs/mute` 展開済み。

### 引き継ぎ文書との差分（重要）
1. `tools/rvc_for_realtime.py` は**存在しない** → **`infer/rtrvc.py`** の `RVC` クラス
2. **fairseq は依存から消えた**。HuBERT は transformers で `assets/hubert_base` からローカル読み
3. upstream は **torch 2.7.1+cu128 固定（"must stay below 2.8"）**、requirements は py312 用のみ。
   ただしリアルタイム推論の部分集合は 3.10 で全部入る（検証済み）
4. **`protect` / `rms_mix_rate` はリアルタイム経路に存在しない**（`infer/cli.py` とバッチ側のみ）
5. **10ms グリッド制約**: `skip_head` / `return_length` は `zc = sr//100` の 10ms 単位。
   さらにピッチキャッシュ前進量 `block_frame_16k // 160` も整数でないと f0 が毎ブロックずれる。
   → **block-ms / crossfade-ms / extra-ms は必ず 10 の倍数**。
   `B=30 / X=10 / EXTRA=100` ならアルゴリズム遅延 B+X=40ms で従来と同値。

### 実装済みの接続コード
- `engines.RVCTorchEngine` — torch も RVC も import しない（callable 注入）。10ms グリッドを起動時に ValueError で弾く。単体テスト済み
- `engines.BaseEngine.convert(x, sr, tail=...)` を追加 — `return_length` で vocoder の合成量を
  窓136ms→末尾40msに削れる（**約3.4倍**の差）。`infer<32ms` を狙うなら必須と判断
- `rvc_backend.RVCBackend` — RVC 側グルー。**CUDA は非同期なので明示 `torch.cuda.synchronize()`**
  （入れないと infer 計測が過小に出る）
- `realtime.py --engine rvc` 配線済み。`--rvc-model / --rvc-index / --rvc-index-rate /
  --rvc-key / --f0-method {rmvpe,fcpe} / --rvc-root`。engine=rvc のとき
  block/crossfade/extra は 30/10/100、warmup は 8回に自動切替

## 2. 未完了

- **話者モデル `.pth` が無い**（自分の声で学習するのはこれから）。
  `--engine rvc` は `assets/weights` に .pth が無いため exit=4 で正しく停止する状態
- **infer 時間の実測は保留**（ユーザー判断）。目標は `infer < 32ms`（RTF<1、実運用 0.6以下）
- 口→仮想ケーブルで TOTAL 100ms 以下の目標は、デバイス 46.67ms を受け入れた前提だと
  `46.67 + 40(B+X) + infer + out-buf` なので **infer をほぼ 0 にしても 90ms 台**。要再評価

## 3. 次にマイクを有線（Shure SE215 インラインマイク）に替える場合の注意

**デバイス番号が変わる。23 を決め打ちしてはいけない。**
SE215 は Realtek のヘッドセットジャックに挿さるので、
- 挿すまで WASAPI 側に Realtek マイクが現れない（今の一覧に無いのはそのため）
- 挿すと番号が振り直され、既存の 18〜24 もずれる可能性がある

必ず `python realtime.py --list-devices` で採り直すこと。
また、有線に替えても**共有 duplex なら 22ms のまま**の可能性が高い（22ms は
PortAudio の共有モード既定で、デバイス固有値ではない）。
10ms 級を取るには「マイク交換」と「duplex 1本を捨てて入出力を別ストリーム化」の
**両方**が要る。片方だけでは効果がない。

## 4. 声の権利（非交渉）
学習に使ってよいのは自分の声か、明示的に許諾された声素材のみ。
配布素材でも規約は個別。他人の声の無断クローンは禁止。
そのため他人の RVC モデルを落として計測することはしていない。
