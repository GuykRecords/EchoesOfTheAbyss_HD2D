# 引き継ぎメモ — リアルタイム VC

このファイルが現状の一次情報。作業を再開するときは最初にここを読む。

**最終更新: 2026-08-26 の実機検証を反映**

---

## 確定済み環境

| 項目 | 値 |
|---|---|
| GPU | RTX 4070 12GB (Ada, sm_89) |
| PC | GALLERIA XA7C-R47-WH / i7-14700 / RAM 32GB / Win11 25H2 |
| RVC 用 venv | `D:\Claude\Project\.venv-rvc` — Python 3.10.9, torch **2.7.1+cu128**。**rtvc もここで動く（2026-09-02 再確認）** |
| ~~計測用 venv~~ | `D:\Claude\Project\.venv` は **2026-09-02 に退避**（`.venv-rvc` で全部動くため）|
| 入力 | device 23 = INZONE Buds Chat mic (WASAPI 共有) |
| 出力 | device 22 = CABLE Input (WASAPI 共有)。VB-CABLE 導入済み |
| コード | このリポジトリの `tools/rtvc/` |

> **`.venv-rvc` で計測ツールも全部動く。** 2026-09-02 に再確認:
> テスト 132 件通過、実機計測も `TOTAL 94.67ms / under 0 / over 0 / drop 0` で
> `.venv` と完全一致。RVC を使うときもプロセスは 1 つで済む。
>
> **`.venv`（4.4GB）は 2026-09-02 に退避済み。** 実行環境は `.venv-rvc` 一本。
>
> ```powershell
> D:\Claude\Project\.venv-rvc\Scripts\Activate.ps1
> ```

---

## 実測ベースライン（WASAPI 共有 / INZONE Buds Chat mic → CABLE Input）

**2026-08-31 にリポジトリ版で再現確認済み。内訳まで完全一致。**

| 構成 | TOTAL | RTF | under/over/drop |
|---|---|---|---|
| passthrough B32 / X8 / prefill 8ms | **94.67 ms** | 0.00 | 0 / 0 / 0 |
| passthrough B32 / X8 / prefill 0（既定） | **86.67 ms** | 0.00 | 0 / 0 / 0 |
| fixed 25ms B32 / X8 | **126.00 ms** | 0.79 | 0 / 0 / 0 |
| fixed 25ms B30 / X10 / EXTRA100（RVC 互換窓） | **122.70 ms** | 0.85 | 0 / 0 / 0 |

内訳は `in-dev 22.00 + block + infer + xfade + out-buf + out-dev 24.67`。

**RTF 0.85 でも under/over/drop がすべて 0。** CPU 側の余裕は確認済みで、
残る不確定要素は RVC の実推論時間だけ。

### `out-buf` は買うもので、ついてくるものではない

既定（`--prefill-ms 0`）だと `out-buf 0.00` / TOTAL 86.67ms で走る。**速くなったのではなく、
出力側の余裕がゼロという意味**で、出力リングは毎周期きっちり空になっている。
passthrough は推論時間が実質 0 なので成立するが、RVC を載せて推論時間がばらつけば
ここが最初に破綻する。

`--prefill-ms 8` を足すと `out-buf 8.00` / TOTAL 94.67ms になる。この 8ms は
遅延として正直に計上される。**RVC では推論時間のばらつき（peak − ema）を目安に
prefill を決めること。**

---

## WASAPI 排他モード — 検証完了。**再試行は不要。**

| 構成 | 結果 |
|---|---|
| in=23 排他 単独 | 開くが **実効 51079 Hz（+6.42%）で破綻** |
| out=22 排他 単独 | lat **4.50ms** / 48000.9 Hz で正常 |
| in=23 を含む duplex 排他 | **DEAD**（コールバック 0 回、例外なし） |
| in=24 排他 duplex | OK（duplex 排他自体は可能） |
| duplex 排他が成立する構成の報告値 | 42.67 / 45.33ms = **共有より悪化** |
| 共有 duplex + 明示 latency 0.001〜0.01 | 全て 22.00 / 24.67ms から動かない |

**犯人は INZONE Buds（ワイヤレス）の排他入力単独。**
PortAudio の制限でもデバイス 22 でもない。Realtek Mic の WDM-KS は +0.01% / lat 11.00ms で正常だった。

→ **duplex 1 本を維持する限り、デバイス側 46.67ms は動かせない。**
→ ユーザー判断で「**46.67ms を受け入れて進む**」に決定済み。

この検証中に、排他モードが**コールバック 0 回のまま `under=0` と表示される**
サイレント失敗が見つかっている。ツール側は起動確認と途中ストール検出を持ち、
どちらも `StreamDead` で `exit=3` になる（`tests/test_session.py` で回帰テスト済み）。

---

## RVC 本体（`D:\Claude\Project\RVC`, HEAD 81eed5e / 2026-08-04）

`.venv-rvc` に torch 2.7.1+cu128 / torchaudio 2.7.1。`is_half=True` 確認済み。
numpy 1.26.4 / transformers 4.49.0 / faiss-cpu 1.15.0 / torchfcpe 0.0.4 /
praat-parselmouth 0.4.7。WebUI・訓練 UI・pymss は未導入。

import 実測で `rtrvc.RVC` / `infer.hubert` / `SynthesizerTrnMs768NSFsid` / `RMVPE` /
`Config` すべて通過。**HuBERT を fp16/cuda:0 で、RMVPE も実ロード成功（VRAM 368MB）。**

ダウンロード済み資産（約 2.8GB, HuggingFace `lj1995/VoiceConversionWebUI`）:
`assets/hubert_base` 181M / `assets/rmvpe` 173M / `assets/pretrained` 1.1G /
`assets/pretrained_v2` 1.3G / `logs/mute` 展開済み。

→ 接続手順は [`RVC_INTEGRATION.md`](RVC_INTEGRATION.md)。**旧メモとの差分が大きいので必ず読むこと。**

---

## 次のタスク

### 1. 話者モデル `.pth` を用意する ← いまここ

**確認用なら訓練は不要かもしれない。** つくよみちゃんの**公式 RVC モデル**が配布されている
（https://tyc.rei-yumesaki.net/work/software/rvc/ ）。外部に公開しない用途なら
コーパスの規約①「何の制限もありません」に該当するので、`.pth` を `assets/weights` に
置くだけで `--engine rvc` の実測に入れる。→ [`VOICE_TRAINING.md`](VOICE_TRAINING.md) §0

自分の声のモデルは、経路が確定してから作れば無駄がない。

**自分の声で学習する。** これが無いと `--engine rvc` は `exit=4` で正しく停止する。

→ 手順は [`VOICE_TRAINING.md`](VOICE_TRAINING.md)。録音 → `check_dataset.py` で判定 →
`.venv-train` を別に作る → 訓練 → `assets/weights` に置く。

**モデルのサンプルレートは 48k を選ぶこと。** 40k だと非整数比のリサンプルになり、
実測で 20dB 損する。

事前学習済みモデル（`assets/pretrained_v2`）はダウンロード済みなので、
必要なのは自分の声の収録と訓練のみ。他人の RVC モデルを落として計測することは
していない（→ 声の権利）。

### 2. infer 時間の実測 — ✅ **完了（2026-09-02）**

**つくよみちゃん公式 RVC モデル（v2 / tgt_sr 40000 / fp16 / cuda:0）、窓 30/10/100、
`--offline` で計測：**

```
engine rvc: ema 9.282ms  peak 10.964ms  RTF 0.309
warmup    : steady 11.07ms  peak 12312.35ms
```

**目標（infer < 32ms、実運用 RTF 0.6 以下）を大きく下回った。** RVC 側に余裕がある。

> **warmup の初回は 12.3 秒。** 内訳は RMVPE のロード（`pitch=11.687s`）。
> warmup が無ければ最初の一言が 12 秒遅れる。旧メモの「1 秒遅れる」は過小評価だった。
> 定常推定（11.07ms）はこの初回を正しく除外しており、プリフィルもそこから決まる。

### 2-B. 実機での TOTAL — ✅ **計測完了（2026-09-02）**

**口 → RVC → 仮想ケーブル、30 秒 / 999 回の推論：**

```
TOTAL 105.26 ms
= in-dev 22.00 + block 30.00 + infer 10.51 + xfade 10.00 + out-buf 8.08 + out-dev 24.67

infer ema 10.512ms  peak 13.124ms  RTF 0.350
under 0  over 0  drop-in 0  drop-out 0
```

`--offline` の 9.28ms から 10.51ms に上がるのは、オーディオコールバックと
競合するぶん。それでも RTF 0.35、ピークでも 0.44 で余裕がある。

> warmup の初回は **2 回目以降 2.09 秒**（1 回目は 12.3 秒）。差は RMVPE が
> OS のファイルキャッシュに乗るかどうか。**PC 起動後の初回だけ 12 秒**と考えてよい。

### 2-C. 100ms を切るには（未実施）

あと **5.3ms**。削る余地は 2 つ：

| 手 | 効果 | 代償 |
|---|---|---|
| `--block-ms 20`（10ms グリッド維持） | **−10ms** | RTF 0.35 → 約 0.53、ピーク 0.65 |
| `--prefill-ms` を auto より小さく | 数 ms | ジッタ耐性が減る。`under` を見ながら |

`B=20 / X=10` ならアルゴリズム遅延 30ms、**TOTAL 95ms 前後**の見込み。
ピーク RTF が 0.65 まで上がるので、`under` が 0 を保てるかが判定条件。

### 3. TOTAL 目標の再評価

デバイス 46.67ms を受け入れた前提では

```
46.67 (device) + 40 (B+X) + infer + out-buf
```

なので **infer をほぼ 0 にしても 90ms 台**。当初の「TOTAL 100ms 以下」は
達成はできるが余裕がほとんど無い。Discord 受信側の 40〜100ms は削れないので
目標には含めない。

---

## 有線マイク（Shure SE215 インラインマイク）に替える場合

**デバイス番号が変わる。23 を決め打ちしてはいけない。**

SE215 は Realtek のヘッドセットジャックに挿さるので:

- 挿すまで WASAPI 側に Realtek マイクが現れない（今の一覧に無いのはそのため）
- 挿すと番号が振り直され、既存の 18〜24 もずれる可能性がある

必ず `python realtime.py --list-devices` で採り直すこと。

さらに重要な点：**有線に替えても共有 duplex なら 22ms のまま**の可能性が高い。
22ms は PortAudio の共有モード既定値であって、デバイス固有値ではない。

10ms 級を取るには
「**マイク交換**」と「**duplex 1 本を捨てて入出力を別ストリーム化**」の**両方**が要る。
片方だけでは効果がない。そして別ストリーム化はクロック独立という別の問題を持ち込む
（→ 不変条件 4）ので、トレードオフを理解した上でやること。

---

## 設計上の不変条件（変更しないこと）

1. 窓 = `[EXTRA | X | B]`。EXTRA は遅延に効かず計算量にだけ効く。この非対称性が設計の核。
   - 遅延を削る → `B` / `X` を削る
   - 音が破綻 → `EXTRA` を伸ばす
   - **この 2 つを混同しない**
2. アルゴリズム遅延 = `B + X + S`（S は SOLA の探索幅）
2-B. **ニューラルボコーダーには SOLA が必須。** 窓ごとに位相が揃わないので、
   固定位置のクロスフェードでは打ち消し合う（実測で振幅が期待値の 0.64 まで落ちる）。
   `--sola-search-ms` を 0 にすると音が言葉として成立しない。
   S は**出力される声の基本周期より長く**取る（高い声 10ms / 低い男声 20ms）
3. オーディオコールバックで推論しない（リング操作のみ）
4. 入出力は 1 本の duplex ストリーム（別々に開くとクロック独立でバッファが増える）
5. `out-buf` は**最小占有量**で計上（平均だと B/2 のドレイン分を含み `block` と二重計上。
   実測で 16ms 過大に出ていた）
6. 等パワークロスフェードは維持。passthrough で +3dB 盛るのは原理どおりで正常
   （実測 0.207 = 0.5×(√2−1) と解析値が一致）
7. `under` / `over` は最初の変換完了後から数える。それ以前は出力リングが空なのが
   当たり前で、数えると `under` が構造上ゼロになれない
8. Windows では `timeBeginPeriod(1)` を自前で握る。既定の ~15.6ms 刻みだと
   warmup 実測が 33〜37ms とぶれ、そこから算出するプリフィルが過大になる

---

## 声の権利（非交渉）

学習に使ってよいのは**自分の声か、明示的に許諾された声素材のみ**。

配布素材でも規約は個別（例：あみたろの声素材は変換していることの明記が条件）。
他人の声の無断クローンは禁止。そのため他人の RVC モデルを落として計測することはしていない。

将来の用途（AI VTuber、ボイス素材の販売など）では、**素材ごとに商用可否を確認してから使う**。
「無料配布だから自由」ではない。
