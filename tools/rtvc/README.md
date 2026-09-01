# rtvc — リアルタイム音声変換 I/O 計測ツール

マイク → (変換) → 仮想ケーブル の経路で、**遅延がどこで何ミリ秒発生しているか**を
分解して表示するツール。RVC 本体を載せる前に「土台が何 ms か」を確定させるために使う。

```
口 ─→ [入力デバイス] ─→ [ブロック待ち B] ─→ [推論] ─→ [クロスフェード X] ─→ [出力バッファ] ─→ [出力デバイス] ─→ Discord/OBS
        in-dev            block              infer        xfade                out-buf            out-dev
```

この 6 つを毎回表示し、合計を `TOTAL` として出す。削るべき場所が数字で分かる。

---

## 1. すぐ使うコマンド

> **⚠ デバイス番号（index）は固定ではありません。**
> ヘッドセットを挿す、ワイヤレス機器がスリープする、それだけで
> Windows は全デバイスを振り直します。昨日動いたコマンドが今日は
> 別のデバイスを指している、が普通に起きます。
> **名前 + `--host-api` で指定してください。**

```powershell
# 家の PC（D:\Claude\Project\.venv-rvc を有効化した状態で）
cd <このリポジトリ>\tools\rtvc

# デバイス一覧（WASAPI だけに絞る）
python realtime.py --list-devices --host-api WASAPI

# 名前で指定（番号が振り直されても壊れない）
python realtime.py --host-api WASAPI --in-device "Realtek" --out-device "CABLE Input"

# ベースライン計測（passthrough = 変換なし）
python realtime.py --engine passthrough --in-device 23 --out-device 22 --sr 48000 --io-block 128 --block-ms 32 --crossfade-ms 8

# 「25ms かかるモデル」を載せた想定での耐久テスト
python realtime.py --engine fixed --fixed-cost-ms 25 --in-device 23 --out-device 22 --sr 48000 --io-block 128 --block-ms 32 --crossfade-ms 8

# RVC（窓は自動で 30/10/100 に切り替わる）
python realtime.py --engine rvc --in-device 23 --out-device 22 --sr 48000 --io-block 128 --rvc-root D:\Claude\Project\RVC --rvc-model <話者>.pth
```

| 終了コード | 意味 |
|---|---|
| 0 | 正常終了 |
| 2 | 窓の指定が不正（10ms グリッド違反、X > B など） |
| 3 | ストリームが開いたのに音が流れない／途中で止まった |
| 4 | RVC の初期化に失敗（`.pth` が無い等） |

10 秒ほど動かして `Ctrl+C`。終了時にサマリが出る。

### 音声デバイスが無い環境（クラウド / CI / 会社の PC）

```bash
python realtime.py --offline --offline-seconds 10 --engine passthrough
python realtime.py --offline --offline-input voice.wav --offline-out converted.wav
```

`--offline` は**実機と同じ処理コード**を、合成信号または WAV に対して流す。
音は出ないが、窓・クロスフェード・DSP が壊れていないことはここで判定できる。

---

## 2. 出力の読み方

```
[  2.0s] in-dev  22.00 | block  32.00 | infer   0.02 | xfade  8.00 | out-buf  8.00 | out-dev  24.67 | TOTAL   94.69 ms | RTF 0.001 | under 0 over 0 drop 0 | loop 0.278ms
```

| 項目 | 意味 | 削り方 |
|---|---|---|
| `in-dev` | 入力デバイスの遅延（PortAudio 申告値） | `--exclusive` / `--io-block` を小さく / 有線マイクへ |
| `block` | `--block-ms`。B サンプル溜まるまでの待ち | `--block-ms` を下げる（推論の余裕と引き換え） |
| `infer` | 変換の実測時間（EMA） | 軽いモデル / f0 手法を変える |
| `xfade` | `--crossfade-ms`。のりしろ分の先送り | `--crossfade-ms` を下げる（下げすぎると継ぎ目が鳴る） |
| `out-buf` | 出力リングの**最小**占有量 | 小さいほど良いが 0 に近いと `under` が出る |
| `out-dev` | 出力デバイスの遅延 | `in-dev` と同じ |
| `RTF` | `infer / block`。1.0 でギリギリ | 実運用は 0.6 以下を目標 |
| `under` | 変換が間に合わず無音を出した回数 | **0 でなければ設定が破綻している**。`--prefill-ms 8` を試す |
| `over` | 入力が溢れた回数 | 同上 |
| `drop` | リングバッファが捨てたサンプル数 | 同上 |

`out-buf` は平均ではなく**最小占有量**で計上している。平均にすると `block` と二重計上になる。

`under` / `over` の計上は、ワーカーが最初のブロックを変換し終えてから始まる。
それ以前は出力リングが空なのが当たり前で、そこを数えると `under` は
**構造上ゼロになれない数字**になり、合否判定に使えなくなる。

### 実測ベースライン（RTX 4070 / i7-14700 / Win11、WASAPI 共有、io-block 128）

| 構成 | TOTAL | RTF | under/over/drop |
|---|---|---|---|
| passthrough B32 / X8 | 94.67 ms | 0.01 | 0 / 0 / 0 |
| fixed 25ms B32 / X8 | 126.00 ms | 0.79 | 0 / 0 / 0 |
| fixed 25ms B30 / X10 / EXTRA100（RVC 互換窓） | 122.70 ms | 0.85 | 0 / 0 / 0 |

デバイス側（in-dev 22.00 + out-dev 24.67 = **46.67ms**）が全体の半分。

**WASAPI 排他モードは検証済みで、改善しないことが確定している**
（ワイヤレスの INZONE Buds が排他入力で +6.42% のレート破綻を起こす）。
→ [`docs/HANDOVER.md`](docs/HANDOVER.md) 。**再計測しないこと。**

---

## 3. 設計の核心 — 窓 `[EXTRA | X | B]`

毎回このかたちの窓をモデルに渡す。

```
      ← 過去の音（もう鳴った） →   ← のりしろ →   ← 新しい音 →
     ┌───────────────────────────┬─────────────┬────────────┐
     │          EXTRA            │      X      │     B      │
     └───────────────────────────┴─────────────┴────────────┘
       遅延に効かない / 計算量に効く    遅延に効く   遅延に効く
```

* `EXTRA`（`--extra-ms`, 既定 500ms / RVC では 100ms）… すでに過ぎた音。モデルに文脈を与えるが**遅延はゼロ**。計算量だけ増える。
* `X`（`--crossfade-ms`）… 前回の出力と重ねて繋ぐ区間。
* `B`（`--block-ms`）… 今回新しく入ってきた音。

**アルゴリズム遅延 = B + X。EXTRA は入らない。** この非対称性が設計の核。

- 遅延を減らしたい → `B` と `X` を削る
- 音が破綻する（こもる・途切れる） → `EXTRA` を伸ばす（0.5s → 1.0s）

この 2 つを混同すると「遅いのに音も悪い」状態になる。

### のりしろ（crossfade）の仕組み

モデルは窓全体を変換するが、使うのは末尾 `X + B` サンプルだけ。
その先頭 `X` サンプルは前回の末尾 `X` サンプルと**同じ時刻の音**なので、
等パワー窓（cos / sin）で混ぜて繋ぐ。混ぜ終わった残りの `X` を次回用に取っておく。

> passthrough で聴くとクロスフェード区間だけ +3dB 大きくなる。
> 同じ信号を等パワーで混ぜれば √2 倍になるので、これは**原理どおりの正常動作**
> （実測 0.207 = 0.5×(√2−1) と解析値が一致）。

### エンジンには「末尾だけ」を頼む

ループが実際に使うのは末尾 `X + B` だけなので、`convert()` は `tail` を渡す。
これを尊重できるエンジン（`supports_tail = True`）は合成量を
**窓 140ms → 末尾 40ms** に削れる。RVC の vocoder で実測**約 3.4 倍**の差になる。

出力音は 1 サンプルも変わらない。変わっていないことはテストで担保している。

---

## 4. 変更してはいけない不変条件

これらは全部「一度壊して痛い目を見た」項目。

1. **オーディオコールバックで推論しない。** コールバックはリングバッファへの読み書きだけ。
   コールバック内でリアルタイムより遅い処理を 1 回でもやると即ドロップアウトする。
2. **入出力は 1 本の duplex ストリーム。** 別々に開くとクロックが独立し、
   ずれを吸収するためのバッファが際限なく増える＝取り返せない遅延になる。
3. **`out-buf` は最小占有量で計上する。** 平均だと `block` と二重計上。
4. **等パワークロスフェードを維持する。** 等振幅にすると音量が凹む。
5. **`EXTRA` は遅延に数えない。** 過去の音だから。
6. **リングバッファが溢れたら古い方を捨てる。** 新しい方を捨てると
   バックログの後ろで詰まり続け、二度とリアルタイムに戻れない。
7. **`under` / `over` は最初の変換完了後から数える。** それ以前は出力リングが空なのが
   当たり前で、数えると `under` が構造上ゼロになれず合否判定に使えなくなる。
8. **ストリームが開いたことを成功と見なさない。** コールバックが来ているかを確認する。
   排他モードは「開くが 1 度も呼ばれない」形で静かに失敗し、全カウンタ 0 ＝ 満点に見える。
9. **Windows では `timeBeginPeriod(1)` を自前で握る。** PortAudio はストリームを
   開いている間しか上げないので、その前に走る warmup が粗いクロックで測られる。

---

## 5. ファイル構成

| ファイル | 中身 |
|---|---|
| `rtvc/dsp.py` | `HighPass` / `NoiseGate` / `equal_power_windows` / `SoftLimiter` / `Resampler` / `StreamResampler` |
| `rtvc/audio_io.py` | `RingBuffer`（ロック付き・古い方を捨てる） / `AudioIO`（duplex 1 本） |
| `rtvc/engines.py` | `BaseEngine`（EMA 計測・warmup） / `PassthroughEngine` / `FixedCostEngine` / `RVCTorchEngine` |
| `rtvc/realtime.py` | `WindowProcessor`（窓とクロスフェード） / `RealtimeSession`（スレッド） / `run_offline` / CLI |
| `rtvc/timing.py` | Windows のスケジューラ刻みを 1ms に上げる（既定 ~15.6ms では warmup が測れない） |
| `rtvc/rvc_backend.py` | RVC 側のグルー。**唯一 torch と RVC に依存**する（遅延 import） |
| `scripts/inventory_local.py` | ローカル作業ディレクトリの棚卸し（読み取り専用） |
| `scripts/check_dataset.py` | 学習用に録った音声が使い物になるかの判定（読み取り専用） |
| `realtime.py` | 上を呼ぶだけのランチャ。`python realtime.py ...` がそのまま動く |
| `tests/` | 音声デバイス無しで回る受け入れテスト |

`rtvc/audio_io.py` だけが `sounddevice` を必要とし、しかも**遅延インポート**にしてある。
そのため音声環境が一切ない機械（CI・クラウド）でも他のモジュールはそのまま読み込める。

---

## 6. テスト

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

148 件。音声デバイスも GPU も不要。GitHub Actions で push のたびに自動実行される
（`.github/workflows/rtvc-tests.yml`）。

うち 9 件（`tests/test_session.py`）は、PortAudio のコールバックを偽のフィーダから
呼ぶことで**ワーカースレッドと遅延計測そのもの**を実機なしで動かしている。
「ストリームは開いたのにコールバックが 1 度も来ない」（排他モードのサイレント失敗）を
検出できることも、ここで回帰テストしている。

中心になるのは `test_passthrough_output_matches_the_exact_oracle`。
passthrough なら出力は「入力を X だけ遅らせて、ブロック境界の X サンプルに
等パワー窓の和を掛けたもの」に**厳密に一致する**はずで、窓の組み立て・
末尾の切り出し・のりしろの持ち回りのどれか一つでも間違えるとこの等式が崩れる。
「聴いた感じ大丈夫」ではなく算数で判定している。

---

## 7. 関連ドキュメント

| ドキュメント | 内容 |
|---|---|
| [`docs/HANDOVER.md`](docs/HANDOVER.md) | 環境・実測値・次のタスク・声の権利（引き継ぎメモ） |
| [`docs/RVC_INTEGRATION.md`](docs/RVC_INTEGRATION.md) | RVC 本体の導入手順と `RVCTorchEngine` の繋ぎ方 |
| [`docs/LOCAL_CLEANUP.md`](docs/LOCAL_CLEANUP.md) | 家の PC の `D:\Claude\Project` 配下を整理する手順 |
| [`docs/VOICE_TRAINING.md`](docs/VOICE_TRAINING.md) | 話者モデルの作り方（録音・判定・訓練・接続） |
| [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md) | 完了条件（合否を人間が判定できる形で固定したもの） |

---

## 8. 声の権利（非交渉）

**学習に使ってよいのは自分の声か、明示的に許諾された声素材のみ。**

配布素材でも規約は個別に違う（例：あみたろの声素材は「変換していること」の明記が条件）。
他人の声の無断クローンは禁止。副業・商用で使う場合は素材ごとの規約を都度確認すること。
