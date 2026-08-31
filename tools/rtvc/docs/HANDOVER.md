# 引き継ぎメモ — リアルタイム VC

このファイルが現状の一次情報。作業を再開するときは最初にここを読む。

---

## 確定済み環境

| 項目 | 値 |
|---|---|
| GPU | RTX 4070 12GB (Ada, sm_89, driver 610.88) ※Blackwell ではない |
| PC | GALLERIA XA7C-R47-WH / i7-14700 / RAM 32GB / Win11 25H2 |
| venv（計測用） | `D:\Claude\Project\.venv` — Python 3.10.9, torch 2.11.0+cu128, sounddevice / scipy / soxr |
| 入力 | device 23 = INZONE Buds Chat mic (WASAPI)。USB ドングル YY2977 のワイヤレス |
| 出力 | device 22 = CABLE Input (WASAPI)。VB-CABLE 導入済み |
| コード | このリポジトリの `tools/rtvc/`（旧 `D:\Claude\Project\rtvc` から移管） |

---

## 実測ベースライン

`--engine passthrough --io-block 128 --block-ms 32 --crossfade-ms 8`

```
TOTAL 94.67ms
  = in-dev 22.00 + block 32.00 + infer 0 + xfade 8.00 + out-buf 8.00 + out-dev 24.67
under / over / drop すべて 0
worker loop ema 0.278ms   ← CPU 側の余裕は十分ある
```

**デバイス側 46.67ms が全体の約半分。最大の削りどころ。**

---

## 次のタスク（この順で）

### 1. 未実施の計測 2 本 ← いまここ

```powershell
# ① 25ms のモデルを載せた想定で耐えるか
python realtime.py --engine fixed --fixed-cost-ms 25 --in-device 23 --out-device 22 `
  --sr 48000 --io-block 128 --block-ms 32 --crossfade-ms 8

# ② WASAPI 排他モードでデバイス遅延が減るか
python realtime.py --engine passthrough --in-device 23 --out-device 22 `
  --sr 48000 --io-block 128 --block-ms 32 --crossfade-ms 8 --exclusive
```

判断基準：

- **①で `under` / `drop` が 0 なら**、RTF 0.78 まで耐える構成だと確定する。
- **②が通れば** in / out が各 5〜10ms まで下がる見込み。

排他モードが蹴られた場合の切り分け順：

1. 入力だけ排他 / 出力だけ排他、と片側ずつ試す（どちらのデバイスが拒否しているか特定）
2. それでも駄目なら有線マイクへ（USB オーディオデバイス、または device 29 = Realtek Mic の WDM-KS）

ワイヤレス（INZONE Buds）は原理的にデバイス遅延が大きい。**最終的には有線が答えになる可能性が高い。**

### 2. RVC 本体の導入

→ [`RVC_INTEGRATION.md`](RVC_INTEGRATION.md) に手順を分離した。

**最重要の制約：現在の `.venv` には絶対に入れない。**
numpy 2.2.6 と RVC 要求の 1.23.5 + fairseq が衝突して計測環境ごと壊れる。
`D:\Claude\Project\.venv-rvc` を別に作る。

### 3. `RVCTorchEngine` の接続

`rtvc/engines.py` に受け口は実装済み。`infer_fn` を注入するだけ。
→ 詳細は [`RVC_INTEGRATION.md`](RVC_INTEGRATION.md)

### 4. 目標値

| 指標 | 目標 |
|---|---|
| `infer` | < 32ms（RTF < 1）。実運用は RTF 0.6 以下 |
| `TOTAL` | 口 → 仮想ケーブルで 100ms 以下 |

Discord 自体が受信側で 40〜100ms 持つが、**それは削れないので目標に含めない。**

---

## 設計上の不変条件（変更しないこと）

1. 窓 = `[EXTRA | X | B]`。EXTRA は遅延に効かず計算量にだけ効く。この非対称性が設計の核。
   - 遅延を削る → `B` / `X` を削る
   - 音が破綻 → `EXTRA` を伸ばす（0.5 → 1.0s）
   - **この 2 つを混同しない**
2. アルゴリズム遅延 = `B + X`
3. オーディオコールバックで推論しない（リング操作のみ）
4. 入出力は 1 本の duplex ストリーム（別々に開くとクロック独立でバッファが増える）
5. `out-buf` は最小占有量で計上（平均だと `block` と二重計上）
6. 等パワークロスフェードは維持。passthrough で +3dB 盛るのは原理どおりで正常

---

## 声の権利（非交渉）

学習に使ってよいのは**自分の声か、明示的に許諾された声素材のみ**。

配布素材でも規約は個別（例：あみたろの声素材は変換していることの明記が条件）。
他人の声の無断クローンは禁止。

将来の用途（AI VTuber、ボイス素材の販売など）では、**素材ごとに商用可否を確認してから使う**。
「無料配布だから自由」ではない。
