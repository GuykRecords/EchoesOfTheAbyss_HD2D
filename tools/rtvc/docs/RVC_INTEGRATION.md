# RVC 本体の導入と接続

`rtvc/engines.py` の `RVCTorchEngine` は**受け口だけ**が実装済みで、
RVC / torch / fairseq を一切 import しない。だからこのリポジトリのテストは
ML スタック無しで回る。ここでは「実機で何を繋ぐか」を書く。

---

## 0. 最重要の制約

> **現在の `D:\Claude\Project\.venv` には RVC を入れない。**

計測用 venv は numpy 2.2.6 で動いている。RVC は numpy 1.23.5 + fairseq を要求する。
同居させると pip が numpy をダウングレードし、**計測環境ごと壊れる**。

`D:\Claude\Project\.venv-rvc` を別に作る。Python 3.10 は維持。

---

## 1. 環境構築

```powershell
# 別 venv を作る（計測用とは完全に分離）
py -3.10 -m venv D:\Claude\Project\.venv-rvc
D:\Claude\Project\.venv-rvc\Scripts\Activate.ps1

# Ada (sm_89) なので安定版 torch cu128 で良い。
# nightly は不要。sm_120 (Blackwell) 向けの手順は無視してよい。
pip install torch --index-url https://download.pytorch.org/whl/cu128

# RVC 本体
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI D:\Claude\Project\RVC
```

> **パスは ASCII のみ。** 日本語やスペースを含むパスに置くと落ちる既知の問題がある。
> `D:\Claude\Project\RVC` は条件を満たしている。

---

## 2. 接続先を間違えないこと

| 使う | 使わない |
|---|---|
| `tools/rvc_for_realtime.py` の `RVC` クラス | `infer/modules/vc/pipeline.py` |
| リアルタイム用。f0 キャッシュが効く | バッチ用。ブロックごとに f0 を計算し直して速度が出ない |

ここを取り違えると「実装は正しいのに RTF が 1 を超える」という
原因の分かりにくい詰まり方をする。

---

## 3. 繋ぎ方

`RVCTorchEngine` が要求するのは、この形の callable ひとつだけ。

```python
infer_fn(wav16k: np.ndarray) -> np.ndarray   # 戻り値は model_sr のオーディオ
```

エンジン側が持っているのは「リアルタイム経路に属する部分」だけ:

- ストリーム SR（48000）→ モデル入力 SR（16000）へのリサンプル
- モデル出力 SR（40000 など）→ ストリーム SR へのリサンプル
- 長さの契約（入力窓と同じサンプル数を返す）の担保

つまり RVC 側の依存地獄は `infer_fn` を作るモジュールに閉じ込められる。

```python
from rtvc.engines import RVCTorchEngine
from rtvc.realtime import WindowProcessor, RealtimeSession
from rtvc.audio_io import AudioIO

rvc = ...  # tools/rvc_for_realtime.py の RVC を初期化

def infer_fn(wav16k):
    return rvc.infer(wav16k, ...)   # 実引数は RVC 側のシグネチャに合わせる

engine = RVCTorchEngine(infer_fn, stream_sr=48000, model_sr=40000)
proc = WindowProcessor(engine, sr=48000, block=1536, crossfade=384, extra=24000)
proc.warmup(8)          # ← 本番と同じ窓長で 8 回。忘れると最初の一言だけ 1 秒遅れる
```

### warmup を飛ばさない

`BaseEngine.warmup()` は**本番と同じ窓長で** 8 回空回しする。
CUDA コンテキストの遅延生成・カーネルのオートチューニング・cuDNN のアルゴリズム探索が
全部「最初の一発目」に乗るので、忘れると最初の一言だけ 1 秒近く遅れて出る。
**窓長が違うと同じコストが再発する**ので、必ず本番の窓長で行う。

---

## 4. 推論パラメータ初期値

| パラメータ | 初期値 | 理由 |
|---|---|---|
| `f0_method` | `rmvpe` | 速度不足なら `fcpe`。**`harvest` / `crepe` はリアルタイム不可** |
| `index_rate` | 0.0〜0.3 | faiss 検索はブロック毎だとコストが高い |
| `protect` | 0.5 | |
| `rms_mix_rate` | 0.25 | |
| `fp16` | `True` | Ada なので有効 |

---

## 5. 接続前の動作確認（GPU 不要）

`infer_fn` の形が合っているかは、偽の推論関数で先に確かめられる。

```python
import numpy as np
from rtvc.engines import RVCTorchEngine

def fake_infer(wav16k):
    # 本物と同じ「16k で受けて model_sr で返す」形だけ真似る
    return np.zeros(int(wav16k.size * 40000 / 16000), dtype=np.float32)

engine = RVCTorchEngine(fake_infer, stream_sr=48000, model_sr=40000)
out = engine.convert(np.zeros(25920, dtype=np.float32), 48000)
assert out.size == 25920
```

同じ内容が `tests/test_pipeline.py::test_rvc_engine_keeps_the_length_contract_across_sample_rates`
として自動テストに入っている。

---

## 6. 速度が出なかったときの順番

1. `f0_method` を `rmvpe` → `fcpe` に落とす
2. `index_rate` を 0.0 にする（faiss を切る）
3. `--extra-ms` を 500 → 300 に減らす（**遅延ではなく計算量が減る**）
4. `--block-ms` を 32 → 48 に**増やす**（1 ブロックあたりの推論回数が減り RTF が下がる。ただし遅延は増える）
5. それでも駄目ならモデルを軽いものに変える

**`--crossfade-ms` を削って速度を稼ごうとしないこと。** X は計算量にほぼ効かず、
削ると継ぎ目が鳴るだけで損しかしない。
