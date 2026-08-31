# 家の PC を整理する手順（今夜やること）

`D:\Claude\Project` 配下を、リポジトリ管理に移行した状態に合わせて整理する。

> **前提：このドキュメントの手順は、何も消さないところから始まる。**
> 削除は最後に、中身を目で見てから、自分の手で行う。
> スクリプトが勝手に消すことはない。

---

## 全体像

| 場所 | これから | 理由 |
|---|---|---|
| `D:\Claude\Project\.venv` | **残す** | 計測用。numpy 2.x + torch cu128。ここに RVC を入れると壊れる |
| `D:\Claude\Project\.venv-rvc` | **残す / これから作る** | RVC 専用。numpy 1.23.5 + fairseq |
| `D:\Claude\Project\RVC` | **残す** | RVC 本体の clone。巨大かつ別ライセンスなのでリポジトリには入れない |
| `D:\Claude\Project\rtvc` | **退避 → 後で削除** | このリポジトリの `tools/rtvc/` に移管済み |
| このリポジトリの clone | **新設** | 今後の作業場所 |

**venv と RVC 本体はリポジトリに入れない。** サイズが大きく、環境依存で、
別ライセンス。Git で管理して得るものが無い。

---

## 手順

### Step 1. リポジトリを clone する

```powershell
# 置き場所は好きなところでよい。例:
git clone <このリポジトリの URL> D:\Claude\Project\EchoesOfTheAbyss_HD2D
cd D:\Claude\Project\EchoesOfTheAbyss_HD2D
git checkout claude/rtvc-realtime-audio-io-cc68yp
```

### Step 2. 棚卸し（読み取り専用・何も消えない）

```powershell
cd D:\Claude\Project\EchoesOfTheAbyss_HD2D\tools\rtvc
python scripts\inventory_local.py
```

> PowerShell ではなく Python なのは、日本語を含む `.ps1` が Windows PowerShell 5.1 の
> 既定コードページで文字化けして構文エラーになるため（実際にそれで一度失敗した）。

出るもの：

- `D:\Claude\Project` 直下の一覧（サイズ・更新日・残す/要判断の仕分け）
- 旧 `rtvc` の各ファイルとリポジトリ版の SHA256 比較
  - `[一致]` … 同じ内容。捨ててよい
  - `[差分あり]` … ローカルで直した可能性あり。**中身を見る**
  - `[ローカルのみ]` … リポジトリに無いファイル。**中身を見る**

### Step 3. 差分があったら先に拾う

`[差分あり]` や `[ローカルのみ]` が出た場合、消す前に中身を確認する。
残すべきものがあれば、リポジトリ側にコピーしてコミットする。

```powershell
# 例: 差分を目で見る
code --diff D:\Claude\Project\rtvc\dsp.py `
            D:\Claude\Project\EchoesOfTheAbyss_HD2D\tools\rtvc\rtvc\dsp.py
```

**ここを飛ばして消すと、実機でしか分からなかった修正が消える。**
このプロジェクトの価値のかなりの部分は実機で分かったことなので、必ず確認する。

### Step 4. 動くことを確認する

移行先が動くと確認できるまで、古い方は消さない。

```powershell
D:\Claude\Project\.venv\Scripts\Activate.ps1
cd D:\Claude\Project\EchoesOfTheAbyss_HD2D\tools\rtvc

# まず音声デバイス無しで通ることを確認（数秒で終わる）
python -m pytest tests\ -q

# デバイス一覧が前と同じか
python realtime.py --list-devices --host-api WASAPI

# ベースライン再現（12 秒で自動終了）
python realtime.py --host-api WASAPI --in-device "INZONE Buds - Chat" `
  --out-device "CABLE Input" --duration 12 --prefill-ms 8
```

> **デバイスは番号ではなく名前で指定する。** index は Windows が再列挙するたびにずれる。

`TOTAL` が **94.67ms**、`under / over / drop` が **すべて 0** なら移行成功
（2026-08-31 に確認済み）。
違っていたら古い方を消さずにそのまま報告する。

### Step 5. 旧ディレクトリを退避する（削除ではない）

```powershell
python scripts\inventory_local.py --proposal cleanup-proposal.ps1

# 中身を読む
type cleanup-proposal.ps1

# 納得したら自分で実行する
powershell -ExecutionPolicy Bypass -File cleanup-proposal.ps1
```

書き出される退避案には `Remove-Item` が一切含まれない（`Rename-Item` だけ）ことを
自動テストで担保してある。

`D:\Claude\Project\rtvc` → `D:\Claude\Project\rtvc._archived_YYYYMMDD` に**改名**するだけ。
中身は消えない。

### Step 6. 1〜2 週間後に削除

新しい方で問題なく作業できている状態が続いたら、退避先を手で消す。
**急がなくてよい。数十 MB のために取り返しのつかないことをする理由は無い。**

---

## やってはいけないこと

- `.venv` に RVC の依存を入れる（numpy が 1.23.5 に落ちて計測環境が壊れる）
- venv や `RVC/` を git に追加する（リポジトリが数 GB になる）
- 確認前に `rtvc` を削除する
- `--exclusive` が通らないのを理由に設定をあれこれ同時に変える
  （1 つずつ変えないと、何が効いたのか分からなくなる）

---

## 整理が終わった後の作業

[`HANDOVER.md`](HANDOVER.md) の「次のタスク」の **1. 未実施の計測 2 本** に進む。
