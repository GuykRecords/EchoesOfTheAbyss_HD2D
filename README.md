# EchoesOfTheAbyss_HD2D

このリポジトリには、いま 2 つの独立したものが入っています。

| 場所 | 中身 | 言語 / 環境 |
|---|---|---|
| `Assets/`, `Packages/`, `ProjectSettings/` | HD-2D ゲーム本体 | Unity (URP) |
| [`tools/rtvc/`](tools/rtvc/) | リアルタイム音声変換の I/O 計測ツール | Python 3.10+ |

**この 2 つは依存関係がありません。** `tools/rtvc/` は Unity を必要とせず、
Unity 側も `tools/rtvc/` を参照していません。将来 `tools/rtvc/` を独立した
リポジトリに切り出す場合も、そのディレクトリを丸ごと移すだけで済みます。

```bash
# 独立リポジトリへ切り出すとき（履歴つき）
git subtree split --prefix=tools/rtvc -b rtvc-only
```

---

## tools/rtvc — リアルタイム音声変換

マイク → 変換 → 仮想ケーブル（Discord / OBS）の経路で、
遅延がどこで何ミリ秒発生しているかを分解して計測するツール。

→ [`tools/rtvc/README.md`](tools/rtvc/README.md)

音声デバイスも GPU も無い環境（CI・クラウド）でも回るテストが 179 件あり、
`tools/rtvc/**` が変更されるたびに GitHub Actions で自動実行されます。
