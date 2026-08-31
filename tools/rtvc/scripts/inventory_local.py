#!/usr/bin/env python3
"""ローカル作業ディレクトリの棚卸し。読み取り専用。何も削除しない。

リポジトリへ移管したあと、ローカルに何が残っていて、そのうち何が
リポジトリ側に無いのかを一覧にする。

削除は行わない。退避案は ``--proposal`` でスクリプトとして書き出せるので、
中身を目で確認してから自分で実行すること。

    python scripts/inventory_local.py
    python scripts/inventory_local.py --proposal cleanup-proposal.ps1

PowerShell ではなく Python なのは、日本語を含む .ps1 が
Windows PowerShell 5.1 の既定コードページで文字化けして構文エラーになるため。
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 既知のディレクトリの扱い。判断の根拠を残すために、理由まで書いておく。
KNOWN: Dict[str, Tuple[str, str]] = {
    ".venv": ("残す", "計測用 venv。RVC の依存を入れると壊れる"),
    ".venv-rvc": ("残す", "RVC 専用 venv。torch 2.7.1+cu128"),
    "RVC": ("残す", "RVC 本体の clone。巨大かつ別ライセンスなのでリポジトリには入れない"),
    "rtvc": ("要判断", "リポジトリ tools/rtvc へ移管済み。差分が無ければ退避してよい"),
}

SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", "node_modules"}


def default_root() -> Path:
    return Path(r"D:\Claude\Project") if os.name == "nt" else Path.cwd()


def dir_size_mb(path: Path, budget: int = 200_000) -> Optional[float]:
    """ざっくりのサイズ。巨大ディレクトリで止まらないよう件数で打ち切る。"""
    total = 0
    seen = 0
    try:
        for root, dirs, files in os.walk(path, onerror=lambda e: None):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in files:
                seen += 1
                if seen > budget:
                    return round(total / 1024 / 1024, 1)
                try:
                    total += (Path(root) / name).stat().st_size
                except OSError:
                    pass
    except OSError:
        return None
    return round(total / 1024 / 1024, 1)


def sha256(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def relative_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            out.append((Path(dirpath) / name).relative_to(root))
    return sorted(out)


def inventory(root: Path) -> List[dict]:
    rows = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        verdict, why = KNOWN.get(entry.name, ("未分類", "心当たりが無ければ中身を確認してから判断"))
        try:
            mtime = dt.datetime.fromtimestamp(entry.stat().st_mtime).strftime("%Y-%m-%d")
        except OSError:
            mtime = "?"
        size = (dir_size_mb(entry) if entry.is_dir()
                else round(entry.stat().st_size / 1024 / 1024, 2))
        rows.append({"name": entry.name, "kind": "dir" if entry.is_dir() else "file",
                     "size": size, "mtime": mtime, "verdict": verdict, "why": why})
    return rows


def compare(old: Path, repo: Path) -> Tuple[List[str], List[str], List[str]]:
    """旧ディレクトリの各ファイルをリポジトリ版と照合する。"""
    same, differ, local_only = [], [], []
    for rel in relative_files(old):
        # リポジトリ側は rtvc/ パッケージ配下に置き直してある
        candidates = [repo / rel, repo / "rtvc" / rel]
        target = next((c for c in candidates if c.is_file()), None)
        if target is None:
            local_only.append(str(rel))
        elif sha256(old / rel) == sha256(target):
            same.append(str(rel))
        else:
            differ.append(str(rel))
    return same, differ, local_only


def print_table(rows: List[dict]) -> None:
    # 判定と理由は日本語（全角）なので幅揃えを諦め、区切り文字で読ませる。
    width = max([len(r["name"]) for r in rows] + [4])
    print(f"{'name'.ljust(width)}  kind  {'size(MB)':>9}  {'updated':<10} | 判定 | 理由")
    print("-" * (width + 40))
    for r in sorted(rows, key=lambda r: -(r["size"] or 0)):
        size = "?" if r["size"] is None else f"{r['size']:.1f}"
        print(f"{r['name'].ljust(width)}  {r['kind']:<4}  {size:>9}  "
              f"{r['mtime']:<10} | {r['verdict']} | {r['why']}")


def write_proposal(path: Path, old: Path) -> None:
    stamp = dt.date.today().strftime("%Y%m%d")
    archive = f"{old.name}._archived_{stamp}"
    path.write_text(
        "# 自動生成された「退避案」。実行前に必ず中身を読むこと。\n"
        "# 削除ではなく rename で退避する。1〜2 週間動かして問題が無ければ手で消す。\n"
        "$ErrorActionPreference = 'Stop'\n"
        f"$old = '{old}'\n"
        "if (Test-Path -LiteralPath $old) {\n"
        f"    Rename-Item -LiteralPath $old -NewName '{archive}'\n"
        f"    Write-Host '退避しました: {archive}'\n"
        "} else {\n"
        "    Write-Host \"$old は存在しません。何もしません。\"\n"
        "}\n",
        encoding="utf-8-sig",  # Windows PowerShell 5.1 は BOM が無いと日本語を誤読する
    )


def main(argv: Optional[List[str]] = None) -> int:
    repo_default = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="ローカル作業ディレクトリの棚卸し（読み取り専用）")
    ap.add_argument("--root", type=Path, default=default_root(),
                    help="調べる場所（既定: D:\\Claude\\Project）")
    ap.add_argument("--repo", type=Path, default=repo_default,
                    help="比較対象のリポジトリ側 tools/rtvc")
    ap.add_argument("--old-name", default="rtvc", help="旧作業ディレクトリの名前")
    ap.add_argument("--proposal", type=Path, default=None,
                    help="退避案をこのパスに書き出す（実行はしない）")
    args = ap.parse_args(argv)

    print("=== ローカル棚卸し（読み取り専用・何も削除しません）===")
    print(f"対象:         {args.root}")
    print(f"リポジトリ側: {args.repo}\n")

    if not args.root.is_dir():
        print(f"error: {args.root} が見つかりません。--root で正しいパスを指定してください。",
              file=sys.stderr)
        return 1

    print_table(inventory(args.root))

    old = args.root / args.old_name
    print(f"\n=== 旧 {args.old_name} とリポジトリ版の比較 ===")
    if not old.is_dir():
        print(f"{old} は存在しません。移管済みか、まだ作っていないかのどちらかです。")
        return 0

    same, differ, local_only = compare(old, args.repo)
    for rel in same:
        print(f"  [一致]         {rel}")
    for rel in differ:
        print(f"  [差分あり]     {rel}   <- 中身を確認する")
    for rel in local_only:
        print(f"  [ローカルのみ] {rel}   <- 中身を確認する")

    print(f"\n一致 {len(same)} / 差分あり {len(differ)} / ローカルのみ {len(local_only)}")
    if differ or local_only:
        print("\n拾い上げるものがあります。消す前に上の [差分あり] と [ローカルのみ] を確認してください。")
    else:
        print("\nローカル固有のものはありません。旧ディレクトリは退避してよい状態です。")

    if args.proposal:
        write_proposal(args.proposal, old)
        print(f"\n退避案を書き出しました: {args.proposal}")
        print("中身を読んでから、自分で実行してください。このスクリプトは実行しません。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
