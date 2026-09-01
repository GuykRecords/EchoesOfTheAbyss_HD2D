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
    ".venv-rvc": ("残す", "RVC 専用 venv。torch 2.7.1+cu128。rtvc もこちらで動く"),
    "RVC": ("残す", "RVC 本体の clone。巨大かつ別ライセンスなのでリポジトリには入れない"),
    "rtvc": ("要判断", "リポジトリ tools/rtvc へ移管済み。差分が無ければ退避してよい"),
    # 2026-09-01 の --peek で正体が判明したもの
    "EchoesOfTheAbyss_HD2D": ("残す", "このリポジトリの clone。GitHub 管理済み"),
    "Project Saikyo AI Vtuber": ("残す", "GitHub 管理済み (GuykRecords/project-saikyo-ai-vtuber)"),
    "discord-voice": ("残す", "Discord 用 VC 環境 (VCClient/Beatrice)。rtvc とは別実装"),
    "ComfyUI": ("残す", "ComfyUI 連携ツールキット。本体は D:\\ComfyUI で別物"),
    "project_handoff": ("要判断", "AI VTuber の旧引き継ぎ。Saikyo リポジトリと重複の可能性"),
    "models": ("要判断", "空のディレクトリ"),
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
        if "._archived_" in entry.name:
            verdict, why = ("退避済み", "様子を見て問題なければ手で削除してよい")
        else:
            verdict, why = KNOWN.get(entry.name,
                                     ("未分類", "心当たりが無ければ中身を確認してから判断"))
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


# --------------------------------------------------------------------------
# 中身を覗く（--peek）
# --------------------------------------------------------------------------

PEEK_TEXT_SUFFIXES = {".md", ".txt", ".rst"}
# 秘密が入りがちなファイルは名前だけ出して中身は読まない
SECRET_HINTS = ("env", "secret", "token", "key", "credential", "password")


def read_lines(path: Path, limit: int) -> List[str]:
    """テキストの冒頭を読む。**UTF-8 と明示すること。**

    Windows の既定は cp932 で、UTF-8 の日本語ドキュメントを黙って化けさせる
    （実際にそれで棚卸しの出力が全部読めなくなった）。
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ["(読めない)"]
    return text.splitlines()[:limit]


def looks_secret(name: str) -> bool:
    low = name.lower()
    return any(h in low for h in SECRET_HINTS)


def git_remote(path: Path) -> Optional[str]:
    config = path / ".git" / "config"
    if not config.is_file():
        return None
    try:
        for line in config.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("url = "):
                return line[len("url = "):]
    except OSError:
        pass
    return "(git リポジトリ・remote 不明)"


def peek(path: Path, max_entries: int = 20, max_lines: int = 8) -> None:
    """1 つのフォルダの正体を掴むための要約。中身は .md などしか読まない。"""
    print(f"\n{'=' * 70}\n{path.name}\n{'=' * 70}")
    if path.is_file():
        print(f"  ファイル {path.stat().st_size} bytes")
        if path.suffix.lower() in PEEK_TEXT_SUFFIXES and not looks_secret(path.name):
            for line in read_lines(path, max_lines):
                print(f"    | {line}")
        return

    remote = git_remote(path)
    if remote:
        print(f"  git    : {remote}")
    if (path / "pyvenv.cfg").is_file():
        print("  種類   : Python の venv（リポジトリには入れない）")
    for marker, what in (("package.json", "Node.js プロジェクト"),
                         ("requirements.txt", "Python プロジェクト"),
                         ("pyproject.toml", "Python パッケージ"),
                         ("docker-compose.yml", "Docker 構成")):
        if (path / marker).is_file():
            print(f"  種類   : {what}（{marker} あり）")

    entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    print(f"  直下   : {len(entries)} 個")
    for e in entries[:max_entries]:
        print(f"    {'[dir] ' if e.is_dir() else '      '}{e.name}")
    if len(entries) > max_entries:
        print(f"    ... 他 {len(entries) - max_entries} 個")

    counts: Dict[str, int] = {}
    biggest: List[Tuple[int, str]] = []
    for dirpath, dirnames, filenames in os.walk(path, onerror=lambda e: None):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            suffix = Path(name).suffix.lower() or "(拡張子なし)"
            counts[suffix] = counts.get(suffix, 0) + 1
            try:
                size = (Path(dirpath) / name).stat().st_size
            except OSError:
                continue
            biggest.append((size, str((Path(dirpath) / name).relative_to(path))))
    if counts:
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:8]
        print("  種類別 : " + "  ".join(f"{k} {v}" for k, v in top))
    for size, rel in sorted(biggest, reverse=True)[:5]:
        print(f"  大きい : {size / 1024 / 1024:8.1f} MB  {rel}")

    for e in entries:
        if (e.is_file() and e.suffix.lower() in PEEK_TEXT_SUFFIXES
                and not looks_secret(e.name)):
            print(f"  --- {e.name} の冒頭 ---")
            for line in read_lines(e, max_lines):
                print(f"    | {line}")
            break


# --------------------------------------------------------------------------
# 同じ中身のファイルを探す（--dupes）
# --------------------------------------------------------------------------

#: 最上位で丸ごと除くもの。
DUPES_SKIP_TOP = {".venv", ".venv-rvc", "RVC", "node_modules"}
#: **深さを問わず**除くディレクトリ名。片付けたいのは書類であって、
#: 配布物に同梱された同じライブラリが何度も出てきても何の役にも立たない
#: （最上位しか見ていなかったとき discord-voice/.venv だけで 100 組以上出た）。
DUPES_SKIP_ANY = {
    ".venv", "venv", "site-packages", "node_modules", "__pycache__", ".git",
    "_internal", "_vendor", "dist", "build", "Scripts", "bin",
}
DUPES_MAX_BYTES = 5 * 1024 * 1024


def find_duplicates(root: Path, max_bytes: int = DUPES_MAX_BYTES) -> Dict[str, List[str]]:
    """同一内容のファイルを探す。どのコピーが原本かは人間が決める。"""
    by_hash: Dict[str, List[str]] = {}
    for entry in sorted(root.iterdir()):
        if entry.name in DUPES_SKIP_TOP or entry.name.startswith("."):
            continue
        targets = [entry] if entry.is_file() else [
            root / entry.name / rel for rel in relative_files(entry)
            if not (set(rel.parts[:-1]) & DUPES_SKIP_ANY)
        ]
        for path in targets:
            try:
                if not path.is_file() or path.stat().st_size == 0:
                    continue
                if path.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue
            digest = sha256(path)
            if digest:
                by_hash.setdefault(digest, []).append(str(path.relative_to(root)))
    return {d: paths for d, paths in by_hash.items() if len(paths) > 1}


def print_table(rows: List[dict]) -> None:
    # 判定と理由は日本語（全角）なので幅揃えを諦め、区切り文字で読ませる。
    width = max([len(r["name"]) for r in rows] + [4])
    print(f"{'name'.ljust(width)}  kind  {'size(MB)':>9}  {'updated':<10} | 判定 | 理由")
    print("-" * (width + 40))
    for r in sorted(rows, key=lambda r: -(r["size"] or 0)):
        size = "?" if r["size"] is None else f"{r['size']:.1f}"
        print(f"{r['name'].ljust(width)}  {r['kind']:<4}  {size:>9}  "
              f"{r['mtime']:<10} | {r['verdict']} | {r['why']}")


def write_proposal(path: Path, targets: List[Path]) -> None:
    """退避案を書き出す。**rename しかしない。** 実行はしない。"""
    stamp = dt.date.today().strftime("%Y%m%d")
    lines = [
        "# 自動生成された「退避案」。実行前に必ず中身を読むこと。",
        "# 削除ではなく rename で退避する。1〜2 週間動かして問題が無ければ手で消す。",
        "$ErrorActionPreference = 'Stop'",
        "",
    ]
    for target in targets:
        archive = f"{target.name}._archived_{stamp}"
        lines += [
            f"$old = '{target}'",
            "if (Test-Path -LiteralPath $old) {",
            f"    Rename-Item -LiteralPath $old -NewName '{archive}'",
            f"    Write-Host '退避しました: {archive}'",
            "} else {",
            '    Write-Host "$old は存在しません。何もしません。"',
            "}",
            "",
        ]
    path.write_text("\n".join(lines),
                    encoding="utf-8-sig")  # PowerShell 5.1 は BOM 無しだと日本語を誤読する


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
    ap.add_argument("--peek", nargs="*", default=None, metavar="NAME",
                    help="中身を覗く。名前を省くと『未分類』のもの全部")
    ap.add_argument("--archive", nargs="+", default=None, metavar="NAME",
                    help="--proposal に含める退避対象（既定: 旧作業ディレクトリのみ）")
    ap.add_argument("--dupes", action="store_true",
                    help="同じ中身のファイルを探す（venv と RVC は対象外）")
    args = ap.parse_args(argv)

    print("=== ローカル棚卸し（読み取り専用・何も削除しません）===")
    print(f"対象:         {args.root}")
    print(f"リポジトリ側: {args.repo}\n")

    if not args.root.is_dir():
        print(f"error: {args.root} が見つかりません。--root で正しいパスを指定してください。",
              file=sys.stderr)
        return 1

    rows = inventory(args.root)
    print_table(rows)

    if args.dupes:
        groups = find_duplicates(args.root)
        print(f"\n=== 同じ中身のファイル ===")
        if not groups:
            print("  重複はありません。")
            return 0
        for paths in sorted(groups.values(), key=lambda p: p[0]):
            print()
            for path in sorted(paths):
                print(f"  {path}")
        print(f"\n{len(groups)} 組。どれを原本として残すかは人間が決めること。")
        return 0

    if args.peek is not None:
        names = args.peek or [r["name"] for r in rows if r["verdict"] == "未分類"]
        for name in names:
            target = args.root / name
            if target.exists():
                peek(target)
            else:
                print(f"\n{name}: 見つかりません")
        return 0

    old = args.root / args.old_name
    print(f"\n=== 旧 {args.old_name} とリポジトリ版の比較 ===")
    if not old.is_dir():
        print(f"{old} は存在しません。移管済みか、まだ作っていないかのどちらかです。")
        if args.proposal and args.archive:
            targets = [args.root / name for name in args.archive]
            write_proposal(args.proposal, targets)
            print(f"\n退避案を書き出しました: {args.proposal}")
            print("中身を読んでから、自分で実行してください。このスクリプトは実行しません。")
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
        targets = ([args.root / name for name in args.archive] if args.archive
                   else [old])
        missing = [t for t in targets if not t.exists()]
        if missing:
            print("\n注意: 見つからない退避対象があります: "
                  + ", ".join(t.name for t in missing), file=sys.stderr)
        write_proposal(args.proposal, targets)
        print(f"\n退避案を書き出しました: {args.proposal}")
        print("中身を読んでから、自分で実行してください。このスクリプトは実行しません。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
