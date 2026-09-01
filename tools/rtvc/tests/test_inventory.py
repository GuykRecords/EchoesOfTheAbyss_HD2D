"""The local-cleanup helper must never delete, and must not miss a local-only file.

Getting this wrong loses work that only exists on one machine, so the
comparison is tested rather than trusted.
"""

import os

import inventory_local as inv
import pytest


@pytest.fixture
def tree(tmp_path):
    """A local working copy beside the repo version it was migrated into."""
    old = tmp_path / "project" / "rtvc"
    repo = tmp_path / "repo"
    (old / "sub").mkdir(parents=True)
    (repo / "rtvc").mkdir(parents=True)

    (old / "dsp.py").write_text("identical\n")
    (repo / "rtvc" / "dsp.py").write_text("identical\n")       # moved into the package

    (old / "engines.py").write_text("local edit\n")
    (repo / "rtvc" / "engines.py").write_text("repo version\n")

    (old / "rvc_backend.py").write_text("only on this machine\n")

    (old / "realtime.py").write_text("shim\n")
    (repo / "realtime.py").write_text("shim\n")                 # stayed at the top level

    (old / "__pycache__").mkdir()
    (old / "__pycache__" / "dsp.cpython-310.pyc").write_bytes(b"\x00")
    return old, repo


def test_comparison_sorts_files_into_same_differ_and_local_only(tree):
    old, repo = tree
    same, differ, local_only = inv.compare(old, repo)
    assert sorted(same) == ["dsp.py", "realtime.py"]
    assert differ == ["engines.py"]
    assert local_only == ["rvc_backend.py"]


def test_a_file_only_on_this_machine_is_never_reported_as_matching(tree):
    """The one that matters: rvc_backend.py exists nowhere else."""
    old, repo = tree
    same, differ, local_only = inv.compare(old, repo)
    assert "rvc_backend.py" not in same and "rvc_backend.py" not in differ


def test_caches_are_ignored(tree):
    old, _ = tree
    assert all("__pycache__" not in str(rel) for rel in inv.relative_files(old))


def test_the_report_changes_nothing_on_disk(tree, capsys):
    old, repo = tree
    before = {p: p.read_bytes() for p in old.rglob("*") if p.is_file()}
    assert inv.main(["--root", str(old.parent), "--repo", str(repo)]) == 0
    after = {p: p.read_bytes() for p in old.rglob("*") if p.is_file()}
    assert before == after, "the inventory must be read-only"
    assert "何も削除しません" in capsys.readouterr().out


def test_the_proposal_archives_by_renaming_and_is_not_executed(tree, tmp_path):
    old, repo = tree
    proposal = tmp_path / "cleanup.ps1"
    inv.main(["--root", str(old.parent), "--repo", str(repo), "--proposal", str(proposal)])

    text = proposal.read_text(encoding="utf-8-sig")
    assert "Rename-Item" in text
    assert "Remove-Item" not in text, "the proposal must never contain a delete"
    assert old.is_dir(), "writing the proposal must not touch the directory"
    # Windows PowerShell 5.1 misreads Japanese in a BOM-less script.
    assert proposal.read_bytes().startswith(b"\xef\xbb\xbf")


def test_a_missing_root_is_reported_rather_than_crashing(tmp_path, capsys):
    assert inv.main(["--root", str(tmp_path / "nope")]) == 1
    assert "見つかりません" in capsys.readouterr().err


# --------------------------------------------------------------------------
# --peek: work out what an unfamiliar folder is, without reading what it holds
# --------------------------------------------------------------------------


@pytest.fixture
def mystery(tmp_path):
    root = tmp_path / "project"
    # A name the inventory has no entry for, which is the case peek exists for.
    folder = root / "some-old-experiment"
    (folder / "src").mkdir(parents=True)
    (folder / ".git").mkdir()
    (folder / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/someone/discord-voice\n')
    (folder / "README.md").write_text("# discord voice bot\n2 行目\n")
    (folder / "package.json").write_text("{}")
    (folder / ".env").write_text("DISCORD_TOKEN=super-secret-value\n")
    (folder / "src" / "bot.js").write_text("console.log(1)\n")
    return root, folder


def test_peek_identifies_a_folder_from_its_markers(mystery, capsys):
    root, folder = mystery
    inv.peek(folder)
    out = capsys.readouterr().out
    assert "https://github.com/someone/discord-voice" in out
    assert "Node.js" in out
    assert "# discord voice bot" in out, "the README is what tells a person what this is"


def test_peek_never_prints_the_contents_of_a_secret_looking_file(mystery, capsys):
    """Names are enough to identify a folder; credentials must not be echoed."""
    root, folder = mystery
    inv.peek(folder)
    out = capsys.readouterr().out
    assert "super-secret-value" not in out
    assert "DISCORD_TOKEN" not in out


def test_peek_changes_nothing(mystery):
    root, folder = mystery
    before = {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    inv.peek(folder)
    after = {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    assert before == after


def test_peek_without_names_covers_everything_unclassified(mystery, capsys):
    root, _ = mystery
    (root / ".venv").mkdir()
    assert inv.main(["--root", str(root), "--peek"]) == 0
    out = capsys.readouterr().out
    assert "some-old-experiment" in out
    assert "=" * 70 in out


# --------------------------------------------------------------------------
# Reading documents, and finding the copies of them
# --------------------------------------------------------------------------


def test_utf8_documents_are_read_as_utf8(tmp_path, capsys):
    """Windows defaults to cp932, which silently mangles every Japanese doc."""
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "README.md").write_text("# 土台ドキュメント\n作成日: 2026-07-06\n",
                                      encoding="utf-8")
    inv.peek(folder)
    out = capsys.readouterr().out
    assert "# 土台ドキュメント" in out
    assert "繝" not in out, "cp932 mojibake"


def test_duplicate_documents_are_found_across_folders(tmp_path):
    root = tmp_path / "project"
    (root / "handoff" / "docs").mkdir(parents=True)
    (root / "saikyo").mkdir()
    body = "# 土台\n本文\n"
    for rel in ("foundation.md", "handoff/docs/foundation.md", "saikyo/foundation.md"):
        (root / rel).write_text(body, encoding="utf-8")
    (root / "saikyo" / "other.md").write_text("違う中身\n", encoding="utf-8")

    groups = inv.find_duplicates(root)
    assert len(groups) == 1
    paths = sorted(next(iter(groups.values())))
    assert len(paths) == 3
    assert any("saikyo" in p for p in paths)


def test_duplicate_search_skips_environments_and_empty_files(tmp_path):
    """Cleanup is about documents; a venv full of identical stubs is noise."""
    root = tmp_path / "project"
    (root / ".venv" / "a").mkdir(parents=True)
    (root / "RVC").mkdir()
    (root / ".venv" / "a" / "x.py").write_text("same\n")
    (root / "RVC" / "x.py").write_text("same\n")
    (root / "empty_a.txt").write_text("")
    (root / "empty_b.txt").write_text("")
    assert inv.find_duplicates(root) == {}


def test_an_archived_directory_is_labelled_as_such(tmp_path):
    root = tmp_path / "project"
    (root / "rtvc._archived_20260901").mkdir(parents=True)
    row = next(r for r in inv.inventory(root) if "archived" in r["name"])
    assert row["verdict"] == "退避済み"


def test_duplicate_search_skips_environments_at_any_depth(tmp_path):
    """A bundled app carries its own .venv; its vendored copies are pure noise."""
    root = tmp_path / "project"
    nested = root / "discord-voice" / ".venv" / "Lib" / "site-packages" / "idna"
    vendored = root / "discord-voice" / "vcclient" / "dist" / "main" / "_internal"
    nested.mkdir(parents=True)
    vendored.mkdir(parents=True)
    (nested / "core.py").write_text("vendored library\n")
    (vendored / "core.py").write_text("vendored library\n")
    (root / "discord-voice" / "README.md").write_text("real doc\n")
    (root / "notes.md").write_text("real doc\n")

    groups = inv.find_duplicates(root)
    assert len(groups) == 1, "only the two real documents should pair up"
    paths = sorted(next(iter(groups.values())))
    assert paths == ["discord-voice/README.md".replace("/", os.sep), "notes.md"]


def test_a_proposal_can_archive_several_things_at_once(tmp_path):
    root = tmp_path / "project"
    (root / "project_handoff").mkdir(parents=True)
    (root / "models").mkdir()
    (root / "old.md").write_text("x")
    proposal = tmp_path / "cleanup.ps1"

    inv.write_proposal(proposal, [root / "project_handoff", root / "models",
                                  root / "old.md"])
    text = proposal.read_text(encoding="utf-8-sig")
    assert text.count("Rename-Item") == 3
    assert "Remove-Item" not in text
    for name in ("project_handoff", "models", "old.md"):
        assert name in text
    assert (root / "project_handoff").is_dir(), "writing a proposal changes nothing"


def test_archive_targets_that_do_not_exist_are_flagged(tmp_path, capsys):
    root = tmp_path / "project"
    (root / "rtvc").mkdir(parents=True)
    (root / "models").mkdir()
    proposal = tmp_path / "cleanup.ps1"
    inv.main(["--root", str(root), "--repo", str(tmp_path / "repo"),
              "--proposal", str(proposal), "--archive", "models", "ghost"])
    assert "ghost" in capsys.readouterr().err
    assert proposal.exists(), "the rest of the proposal is still written"
