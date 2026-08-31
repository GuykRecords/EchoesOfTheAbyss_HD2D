"""The local-cleanup helper must never delete, and must not miss a local-only file.

Getting this wrong loses work that only exists on one machine, so the
comparison is tested rather than trusted.
"""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "inventory_local",
    Path(__file__).resolve().parent.parent / "scripts" / "inventory_local.py",
)
inv = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(inv)


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
