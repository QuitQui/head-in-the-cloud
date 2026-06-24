"""Tests for headinthecloud.packer — blacklist-only design."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from headinthecloud import packer


def _archive_members(archive: Path) -> list[str]:
    with tarfile.open(archive, "r:gz") as tar:
        return tar.getnames()


# --- core: include-everything-by-default ------------------------------------

def test_pack_creates_archive(tmp_path):
    (tmp_path / "train.py").write_text("print('hello')")
    result = packer.pack(tmp_path, quiet=True)
    assert result.exists()
    assert result.name.endswith(".tar.gz")


def test_pack_includes_all_file_types_by_default(tmp_path):
    for name in ("train.py", "model.pt", "README.md", "config.cfg",
                 "data.npz", "weights.safetensors", "notes.txt"):
        (tmp_path / name).write_text("x")
    result = packer.pack(tmp_path, quiet=True)
    members = _archive_members(result)
    for name in ("train.py", "model.pt", "README.md", "config.cfg",
                 "data.npz", "weights.safetensors", "notes.txt"):
        assert name in members, f"{name} should be included by default"


def test_pack_includes_subdirectory_structure(tmp_path):
    sub = tmp_path / "src"
    sub.mkdir()
    (sub / "model.py").write_text("class Model: pass")
    result = packer.pack(tmp_path, quiet=True)
    assert "src/model.py" in _archive_members(result)


# --- default excludes (environment junk only) -------------------------------

def test_pack_excludes_venv(tmp_path):
    (tmp_path / "train.py").write_text("x")
    venv = tmp_path / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "something.py").write_text("venv file")
    result = packer.pack(tmp_path, quiet=True)
    assert not any(".venv" in m for m in _archive_members(result))


def test_pack_excludes_pycache(tmp_path):
    (tmp_path / "train.py").write_text("x")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "train.cpython-312.pyc").write_bytes(b"fake")
    result = packer.pack(tmp_path, quiet=True)
    assert not any("__pycache__" in m for m in _archive_members(result))


def test_pack_excludes_git(tmp_path):
    (tmp_path / "train.py").write_text("x")
    git = tmp_path / ".git" / "objects"
    git.mkdir(parents=True)
    (git / "abc").write_bytes(b"\x00")
    result = packer.pack(tmp_path, quiet=True)
    assert not any(".git" in m for m in _archive_members(result))


def test_pack_does_NOT_exclude_pt_by_default(tmp_path):
    (tmp_path / "train.py").write_text("x")
    (tmp_path / "checkpoint.pt").write_bytes(b"\x00" * 16)
    result = packer.pack(tmp_path, quiet=True)
    assert "checkpoint.pt" in _archive_members(result)


# --- .gpuignore -------------------------------------------------------------

def test_gpuignore_excludes(tmp_path):
    (tmp_path / "train.py").write_text("x")
    (tmp_path / "big_data.csv").write_text("a,b,c")
    (tmp_path / ".gpuignore").write_text("*.csv\n")
    result = packer.pack(tmp_path, quiet=True)
    members = _archive_members(result)
    assert "train.py" in members
    assert "big_data.csv" not in members


def test_gpuignore_negation(tmp_path):
    (tmp_path / "train.py").write_text("x")
    (tmp_path / "large.bin").write_bytes(b"\x00" * 8)
    (tmp_path / "keep.bin").write_bytes(b"\x00" * 8)
    (tmp_path / ".gpuignore").write_text("*.bin\n!keep.bin\n")
    result = packer.pack(tmp_path, quiet=True)
    members = _archive_members(result)
    assert "large.bin" not in members
    assert "keep.bin" in members


def test_gpuignore_directory_exclude(tmp_path):
    (tmp_path / "train.py").write_text("x")
    output = tmp_path / "output"
    output.mkdir()
    (output / "result.json").write_text("{}")
    (tmp_path / ".gpuignore").write_text("output/\n")
    result = packer.pack(tmp_path, quiet=True)
    members = _archive_members(result)
    assert "train.py" in members
    assert not any("output" in m for m in members)


def test_explicit_ignore_file(tmp_path):
    (tmp_path / "train.py").write_text("x")
    (tmp_path / "secret.json").write_text('{"key": "val"}')
    ignore = tmp_path / "custom.ignore"
    ignore.write_text("secret.json\n")
    result = packer.pack(tmp_path, ignore_file=ignore, quiet=True)
    members = _archive_members(result)
    assert "train.py" in members
    assert "secret.json" not in members


def test_gpuignore_comments_and_blanks(tmp_path):
    (tmp_path / "train.py").write_text("x")
    (tmp_path / "keep.txt").write_text("important")
    (tmp_path / ".gpuignore").write_text("# comment\n\n# blank above\n")
    result = packer.pack(tmp_path, quiet=True)
    members = _archive_members(result)
    assert "train.py" in members
    assert "keep.txt" in members


# --- negation edge cases ----------------------------------------------------

def test_negation_overrides_default_exclude(tmp_path):
    (tmp_path / "train.py").write_text("x")
    (tmp_path / ".DS_Store").write_bytes(b"\x00")
    (tmp_path / ".gpuignore").write_text("!.DS_Store\n")
    result = packer.pack(tmp_path, quiet=True)
    assert ".DS_Store" in _archive_members(result)


def test_last_matching_pattern_wins(tmp_path):
    (tmp_path / "data.csv").write_text("a,b")
    (tmp_path / ".gpuignore").write_text("*.csv\n!data.csv\n*.csv\n")
    result = packer.pack(tmp_path, quiet=True)
    assert "data.csv" not in _archive_members(result)


# --- empty dir --------------------------------------------------------------

def test_pack_empty_directory(tmp_path):
    result = packer.pack(tmp_path, quiet=True)
    assert result.exists()
    assert _archive_members(result) == []


# --- summary output ---------------------------------------------------------

def test_pack_prints_summary(tmp_path, capsys):
    (tmp_path / "train.py").write_text("x = 1")
    packer.pack(tmp_path)
    captured = capsys.readouterr()
    assert "[hitc pack]" in captured.err
    assert "1 files" in captured.err


def test_pack_quiet_suppresses_summary(tmp_path, capsys):
    (tmp_path / "train.py").write_text("x = 1")
    packer.pack(tmp_path, quiet=True)
    captured = capsys.readouterr()
    assert captured.err == ""
