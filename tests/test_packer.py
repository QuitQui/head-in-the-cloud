"""Tests for headinthecloud.packer — Phase 2."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from headinthecloud import packer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _archive_members(archive: Path) -> list[str]:
    """Return the list of member names inside a tar.gz archive."""
    with tarfile.open(archive, "r:gz") as tar:
        return tar.getnames()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pack_creates_archive(tmp_path):
    """pack() returns a path that exists and has a .tar.gz extension."""
    (tmp_path / "train.py").write_text("print('hello')")

    result = packer.pack(tmp_path)

    assert result.exists(), "archive file should exist on disk"
    assert result.name.endswith(".tar.gz"), "archive should be a .tar.gz file"
    assert result.name.startswith("project_"), "archive name should start with 'project_'"


def test_pack_respects_gpuignore(tmp_path):
    """Files matching .gpuignore patterns are excluded from the archive."""
    (tmp_path / "train.py").write_text("# training script")
    (tmp_path / "run.log").write_text("epoch 1 loss=0.5")
    (tmp_path / ".gpuignore").write_text("*.log\n")

    result = packer.pack(tmp_path)
    members = _archive_members(result)

    assert "train.py" in members, "train.py should be included"
    assert "run.log" not in members, "run.log should be excluded by .gpuignore"


def test_pack_excludes_large_pt_files(tmp_path):
    """Model weight files (*.pt etc.) are excluded by default."""
    (tmp_path / "train.py").write_text("# training script")
    (tmp_path / "model.pt").write_bytes(b"\x00" * 16)   # tiny but real extension

    result = packer.pack(tmp_path)
    members = _archive_members(result)

    assert "train.py" in members, "train.py should be included"
    assert "model.pt" not in members, "model.pt should be excluded by default rules"


def test_pack_includes_python_and_config_files(tmp_path):
    """Python sources, YAML configs, and requirements files are all included."""
    (tmp_path / "train.py").write_text("# entry point")
    (tmp_path / "config.yaml").write_text("lr: 0.001\n")
    (tmp_path / "requirements.txt").write_text("torch\n")

    result = packer.pack(tmp_path)
    members = _archive_members(result)

    assert "train.py" in members, "train.py should be included"
    assert "config.yaml" in members, "config.yaml should be included"
    assert "requirements.txt" in members, "requirements.txt should be included"


# ---------------------------------------------------------------------------
# Edge-case tests (bonus coverage)
# ---------------------------------------------------------------------------

def test_pack_empty_directory(tmp_path):
    """pack() on a directory with no matching files creates an empty archive."""
    result = packer.pack(tmp_path)

    assert result.exists()
    assert result.name.endswith(".tar.gz")
    assert _archive_members(result) == []


def test_pack_excludes_pycache(tmp_path):
    """__pycache__ directories are always excluded."""
    (tmp_path / "train.py").write_text("x = 1")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "train.cpython-312.pyc").write_bytes(b"fake bytecode")

    result = packer.pack(tmp_path)
    members = _archive_members(result)

    assert not any("__pycache__" in m for m in members), \
        "__pycache__ entries should never appear in the archive"


def test_pack_excludes_venv(tmp_path):
    """.venv directories are always excluded."""
    (tmp_path / "train.py").write_text("x = 1")
    venv = tmp_path / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "something.py").write_text("venv file")

    result = packer.pack(tmp_path)
    members = _archive_members(result)

    assert not any(".venv" in m for m in members), \
        ".venv entries should never appear in the archive"


def test_pack_explicit_ignore_file(tmp_path):
    """A custom ignore_file path is respected when explicitly provided."""
    (tmp_path / "train.py").write_text("# train")
    (tmp_path / "secret.json").write_text('{"key": "value"}')

    ignore = tmp_path / "custom.ignore"
    ignore.write_text("secret.json\n")

    result = packer.pack(tmp_path, ignore_file=ignore)
    members = _archive_members(result)

    assert "train.py" in members
    assert "secret.json" not in members


def test_pack_preserves_subdirectory_structure(tmp_path):
    """Files in subdirectories appear with their relative path in the archive."""
    sub = tmp_path / "src"
    sub.mkdir()
    (sub / "model.py").write_text("class Model: pass")

    result = packer.pack(tmp_path)
    members = _archive_members(result)

    assert "src/model.py" in members, \
        "subdirectory structure should be preserved in the archive"


def test_pack_other_weight_extensions_excluded(tmp_path):
    """*.pth, *.ckpt, *.safetensors, *.bin are all excluded by default."""
    (tmp_path / "train.py").write_text("x = 1")
    for ext in ("model.pth", "checkpoint.ckpt", "weights.safetensors", "model.bin"):
        (tmp_path / ext).write_bytes(b"\x00" * 8)

    result = packer.pack(tmp_path)
    members = _archive_members(result)

    for ext_file in ("model.pth", "checkpoint.ckpt", "weights.safetensors", "model.bin"):
        assert ext_file not in members, f"{ext_file} should be excluded"
