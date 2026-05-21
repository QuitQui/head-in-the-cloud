"""Tests for headinthecloud.collector."""

from __future__ import annotations

import re
import zipfile

from headinthecloud import collector


def test_collect_creates_zip(tmp_path):
    kernel_dir = tmp_path / "out"
    kernel_dir.mkdir()
    (kernel_dir / "model.pt").write_bytes(b"\x00" * 8)

    result = collector.collect(kernel_dir, tmp_path / "results")

    assert result.exists()
    assert result.suffix == ".zip"


def test_collect_zip_contains_all_output_files(tmp_path):
    kernel_dir = tmp_path / "out"
    kernel_dir.mkdir()
    (kernel_dir / "results.csv").write_text("a,b\n1,2\n")
    (kernel_dir / "log.txt").write_text("epoch 1\n")

    result = collector.collect(kernel_dir, tmp_path / "results")

    with zipfile.ZipFile(result) as zf:
        names = zf.namelist()
    assert "results.csv" in names
    assert "log.txt" in names


def test_collect_creates_output_dir_if_missing(tmp_path):
    kernel_dir = tmp_path / "out"
    kernel_dir.mkdir()
    output_dir = tmp_path / "deep" / "nested" / "output"

    result = collector.collect(kernel_dir, output_dir)

    assert output_dir.exists()
    assert result.exists()


def test_collect_zip_filename_has_timestamp(tmp_path):
    kernel_dir = tmp_path / "out"
    kernel_dir.mkdir()

    result = collector.collect(kernel_dir, tmp_path / "results")

    assert re.fullmatch(r"results_\d{8}_\d{6}\.zip", result.name)


def test_collect_empty_kernel_dir(tmp_path):
    kernel_dir = tmp_path / "out"
    kernel_dir.mkdir()

    result = collector.collect(kernel_dir, tmp_path / "results")

    with zipfile.ZipFile(result) as zf:
        assert zf.namelist() == []


def test_collect_preserves_subdirectory_structure(tmp_path):
    kernel_dir = tmp_path / "out"
    sub = kernel_dir / "checkpoints"
    sub.mkdir(parents=True)
    (sub / "epoch_1.pt").write_bytes(b"\x00")

    result = collector.collect(kernel_dir, tmp_path / "results")

    with zipfile.ZipFile(result) as zf:
        assert "checkpoints/epoch_1.pt" in zf.namelist()
