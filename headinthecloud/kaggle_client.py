"""Kaggle API wrapper: dataset upload, kernel launch, poll, output download.

Implemented in Phase 2 (feat/kaggle-client branch).
"""

from __future__ import annotations

from pathlib import Path


def upload_dataset(archive: Path, dataset_slug: str) -> None:
    """Upload archive as a Kaggle dataset version."""
    raise NotImplementedError("kaggle_client not yet implemented")


def run_kernel(script: str, dataset_slug: str, kernel_slug: str) -> str:
    """Create or update a Kaggle kernel that runs script. Returns kernel ref."""
    raise NotImplementedError("kaggle_client not yet implemented")


def poll_kernel(kernel_ref: str, interval: int = 30) -> str:
    """Block until kernel completes. Returns final status ('complete'/'error')."""
    raise NotImplementedError("kaggle_client not yet implemented")


def download_output(kernel_ref: str, dest_dir: Path) -> list[Path]:
    """Download all kernel output files into dest_dir. Returns file list."""
    raise NotImplementedError("kaggle_client not yet implemented")
