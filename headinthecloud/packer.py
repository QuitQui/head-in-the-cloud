"""Pack local training files into a tar.gz for upload.

Implemented in Phase 2 (feat/packer branch).
"""

from __future__ import annotations

from pathlib import Path


def pack(project_dir: Path, ignore_file: Path | None = None) -> Path:
    """Bundle training-relevant files from project_dir into a tar.gz.

    Returns the path to the created archive.
    """
    raise NotImplementedError("packer not yet implemented")
