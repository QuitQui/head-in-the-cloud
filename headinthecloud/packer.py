"""Pack local training files into a tar.gz for upload.

Implemented in Phase 2 (feat/packer branch).
"""

from __future__ import annotations

import fnmatch
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Extensions considered training-relevant
INCLUDE_EXTENSIONS = {
    ".py",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".txt",
}

# Default exclusion patterns — always applied even without a .gpuignore
DEFAULT_EXCLUDES = [
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    ".venv/",
    "venv/",
    "env/",
    ".git/",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.safetensors",
    "*.bin",
    "data/",
    "datasets/",
    "*.csv",
    "*.parquet",
    "output/",
    "results/",
    ".DS_Store",
]


def _load_patterns(ignore_file: Path) -> list[str]:
    """Read patterns from a .gpuignore-style file, stripping comments and blanks."""
    patterns: list[str] = []
    for line in ignore_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(line)
    return patterns


def _is_excluded(rel_path: Path, patterns: list[str]) -> bool:
    """Return True if *rel_path* matches any exclusion pattern.

    Supports two kinds of patterns:
    - Directory patterns ending in '/' — match any path component equal to
      the directory name (e.g. ``__pycache__/`` excludes anything whose
      parts contain ``__pycache__``).
    - Glob patterns — matched against the file name and the full relative
      path string using fnmatch.
    """
    parts = rel_path.parts
    name = rel_path.name
    rel_str = str(rel_path)

    for pattern in patterns:
        if pattern.endswith("/"):
            # Directory pattern — exclude if any part of the path equals
            # the directory name (without trailing slash).
            dir_name = pattern.rstrip("/")
            if dir_name in parts:
                return True
        else:
            # Glob pattern — match against the file name and the full path.
            if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel_str, pattern):
                return True

    return False


def _collect_files(project_dir: Path, patterns: list[str]) -> list[Path]:
    """Walk project_dir and return files that pass the exclusion filter."""
    collected: list[Path] = []
    for path in sorted(project_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(project_dir)
        if _is_excluded(rel, patterns):
            continue
        if path.suffix.lower() in INCLUDE_EXTENSIONS or _is_requirements(path.name):
            collected.append(path)
    return collected


def _is_requirements(name: str) -> bool:
    """True for requirements*.txt filenames."""
    return fnmatch.fnmatch(name, "requirements*.txt")


def pack(project_dir: Path, ignore_file: Path | None = None) -> Path:
    """Bundle training-relevant files from project_dir into a tar.gz.

    Args:
        project_dir: Root directory of the training project.
        ignore_file: Path to a .gpuignore file.  If None, looks for
            ``project_dir/.gpuignore``.  Missing file is silently ignored.

    Returns:
        Path to the created ``project_<timestamp>.tar.gz`` archive.
    """
    project_dir = Path(project_dir).resolve()

    # Build the combined exclusion pattern list.
    patterns: list[str] = list(DEFAULT_EXCLUDES)

    if ignore_file is None:
        candidate = project_dir / ".gpuignore"
        if candidate.is_file():
            ignore_file = candidate

    if ignore_file is not None and Path(ignore_file).is_file():
        patterns.extend(_load_patterns(Path(ignore_file)))

    files = _collect_files(project_dir, patterns)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_name = f"project_{timestamp}.tar.gz"
    archive_path = Path(tempfile.gettempdir()) / archive_name

    with tarfile.open(archive_path, "w:gz") as tar:
        for file_path in files:
            arcname = str(file_path.relative_to(project_dir))
            tar.add(file_path, arcname=arcname)

    return archive_path
