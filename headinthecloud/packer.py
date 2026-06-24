"""Pack a local project directory into a tar.gz for upload to a remote GPU.

Design: **blacklist-only** (like .gitignore / .dockerignore). Every file is
included by default; only files matching an exclusion pattern are dropped.
This avoids the "silently missing file" class of bugs that a whitelist causes.

Exclusion sources (applied in order):
  1. ``DEFAULT_EXCLUDES`` — always applied (environment junk that no remote
     runner needs: ``.venv/``, ``.git/``, ``__pycache__/``, etc.).
  2. ``.gpuignore`` in the project root (or a custom ``ignore_file``) —
     project-specific exclusions, same syntax as ``.gitignore``:
       - ``pattern``  excludes matching files
       - ``!pattern`` un-excludes (overrides a previous exclude)
       - ``dir/``     excludes entire directories
       - ``# comment`` and blank lines are ignored

A packing summary is printed to stderr so the user can verify what's included.
"""

from __future__ import annotations

import fnmatch
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Genuine environment junk — things no remote GPU runner ever needs.
# Intentionally minimal: if in doubt, leave it OUT of this list and let the
# user add it to .gpuignore. Model weights (*.pt etc.) are NOT excluded by
# default — they are often needed (e.g. pretrained checkpoints for fine-tuning).
DEFAULT_EXCLUDES = [
    ".venv/",
    "venv/",
    "env/",
    ".git/",
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    ".DS_Store",
    ".worktrees/",
    "node_modules/",
]

# Size warning threshold (bytes). Printed to stderr; does not block packing.
SIZE_WARNING_BYTES = 500 * 1024 * 1024  # 500 MB


def _load_patterns(ignore_file: Path) -> list[str]:
    """Read patterns from a .gpuignore file, preserving order (including ``!``)."""
    patterns: list[str] = []
    for line in ignore_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(line)
    return patterns


def _matches_pattern(rel_path: Path, pattern: str) -> bool:
    """Check if rel_path matches a single pattern (without ``!`` prefix)."""
    parts = rel_path.parts
    name = rel_path.name
    rel_str = str(rel_path)

    if pattern.endswith("/"):
        dir_name = pattern.rstrip("/")
        return dir_name in parts
    return fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel_str, pattern)


def _is_excluded(rel_path: Path, patterns: list[str]) -> bool:
    """Evaluate the pattern list in order; ``!pattern`` un-excludes.

    Last matching pattern wins (same semantics as .gitignore).
    """
    excluded = False
    for pattern in patterns:
        if pattern.startswith("!"):
            if _matches_pattern(rel_path, pattern[1:]):
                excluded = False
        else:
            if _matches_pattern(rel_path, pattern):
                excluded = True
    return excluded


def _collect_files(project_dir: Path, patterns: list[str]) -> list[Path]:
    """Walk project_dir and return all files not excluded by patterns."""
    collected: list[Path] = []
    for path in sorted(project_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(project_dir)
        if not _is_excluded(rel, patterns):
            collected.append(path)
    return collected


def pack(project_dir: Path, ignore_file: Path | None = None,
         quiet: bool = False) -> Path:
    """Bundle all non-excluded files from project_dir into a tar.gz.

    Args:
        project_dir: Root directory of the project.
        ignore_file: Path to a .gpuignore file. If None, looks for
            ``project_dir/.gpuignore``. Missing file is silently ignored.
        quiet: Suppress the packing summary.

    Returns:
        Path to the created ``project_<timestamp>.tar.gz`` archive.
    """
    project_dir = Path(project_dir).resolve()

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

    total_bytes = 0
    with tarfile.open(archive_path, "w:gz") as tar:
        for file_path in files:
            arcname = str(file_path.relative_to(project_dir))
            tar.add(file_path, arcname=arcname)
            total_bytes += file_path.stat().st_size

    if not quiet:
        mb = total_bytes / (1024 * 1024)
        print(f"[hitc pack] {len(files)} files, {mb:.1f} MB "
              f"(from {project_dir.name}/)", file=sys.stderr, flush=True)
        if total_bytes > SIZE_WARNING_BYTES:
            print(f"[hitc pack] WARNING: archive is {mb:.0f} MB "
                  f"(>{SIZE_WARNING_BYTES // (1024*1024)} MB). Consider adding "
                  f"exclusions to .gpuignore.", file=sys.stderr, flush=True)

    return archive_path
