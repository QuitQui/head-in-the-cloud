"""Collect output files after a kernel run and zip them locally.

Implemented in Phase 2 (feat/collector-notifier branch).
"""

from __future__ import annotations

from pathlib import Path


def collect(kernel_output_dir: Path, output_dir: Path) -> Path:
    """Detect new/modified files in kernel_output_dir and package them.

    Returns path to the results_<timestamp>.zip written to output_dir.
    """
    raise NotImplementedError("collector not yet implemented")
