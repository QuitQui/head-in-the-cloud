"""Collect output files after a kernel run and zip them locally."""

from __future__ import annotations

import zipfile
from datetime import datetime, timezone
from pathlib import Path


def collect(kernel_output_dir: Path, output_dir: Path) -> Path:
    """Zip all files in kernel_output_dir and write results_<timestamp>.zip to output_dir.

    Creates output_dir if it does not exist. Returns the path to the zip.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    zip_path = output_dir / f"results_{timestamp}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(kernel_output_dir.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(kernel_output_dir))

    return zip_path
