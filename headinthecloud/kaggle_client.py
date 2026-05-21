"""Kaggle API wrapper: dataset upload, kernel launch, poll, output download."""

from __future__ import annotations

import json
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

api: Any = None


class ApiException(Exception):
    """Wraps Kaggle API errors; exposes a .status attribute for callers."""

    def __init__(self, status: int = 0, **kwargs: object) -> None:
        self.status = status
        super().__init__(f"Kaggle API error: status={status}")


_TERMINAL = {"complete", "error", "cancelled"}


def _get_api() -> Any:
    global api
    if api is None:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
    return api


def _extract_status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _safe_extract_tar(archive: Path, dest: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        dest_root = dest.resolve()
        for member in tar.getmembers():
            member_path = (dest / member.name).resolve()
            if member_path != dest_root and dest_root not in member_path.parents:
                raise ValueError(f"Unsafe archive member path: {member.name}")
        try:
            tar.extractall(dest, filter="data")
        except TypeError:
            tar.extractall(dest)


def upload_dataset(archive: Path, dataset_slug: str) -> None:
    """Upload archive as a new or updated Kaggle dataset version."""
    api = _get_api()
    api.authenticate()
    username = api.get_config_value("username")
    full_slug = f"{username}/{dataset_slug}"

    # mkdtemp so the dir outlives the function (Kaggle API reads it after call returns)
    tmp_dir = Path(tempfile.mkdtemp(prefix="hitc_ds_"))
    _safe_extract_tar(archive, tmp_dir)
    (tmp_dir / "dataset-metadata.json").write_text(
        json.dumps({"id": full_slug, "title": dataset_slug, "licenses": [{"name": "unknown"}]})
    )
    try:
        api.dataset_create_version(str(tmp_dir), version_notes="hitc upload", quiet=True)
    except Exception as exc:
        if _extract_status_code(exc) == 404:
            api.dataset_create_new(str(tmp_dir), public=False, quiet=True)
        else:
            raise


def run_kernel(script: str, dataset_slug: str, kernel_slug: str) -> str:
    """Push a Kaggle kernel that mounts dataset_slug and runs script.

    Returns the kernel ref ("username/kernel_slug").
    """
    api = _get_api()
    api.authenticate()
    username = api.get_config_value("username")
    kernel_ref = f"{username}/{kernel_slug}"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        runner = (
            "import subprocess, shutil, os\n"
            "from pathlib import Path\n\n"
            f"src = Path('/kaggle/input/{dataset_slug}')\n"
            "dst = Path('/kaggle/working')\n"
            "for f in src.iterdir():\n"
            "    shutil.copy2(f, dst / f.name)\n\n"
            "os.chdir('/kaggle/working')\n"
            f"subprocess.run(['python', '{script}'], check=True)\n"
        )
        (tmp_dir / "runner.py").write_text(runner)
        (tmp_dir / "kernel-metadata.json").write_text(
            json.dumps({
                "id": kernel_ref,
                "title": "HITC Runner",
                "code_file": "runner.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": False,
                "dataset_sources": [f"{username}/{dataset_slug}"],
            })
        )
        api.kernels_push(str(tmp_dir))

    return kernel_ref


def poll_kernel(kernel_ref: str, interval: int = 30) -> str:
    """Block until kernel reaches a terminal status; return that status."""
    api = _get_api()
    api.authenticate()
    while True:
        status = api.kernels_status(kernel_ref).status
        if status in _TERMINAL:
            return status
        time.sleep(interval)


def download_output(kernel_ref: str, dest_dir: Path) -> list[Path]:
    """Download kernel output files into dest_dir; return list of files."""
    api = _get_api()
    api.authenticate()
    api.kernels_output(kernel_ref, path=str(dest_dir))
    return list(dest_dir.iterdir())
