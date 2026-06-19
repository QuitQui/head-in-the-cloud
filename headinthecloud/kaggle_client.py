"""Kaggle API wrapper: dataset upload, kernel launch, poll, output download."""

from __future__ import annotations

import json
import tarfile
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

api: Any = None
_api_lock = threading.Lock()


class ApiException(Exception):
    """Wraps Kaggle API errors; exposes a .status attribute for callers."""

    def __init__(self, status: int = 0, **kwargs: object) -> None:
        self.status = status
        super().__init__(f"Kaggle API error: status={status}")


_TERMINAL = {"COMPLETE", "ERROR", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED"}


def _get_api() -> Any:
    global api
    with _api_lock:
        if api is None:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
        api.authenticate()
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
            is_within_dest = member_path == dest_root or dest_root in member_path.parents
            if not is_within_dest:
                raise ValueError(f"Unsafe archive member path: {member.name}")
        extract_kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(dest, **extract_kwargs)


def upload_dataset(archive: Path, dataset_slug: str) -> None:
    """Upload archive as a new or updated Kaggle dataset version.

    The tar.gz is uploaded as a single file (workspace.tar.gz) rather than
    extracted, because the Kaggle dataset API does not recursively upload
    subdirectories — only root-level files in the folder are included.
    """
    import shutil

    api = _get_api()
    username = api.get_config_value("username")
    full_slug = f"{username}/{dataset_slug}"

    # mkdtemp so the dir outlives the function (Kaggle API reads it after call returns)
    tmp_dir = Path(tempfile.mkdtemp(prefix="hitc_ds_"))
    shutil.copy2(archive, tmp_dir / "workspace.tar.gz")
    (tmp_dir / "dataset-metadata.json").write_text(
        json.dumps({"id": full_slug, "title": dataset_slug, "licenses": [{"name": "unknown"}]})
    )
    try:
        api.dataset_create_version(folder=str(tmp_dir), version_notes="hitc upload", quiet=True)
    except Exception as exc:
        # Kaggle API returns 404 (dataset missing) or 403 (resource not found in newer SDK)
        if _extract_status_code(exc) in (403, 404):
            api.dataset_create_new(folder=str(tmp_dir), public=False, quiet=True)
        else:
            raise


def run_kernel(script: str, dataset_slug: str, kernel_slug: str,
               env: dict[str, str] | None = None,
               machine_shape: str | None = None) -> str:
    """Push a Kaggle kernel that mounts dataset_slug and runs script.

    ``env`` maps environment variable names to values that are written into the
    generated runner as ``os.environ`` assignments (e.g. forwarding a secret like
    WANDB_API_KEY). Values are baked into the private runner and never printed.

    ``machine_shape`` selects the accelerator (``NvidiaTeslaT4`` /
    ``NvidiaTeslaP100`` / ``Tpu1VmV38``). Kaggle's legacy ``enable_gpu`` default
    assigns a P100, whose sm_60 kernels were dropped from the current base
    ``torch`` build; request ``NvidiaTeslaT4`` (sm_75) to stay compatible.

    Returns the kernel ref ("username/kernel_slug").
    """
    api = _get_api()
    username = api.get_config_value("username")
    kernel_ref = f"{username}/{kernel_slug}"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # Dataset uploads contain a single workspace.tar.gz. Extract it into
        # /kaggle/working/ (writable) before running the user script. If Kaggle
        # ever auto-extracts this file for us, fall back to copying the tree.
        env_lines = ""
        if env:
            for _k, _v in env.items():
                env_lines += f"os.environ[{json.dumps(_k)}] = {json.dumps(_v)}\n"
            env_lines += "\n"
        runner = (
            "import subprocess, shutil, os, sys, tarfile\n"
            "from pathlib import Path\n\n"
            + env_lines
            + f"src = Path('/kaggle/input/{dataset_slug}')\n"
            "dst = Path('/kaggle/working')\n"
            "archive = src / 'workspace.tar.gz'\n"
            "if archive.exists():\n"
            "    with tarfile.open(archive, 'r:gz') as tar:\n"
            "        tar.extractall(dst)\n"
            "else:\n"
            "    shutil.copytree(str(src), str(dst), dirs_exist_ok=True)\n\n"
            "os.chdir('/kaggle/working')\n"
            f"subprocess.run([sys.executable, '{script}'], check=True)\n"
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
                "enable_internet": True,
                "dataset_sources": [f"{username}/{dataset_slug}"],
                **({"machine_shape": machine_shape} if machine_shape else {}),
            })
        )
        api.kernels_push(str(tmp_dir))

    return kernel_ref


def poll_kernel(kernel_ref: str, interval: int = 30) -> str:
    """Block until kernel reaches a terminal status; return that status."""
    api = _get_api()
    while True:
        status = api.kernels_status(kernel_ref).status
        status_name = status.name if hasattr(status, "name") else str(status).upper()
        if status_name in _TERMINAL:
            return status_name.lower()
        time.sleep(interval)


def download_output(kernel_ref: str, dest_dir: Path) -> list[Path]:
    """Download kernel output files into dest_dir; return list of files."""
    api = _get_api()
    api.kernels_output(kernel_ref, path=str(dest_dir))
    return list(dest_dir.iterdir())
