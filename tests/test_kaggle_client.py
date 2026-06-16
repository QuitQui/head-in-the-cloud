"""Tests for headinthecloud.kaggle_client — Phase 2 implementation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from headinthecloud.kaggle_client import ApiException


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tar(tmp_path: Path) -> Path:
    """Create a minimal tar.gz archive for upload tests."""
    import tarfile

    data_file = tmp_path / "data.csv"
    data_file.write_text("col1,col2\n1,2\n")
    archive = tmp_path / "dataset.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(data_file, arcname="data.csv")
    return archive


def test_extract_status_code_helper():
    from headinthecloud import kaggle_client

    class _WithStatus(Exception):
        status = 404

    class _Response:
        status_code = 500

    class _WithResponse(Exception):
        response = _Response()

    assert kaggle_client._extract_status_code(_WithStatus()) == 404
    assert kaggle_client._extract_status_code(_WithResponse()) == 500
    assert kaggle_client._extract_status_code(Exception("x")) is None


def test_safe_extract_tar_uses_data_filter_when_available(tmp_path, mocker):
    from headinthecloud import kaggle_client

    fake_tar = mocker.MagicMock()
    fake_tar.getmembers.return_value = []

    fake_ctx = mocker.MagicMock()
    fake_ctx.__enter__.return_value = fake_tar
    fake_ctx.__exit__.return_value = False

    mocker.patch("headinthecloud.kaggle_client.tarfile.open", return_value=fake_ctx)
    mocker.patch("headinthecloud.kaggle_client.tarfile.data_filter", object(), create=True)

    kaggle_client._safe_extract_tar(tmp_path / "dummy.tar.gz", tmp_path)

    fake_tar.extractall.assert_called_once_with(tmp_path, filter="data")


# ---------------------------------------------------------------------------
# upload_dataset
# ---------------------------------------------------------------------------

class TestUploadDataset:
    def test_upload_dataset_calls_create_new_on_first_upload(self, tmp_path, mocker):
        """When the dataset does not yet exist, dataset_create_new is called."""
        archive = _make_tar(tmp_path)

        mock_api = mocker.MagicMock()
        mock_api.get_config_value.return_value = "testuser"
        mock_api.dataset_create_version.side_effect = ApiException(status=404)
        mocker.patch("headinthecloud.kaggle_client.api", mock_api)

        from headinthecloud import kaggle_client
        kaggle_client.upload_dataset(archive, "ws")

        mock_api.dataset_create_new.assert_called_once()
        call_kwargs = mock_api.dataset_create_new.call_args
        folder_arg = call_kwargs[1].get("folder") or call_kwargs[0][0]
        assert Path(folder_arg).is_dir()

    def test_upload_dataset_calls_create_version_on_update(self, tmp_path, mocker):
        """When the dataset exists, dataset_create_version is called."""
        archive = _make_tar(tmp_path)

        mock_api = mocker.MagicMock()
        mock_api.get_config_value.return_value = "testuser"
        mocker.patch("headinthecloud.kaggle_client.api", mock_api)

        from headinthecloud import kaggle_client
        kaggle_client.upload_dataset(archive, "ws")

        mock_api.dataset_create_version.assert_called_once()

    def test_upload_dataset_unzips_archive(self, tmp_path, mocker):
        """The archive contents appear in the temp directory passed to the API."""
        archive = _make_tar(tmp_path)

        captured_paths: list[Path] = []

        mock_api = mocker.MagicMock()
        mock_api.get_config_value.return_value = "testuser"

        def _capture_version(*args, **kwargs):
            if "folder" in kwargs:
                folder = kwargs["folder"]
            elif "path" in kwargs:
                folder = kwargs["path"]
            else:
                folder = args[0]
            captured_paths.append(Path(folder))

        mock_api.dataset_create_version.side_effect = _capture_version
        mocker.patch("headinthecloud.kaggle_client.api", mock_api)

        from headinthecloud import kaggle_client
        kaggle_client.upload_dataset(archive, "ws")

        assert captured_paths, "create_version was never called with a path"
        unpacked_dir = captured_paths[0]
        assert (unpacked_dir / "data.csv").exists()

    def test_upload_dataset_propagates_api_error(self, tmp_path, mocker):
        """Non-404/403 API errors propagate without being swallowed."""
        archive = _make_tar(tmp_path)

        mock_api = mocker.MagicMock()
        mock_api.get_config_value.return_value = "testuser"
        mock_api.dataset_create_version.side_effect = ApiException(status=500)
        mocker.patch("headinthecloud.kaggle_client.api", mock_api)

        from headinthecloud import kaggle_client
        with pytest.raises(ApiException):
            kaggle_client.upload_dataset(archive, "ws")

    def test_upload_dataset_calls_create_new_on_403(self, tmp_path, mocker):
        """Kaggle newer SDK returns 403 for missing dataset; falls back to create_new."""
        archive = _make_tar(tmp_path)

        mock_api = mocker.MagicMock()
        mock_api.get_config_value.return_value = "testuser"
        mock_api.dataset_create_version.side_effect = ApiException(status=403)
        mocker.patch("headinthecloud.kaggle_client.api", mock_api)

        from headinthecloud import kaggle_client
        kaggle_client.upload_dataset(archive, "ws")

        mock_api.dataset_create_new.assert_called_once()

    def test_upload_dataset_rejects_path_traversal_archive(self, tmp_path, mocker):
        """Unsafe tar member paths are rejected."""
        import tarfile

        archive = tmp_path / "dataset.tar.gz"
        safe = tmp_path / "safe.txt"
        safe.write_text("ok")
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(safe, arcname="../evil.txt")

        mock_api = mocker.MagicMock()
        mocker.patch("headinthecloud.kaggle_client.api", mock_api)

        from headinthecloud import kaggle_client

        with pytest.raises(ValueError, match="Unsafe archive member path"):
            kaggle_client.upload_dataset(archive, "ws")


# ---------------------------------------------------------------------------
# run_kernel
# ---------------------------------------------------------------------------

class TestRunKernel:
    def test_run_kernel_creates_kernel(self, mocker):
        """kernels_push is called and the kernel ref is returned."""
        mock_api = mocker.MagicMock()
        mock_api.get_config_value.return_value = "testuser"
        mocker.patch("headinthecloud.kaggle_client.api", mock_api)

        from headinthecloud import kaggle_client
        ref = kaggle_client.run_kernel(
            script="train.py",
            dataset_slug="ws",
            kernel_slug="hitc-runner",
        )

        mock_api.kernels_push.assert_called_once()
        assert ref == "testuser/hitc-runner"

    def test_run_kernel_metadata_contains_dataset_source(self, tmp_path, mocker):
        """kernel-metadata.json written to temp dir lists the dataset source."""
        written_metadata: list[dict] = []

        mock_api = mocker.MagicMock()
        mock_api.get_config_value.return_value = "testuser"

        def _capture_push(folder):
            meta_path = Path(folder) / "kernel-metadata.json"
            if meta_path.exists():
                written_metadata.append(json.loads(meta_path.read_text()))

        mock_api.kernels_push.side_effect = _capture_push
        mocker.patch("headinthecloud.kaggle_client.api", mock_api)

        from headinthecloud import kaggle_client
        kaggle_client.run_kernel(
            script="train.py",
            dataset_slug="ws",
            kernel_slug="hitc-runner",
        )

        assert written_metadata, "kernels_push was never called with a folder"
        meta = written_metadata[0]
        assert "testuser/ws" in meta.get("dataset_sources", [])

    def test_run_kernel_metadata_sets_gpu_and_internet(self, mocker):
        """kernel-metadata.json sets enable_gpu=true and enable_internet=true.

        Internet is enabled so kernel scripts can pip-install extras (e.g.
        torch_geometric, ortools) that are not pre-installed on the Kaggle
        GPU image.
        """
        written_metadata: list[dict] = []

        mock_api = mocker.MagicMock()
        mock_api.get_config_value.return_value = "testuser"

        def _capture_push(folder):
            meta_path = Path(folder) / "kernel-metadata.json"
            if meta_path.exists():
                written_metadata.append(json.loads(meta_path.read_text()))

        mock_api.kernels_push.side_effect = _capture_push
        mocker.patch("headinthecloud.kaggle_client.api", mock_api)

        from headinthecloud import kaggle_client
        kaggle_client.run_kernel(
            script="train.py",
            dataset_slug="ws",
            kernel_slug="hitc-runner",
        )

        meta = written_metadata[0]
        assert meta.get("enable_gpu") is True
        assert meta.get("enable_internet") is True

    def test_run_kernel_metadata_type_is_script(self, mocker):
        """kernel-metadata.json sets kernel_type='script' and language='python'."""
        written_metadata: list[dict] = []

        mock_api = mocker.MagicMock()
        mock_api.get_config_value.return_value = "testuser"

        def _capture_push(folder):
            meta_path = Path(folder) / "kernel-metadata.json"
            if meta_path.exists():
                written_metadata.append(json.loads(meta_path.read_text()))

        mock_api.kernels_push.side_effect = _capture_push
        mocker.patch("headinthecloud.kaggle_client.api", mock_api)

        from headinthecloud import kaggle_client
        kaggle_client.run_kernel(
            script="train.py",
            dataset_slug="ws",
            kernel_slug="hitc-runner",
        )

        meta = written_metadata[0]
        assert meta.get("kernel_type") == "script"
        assert meta.get("language") == "python"

    def test_run_kernel_script_copies_and_runs(self, mocker):
        """The pushed kernel script copies dataset files then runs the user script."""
        written_scripts: list[str] = []

        mock_api = mocker.MagicMock()
        mock_api.get_config_value.return_value = "testuser"

        def _capture_push(folder):
            for f in Path(folder).iterdir():
                if f.suffix == ".py":
                    written_scripts.append(f.read_text())

        mock_api.kernels_push.side_effect = _capture_push
        mocker.patch("headinthecloud.kaggle_client.api", mock_api)

        from headinthecloud import kaggle_client
        kaggle_client.run_kernel(
            script="train.py",
            dataset_slug="ws",
            kernel_slug="hitc-runner",
        )

        assert written_scripts, "no Python script was written to the push folder"
        runner_src = written_scripts[0]
        assert "/kaggle/working" in runner_src
        assert "train.py" in runner_src


# ---------------------------------------------------------------------------
# poll_kernel
# ---------------------------------------------------------------------------

class TestPollKernel:
    def test_poll_kernel_returns_on_complete(self, mocker):
        """Returns 'complete' immediately when the first status is 'complete'."""
        mock_api = mocker.MagicMock()
        mock_api.kernels_status.return_value = mocker.MagicMock(status="complete")
        mocker.patch("headinthecloud.kaggle_client.api", mock_api)
        mocker.patch("headinthecloud.kaggle_client.time.sleep")

        from headinthecloud import kaggle_client
        result = kaggle_client.poll_kernel("testuser/hitc-runner", interval=1)

        assert result == "complete"
        mock_api.kernels_status.assert_called_once_with("testuser/hitc-runner")

    def test_poll_kernel_raises_on_error(self, mocker):
        """Returns 'error' when the kernel status is 'error'."""
        mock_api = mocker.MagicMock()
        mock_api.kernels_status.return_value = mocker.MagicMock(status="error")
        mocker.patch("headinthecloud.kaggle_client.api", mock_api)
        mocker.patch("headinthecloud.kaggle_client.time.sleep")

        from headinthecloud import kaggle_client
        result = kaggle_client.poll_kernel("testuser/hitc-runner", interval=1)

        assert result == "error"

    def test_poll_kernel_returns_on_cancelled(self, mocker):
        """Returns 'cancelled' when the kernel status is 'cancelled'."""
        mock_api = mocker.MagicMock()
        mock_api.kernels_status.return_value = mocker.MagicMock(status="cancelled")
        mocker.patch("headinthecloud.kaggle_client.api", mock_api)
        mocker.patch("headinthecloud.kaggle_client.time.sleep")

        from headinthecloud import kaggle_client
        result = kaggle_client.poll_kernel("testuser/hitc-runner", interval=1)

        assert result == "cancelled"

    def test_poll_kernel_loops_until_terminal(self, mocker):
        """Keeps polling when status is non-terminal, stops when terminal."""
        mock_api = mocker.MagicMock()
        mock_api.kernels_status.side_effect = [
            mocker.MagicMock(status="running"),
            mocker.MagicMock(status="running"),
            mocker.MagicMock(status="complete"),
        ]
        mocker.patch("headinthecloud.kaggle_client.api", mock_api)
        mock_sleep = mocker.patch("headinthecloud.kaggle_client.time.sleep")

        from headinthecloud import kaggle_client
        result = kaggle_client.poll_kernel("testuser/hitc-runner", interval=5)

        assert result == "complete"
        assert mock_api.kernels_status.call_count == 3
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(5)

    def test_poll_kernel_passes_full_ref(self, mocker):
        """The full kernel ref is passed to kernels_status without splitting."""
        mock_api = mocker.MagicMock()
        mock_api.kernels_status.return_value = mocker.MagicMock(status="complete")
        mocker.patch("headinthecloud.kaggle_client.api", mock_api)
        mocker.patch("headinthecloud.kaggle_client.time.sleep")

        from headinthecloud import kaggle_client
        kaggle_client.poll_kernel("myorg/my-kernel", interval=1)

        mock_api.kernels_status.assert_called_once_with("myorg/my-kernel")


# ---------------------------------------------------------------------------
# download_output
# ---------------------------------------------------------------------------

class TestDownloadOutput:
    def test_download_output_fetches_files(self, tmp_path, mocker):
        """kernels_output is called with the full kernel ref and dest dir."""
        mock_api = mocker.MagicMock()
        mocker.patch("headinthecloud.kaggle_client.api", mock_api)

        out_file = tmp_path / "result.csv"
        out_file.write_text("a,b\n1,2\n")

        from headinthecloud import kaggle_client
        files = kaggle_client.download_output("testuser/hitc-runner", tmp_path)

        mock_api.kernels_output.assert_called_once_with(
            "testuser/hitc-runner", path=str(tmp_path)
        )
        assert isinstance(files, list)

    def test_download_output_returns_paths_in_dest_dir(self, tmp_path, mocker):
        """All returned paths live inside dest_dir."""
        mock_api = mocker.MagicMock()
        mocker.patch("headinthecloud.kaggle_client.api", mock_api)

        (tmp_path / "output1.csv").write_text("x\n1\n")
        (tmp_path / "output2.log").write_text("done\n")

        from headinthecloud import kaggle_client
        files = kaggle_client.download_output("testuser/hitc-runner", tmp_path)

        for f in files:
            assert f.parent == tmp_path

    def test_download_output_empty_dir_returns_empty_list(self, tmp_path, mocker):
        """Returns an empty list when the API writes nothing."""
        mock_api = mocker.MagicMock()
        mocker.patch("headinthecloud.kaggle_client.api", mock_api)

        from headinthecloud import kaggle_client
        files = kaggle_client.download_output("testuser/hitc-runner", tmp_path)

        assert files == []

    def test_download_output_passes_full_ref(self, tmp_path, mocker):
        """The full kernel ref is passed to kernels_output without splitting."""
        mock_api = mocker.MagicMock()
        mocker.patch("headinthecloud.kaggle_client.api", mock_api)

        from headinthecloud import kaggle_client
        kaggle_client.download_output("myorg/my-kernel", tmp_path)

        mock_api.kernels_output.assert_called_once_with(
            "myorg/my-kernel", path=str(tmp_path)
        )
