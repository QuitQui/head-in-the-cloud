import pytest
from click.testing import CliRunner
import tomli_w

from headinthecloud import cli, config


def test_load_does_not_mutate_defaults_when_config_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_CONFIG_FILE", tmp_path / "config.toml")

    loaded = config.load()
    loaded["default"]["platform"] = "local"

    assert config._DEFAULTS["default"]["platform"] == "kaggle"


def test_load_does_not_mutate_defaults_when_config_exists(tmp_path, monkeypatch):
    config_file = tmp_path / "config.toml"
    with config_file.open("wb") as f:
        tomli_w.dump({"default": {"platform": "local"}}, f)
    monkeypatch.setattr(config, "_CONFIG_FILE", config_file)

    loaded = config.load()
    loaded["default"]["platform"] = "aws"

    assert config._DEFAULTS["default"]["platform"] == "kaggle"


@pytest.mark.parametrize("platform", ["aws", "gcp", "local"])
def test_run_rejects_unsupported_platform(tmp_path, platform):
    script = tmp_path / "train.py"
    script.write_text("print('ok')\n")

    result = CliRunner().invoke(cli.main, ["run", str(script), "--platform", platform])

    assert result.exit_code != 0
    assert "Unsupported platform" in result.output
    assert f"'{platform}'" in result.output


def test_run_accepts_kaggle_platform(tmp_path, monkeypatch):
    script = tmp_path / "train.py"
    script.write_text("print('ok')\n")

    monkeypatch.setattr(cli.packer, "pack", lambda project_dir: tmp_path / "project.zip")
    monkeypatch.setattr(cli.kaggle_client, "upload_dataset", lambda archive, dataset_slug: None)
    monkeypatch.setattr(cli.kaggle_client, "run_kernel", lambda script_name, dataset_slug, kernel_slug, env=None, machine_shape=None: "kernel/ref")
    monkeypatch.setattr(cli.kaggle_client, "poll_kernel", lambda kernel_ref: "complete")
    monkeypatch.setattr(cli.kaggle_client, "download_output", lambda kernel_ref, output_dir: [])
    monkeypatch.setattr(cli.collector, "collect", lambda tmp_dir, output_dir: output_dir / "results.zip")
    monkeypatch.setattr(cli.notifier, "notify", lambda message: None)

    result = CliRunner().invoke(cli.main, ["run", str(script), "--platform", "kaggle"])

    assert result.exit_code == 0
    assert "[hitc] Done. Results:" in result.output
