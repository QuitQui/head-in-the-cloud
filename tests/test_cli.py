from pathlib import Path

from click.testing import CliRunner

from headinthecloud.cli import main


def test_help_shows_config_command():
    runner = CliRunner()

    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "config" in result.output
    assert "cfg" not in result.output


def _mock_pipeline(mocker):
    """Stub the whole hitc pipeline so run() is exercised without network."""
    mocker.patch("headinthecloud.cli.packer.pack", return_value="archive.tar.gz")
    mocker.patch("headinthecloud.cli.kaggle_client.upload_dataset")
    rk = mocker.patch("headinthecloud.cli.kaggle_client.run_kernel",
                      return_value="testuser/hitc-runner")
    mocker.patch("headinthecloud.cli.kaggle_client.poll_kernel",
                 return_value="complete")
    mocker.patch("headinthecloud.cli.kaggle_client.download_output")
    mocker.patch("headinthecloud.cli.collector.collect",
                 return_value=Path("results.zip"))
    mocker.patch("headinthecloud.cli.notifier.notify")
    return rk


def test_run_env_forwards_value_from_environment(tmp_path, mocker, monkeypatch):
    from headinthecloud import cli

    script = tmp_path / "train.py"
    script.write_text("print('hi')\n")
    monkeypatch.setenv("WANDB_API_KEY", "secret-value-123")
    rk = _mock_pipeline(mocker)

    runner = CliRunner()
    result = runner.invoke(
        cli.main, ["run", str(script), "-e", "WANDB_API_KEY"])

    assert result.exit_code == 0, result.output
    # value reached run_kernel
    _, kwargs = rk.call_args
    assert kwargs["env"] == {"WANDB_API_KEY": "secret-value-123"}
    # key name may appear in output, but the VALUE must never be printed
    assert "WANDB_API_KEY" in result.output
    assert "secret-value-123" not in result.output


def test_run_env_missing_key_errors(tmp_path, mocker, monkeypatch):
    from headinthecloud import cli

    script = tmp_path / "train.py"
    script.write_text("print('hi')\n")
    monkeypatch.delenv("NOPE_KEY", raising=False)
    _mock_pipeline(mocker)

    runner = CliRunner()
    result = runner.invoke(cli.main, ["run", str(script), "-e", "NOPE_KEY"])

    assert result.exit_code != 0
    assert "NOPE_KEY" in result.output
