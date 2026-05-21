from click.testing import CliRunner

from headinthecloud import cli, config


def test_load_does_not_mutate_defaults_when_config_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_CONFIG_FILE", tmp_path / "config.toml")

    loaded = config.load()
    loaded["default"]["platform"] = "local"

    assert config._DEFAULTS["default"]["platform"] == "kaggle"


def test_run_rejects_unsupported_platform(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("print('ok')\n")

    result = CliRunner().invoke(cli.main, ["run", str(script), "--platform", "aws"])

    assert result.exit_code != 0
    assert "Unsupported platform" in result.output
