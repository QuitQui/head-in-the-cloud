from click.testing import CliRunner

from headinthecloud.cli import main


def test_help_shows_config_command():
    runner = CliRunner()

    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "config" in result.output
    assert "cfg" not in result.output
