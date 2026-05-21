"""CLI entry point for head-in-the-cloud.

Usage:
  hitc run <script.py> [--platform kaggle] [--output ./output]
  hitc config show
  hitc config set <key> <value>
"""

from __future__ import annotations

import click

from headinthecloud import config, packer, kaggle_client, collector, notifier


@click.group()
def main() -> None:
    """head-in-the-cloud: run GPU workloads on Kaggle from your local machine."""


@main.command()
@click.argument("script", type=click.Path(exists=True))
@click.option("--platform", default=None, help="Platform override (default: from config)")
@click.option("--output", default=None, type=click.Path(), help="Local output directory")
def run(script: str, platform: str | None, output: str | None) -> None:
    """Pack local project, run SCRIPT on a remote GPU, collect results."""
    from pathlib import Path

    platform = platform or config.get("platform") or "kaggle"
    if platform != "kaggle":
        raise click.UsageError(f"Unsupported platform: {platform!r}. Only 'kaggle' is currently supported.")
    output_dir = Path(output or config.get("output_dir") or "./output")
    project_dir = Path(script).parent.resolve()
    script_name = Path(script).name

    click.echo(f"[hitc] Packing {project_dir} ...")
    archive = packer.pack(project_dir)

    click.echo(f"[hitc] Uploading to {platform} ...")
    dataset_slug = "hitc-workspace"
    kernel_slug = "hitc-runner"
    kaggle_client.upload_dataset(archive, dataset_slug)

    click.echo(f"[hitc] Launching kernel: {script_name}")
    kernel_ref = kaggle_client.run_kernel(script_name, dataset_slug, kernel_slug)

    click.echo("[hitc] Polling for completion (Ctrl-C to detach) ...")
    status = kaggle_client.poll_kernel(kernel_ref)

    if status == "complete":
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            files = kaggle_client.download_output(kernel_ref, Path(tmp))
            result_zip = collector.collect(Path(tmp), output_dir)
        click.echo(f"[hitc] Done. Results: {result_zip}")
        notifier.notify(f"hitc: job done — {result_zip.name}")
    else:
        click.echo(f"[hitc] Kernel ended with status: {status}", err=True)
        notifier.notify(f"hitc: job FAILED (status={status})")
        raise SystemExit(1)


@main.group()
def cfg() -> None:
    """Manage hitc configuration."""


@cfg.command("show")
def cfg_show() -> None:
    """Print current configuration."""
    config.show()


@cfg.command("set")
@click.argument("key")
@click.argument("value")
@click.option("--section", default="default")
def cfg_set(key: str, value: str, section: str) -> None:
    """Set a config value."""
    config.set_value(key, value, section)
    click.echo(f"[hitc] config: {section}.{key} = {value!r}")
