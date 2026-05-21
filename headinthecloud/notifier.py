"""Send notifications via Link (terminal-to-Slack bridge)."""

from __future__ import annotations

import subprocess


def notify(message: str) -> None:
    """Send message via `link send`. Silently skips if link is not installed."""
    try:
        subprocess.run(["link", "send", message], check=True, capture_output=True)
    except FileNotFoundError:
        pass
