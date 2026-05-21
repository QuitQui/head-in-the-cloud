"""Send notifications via Link (terminal-to-Slack bridge).

Implemented in Phase 2 (feat/collector-notifier branch).
"""

from __future__ import annotations


def notify(message: str) -> None:
    """Send message via `link send`. Silently skips if link is not installed."""
    raise NotImplementedError("notifier not yet implemented")
