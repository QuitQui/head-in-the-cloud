"""Tests for headinthecloud.notifier."""

from __future__ import annotations

import pytest

from headinthecloud import notifier


def test_notify_calls_link_send(mocker):
    mock_run = mocker.patch("subprocess.run")

    notifier.notify("Training complete!")

    mock_run.assert_called_once_with(
        ["link", "send", "Training complete!"],
        check=True,
        capture_output=True,
    )


def test_notify_silent_when_link_not_installed(mocker):
    mocker.patch("subprocess.run", side_effect=FileNotFoundError)

    notifier.notify("Test message")  # must not raise


def test_notify_empty_message(mocker):
    mock_run = mocker.patch("subprocess.run")

    notifier.notify("")

    mock_run.assert_called_once_with(["link", "send", ""], check=True, capture_output=True)


def test_notify_propagates_unexpected_exceptions(mocker):
    mocker.patch("subprocess.run", side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        notifier.notify("Test message")
