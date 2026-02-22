"""Tests for alert system."""

from __future__ import annotations

from unittest.mock import patch

from mlbv1.alerts.discord import DiscordAlert
from mlbv1.alerts.manager import AlertManager


class TestDiscordAlert:
    """Tests for Discord webhook alerts."""

    def test_init_requires_url(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="required"):
            DiscordAlert(webhook_url="")

    def test_build_embeds(self) -> None:
        alert = DiscordAlert(webhook_url="https://discord.com/api/webhooks/test/test")
        predictions = [
            {
                "home_team": "NYY",
                "away_team": "BOS",
                "prediction": 1,
                "probability": 0.72,
                "spread": -1.5,
                "edge": 0.08,
                "recommended_bet": 150,
            },
        ]
        embeds = alert._build_embeds(predictions, "rf", "run-123")
        assert len(embeds) == 1
        assert "NYY" in embeds[0]["title"]
        assert embeds[0]["color"] == 0x2ECC71  # HIGH confidence

    def test_send_predictions_empty(self) -> None:
        alert = DiscordAlert(webhook_url="https://discord.com/api/webhooks/test/test")
        result = alert.send_predictions([], "rf", "run-123")
        assert result is True  # No predictions = success


class TestAlertManager:
    """Tests for unified alert manager."""

    @patch.dict("os.environ", {"DISCORD_WEBHOOK_URL": "", "SMTP_HOST": ""})
    def test_no_channels(self) -> None:
        mgr = AlertManager()
        assert not mgr.has_channels

    @patch.dict(
        "os.environ",
        {
            "DISCORD_WEBHOOK_URL": "https://discord.com/api/webhooks/test/test",
            "SMTP_HOST": "",
        },
    )
    def test_discord_only(self) -> None:
        mgr = AlertManager()
        assert mgr.has_channels
        assert mgr.discord is not None
        assert mgr.email is None
