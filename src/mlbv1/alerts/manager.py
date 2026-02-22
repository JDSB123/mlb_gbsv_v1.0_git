"""Unified alert manager — dispatches to Discord, email, etc."""

from __future__ import annotations

import logging
import os
from typing import Any

from mlbv1.alerts.discord import DiscordAlert
from mlbv1.alerts.email_sender import EmailAlert

logger = logging.getLogger(__name__)


class AlertManager:
    """Dispatch alerts to all configured channels.

    Environment variables (set in .env or Azure Key Vault):
        DISCORD_WEBHOOK_URL  — Discord webhook URL
        SMTP_HOST            — SMTP server host
        SMTP_PORT            — SMTP server port (default 587)
        SMTP_EMAIL           — sender email
        SMTP_PASSWORD        — sender password
        ALERT_RECIPIENT      — recipient email
    """

    def __init__(self) -> None:
        self.discord: DiscordAlert | None = None
        self.email: EmailAlert | None = None
        self._init_channels()

    def _init_channels(self) -> None:
        webhook = os.getenv("DISCORD_WEBHOOK_URL", "")
        if webhook:
            try:
                self.discord = DiscordAlert(webhook)
                logger.info("Discord alerts enabled")
            except ValueError as exc:
                logger.warning("Discord setup failed: %s", exc)

        smtp_host = os.getenv("SMTP_HOST", "")
        smtp_email = os.getenv("SMTP_EMAIL", "")
        smtp_password = os.getenv("SMTP_PASSWORD", "")
        recipient = os.getenv("ALERT_RECIPIENT", smtp_email)
        if smtp_host and smtp_email:
            self.email = EmailAlert(
                smtp_host=smtp_host,
                smtp_port=int(os.getenv("SMTP_PORT", "587")),
                sender_email=smtp_email,
                sender_password=smtp_password,
                recipient_email=recipient,
            )
            logger.info("Email alerts enabled")

    @property
    def has_channels(self) -> bool:
        return self.discord is not None or self.email is not None

    def send_predictions(
        self,
        predictions: list[dict[str, Any]],
        model_name: str = "ensemble",
        run_id: str = "",
    ) -> None:
        """Send predictions to all configured channels."""
        if self.discord:
            try:
                self.discord.send_predictions(predictions, model_name, run_id)
            except Exception as exc:
                logger.error("Discord prediction alert failed: %s", exc)

        if self.email:
            try:
                self.email.send_predictions(predictions, model_name, run_id)
            except Exception as exc:
                logger.error("Email prediction alert failed: %s", exc)

    def send_daily_summary(
        self,
        stats: dict[str, Any],
        top_picks: list[dict[str, Any]] | None = None,
    ) -> None:
        """Send daily summary to all channels."""
        if self.discord:
            try:
                self.discord.send_daily_summary(stats, top_picks)
            except Exception as exc:
                logger.error("Discord summary alert failed: %s", exc)

        if self.email:
            try:
                self.email.send_daily_summary(stats, top_picks)
            except Exception as exc:
                logger.error("Email summary alert failed: %s", exc)

    def send_alert(self, message: str) -> None:
        """Send a simple alert to all channels."""
        if self.discord:
            try:
                self.discord.send_alert(message)
            except Exception as exc:
                logger.error("Discord alert failed: %s", exc)

        if self.email:
            try:
                self.email.send_alert("MLB GBSV Alert", message)
            except Exception as exc:
                logger.error("Email alert failed: %s", exc)
