"""Unified alert manager — dispatches to Discord, email, etc."""

from __future__ import annotations

import logging
from typing import Any

from mlbv1.alerts.discord import DiscordAlert
from mlbv1.alerts.email_sender import EmailAlert
from mlbv1.alerts.teams import TeamsAlert
from mlbv1.config import AlertConfig, AppConfig

logger = logging.getLogger(__name__)


class AlertManager:
    """Dispatch alerts to all configured channels.

    Configuration is sourced via AppConfig/AlertConfig so local `.env`,
    repo-tracked defaults, and ACA environment variables share one contract.
    """

    def __init__(self, config: AlertConfig | None = None) -> None:
        self.config = config or AppConfig.load().alerts
        self.discord: DiscordAlert | None = None
        self.email: EmailAlert | None = None
        self.teams: TeamsAlert | None = None
        self._init_channels()

    def _init_channels(self) -> None:
        webhook = self.config.discord_webhook_url
        if webhook:
            try:
                self.discord = DiscordAlert(webhook)
                logger.info("Discord alerts enabled")
            except ValueError as exc:
                logger.warning("Discord setup failed: %s", exc)

        smtp_host = self.config.smtp_host
        smtp_login = self.config.smtp_login
        smtp_sender = self.config.smtp_sender
        smtp_password = self.config.smtp_password
        recipient = self.config.smtp_recipient
        if smtp_host and smtp_login and smtp_sender and recipient:
            self.email = EmailAlert(
                smtp_host=smtp_host,
                smtp_port=self.config.smtp_port,
                sender_email=smtp_sender,
                sender_password=smtp_password,
                recipient_email=recipient,
                smtp_username=smtp_login,
            )
            logger.info("Email alerts enabled")

        teams_url = self.config.teams_webhook_url
        teams_group = self.config.teams_group_id
        teams_channel = self.config.teams_channel_id
        if teams_url or (teams_group and teams_channel):
            try:
                self.teams = TeamsAlert(
                    webhook_url=teams_url,
                    group_id=teams_group,
                    channel_id=teams_channel,
                )
                mode = "Graph API" if (teams_group and teams_channel) else "webhook"
                logger.info("Teams alerts enabled (%s)", mode)
            except ValueError as exc:
                logger.warning("Teams setup failed: %s", exc)

    @property
    def has_channels(self) -> bool:
        return self.discord is not None or self.email is not None or self.teams is not None

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

        if self.teams:
            try:
                self.teams.send_predictions(predictions, model_name, run_id)
            except Exception as exc:
                logger.error("Teams prediction alert failed: %s", exc)

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

        if self.teams:
            try:
                self.teams.send_daily_summary(stats, top_picks)
            except Exception as exc:
                logger.error("Teams summary alert failed: %s", exc)

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

        if self.teams:
            try:
                self.teams.send_alert(message)
            except Exception as exc:
                logger.error("Teams alert failed: %s", exc)
