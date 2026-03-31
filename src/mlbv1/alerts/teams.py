"""Microsoft Teams integration for MLB pick slate posting.

Supports two modes:
    1. **Graph API** (preferred, zero-config): Uses DefaultAzureCredential
       (managed identity in ACA, az CLI locally) to post Adaptive Cards
       directly to a Teams channel.
       Requires: TEAMS_GROUP_ID + TEAMS_CHANNEL_ID in .env.
    2. **Webhook** (fallback): Posts via Power Automate Workflow webhook.
       Requires: TEAMS_WEBHOOK_URL in .env.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import urllib.request
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"
_GRAPH_SCOPE = "https://graph.microsoft.com/.default"


def _get_graph_token() -> str | None:
    """Obtain a Microsoft Graph token.

    Tries (in order):
    1. azure.identity.DefaultAzureCredential (managed identity in ACA, az CLI locally)
    2. az CLI subprocess fallback
    """
    # Try DefaultAzureCredential first (works in ACA + local)
    try:
        from azure.identity import DefaultAzureCredential

        credential = DefaultAzureCredential()
        token = credential.get_token(_GRAPH_SCOPE)
        if token and token.token:
            return token.token
    except Exception as exc:
        logger.debug("DefaultAzureCredential failed: %s", exc)

    # Fallback: az CLI subprocess (local dev)
    try:
        result = subprocess.run(
            ["az", "account", "get-access-token", "--resource",
             "https://graph.microsoft.com", "--query", "accessToken", "-o", "tsv"],
            capture_output=True, text=True, timeout=15,
            shell=_IS_WINDOWS,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


class TeamsAlert:
    """Post pick slates to a Microsoft Teams channel.

    Tries Graph API first, falls back to webhook.
    """

    def __init__(
        self,
        webhook_url: str = "",
        slate_base_url: str = "",
        group_id: str = "",
        channel_id: str = "",
    ) -> None:
        self.webhook_url = webhook_url
        self.slate_base_url = slate_base_url.rstrip("/")
        self.group_id = group_id or os.getenv("TEAMS_GROUP_ID", "")
        self.channel_id = channel_id or os.getenv("TEAMS_CHANNEL_ID", "")
        self._graph_token: str | None = None

        if not webhook_url and not (self.group_id and self.channel_id):
            raise ValueError(
                "Teams requires either TEAMS_WEBHOOK_URL or "
                "TEAMS_GROUP_ID + TEAMS_CHANNEL_ID"
            )

    # ── Public API ───────────────────────────────────────────────────

    def send_slate(
        self,
        picks: list[dict[str, Any]],
        date: str,
        slate_url: str = "",
    ) -> bool:
        """Post an Adaptive Card summary of the pick slate to Teams."""
        recommended = [
            p
            for p in picks
            if str(p.get("is_recommended", "")).lower() == "true"
        ]
        url = slate_url or (f"{self.slate_base_url}/slate?date={date}" if self.slate_base_url else "")
        card = self._build_slate_card(picks, recommended, date, url)

        payload = {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "contentUrl": None,
                    "content": card,
                }
            ],
        }
        return self._post(payload)

    def send_alert(self, message: str) -> bool:
        """Post a simple text alert to Teams."""
        card = {
            "type": "AdaptiveCard",
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "version": "1.4",
            "body": [
                {
                    "type": "TextBlock",
                    "text": "⚾ GBSV Alert",
                    "weight": "Bolder",
                    "size": "Medium",
                    "color": "Attention",
                },
                {"type": "TextBlock", "text": message, "wrap": True},
            ],
        }
        payload = {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "contentUrl": None,
                    "content": card,
                }
            ],
        }
        return self._post(payload)

    def send_daily_summary(
        self,
        stats: dict[str, Any],
        top_picks: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Post a daily ROI summary to Teams."""
        roi = stats.get("roi", 0)

        facts = [
            {"title": "Total Bets", "value": str(stats.get("total_bets", 0))},
            {"title": "Win Rate", "value": f"{stats.get('win_rate', 0):.1%}"},
            {"title": "ROI", "value": f"{roi:.1%}"},
            {"title": "Net Profit", "value": f"${stats.get('net_profit', 0):,.2f}"},
            {"title": "Balance", "value": f"${stats.get('current_balance', 0):,.2f}"},
        ]

        body: list[dict[str, Any]] = [
            {
                "type": "TextBlock",
                "text": "⚾ GBSV Daily Summary",
                "weight": "Bolder",
                "size": "Medium",
            },
            {"type": "FactSet", "facts": facts},
        ]

        if top_picks:
            picks_text = "\n".join(
                f"• **{p['home_team']} vs {p['away_team']}** — "
                f"{p.get('probability', 0):.0%} conf"
                for p in top_picks[:5]
            )
            body.append(
                {"type": "TextBlock", "text": picks_text, "wrap": True}
            )

        card = {
            "type": "AdaptiveCard",
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "version": "1.4",
            "body": body,
        }
        payload = {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "contentUrl": None,
                    "content": card,
                }
            ],
        }
        return self._post(payload)

    def send_predictions(
        self,
        predictions: list[dict[str, Any]],
        model_name: str = "ensemble",
        run_id: str = "",
    ) -> bool:
        """Post prediction results (used by AlertManager interface)."""
        if not predictions:
            return True
        return self.send_slate(predictions, datetime.now(tz=UTC).strftime("%Y-%m-%d"))

    # ── Card builder ─────────────────────────────────────────────────

    def _build_slate_card(
        self,
        all_picks: list[dict[str, Any]],
        recommended: list[dict[str, Any]],
        date: str,
        slate_url: str,
    ) -> dict[str, Any]:
        """Build an Adaptive Card for the pick slate summary."""
        n_games = len({p.get("game", "") for p in all_picks})
        n_total = len(all_picks)
        n_rec = len(recommended)

        # Top picks by EV
        top = sorted(recommended, key=lambda p: -float(p.get("no_vig_ev", 0)))[:5]

        # Header
        body: list[dict[str, Any]] = [
            {
                "type": "ColumnSet",
                "columns": [
                    {
                        "type": "Column",
                        "width": "stretch",
                        "items": [
                            {
                                "type": "TextBlock",
                                "text": "⚾ GBSV MLB SLATE",
                                "weight": "Bolder",
                                "size": "Large",
                                "color": "Accent",
                            },
                            {
                                "type": "TextBlock",
                                "text": date,
                                "spacing": "None",
                                "isSubtle": True,
                            },
                        ],
                    },
                ],
            },
        ]

        # Stats row
        body.append(
            {
                "type": "ColumnSet",
                "separator": True,
                "columns": [
                    {
                        "type": "Column",
                        "width": "1",
                        "items": [
                            {
                                "type": "TextBlock",
                                "text": str(n_games),
                                "weight": "Bolder",
                                "size": "ExtraLarge",
                                "horizontalAlignment": "Center",
                            },
                            {
                                "type": "TextBlock",
                                "text": "GAMES",
                                "size": "Small",
                                "horizontalAlignment": "Center",
                                "isSubtle": True,
                                "spacing": "None",
                            },
                        ],
                    },
                    {
                        "type": "Column",
                        "width": "1",
                        "items": [
                            {
                                "type": "TextBlock",
                                "text": str(n_total),
                                "weight": "Bolder",
                                "size": "ExtraLarge",
                                "horizontalAlignment": "Center",
                            },
                            {
                                "type": "TextBlock",
                                "text": "PICKS",
                                "size": "Small",
                                "horizontalAlignment": "Center",
                                "isSubtle": True,
                                "spacing": "None",
                            },
                        ],
                    },
                    {
                        "type": "Column",
                        "width": "1",
                        "items": [
                            {
                                "type": "TextBlock",
                                "text": str(n_rec),
                                "weight": "Bolder",
                                "size": "ExtraLarge",
                                "horizontalAlignment": "Center",
                                "color": "Good" if n_rec > 0 else "Default",
                            },
                            {
                                "type": "TextBlock",
                                "text": "RECOMMENDED",
                                "size": "Small",
                                "horizontalAlignment": "Center",
                                "isSubtle": True,
                                "spacing": "None",
                            },
                        ],
                    },
                ],
            }
        )

        # Top picks detail
        if top:
            body.append(
                {
                    "type": "TextBlock",
                    "text": "TOP PICKS",
                    "weight": "Bolder",
                    "separator": True,
                    "spacing": "Medium",
                }
            )
            for p in top:
                ev_val = float(p.get("no_vig_ev", 0))
                odds_cur = p.get("odds_current", "")
                confidence = float(p.get("confidence", 0))
                stars = "★" * round(confidence * 5) + "☆" * (5 - round(confidence * 5))

                body.append(
                    {
                        "type": "ColumnSet",
                        "columns": [
                            {
                                "type": "Column",
                                "width": "stretch",
                                "items": [
                                    {
                                        "type": "TextBlock",
                                        "text": f"**{p.get('pick', '')}**  ({p.get('game', '')})",
                                        "wrap": True,
                                    },
                                ],
                            },
                            {
                                "type": "Column",
                                "width": "auto",
                                "items": [
                                    {
                                        "type": "TextBlock",
                                        "text": f"{odds_cur}  |  EV: {ev_val:+.1%}  |  {stars}",
                                        "isSubtle": True,
                                        "size": "Small",
                                    },
                                ],
                            },
                        ],
                    }
                )
        elif n_rec == 0:
            body.append(
                {
                    "type": "TextBlock",
                    "text": "No picks met recommendation threshold today.",
                    "isSubtle": True,
                    "separator": True,
                    "wrap": True,
                }
            )

        # Action buttons
        actions: list[dict[str, Any]] = []
        if slate_url:
            actions.append(
                {
                    "type": "Action.OpenUrl",
                    "title": "📊 View Full Interactive Slate",
                    "url": slate_url,
                    "style": "positive",
                }
            )

        body.append(
            {
                "type": "TextBlock",
                "text": f"Generated {datetime.now(tz=UTC).strftime('%I:%M %p UTC')}",
                "isSubtle": True,
                "size": "Small",
                "spacing": "Medium",
                "separator": True,
            }
        )

        card: dict[str, Any] = {
            "type": "AdaptiveCard",
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "version": "1.4",
            "body": body,
        }
        if actions:
            card["actions"] = actions

        return card

    # ── HTTP post ────────────────────────────────────────────────────

    def _post(self, payload: dict[str, Any]) -> bool:
        """Post to Teams — tries webhook first (reliable from managed identity),
        then Graph API as fallback."""
        # Prefer webhook (works from managed identity / application context)
        if self.webhook_url:
            logger.info("Posting to Teams via webhook")
            if self._post_webhook(payload):
                return True
            logger.warning("Webhook post failed, trying Graph API fallback")

        # Graph API fallback (requires delegated permissions for channel messages)
        if self.group_id and self.channel_id:
            logger.info("Posting to Teams via Graph API")
            if self._post_graph(payload):
                return True
            logger.warning("Graph API post also failed")

        logger.error("No Teams posting method succeeded")
        return False

    def _post_graph(self, payload: dict[str, Any]) -> bool:
        """Post an Adaptive Card message to Teams via Microsoft Graph API."""
        if not self._graph_token:
            self._graph_token = _get_graph_token()
        if not self._graph_token:
            logger.warning("Could not obtain Graph API token via az CLI")
            return False

        # Graph chatMessage format with Adaptive Card attachment
        card = None
        attachments = payload.get("attachments", [])
        if attachments:
            card = attachments[0].get("content")

        if card:
            attach_id = "slate-card"
            graph_payload = {
                "body": {
                    "contentType": "html",
                    "content": f'<attachment id="{attach_id}"></attachment>',
                },
                "attachments": [
                    {
                        "id": attach_id,
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "contentUrl": None,
                        "content": json.dumps(card),
                        "name": None,
                        "thumbnailUrl": None,
                    }
                ],
            }
        else:
            # Plain message fallback
            content = payload.get("body", {}).get("content", "")
            if not content:
                content = json.dumps(payload)
            graph_payload = {"body": {"contentType": "html", "content": content}}

        url = (
            f"https://graph.microsoft.com/v1.0/teams/{self.group_id}"
            f"/channels/{self.channel_id}/messages"
        )
        data = json.dumps(graph_payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._graph_token}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
                if resp.status < 300:
                    logger.info("Teams Graph API post OK (%d)", resp.status)
                    return True
                logger.warning("Teams Graph API returned %d", resp.status)
                return False
        except urllib.error.HTTPError as exc:
            # Token expired — try refreshing once
            if exc.code == 401 and self._graph_token:
                self._graph_token = _get_graph_token()
                if self._graph_token:
                    req.add_header("Authorization", f"Bearer {self._graph_token}")
                    try:
                        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
                            if resp.status < 300:
                                return True
                    except Exception:
                        pass
            logger.error("Teams Graph API error %d: %s", exc.code, exc.reason)
            return False
        except Exception as exc:
            logger.error("Teams Graph API failed: %s", exc)
            return False

    def _post_webhook(self, payload: dict[str, Any]) -> bool:
        """POST JSON payload to the Teams webhook URL."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
                if resp.status < 300:
                    logger.info("Teams webhook OK (%d)", resp.status)
                    return True
                logger.warning("Teams webhook returned %d", resp.status)
                return False
        except Exception as exc:
            logger.error("Teams webhook failed: %s", exc)
            return False
