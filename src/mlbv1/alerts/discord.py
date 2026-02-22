"""Discord webhook alerts for MLB predictions."""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


class DiscordAlert:
    """Send prediction alerts to a Discord channel via webhook."""

    def __init__(self, webhook_url: str) -> None:
        if not webhook_url:
            raise ValueError("Discord webhook URL is required")
        self.webhook_url = webhook_url

    def send_predictions(
        self,
        predictions: list[dict[str, Any]],
        model_name: str = "ensemble",
        run_id: str = "",
    ) -> bool:
        """Send a formatted prediction embed to Discord."""
        if not predictions:
            logger.info("No predictions to send to Discord")
            return True

        embeds = self._build_embeds(predictions, model_name, run_id)

        for batch_start in range(0, len(embeds), 10):
            batch = embeds[batch_start : batch_start + 10]
            payload = {
                "username": "MLB GBSV Bot",
                "avatar_url": "https://img.icons8.com/color/96/baseball.png",
                "embeds": batch,
            }
            if not self._post(payload):
                return False
        return True

    def send_daily_summary(
        self,
        stats: dict[str, Any],
        top_picks: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Send a daily summary embed."""
        fields = [
            {
                "name": "Total Bets",
                "value": str(stats.get("total_bets", 0)),
                "inline": True,
            },
            {
                "name": "Win Rate",
                "value": f"{stats.get('win_rate', 0):.1%}",
                "inline": True,
            },
            {"name": "ROI", "value": f"{stats.get('roi', 0):.1%}", "inline": True},
            {
                "name": "Net Profit",
                "value": f"${stats.get('net_profit', 0):,.2f}",
                "inline": True,
            },
            {
                "name": "Balance",
                "value": f"${stats.get('current_balance', 0):,.2f}",
                "inline": True,
            },
            {
                "name": "Sharpe",
                "value": f"{stats.get('sharpe', 0):.3f}",
                "inline": True,
            },
        ]

        embed: dict[str, Any] = {
            "title": "MLB GBSV — Daily Summary",
            "color": 0x2ECC71 if stats.get("roi", 0) >= 0 else 0xE74C3C,
            "fields": fields,
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "footer": {"text": "MLB GBSV v1.0"},
        }

        if top_picks:
            picks_text = "\n".join(
                f"• **{p['home_team']} vs {p['away_team']}** "
                f"({p.get('side', '?')} {p.get('spread', 0):+.1f}) "
                f"— {p.get('probability', 0):.0%} conf"
                for p in top_picks[:5]
            )
            embed["description"] = f"**Top Picks:**\n{picks_text}"

        return self._post(
            {
                "username": "MLB GBSV Bot",
                "avatar_url": "https://img.icons8.com/color/96/baseball.png",
                "embeds": [embed],
            }
        )

    def send_alert(self, message: str, color: int = 0x3498DB) -> bool:
        """Send a simple text alert."""
        embed = {
            "title": "MLB GBSV Alert",
            "description": message,
            "color": color,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }
        return self._post(
            {
                "username": "MLB GBSV Bot",
                "embeds": [embed],
            }
        )

    # -- private --------------------------------------------------------------

    def _build_embeds(
        self,
        predictions: list[dict[str, Any]],
        model_name: str,
        run_id: str,
    ) -> list[dict[str, Any]]:
        embeds: list[dict[str, Any]] = []
        for pred in predictions:
            prob = pred.get("probability", 0.5)
            side = "HOME" if pred.get("prediction", 0) == 1 else "AWAY"
            confidence = "HIGH" if prob >= 0.65 else "MED" if prob >= 0.55 else "LOW"
            color = (
                0x2ECC71
                if confidence == "HIGH"
                else 0xF39C12 if confidence == "MED" else 0x95A5A6
            )

            embeds.append(
                {
                    "title": f"{pred.get('away_team', '?')} @ {pred.get('home_team', '?')}",
                    "color": color,
                    "fields": [
                        {"name": "Pick", "value": side, "inline": True},
                        {
                            "name": "Spread",
                            "value": f"{pred.get('spread', 0):+.1f}",
                            "inline": True,
                        },
                        {"name": "Confidence", "value": f"{prob:.1%}", "inline": True},
                        {"name": "Model", "value": model_name, "inline": True},
                        {
                            "name": "Edge",
                            "value": f"{pred.get('edge', 0):.1%}",
                            "inline": True,
                        },
                        {
                            "name": "Bet Size",
                            "value": f"${pred.get('recommended_bet', 0):,.0f}",
                            "inline": True,
                        },
                    ],
                    "timestamp": datetime.now(tz=UTC).isoformat(),
                    "footer": {"text": f"Run: {run_id[:8]}... | {confidence}"},
                }
            )
        return embeds

    def _post(self, payload: dict[str, Any]) -> bool:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as resp:
                if resp.status not in (200, 204):
                    logger.warning("Discord webhook returned %d", resp.status)
                    return False
            return True
        except Exception as exc:
            logger.error("Discord webhook failed: %s", exc)
            return False
