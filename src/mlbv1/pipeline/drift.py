"""Model drift detection — checks prediction outputs for systematic bias."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def check_model_drift(
    picks: list[dict[str, Any]],
    alert_callback: Any | None = None,
) -> list[str]:
    """Check model output metrics for drift and return warnings.

    *alert_callback*, if provided, should have a ``send_alert(msg)`` method
    and a ``has_channels`` attribute (e.g. ``AlertManager``).
    """
    if not picks:
        logger.info("No picks to check for drift")
        return []

    drift_warnings: list[str] = []

    # 1. Average predicted total (MLB avg ~8.6)
    totals: list[float] = []
    for p in picks:
        h = float(p.get("exp_home_score", p.get("home_score", 0)))
        a = float(p.get("exp_away_score", p.get("away_score", 0)))
        if h > 0 and a > 0:
            totals.append(h + a)
    if totals:
        avg_total = sum(totals) / len(totals)
        if avg_total > 11.0:
            drift_warnings.append(
                f"Score inflation: avg predicted total {avg_total:.1f} (threshold: 11.0)"
            )

    # 2. Kelly sizing (quarter-Kelly max should be ~25%)
    kellys = [
        float(p.get("kelly", p.get("edge", 0)))
        for p in picks
        if float(p.get("kelly", p.get("edge", 0))) > 0
    ]
    if kellys:
        avg_kelly = sum(kellys) / len(kellys)
        max_kelly = max(kellys)
        if avg_kelly > 0.25:
            drift_warnings.append(f"Kelly inflation: avg {avg_kelly:.1%} (threshold: 25%)")
        if max_kelly > 0.50:
            drift_warnings.append(f"Kelly spike: max {max_kelly:.1%} (threshold: 50%)")

    # 3. Home bias (sides only)
    side_picks = [p for p in picks if p.get("side") in ("home", "away")]
    if len(side_picks) >= 4:
        home_count = sum(1 for p in side_picks if p.get("side") == "home")
        home_pct = home_count / len(side_picks) if side_picks else 0
        if home_pct > 0.80:
            drift_warnings.append(
                f"Home bias: {home_pct:.0%} of side picks favor home "
                f"({home_count}/{len(side_picks)})"
            )

    # 4. Over/Under balance
    over_picks = [p for p in picks if "over" in str(p.get("side", "")).lower()]
    under_picks = [p for p in picks if "under" in str(p.get("side", "")).lower()]
    if len(over_picks) + len(under_picks) >= 4 and len(under_picks) == 0:
        drift_warnings.append(f"Over bias: {len(over_picks)} overs, 0 unders")

    if drift_warnings:
        msg = "\u26a0 MODEL DRIFT DETECTED:\n" + "\n".join(f"  \u2022 {w}" for w in drift_warnings)
        logger.warning(msg)
        if alert_callback and getattr(alert_callback, "has_channels", False):
            alert_callback.send_alert(msg)
    else:
        logger.info("Model drift check passed \u2014 all metrics within thresholds")

    return drift_warnings
