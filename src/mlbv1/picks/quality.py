"""Quality metrics, diversification checks, and publish gates for pick slates."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def compute_quality_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute data-quality and provenance coverage metrics for a slate."""
    if not rows:
        return {
            "row_count": 0,
            "game_count": 0,
            "live_odds_rate": 0.0,
            "default_line_rate": 0.0,
            "default_counter_odds_rate": 0.0,
            "unk_team_rows": 0,
            "duplicate_rows": 0,
            "missing_required_rows": 0,
            "model_count": 0,
        }

    df = pd.DataFrame(rows)
    required = [
        "game", "home_team", "away_team", "segment",
        "market_type", "pick", "odds_current", "model_prob",
    ]
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA
    missing_required_rows = int(df[required].isna().any(axis=1).sum())
    duplicate_rows = int(
        df.duplicated(subset=["game", "segment", "market_type", "pick"]).sum()
    )
    unk_team_rows = int(
        ((df["home_team"] == "UNK") | (df["away_team"] == "UNK")).sum()
    )

    return {
        "row_count": int(len(df)),
        "game_count": int(df["game"].nunique()),
        "live_odds_rate": float((df["odds_quality"] == "live").mean()),
        "default_line_rate": float(df["used_default_line"].astype(bool).mean()),
        "default_counter_odds_rate": float(
            df["used_default_counter_odds"].astype(bool).mean()
        ),
        "unk_team_rows": unk_team_rows,
        "duplicate_rows": duplicate_rows,
        "missing_required_rows": missing_required_rows,
        "model_count": int(df["model_count"].max()),
        **compute_diversification_metrics(df),
    }


def compute_diversification_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Check if recommended picks are diversified across market directions."""
    recs = df[df["is_recommended"] == True]  # noqa: E712
    if recs.empty:
        return {"diversification_warning": False, "rec_direction_breakdown": {}}
    picks = recs["pick"].str.lower()
    n_under = int(picks.str.contains("under").sum())
    n_over = int(picks.str.contains("over").sum())
    n_total_market = n_under + n_over
    n_recs = len(recs)
    one_sided = False
    dominant_direction = None
    if n_total_market >= 2:
        if n_under / n_total_market > 0.75:
            one_sided = True
            dominant_direction = "under"
        elif n_over / n_total_market > 0.75:
            one_sided = True
            dominant_direction = "over"
    return {
        "diversification_warning": one_sided,
        "dominant_direction": dominant_direction,
        "rec_direction_breakdown": {
            "total_recs": n_recs,
            "over_recs": n_over,
            "under_recs": n_under,
            "spread_ml_recs": n_recs - n_total_market,
        },
    }


def run_quality_gates(metrics: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    """Evaluate hard publish gates for canonical slate output."""
    gates = [
        {
            "name": "non_empty_output",
            "passed": metrics["row_count"] > 0,
            "value": metrics["row_count"],
            "threshold": "> 0",
        },
        {
            "name": "no_unknown_teams",
            "passed": metrics["unk_team_rows"] == 0,
            "value": metrics["unk_team_rows"],
            "threshold": "== 0",
        },
        {
            "name": "no_duplicate_market_rows",
            "passed": metrics["duplicate_rows"] == 0,
            "value": metrics["duplicate_rows"],
            "threshold": "== 0",
        },
        {
            "name": "required_fields_present",
            "passed": metrics["missing_required_rows"] == 0,
            "value": metrics["missing_required_rows"],
            "threshold": "== 0",
        },
        {
            "name": "live_odds_coverage",
            "passed": metrics["live_odds_rate"] >= 0.95,
            "value": round(metrics["live_odds_rate"], 4),
            "threshold": ">= 0.95",
        },
        {
            "name": "min_model_count",
            "passed": metrics["model_count"] >= 2,
            "value": metrics["model_count"],
            "threshold": ">= 2",
        },
        {
            "name": "default_line_rate_reasonable",
            "passed": metrics["default_line_rate"] <= 0.60,
            "value": round(metrics["default_line_rate"], 4),
            "threshold": "<= 0.60",
        },
    ]
    passed = all(g["passed"] for g in gates)
    return passed, gates


def write_audit_artifact(path: Path, audit: dict[str, Any]) -> None:
    """Write JSON audit file with data-quality metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
