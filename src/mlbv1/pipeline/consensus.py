"""Consensus pick aggregation and model weighting."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def build_model_weights(
    db: Any,
    lookback_runs: int = 200,
    decay: float = 0.97,
) -> dict[str, float]:
    """Build per-model weights from recent run accuracy history.

    Uses exponential recency decay, where newer runs carry more weight.
    """
    runs = db.get_runs(limit=lookback_runs)
    weighted_sums: dict[str, float] = {}
    weight_totals: dict[str, float] = {}

    for age, run in enumerate(runs):
        model = str(run.get("model_name", "")).strip()
        accuracy_raw = run.get("accuracy")
        if not model or accuracy_raw is None:
            continue
        try:
            accuracy = float(accuracy_raw)
        except (TypeError, ValueError):
            continue
        recency_weight = decay**age
        weighted_sums[model] = weighted_sums.get(model, 0.0) + (accuracy * recency_weight)
        weight_totals[model] = weight_totals.get(model, 0.0) + recency_weight

    weights: dict[str, float] = {}
    for model_name, total in weighted_sums.items():
        denom = weight_totals.get(model_name, 0.0)
        if denom <= 0:
            continue
        weights[model_name] = max(0.05, float(total / denom))

    if weights:
        logger.info(
            "Model weights from recent accuracy (decay=%.2f): %s",
            decay,
            ", ".join(f"{k}={v:.3f}" for k, v in sorted(weights.items())),
        )
    else:
        logger.info("No historical model accuracy found; using equal weights")

    return weights


def build_consensus_picks(
    all_picks: list[dict[str, Any]],
    model_weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Collapse per-model picks into one weighted consensus pick per game."""
    if not all_picks:
        return []

    frame = pd.DataFrame(all_picks)
    weights = model_weights or {}
    frame["model_weight"] = frame["model_name"].map(weights).fillna(1.0).astype(float)
    keys = [
        "game_date", "home_team", "away_team",
        "spread", "home_moneyline", "away_moneyline",
    ]

    consensus: list[dict[str, Any]] = []
    grouped = frame.groupby(keys, dropna=False, sort=False)
    for game_key, group in grouped:
        total_weight = float(group["model_weight"].sum())
        if total_weight <= 0:
            total_weight = float(group.shape[0])
            group = group.assign(model_weight=1.0)

        weighted_home_prob = float(
            (group["probability"] * group["model_weight"]).sum() / total_weight
        )
        weighted_home_pick = float(
            (group["prediction"] * group["model_weight"]).sum() / total_weight
        )
        consensus_prediction = int(round(weighted_home_pick))
        agreement = max(weighted_home_pick, 1.0 - weighted_home_pick)
        model_names = sorted({str(name) for name in group["model_name"].tolist()})

        game_date, home_team, away_team, spread, home_ml, away_ml = game_key
        consensus.append({
            "game_date": str(game_date),
            "home_team": str(home_team),
            "away_team": str(away_team),
            "spread": float(str(spread)),
            "prediction": consensus_prediction,
            "probability": weighted_home_prob,
            "home_moneyline": int(str(home_ml)),
            "away_moneyline": int(str(away_ml)),
            "model_name": "consensus",
            "model_count": int(group.shape[0]),
            "agreement": float(agreement),
            "model_names": model_names,
            "weight_sum": total_weight,
        })

    consensus.sort(key=lambda p: abs(float(p["probability"]) - 0.5), reverse=True)
    return consensus
