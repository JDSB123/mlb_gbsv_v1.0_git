# mypy: ignore-errors
"""Inference utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mlbv1.models.ensemble import EnsembleModel
from mlbv1.models.market_deriver import MarketDeriver
from mlbv1.models.trainer import TrainedModel

logger = logging.getLogger(__name__)

# ── Calibration constants ───────────────────────────────────────────────
# Source: 2024 MLB league averages (Baseball Reference)
LEAGUE_AVG_RUNS_FG = 4.35   # runs per team per full game
LEAGUE_AVG_RUNS_F5 = 2.40   # ~55 % of scoring occurs in first 5 innings
SHRINKAGE_FACTOR = 0.30      # Bayesian regression toward league prior
MAX_DIVERGENCE_FG = 1.5      # max runs gap before market blending activates
MAX_DIVERGENCE_F5 = 1.0      # same threshold for first-5-innings market
MARKET_BLEND_RATE = 0.40     # pull strength toward market beyond threshold
PRED_FLOOR = 0.5             # minimum predicted runs per team (clamp)
PRED_CEILING = 8.0           # maximum predicted runs per team (clamp)
PROB_FLOOR = 0.005           # minimum output probability
PROB_CEILING = 0.995         # maximum output probability


@dataclass(frozen=True)
class PredictionResult:
    """Predictions with probabilities across markets."""

    expected_runs: pd.DataFrame
    market_probabilities: pd.DataFrame


def load_model(path: str) -> TrainedModel | EnsembleModel:
    """Load a trained model from disk."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)
    if not isinstance(model, (TrainedModel, EnsembleModel)):
        raise TypeError("Invalid model file")
    return model


def _calibrate_to_market(
    preds: pd.DataFrame,
    lines: pd.DataFrame,
    home_col: str,
    away_col: str,
    market_col: str,
    max_divergence: float,
) -> pd.DataFrame:
    """Blend model predictions toward market when divergence exceeds threshold."""
    if market_col not in lines.columns:
        return preds
    market_total = pd.to_numeric(lines[market_col], errors="coerce")
    model_total = preds[home_col] + preds[away_col]
    divergence = model_total - market_total
    excess = divergence.abs() - max_divergence
    correction = excess.clip(lower=0) * MARKET_BLEND_RATE * np.sign(-divergence)
    total_safe = model_total.replace(0, 1)
    preds[home_col] = preds[home_col] + correction * (preds[home_col] / total_safe)
    preds[away_col] = preds[away_col] + correction * (preds[away_col] / total_safe)
    preds[[home_col, away_col]] = preds[[home_col, away_col]].clip(
        lower=PRED_FLOOR, upper=PRED_CEILING
    )
    if home_col == "home_score":
        logger.info(
            "Calibrated FG totals: model_raw=%.2f market=%.2f → calibrated=%.2f",
            model_total.mean(),
            market_total.mean(),
            (preds[home_col] + preds[away_col]).mean(),
        )
    return preds


def predict(
    model: TrainedModel | EnsembleModel, X: pd.DataFrame, lines: pd.DataFrame | None = None
) -> PredictionResult:
    """Generate predictions and probabilities with runtime anomaly detection."""
    if lines is None:
        lines = X  # Try to grab lines from X if not explicitly supplied

    features = X[model.feature_names]

    if features.isnull().any().any():
        logger.warning("Anomaly Detected: Input features contain missing data.")
        features = features.fillna(0)

    preds = pd.DataFrame(
        model.predict(features), index=X.index, columns=model.target_names
    )

    preds = preds.clip(lower=PRED_FLOOR, upper=PRED_CEILING)

    # ── League-average regression (Bayesian shrinkage) ───────────────
    # Models trained on synthetic data can systematically under/over-predict.
    # Regress predictions toward MLB league averages to correct global bias.
    for col, prior in [
        ("home_score", LEAGUE_AVG_RUNS_FG),
        ("away_score", LEAGUE_AVG_RUNS_FG),
        ("f5_home_score", LEAGUE_AVG_RUNS_F5),
        ("f5_away_score", LEAGUE_AVG_RUNS_F5),
    ]:
        if col in preds.columns:
            preds[col] = preds[col] * (1 - SHRINKAGE_FACTOR) + prior * SHRINKAGE_FACTOR

    # ── Market-informed calibration ──────────────────────────────────
    # When market total lines are available, anchor extreme divergences.
    # This prevents the model from wildly disagreeing with the market
    # while preserving genuine model edge for reasonable disagreements.
    preds = _calibrate_to_market(
        preds, lines, "home_score", "away_score", "total_runs", MAX_DIVERGENCE_FG
    )
    preds = _calibrate_to_market(
        preds, lines, "f5_home_score", "f5_away_score", "f5_total_runs", MAX_DIVERGENCE_F5
    )

    # Re-clamp after calibration
    preds = preds.clip(lower=PRED_FLOOR, upper=PRED_CEILING)

    f5_home = preds["f5_home_score"]
    f5_away = preds["f5_away_score"]
    fg_home = preds["home_score"]
    fg_away = preds["away_score"]

    market_probs = MarketDeriver.derive_markets(
        f5_home=f5_home, f5_away=f5_away, fg_home=fg_home, fg_away=fg_away, lines=lines
    )

    # Anomaly Detection: Probability bounds check globally across the output DataFrame
    numeric_cols = market_probs.select_dtypes(include=[np.number]).columns
    market_probs[numeric_cols] = market_probs[numeric_cols].clip(
        lower=PROB_FLOOR, upper=PROB_CEILING
    )

    return PredictionResult(expected_runs=preds, market_probabilities=market_probs)

