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

    # Ensure no negative runs predicted (ReLU)
    preds = preds.clip(lower=0.01)

    f5_home = preds["f5_home_score"]
    f5_away = preds["f5_away_score"]
    fg_home = preds["home_score"]
    fg_away = preds["away_score"]

    market_probs = MarketDeriver.derive_markets(
        f5_home=f5_home, f5_away=f5_away, fg_home=fg_home, fg_away=fg_away, lines=lines
    )

    # Anomaly Detection: Probability bounds check globally across the output DataFrame
    numeric_cols = market_probs.select_dtypes(include=[np.number]).columns
    market_probs[numeric_cols] = market_probs[numeric_cols].clip(lower=0.005, upper=0.995)

    return PredictionResult(expected_runs=preds, market_probabilities=market_probs)

