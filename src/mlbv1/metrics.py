"""Evaluation metrics for MLB models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricsReport:
    """Metrics summary."""

    accuracy: float
    roi: float
    sharpe_ratio: float


def sharpe_ratio(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    std = returns.std(ddof=1) if returns.size > 1 else 0.0
    if std == 0.0:
        return 0.0
    return float(returns.mean() / std)


def evaluate(y_true: pd.DataFrame, y_pred: Any) -> MetricsReport:
    """Evaluate model predictions against actuals.

    *accuracy* is reported as negative MSE (higher = better, 0.0 = perfect).
    *roi* and *sharpe_ratio* require bet-level P&L data which is not available
    at model-evaluation time; they are placeholders filled by the tracking layer.
    """
    score_cols = ["f5_home_score", "f5_away_score", "home_score", "away_score"]
    try:
        y_true_clean = y_true[score_cols].fillna(0)
        mse = mean_squared_error(y_true_clean, y_pred)
        acc = -mse
    except (KeyError, ValueError) as exc:
        logger.warning("Metrics evaluation failed: %s", exc)
        acc = -999.0

    return MetricsReport(accuracy=acc, roi=0.0, sharpe_ratio=0.0)
