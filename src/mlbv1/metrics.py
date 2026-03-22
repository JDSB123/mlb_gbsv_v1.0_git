"""Evaluation metrics for MLB models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


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

def evaluate(y_true: pd.DataFrame, y_pred: np.ndarray) -> MetricsReport:
    try:
        y_true_clean = y_true[["f5_home_score", "f5_away_score", "home_score", "away_score"]].fillna(0)
        mse = mean_squared_error(y_true_clean, y_pred)
        acc = -mse 
    except Exception:
        acc = -999.0
    
    roi = 0.00
    sharpe = 0.00
    return MetricsReport(accuracy=acc, roi=roi, sharpe_ratio=sharpe)
