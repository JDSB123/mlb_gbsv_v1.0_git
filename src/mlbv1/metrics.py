"""Evaluation metrics for MLB models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricsReport:
    """Metrics summary."""

    accuracy: float
    roi: float
    sharpe_ratio: float


def accuracy_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    y_true_arr = np.array(list(y_true))
    y_pred_arr = np.array(list(y_pred))
    if y_true_arr.size == 0:
        return 0.0
    return float((y_true_arr == y_pred_arr).mean())


def roi_on_spread(
    y_true: Iterable[int], y_pred: Iterable[int], stake: float = 100.0, vig: float = -110
) -> float:
    y_true_arr = np.array(list(y_true))
    y_pred_arr = np.array(list(y_pred))
    if y_true_arr.size == 0:
        return 0.0
    win_payout = stake * (100.0 / abs(vig))
    returns = np.where(y_true_arr == y_pred_arr, win_payout, -stake)
    return float(returns.sum() / (stake * len(y_true_arr)))


def sharpe_ratio(returns: Iterable[float]) -> float:
    returns_arr = np.array(list(returns))
    if returns_arr.size == 0:
        return 0.0
    std = returns_arr.std(ddof=1) if returns_arr.size > 1 else 0.0
    if std == 0.0:
        return 0.0
    return float(returns_arr.mean() / std)


def evaluate(y_true: Iterable[int], y_pred: Iterable[int]) -> MetricsReport:
    acc = accuracy_score(y_true, y_pred)
    roi = roi_on_spread(y_true, y_pred)
    y_true_arr = np.array(list(y_true))
    y_pred_arr = np.array(list(y_pred))
    win_payout = 100.0 * (100.0 / 110.0)
    returns = np.where(y_true_arr == y_pred_arr, win_payout, -100.0)
    sharpe = sharpe_ratio(returns)
    return MetricsReport(accuracy=acc, roi=roi, sharpe_ratio=sharpe)
