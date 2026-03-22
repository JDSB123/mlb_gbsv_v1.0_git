"""Tests for metrics module."""

from __future__ import annotations

import numpy as np
import pandas as pd
from mlbv1.metrics import evaluate, sharpe_ratio

def test_sharpe_ratio_constant() -> None:
    assert sharpe_ratio(np.array([1.0, 1.0, 1.0])) == 0.0

def test_sharpe_ratio_empty() -> None:
    assert sharpe_ratio(np.array([])) == 0.0

def test_sharpe_ratio_basic() -> None:
    result = sharpe_ratio(np.array([10.0, -5.0, 8.0, -3.0]))
    assert isinstance(result, float)

def test_evaluate_integration() -> None:
    y_true = pd.DataFrame(
        [[1, 2, 3, 4], [0, 1, 0, 1]],
        columns=["f5_home_score", "f5_away_score", "home_score", "away_score"]
    )
    y_pred = np.array([[1, 2, 3, 4], [0, 1, 0, 1]])
    
    report = evaluate(y_true, y_pred)
    assert report.accuracy == -0.0
    assert isinstance(report.roi, float)
    assert isinstance(report.sharpe_ratio, float)
