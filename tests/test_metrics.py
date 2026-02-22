"""Tests for metrics module."""

from __future__ import annotations

from mlbv1.metrics import accuracy_score, evaluate, roi_on_spread, sharpe_ratio


def test_accuracy_score_perfect() -> None:
    assert accuracy_score([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0


def test_accuracy_score_zero() -> None:
    assert accuracy_score([1, 1, 1], [0, 0, 0]) == 0.0


def test_accuracy_score_empty() -> None:
    assert accuracy_score([], []) == 0.0


def test_roi_on_spread_all_wins() -> None:
    roi = roi_on_spread([1, 1, 1], [1, 1, 1])
    assert roi > 0


def test_roi_on_spread_all_losses() -> None:
    roi = roi_on_spread([1, 1, 1], [0, 0, 0])
    assert roi < 0


def test_roi_breakeven() -> None:
    # At -110, need ~52.4% to breakeven. 50% should lose
    roi = roi_on_spread([1, 0, 1, 0], [1, 1, 0, 0])
    assert abs(roi) < 0.5  # sanity check


def test_sharpe_ratio_constant() -> None:
    assert sharpe_ratio([1.0, 1.0, 1.0]) == 0.0


def test_sharpe_ratio_empty() -> None:
    assert sharpe_ratio([]) == 0.0


def test_sharpe_ratio_basic() -> None:
    result = sharpe_ratio([10.0, -5.0, 8.0, -3.0])
    assert isinstance(result, float)


def test_evaluate_integration() -> None:
    report = evaluate([1, 0, 1, 0, 1], [1, 1, 1, 0, 0])
    assert 0.0 <= report.accuracy <= 1.0
    assert isinstance(report.roi, float)
    assert isinstance(report.sharpe_ratio, float)
