"""Tests for mlbv1.models.training_helpers."""

from __future__ import annotations

from mlbv1.models.training_helpers import ALL_MODEL_TYPES, resolve_model_types


def test_all_model_types_has_four() -> None:
    assert len(ALL_MODEL_TYPES) == 4
    assert "random_forest" in ALL_MODEL_TYPES
    assert "xgboost" in ALL_MODEL_TYPES


def test_resolve_all() -> None:
    result = resolve_model_types("all")
    assert result == list(ALL_MODEL_TYPES)


def test_resolve_both() -> None:
    result = resolve_model_types("both")
    assert result == ["random_forest", "ridge_regression"]


def test_resolve_single() -> None:
    assert resolve_model_types("xgboost") == ["xgboost"]
