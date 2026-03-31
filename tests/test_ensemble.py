"""Tests for ensemble models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from mlbv1.data.preprocessor import ProcessedData
from mlbv1.features.engineer import FeatureSet
from mlbv1.models.ensemble import EnsembleModel, EnsembleTrainer


def test_voting_ensemble(large_feature_set: tuple[FeatureSet, ProcessedData]) -> None:
    features, processed = large_feature_set
    X = features.X
    y = processed.target

    rf = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
    rf.fit(X, y)
    lr = Ridge(random_state=42)
    lr.fit(X, y)

    ensemble = EnsembleTrainer.build_voting_ensemble(
        models=[("rf", rf), ("lr", lr)],
        feature_names=features.feature_names,
        target_names=processed.target.columns.tolist()
    )
    preds = ensemble.predict(X)
    assert len(preds) == len(X)
    assert preds.shape[1] == y.shape[1]


def test_stacking_ensemble(large_feature_set: tuple[FeatureSet, ProcessedData]) -> None:
    features, processed = large_feature_set
    X = features.X
    y = processed.target

    rf = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
    lr = Ridge(random_state=42)

    ensemble = EnsembleTrainer.build_stacking_ensemble(
        models=[("rf", rf), ("lr", lr)],
        X_train=X,
        y_train=y,
        feature_names=features.feature_names,
        target_names=processed.target.columns.tolist(),
        cv=3,
    )
    preds = ensemble.predict(X)
    assert len(preds) == len(X)
    assert preds.shape[1] == y.shape[1]


class ConstantRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, value: float = 0.0, n_targets: int = 1) -> None:
        self.value = value
        self.n_targets = n_targets

    def fit(self, X, y):  # type: ignore[no-untyped-def]
        return self

    def predict(self, X):  # type: ignore[no-untyped-def]
        return np.full((len(X), self.n_targets), self.value, dtype=float)


def test_stacking_prediction_uses_meta_model() -> None:
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    ensemble = EnsembleModel(
        name="stacking_ensemble",
        base_models=[
            ("low", ConstantRegressor(value=1.0)),
            ("high", ConstantRegressor(value=3.0)),
        ],
        meta_model=ConstantRegressor(value=7.0),
        meta_scaler=None,
        strategy="stacking",
        feature_names=["feature"],
        target_names=["runs"],
    )

    preds = ensemble.predict(X)

    assert np.allclose(preds, 7.0)


def test_stacking_uses_time_series_split_by_default(monkeypatch) -> None:
    calls: list[int] = []

    class RecordingTimeSeriesSplit:
        def __init__(self, n_splits: int) -> None:
            calls.append(n_splits)
            self.n_splits = n_splits

        def split(self, X, y=None, groups=None):  # type: ignore[no-untyped-def]
            yield np.array([0, 1]), np.array([2, 3])
            yield np.array([0, 1, 2, 3]), np.array([4, 5])

    monkeypatch.setattr("mlbv1.models.ensemble.TimeSeriesSplit", RecordingTimeSeriesSplit)

    X = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
    y = pd.DataFrame(
        {
            "f5_home_score": [1, 2, 3, 4, 5, 6],
            "f5_away_score": [0, 1, 1, 2, 2, 3],
            "home_score": [2, 3, 4, 5, 6, 7],
            "away_score": [1, 1, 2, 2, 3, 3],
        }
    )

    ensemble = EnsembleTrainer.build_stacking_ensemble(
        models=[
            ("rf", RandomForestRegressor(n_estimators=5, random_state=42)),
            ("lr", Ridge(random_state=42)),
        ],
        X_train=X,
        y_train=y,
        feature_names=["feature"],
        target_names=y.columns.tolist(),
        cv=2,
    )

    assert calls == [2]
    assert ensemble.predict(X).shape == (len(X), y.shape[1])
