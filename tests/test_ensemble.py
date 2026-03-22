"""Tests for ensemble models."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from mlbv1.data.preprocessor import ProcessedData
from mlbv1.features.engineer import FeatureSet
from mlbv1.models.ensemble import EnsembleTrainer


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
