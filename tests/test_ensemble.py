"""Tests for ensemble models."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mlbv1.data.preprocessor import ProcessedData
from mlbv1.features.engineer import FeatureSet
from mlbv1.models.ensemble import EnsembleTrainer


def test_voting_ensemble(large_feature_set: tuple[FeatureSet, ProcessedData]) -> None:
    features, processed = large_feature_set
    X = features.X
    y = processed.target

    rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    rf.fit(X, y)
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X, y)

    ensemble = EnsembleTrainer.build_voting_ensemble(
        models=[("rf", rf), ("lr", lr)],
        feature_names=features.feature_names,
    )
    preds = ensemble.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})


def test_voting_ensemble_proba(
    large_feature_set: tuple[FeatureSet, ProcessedData],
) -> None:
    features, processed = large_feature_set
    X = features.X
    y = processed.target

    rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    rf.fit(X, y)
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X, y)

    ensemble = EnsembleTrainer.build_voting_ensemble(
        models=[("rf", rf), ("lr", lr)],
        feature_names=features.feature_names,
    )
    probas = ensemble.predict_proba(X)
    assert probas.shape == (len(X), 2)
    assert (probas >= 0).all()
    assert (probas <= 1).all()


def test_stacking_ensemble(large_feature_set: tuple[FeatureSet, ProcessedData]) -> None:
    features, processed = large_feature_set
    X = features.X
    y = processed.target

    rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    lr = LogisticRegression(max_iter=500, random_state=42)

    ensemble = EnsembleTrainer.build_stacking_ensemble(
        models=[("rf", rf), ("lr", lr)],
        X_train=X,
        y_train=y,
        feature_names=features.feature_names,
        cv=3,
    )
    preds = ensemble.predict(X)
    assert len(preds) == len(X)

    probas = ensemble.predict_proba(X)
    assert probas.shape == (len(X), 2)
