"""Tests for model training and prediction."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from mlbv1.config import LightGBMConfig, RandomForestConfig, RidgeRegressionConfig, XGBoostConfig
from mlbv1.data.loader import SyntheticDataLoader
from mlbv1.data.preprocessor import preprocess
from mlbv1.features.engineer import engineer_features
from mlbv1.models.trainer import ModelTrainer


def test_train_random_forest() -> None:
    df = SyntheticDataLoader(num_games=40).load()
    processed = preprocess(df)
    features = engineer_features(processed.features)
    trainer = ModelTrainer(output_dir="artifacts/test_models")
    model = trainer.train_random_forest(features.X, processed.target, trainer_config())
    acc = trainer.evaluate(model, features.X, processed.target)
    assert acc <= 0


def test_train_ridge_regression_is_self_contained() -> None:
    df = SyntheticDataLoader(num_games=40, seed=7).load()
    processed = preprocess(df)
    features = engineer_features(processed.features)
    trainer = ModelTrainer(output_dir="artifacts/test_models")
    model = trainer.train_ridge_regression(
        features.X,
        processed.target,
        RidgeRegressionConfig(),
    )

    preds = model.predict(features.X)

    assert model.scaler is None
    assert preds.shape == (len(features.X), len(model.target_names))
    assert np.allclose(preds, model.model.predict(features.X))


class _FakeBoostRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def fit(self, X, y):  # type: ignore[no-untyped-def]
        self.n_outputs_ = y.shape[1] if getattr(y, "ndim", 1) > 1 else 1
        return self

    def predict(self, X):  # type: ignore[no-untyped-def]
        if self.n_outputs_ == 1:
            return np.zeros(len(X))
        return np.zeros((len(X), self.n_outputs_))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return dict(self.kwargs)

    def set_params(self, **params):  # type: ignore[no-untyped-def]
        self.kwargs = dict(params)
        return self


def test_train_xgboost_passes_full_config(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeXGBRegressor(_FakeBoostRegressor):
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setitem(sys.modules, "xgboost", SimpleNamespace(XGBRegressor=FakeXGBRegressor))

    df = SyntheticDataLoader(num_games=20, seed=11).load()
    processed = preprocess(df)
    features = engineer_features(processed.features)
    trainer = ModelTrainer(output_dir="artifacts/test_models")
    trainer.train_xgboost(
        features.X,
        processed.target,
        XGBoostConfig(
            n_estimators=17,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.6,
            min_child_weight=5,
            gamma=0.2,
            reg_alpha=0.4,
            reg_lambda=1.8,
            eval_metric="rmse",
            use_label_encoder=False,
        ),
    )

    assert captured["min_child_weight"] == 5
    assert captured["gamma"] == 0.2
    assert captured["reg_alpha"] == 0.4
    assert captured["reg_lambda"] == 1.8
    assert captured["eval_metric"] == "rmse"
    assert captured["use_label_encoder"] is False


def test_train_lightgbm_passes_full_config(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeLGBMRegressor(_FakeBoostRegressor):
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setitem(sys.modules, "lightgbm", SimpleNamespace(LGBMRegressor=FakeLGBMRegressor))

    df = SyntheticDataLoader(num_games=20, seed=13).load()
    processed = preprocess(df)
    features = engineer_features(processed.features)
    trainer = ModelTrainer(output_dir="artifacts/test_models")
    trainer.train_lightgbm(
        features.X,
        processed.target,
        LightGBMConfig(
            n_estimators=19,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.75,
            colsample_bytree=0.65,
            min_child_samples=9,
            reg_alpha=0.15,
            reg_lambda=1.6,
            num_leaves=22,
            verbose=-1,
        ),
    )

    assert captured["min_child_samples"] == 9
    assert captured["reg_alpha"] == 0.15
    assert captured["reg_lambda"] == 1.6
    assert captured["num_leaves"] == 22
    assert captured["verbose"] == -1


def trainer_config() -> RandomForestConfig:
    return RandomForestConfig(n_estimators=10, max_depth=3)
