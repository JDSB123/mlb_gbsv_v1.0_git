"""Model training for MLB spread predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from mlbv1.config import LogisticRegressionConfig, RandomForestConfig


@dataclass(frozen=True)
class TrainedModel:
    """Container for trained model artifacts."""

    name: str
    model: object
    scaler: StandardScaler | None
    feature_names: list[str]


class ModelTrainer:
    """Train scikit-learn models for MLB predictions."""

    def __init__(self, output_dir: str = "artifacts/models") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_random_forest(
        self, X: pd.DataFrame, y: pd.Series, config: RandomForestConfig
    ) -> TrainedModel:
        model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_state,
        )
        model.fit(X, y)
        return TrainedModel(name="random_forest", model=model, scaler=None, feature_names=list(X.columns))

    def train_logistic_regression(
        self, X: pd.DataFrame, y: pd.Series, config: LogisticRegressionConfig
    ) -> TrainedModel:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(
            C=config.C,
            max_iter=config.max_iter,
            random_state=config.random_state,
        )
        model.fit(X_scaled, y)
        return TrainedModel(
            name="logistic_regression", model=model, scaler=scaler, feature_names=list(X.columns)
        )

    def evaluate(self, model: TrainedModel, X: pd.DataFrame, y: pd.Series) -> float:
        if model.scaler:
            preds = model.model.predict(model.scaler.transform(X))
        else:
            preds = model.model.predict(X)
        return float(accuracy_score(y, preds))

    def save(self, model: TrainedModel) -> Path:
        path = self.output_dir / f"{model.name}.pkl"
        with open(path, "wb") as handle:
            pickle.dump(model, handle)
        return path

    def train_all(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        rf_config: RandomForestConfig,
        lr_config: LogisticRegressionConfig,
    ) -> Dict[str, TrainedModel]:
        models = {
            "random_forest": self.train_random_forest(X, y, rf_config),
            "logistic_regression": self.train_logistic_regression(X, y, lr_config),
        }
        return models
