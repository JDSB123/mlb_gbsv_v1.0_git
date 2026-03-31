"""Model training for MLB Regression Multi-Market predictions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlbv1.config import (
    LightGBMConfig,
    RandomForestConfig,
    RidgeRegressionConfig,
    XGBoostConfig,
)

logger = logging.getLogger(__name__)

class RegressorLike(Protocol):
    """Protocol for regressors."""
    def predict(self, X: Any) -> Any:
        ...

@dataclass(frozen=True)
class TrainedModel:
    """Container for trained model artifacts."""
    name: str
    model: RegressorLike
    scaler: StandardScaler | None
    feature_names: list[str]
    target_names: list[str]

    def predict(self, X: pd.DataFrame) -> Any:
        """Predict using the trained feature schema and optional scaler."""
        features = X[self.feature_names]
        if self.scaler is not None:
            features = pd.DataFrame(
                self.scaler.transform(features),
                columns=self.feature_names,
                index=features.index,
            )
        return self.model.predict(features)

class ModelTrainer:
    """Train scikit-learn / xgboost / lightgbm multi-target regressors."""

    def __init__(self, output_dir: str = "artifacts/models") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_names = ["f5_home_score", "f5_away_score", "home_score", "away_score"]

    def _prepare_targets(self, y: Any) -> pd.DataFrame:
        """Ensure targets exist and are aligned."""
        missing = [col for col in self.target_names if col not in y.columns]
        if missing:
            raise ValueError(f"Missing target columns: {missing}")
        return y[self.target_names].fillna(0)

    # ------------------------------------------------------------------
    # Random Forest Regression
    # ------------------------------------------------------------------
    def train_random_forest(
        self, X: pd.DataFrame, y: Any, config: RandomForestConfig
    ) -> TrainedModel:
        y_multi = self._prepare_targets(y)
        base_model = RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_state,
        )
        model = MultiOutputRegressor(base_model)
        model.fit(X, y_multi)
        return TrainedModel(
            name="random_forest",
            model=model,
            scaler=None,
            feature_names=list(X.columns),
            target_names=self.target_names,
        )

    # ------------------------------------------------------------------
    # Ridge Regression
    # ------------------------------------------------------------------
    def train_ridge_regression(
        self, X: pd.DataFrame, y: Any, config: RidgeRegressionConfig
    ) -> TrainedModel:
        y_multi = self._prepare_targets(y)
        base_model = Ridge(
            alpha=1.0 / (config.C + 1e-6),  # C is inverse of regularization strength
            max_iter=config.max_iter,
            random_state=config.random_state,
        )
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", MultiOutputRegressor(base_model)),
            ]
        )
        model.fit(X, y_multi)
        return TrainedModel(
            name="ridge_regression",
            model=model,
            scaler=None,
            feature_names=list(X.columns),
            target_names=self.target_names,
        )

    # Backward-compatible alias
    train_logistic_regression = train_ridge_regression

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------
    def train_xgboost(
        self, X: pd.DataFrame, y: Any, config: XGBoostConfig
    ) -> TrainedModel:
        import xgboost as xgb

        y_multi = self._prepare_targets(y)
        base_model = xgb.XGBRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            min_child_weight=config.min_child_weight,
            gamma=config.gamma,
            reg_alpha=config.reg_alpha,
            reg_lambda=config.reg_lambda,
            random_state=config.random_state,
            use_label_encoder=config.use_label_encoder,
            eval_metric=config.eval_metric,
            objective="reg:squarederror", # appropriate for counts
        )
        model = MultiOutputRegressor(base_model)
        model.fit(X, y_multi)
        return TrainedModel(
            name="xgboost",
            model=model,
            scaler=None,
            feature_names=list(X.columns),
            target_names=self.target_names,
        )

    # ------------------------------------------------------------------
    # LightGBM
    # ------------------------------------------------------------------
    def train_lightgbm(
        self, X: pd.DataFrame, y: Any, config: LightGBMConfig
    ) -> TrainedModel:
        import lightgbm as lgb

        y_multi = self._prepare_targets(y)
        base_model = lgb.LGBMRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            min_child_samples=config.min_child_samples,
            reg_alpha=config.reg_alpha,
            reg_lambda=config.reg_lambda,
            num_leaves=config.num_leaves,
            random_state=config.random_state,
            verbose=config.verbose,
            objective="regression",
        )
        model = MultiOutputRegressor(base_model)
        model.fit(X, y_multi)
        return TrainedModel(
            name="lightgbm",
            model=model,
            scaler=None,
            feature_names=list(X.columns),
            target_names=self.target_names,
        )

    def evaluate(
        self, trained: TrainedModel, X: pd.DataFrame, y: pd.DataFrame
    ) -> float:
        """Evaluate model returning negative MSE as 'accuracy' for compatibility, or an R2 score."""
        y_multi = self._prepare_targets(y)
        preds = trained.predict(X)
        mse = mean_squared_error(y_multi, preds)
        
        logger.info("%s evaluation MSE: %.3f", trained.name, mse)
        # To maintain compatibility with scripts expecting higher-is-better metrics (like 'accuracy')
        # we return pseudo accuracy. Negative MSE keeps higher values (closer to 0) better.
        return -float(mse)

    def save(self, trained: TrainedModel) -> None:
        """Save trained model to disk."""
        path = self.output_dir / f"{trained.name}.pkl"
        joblib.dump(trained, path)
        logger.info("Saved model to %s", path)
