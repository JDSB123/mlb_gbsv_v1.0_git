"""Model training for MLB spread predictions."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from mlbv1.config import (
    LightGBMConfig,
    LogisticRegressionConfig,
    RandomForestConfig,
    TuningConfig,
    XGBoostConfig,
)

logger = logging.getLogger(__name__)


class ClassifierLike(Protocol):
    """Protocol for classifiers supporting probability predictions."""

    def predict(self, X: Any) -> Any:  # noqa: ANN401 - sklearn compatibility
        ...

    def predict_proba(self, X: Any) -> Any:  # noqa: ANN401 - sklearn compatibility
        ...


@dataclass(frozen=True)
class TrainedModel:
    """Container for trained model artifacts."""

    name: str
    model: ClassifierLike
    scaler: StandardScaler | None
    feature_names: list[str]


class ModelTrainer:
    """Train scikit-learn / xgboost / lightgbm models for MLB predictions."""

    def __init__(self, output_dir: str = "artifacts/models") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------------------

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
        return TrainedModel(
            name="random_forest",
            model=model,
            scaler=None,
            feature_names=list(X.columns),
        )

    # ------------------------------------------------------------------
    # Logistic Regression
    # ------------------------------------------------------------------

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
            name="logistic_regression",
            model=model,
            scaler=scaler,
            feature_names=list(X.columns),
        )

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------

    def train_xgboost(
        self, X: pd.DataFrame, y: pd.Series, config: XGBoostConfig
    ) -> TrainedModel:
        from xgboost import XGBClassifier  # lazy import

        model = XGBClassifier(
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
            use_label_encoder=False,
            eval_metric=config.eval_metric,
        )
        model.fit(X, y)
        return TrainedModel(
            name="xgboost",
            model=model,
            scaler=None,
            feature_names=list(X.columns),
        )

    # ------------------------------------------------------------------
    # LightGBM
    # ------------------------------------------------------------------

    def train_lightgbm(
        self, X: pd.DataFrame, y: pd.Series, config: LightGBMConfig
    ) -> TrainedModel:
        from lightgbm import LGBMClassifier  # lazy import

        model = LGBMClassifier(
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
        )
        model.fit(X, y)
        return TrainedModel(
            name="lightgbm",
            model=model,
            scaler=None,
            feature_names=list(X.columns),
        )

    # ------------------------------------------------------------------
    # Hyperparameter Tuning (GridSearchCV)
    # ------------------------------------------------------------------

    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        tuning: TuningConfig,
    ) -> TrainedModel:
        """Run GridSearchCV for *model_name* and return the best estimator."""
        scaler: StandardScaler | None = None
        X_fit: Any = X

        if model_name == "random_forest":
            estimator = RandomForestClassifier(random_state=42)
            param_grid: dict[str, list[Any]] = {
                "n_estimators": [100, 300, 500],
                "max_depth": [4, 8, None],
                "min_samples_split": [2, 5],
            }
        elif model_name == "logistic_regression":
            estimator = LogisticRegression(max_iter=2000, random_state=42)
            param_grid = {"C": [0.01, 0.1, 1.0, 10.0]}
            scaler = StandardScaler()
            X_fit = scaler.fit_transform(X)
        elif model_name == "xgboost":
            from xgboost import XGBClassifier

            estimator = XGBClassifier(
                use_label_encoder=False, eval_metric="logloss", random_state=42
            )
            param_grid = {
                "n_estimators": [100, 300],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
            }
        elif model_name == "lightgbm":
            from lightgbm import LGBMClassifier

            estimator = LGBMClassifier(verbose=-1, random_state=42)
            param_grid = {
                "n_estimators": [100, 300],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1],
                "num_leaves": [15, 31, 63],
            }
        else:
            raise ValueError(f"Unknown model for tuning: {model_name}")

        gs = GridSearchCV(
            estimator,
            param_grid,
            cv=tuning.cv_folds,
            scoring=tuning.scoring,
            n_jobs=tuning.n_jobs,
            verbose=0,
        )
        gs.fit(X_fit, y)
        logger.info(
            "Best %s params: %s  score=%.4f",
            model_name,
            gs.best_params_,
            gs.best_score_,
        )
        return TrainedModel(
            name=model_name,
            model=gs.best_estimator_,
            scaler=scaler,
            feature_names=list(X.columns),
        )

    # ------------------------------------------------------------------
    # Evaluate / Save / Train-all
    # ------------------------------------------------------------------

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
        xgb_config: XGBoostConfig | None = None,
        lgbm_config: LightGBMConfig | None = None,
        tuning: TuningConfig | None = None,
    ) -> dict[str, TrainedModel]:
        """Train all configured model types, optionally with grid search."""
        models: dict[str, TrainedModel] = {}

        if tuning and tuning.enabled:
            for name in ("random_forest", "logistic_regression", "xgboost", "lightgbm"):
                try:
                    models[name] = self.tune_hyperparameters(X, y, name, tuning)
                except Exception as exc:
                    logger.warning("Tuning %s failed: %s", name, exc)
        else:
            models["random_forest"] = self.train_random_forest(X, y, rf_config)
            models["logistic_regression"] = self.train_logistic_regression(
                X, y, lr_config
            )
            if xgb_config:
                try:
                    models["xgboost"] = self.train_xgboost(X, y, xgb_config)
                except ImportError:
                    logger.warning("xgboost not installed — skipping")
            if lgbm_config:
                try:
                    models["lightgbm"] = self.train_lightgbm(X, y, lgbm_config)
                except ImportError:
                    logger.warning("lightgbm not installed — skipping")

        return models
