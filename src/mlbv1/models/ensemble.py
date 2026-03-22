"""Ensemble models — regression voting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class RegressorLike(Protocol):
    """Protocol for regressors."""
    def predict(self, X: Any) -> Any: ...
    def fit(self, X: Any, y: Any) -> Any: ...

@dataclass
class EnsembleModel:
    """An ensemble of multiple regressors."""
    name: str
    base_models: list[tuple[str, RegressorLike]]
    meta_model: RegressorLike | None
    meta_scaler: StandardScaler | None
    strategy: str  # "voting" 
    feature_names: list[str]
    target_names: list[str]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict expected runs using average ensemble strategy."""
        preds = np.stack([m.predict(X) for _, m in self.base_models], axis=0)
        # preds shape: (n_models, n_samples, n_targets)
        avg = preds.mean(axis=0)
        return avg

class EnsembleTrainer:
    """Build ensemble models from trained base regressors."""

    @staticmethod
    def build_voting_ensemble(
        models: list[tuple[str, RegressorLike]],
        feature_names: list[str],
        target_names: list[str]
    ) -> EnsembleModel:
        """Simple averaging ensemble."""
        return EnsembleModel(
            name="voting_ensemble",
            base_models=models,
            meta_model=None,
            meta_scaler=None,
            strategy="voting",
            feature_names=feature_names,
            target_names=target_names
        )

    @staticmethod
    def build_stacking_ensemble(
        models: list[tuple[str, RegressorLike]],
        X_train: pd.DataFrame,
        y_train: Any,
        feature_names: list[str],
        target_names: list[str],
        cv: int = 5,
    ) -> EnsembleModel:
        """Stacking ensemble with ridge meta-learner."""
        logger.info(
            "Building stacking ensemble with %d base models, %d-fold CV",
            len(models),
            cv,
        )
        meta_features_list: list[np.ndarray] = []
        for name, model in models:
            try:
                oof = cross_val_predict(
                    model, X_train, y_train, cv=cv, method="predict"
                )
                meta_features_list.append(oof)
            except Exception as e:
                logger.warning("cross_val_predict failed for %s: %s", name, e)
                # Fallback to in-sample
                meta_features_list.append(model.predict(X_train))

        meta_X = np.concatenate(meta_features_list, axis=1) # (n_samples, n_models * n_targets)
        scaler = StandardScaler()
        meta_X_scaled = scaler.fit_transform(meta_X)

        base_meta = Ridge(alpha=1.0, random_state=42)
        meta_model = MultiOutputRegressor(base_meta)
        meta_model.fit(meta_X_scaled, y_train)

        for _, model in models:
            model.fit(X_train, y_train)

        return EnsembleModel(
            name="stacking_ensemble",
            base_models=models,
            meta_model=meta_model,
            meta_scaler=scaler,
            strategy="stacking",
            feature_names=feature_names,
            target_names=target_names
        )
