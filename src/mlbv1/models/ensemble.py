"""Ensemble models — regression voting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from mlbv1.models.trainer import RegressorLike

logger = logging.getLogger(__name__)


def _coerce_predictions(predictions: Any) -> np.ndarray:
    """Normalize model outputs to a 2D numpy array."""
    array = np.asarray(predictions)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array

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
    weights: list[float] | None = None  # per-model weights for voting

    def _stack_base_predictions(self, X: pd.DataFrame) -> list[np.ndarray]:
        features = X[self.feature_names]
        return [_coerce_predictions(model.predict(features)) for _, model in self.base_models]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict expected runs using the configured ensemble strategy."""
        base_predictions = self._stack_base_predictions(X)

        if self.strategy == "stacking" and self.meta_model is not None:
            meta_features = np.concatenate(base_predictions, axis=1)
            if self.meta_scaler is not None:
                meta_features = self.meta_scaler.transform(meta_features)
            return _coerce_predictions(self.meta_model.predict(meta_features))

        preds = np.stack(base_predictions, axis=0)
        if self.weights is not None:
            w = np.array(self.weights).reshape(-1, *([1] * (preds.ndim - 1)))
            return (preds * w).sum(axis=0)
        return preds.mean(axis=0)

class EnsembleTrainer:
    """Build ensemble models from trained base regressors."""

    @staticmethod
    def build_voting_ensemble(
        models: list[tuple[str, RegressorLike]],
        feature_names: list[str],
        target_names: list[str],
        weights: list[float] | None = None,
    ) -> EnsembleModel:
        """Averaging ensemble, optionally weighted per model.

        *weights* should sum to 1.0.  When ``None``, equal weights are used.
        """
        if weights is not None:
            if len(weights) != len(models):
                raise ValueError("len(weights) must equal len(models)")
            wsum = sum(weights)
            if abs(wsum - 1.0) > 1e-6:
                weights = [w / wsum for w in weights]
        return EnsembleModel(
            name="voting_ensemble",
            base_models=models,
            meta_model=None,
            meta_scaler=None,
            strategy="voting",
            feature_names=feature_names,
            target_names=target_names,
            weights=weights,
        )

    @staticmethod
    def build_stacking_ensemble(
        models: list[tuple[str, RegressorLike]],
        X_train: pd.DataFrame,
        y_train: Any,
        feature_names: list[str],
        target_names: list[str],
        cv: int | TimeSeriesSplit = 5,
    ) -> EnsembleModel:
        """Stacking ensemble with ridge meta-learner and time-series OOF predictions."""
        logger.info(
            "Building stacking ensemble with %d base models, %d-fold CV",
            len(models),
            cv if isinstance(cv, int) else cv.n_splits,
        )
        splitter = TimeSeriesSplit(n_splits=cv) if isinstance(cv, int) else cv
        if isinstance(y_train, pd.DataFrame) and set(target_names).issubset(y_train.columns):
            target_frame = y_train[target_names]
        elif isinstance(y_train, pd.DataFrame):
            target_frame = y_train
        else:
            target_frame = pd.DataFrame(_coerce_predictions(y_train))
        meta_features_list: list[np.ndarray] = []
        valid_mask = np.ones(len(X_train), dtype=bool)
        for name, model in models:
            oof = np.full((len(X_train), target_frame.shape[1]), np.nan, dtype=float)
            try:
                for train_idx, val_idx in splitter.split(X_train):
                    fold_model = clone(model)
                    fold_model.fit(X_train.iloc[train_idx], target_frame.iloc[train_idx])
                    oof[val_idx] = _coerce_predictions(
                        fold_model.predict(X_train.iloc[val_idx])
                    )
            except Exception as exc:
                logger.warning("time-series stacking failed for %s: %s", name, exc)
                oof = _coerce_predictions(model.predict(X_train))
            meta_features_list.append(oof)
            valid_mask &= np.isfinite(oof).all(axis=1)

        if not valid_mask.any():
            raise ValueError("Stacking ensemble requires at least one held-out validation fold")

        meta_X = np.concatenate(
            [predictions[valid_mask] for predictions in meta_features_list], axis=1
        )
        meta_y = _coerce_predictions(target_frame)[valid_mask]
        scaler = StandardScaler()
        meta_X_scaled = scaler.fit_transform(meta_X)

        base_meta = Ridge(alpha=1.0, random_state=42)
        meta_model = MultiOutputRegressor(base_meta)
        meta_model.fit(meta_X_scaled, meta_y)

        fitted_models: list[tuple[str, RegressorLike]] = []
        for name, model in models:
            fitted_model = clone(model)
            fitted_model.fit(X_train, target_frame)
            fitted_models.append((name, fitted_model))

        return EnsembleModel(
            name="stacking_ensemble",
            base_models=fitted_models,
            meta_model=meta_model,
            meta_scaler=scaler,
            strategy="stacking",
            feature_names=feature_names,
            target_names=target_names
        )
