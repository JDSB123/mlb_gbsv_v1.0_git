"""Ensemble models — stacking, voting, blending."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ClassifierLike(Protocol):
    """Protocol for classifiers supporting probability predictions."""

    def predict(self, X: Any) -> Any: ...
    def predict_proba(self, X: Any) -> Any: ...
    def fit(self, X: Any, y: Any) -> Any: ...


@dataclass
class EnsembleModel:
    """An ensemble of multiple classifiers."""

    name: str
    base_models: list[tuple[str, ClassifierLike]]
    meta_model: ClassifierLike | None
    meta_scaler: StandardScaler | None
    strategy: str  # "stacking" | "voting" | "blending"
    feature_names: list[str]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using ensemble strategy."""
        if self.strategy == "voting":
            return self._voting_predict(X)
        if self.strategy == "stacking":
            return self._stacking_predict(X)
        return self._voting_predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using ensemble strategy."""
        if self.strategy == "voting":
            return self._voting_proba(X)
        if self.strategy == "stacking":
            return self._stacking_proba(X)
        return self._voting_proba(X)

    # -- voting ---------------------------------------------------------------

    def _voting_predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.column_stack([m.predict(X) for _, m in self.base_models])
        # majority vote
        result: np.ndarray = np.round(preds.mean(axis=1)).astype(int)
        return result

    def _voting_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas = np.stack(
            [m.predict_proba(X)[:, 1] for _, m in self.base_models], axis=1
        )
        avg = probas.mean(axis=1)
        result: np.ndarray = np.column_stack([1 - avg, avg])
        return result

    # -- stacking -------------------------------------------------------------

    def _stacking_predict(self, X: pd.DataFrame) -> np.ndarray:
        meta_features = self._build_meta_features(X)
        if self.meta_scaler:
            meta_features = self.meta_scaler.transform(meta_features)
        if self.meta_model is None:
            raise ValueError("Stacking requires a trained meta_model")
        result: np.ndarray = self.meta_model.predict(meta_features)
        return result

    def _stacking_proba(self, X: pd.DataFrame) -> np.ndarray:
        meta_features = self._build_meta_features(X)
        if self.meta_scaler:
            meta_features = self.meta_scaler.transform(meta_features)
        if self.meta_model is None:
            raise ValueError("Stacking requires a trained meta_model")
        result: np.ndarray = self.meta_model.predict_proba(meta_features)
        return result

    def _build_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        probas = [m.predict_proba(X)[:, 1] for _, m in self.base_models]
        return np.column_stack(probas)


class EnsembleTrainer:
    """Build ensemble models from trained base classifiers."""

    @staticmethod
    def build_voting_ensemble(
        models: list[tuple[str, ClassifierLike]],
        feature_names: list[str],
    ) -> EnsembleModel:
        """Simple soft-voting ensemble."""
        return EnsembleModel(
            name="voting_ensemble",
            base_models=models,
            meta_model=None,
            meta_scaler=None,
            strategy="voting",
            feature_names=feature_names,
        )

    @staticmethod
    def build_stacking_ensemble(
        models: list[tuple[str, ClassifierLike]],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_names: list[str],
        cv: int = 5,
    ) -> EnsembleModel:
        """Stacking ensemble with logistic regression meta-learner.

        Uses out-of-fold predictions from each base model as
        meta-features for the second-level model.
        """
        logger.info(
            "Building stacking ensemble with %d base models, %d-fold CV",
            len(models),
            cv,
        )
        meta_features_list: list[np.ndarray] = []
        for name, model in models:
            try:
                oof_proba = cross_val_predict(
                    model, X_train, y_train, cv=cv, method="predict_proba"
                )[:, 1]
            except Exception:
                logger.warning("cross_val_predict failed for %s — using predict", name)
                oof_proba = cross_val_predict(
                    model, X_train, y_train, cv=cv, method="predict"
                ).astype(float)
            meta_features_list.append(oof_proba)

        meta_X = np.column_stack(meta_features_list)
        scaler = StandardScaler()
        meta_X_scaled = scaler.fit_transform(meta_X)

        meta_model = LogisticRegression(max_iter=1000, random_state=42)
        meta_model.fit(meta_X_scaled, y_train)

        # Refit all base models on full training data
        for _, model in models:
            model.fit(X_train, y_train)

        return EnsembleModel(
            name="stacking_ensemble",
            base_models=models,
            meta_model=meta_model,
            meta_scaler=scaler,
            strategy="stacking",
            feature_names=feature_names,
        )
