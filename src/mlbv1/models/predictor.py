"""Inference utilities."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mlbv1.models.trainer import TrainedModel


@dataclass(frozen=True)
class PredictionResult:
    """Predictions with probabilities."""

    predictions: pd.Series
    probabilities: pd.Series


def load_model(path: str) -> TrainedModel:
    """Load a trained model from disk."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    with open(model_path, "rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, TrainedModel):
        raise TypeError("Invalid model file")
    return model


def predict(model: TrainedModel, X: pd.DataFrame) -> PredictionResult:
    """Generate predictions and probabilities."""
    features = X[model.feature_names]
    if model.scaler:
        features = pd.DataFrame(
            model.scaler.transform(features), columns=model.feature_names
        )
    preds = pd.Series(model.model.predict(features), index=X.index)
    proba = pd.Series(model.model.predict_proba(features)[:, 1], index=X.index)
    return PredictionResult(predictions=preds, probabilities=proba)
