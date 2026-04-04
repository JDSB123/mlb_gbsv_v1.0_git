"""Shared model training utilities.

Eliminates duplication across train.py, backtest.py, and bootstrap_models.py
by providing a data-driven training interface.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from mlbv1.models.trainer import ModelTrainer, TrainedModel

logger = logging.getLogger(__name__)

# Single source of truth for supported model types.
ALL_MODEL_TYPES: tuple[str, ...] = (
    "random_forest",
    "ridge_regression",
    "xgboost",
    "lightgbm",
)


class TrainingJob(NamedTuple):
    """Declarative specification for one model training run."""

    model_type: str
    trainer_method: Callable[..., TrainedModel]
    config: Any


def get_training_jobs(trainer: ModelTrainer, config: Any) -> list[TrainingJob]:
    """Build the standard list of training jobs from an AppConfig."""
    return [
        TrainingJob("random_forest", trainer.train_random_forest, config.model.random_forest),
        TrainingJob("ridge_regression", trainer.train_ridge_regression, config.model.ridge_regression),
        TrainingJob("xgboost", trainer.train_xgboost, config.model.xgboost),
        TrainingJob("lightgbm", trainer.train_lightgbm, config.model.lightgbm),
    ]


def train_model_safe(
    job: TrainingJob, X: pd.DataFrame, y: Any
) -> TrainedModel | None:
    """Execute a single training job, catching ImportError for optional libs."""
    try:
        return job.trainer_method(X, y, job.config)
    except ImportError:
        logger.warning("Skipping %s: library not installed", job.model_type)
        return None


def resolve_model_types(model_arg: str) -> list[str]:
    """Parse a CLI model selection into concrete model type names.

    Accepts: "all", "both" (RF + Ridge), or a single model name.
    """
    if model_arg == "all":
        return list(ALL_MODEL_TYPES)
    if model_arg == "both":
        return ["random_forest", "ridge_regression"]
    return [model_arg]
