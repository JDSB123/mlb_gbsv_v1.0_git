"""Bootstrap baseline model artifacts for environments without local model files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mlbv1.config import (
    LightGBMConfig,
    RandomForestConfig,
    RidgeRegressionConfig,
    XGBoostConfig,
)
from mlbv1.data.loader import SyntheticDataLoader
from mlbv1.data.preprocessor import preprocess, train_test_split_time
from mlbv1.features.engineer import engineer_features
from mlbv1.models.trainer import ModelTrainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_MODELS = ("random_forest", "ridge_regression", "xgboost", "lightgbm")


def _missing_models(output_dir: Path) -> list[str]:
    return [name for name in REQUIRED_MODELS if not (output_dir / f"{name}.pkl").exists()]


def _build_training_matrices(num_games: int) -> tuple:
    df = SyntheticDataLoader(num_games=num_games, seed=42).load()
    processed = preprocess(df)
    features = engineer_features(processed.features, short_window=5, long_window=20)

    train_df, _ = train_test_split_time(processed.features)
    train_features = features.X.loc[train_df.index]
    train_target = processed.target.loc[train_df.index]
    return train_features, train_target


def _train_and_save_models(output_dir: Path, num_games: int) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = ModelTrainer(output_dir=str(output_dir))
    X_train, y_train = _build_training_matrices(num_games=num_games)

    trained: list[str] = []

    jobs = [
        ("random_forest", trainer.train_random_forest, RandomForestConfig(n_estimators=120, max_depth=8)),
        ("ridge_regression", trainer.train_ridge_regression, RidgeRegressionConfig(max_iter=1000)),
        ("xgboost", trainer.train_xgboost, XGBoostConfig(n_estimators=120)),
        ("lightgbm", trainer.train_lightgbm, LightGBMConfig(n_estimators=120)),
    ]

    for name, train_fn, cfg in jobs:
        try:
            model = train_fn(X_train, y_train, cfg)  # type: ignore[misc]
            trainer.save(model)
            trained.append(name)
            logger.info("Bootstrapped model: %s", name)
        except ImportError as exc:
            logger.warning("Skipping %s bootstrap: %s", name, exc)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed bootstrapping %s: %s", name, exc)

    return trained


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap baseline models when artifacts/models is missing."
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/models",
        help="Directory where model .pkl files should be created.",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=500,
        help="Synthetic games used for quick baseline training.",
    )
    parser.add_argument(
        "--if-missing",
        action="store_true",
        help="Only train if any required model file is missing.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    missing_before = _missing_models(output_dir)
    if args.if_missing and not missing_before:
        logger.info("All required models already exist in %s; skipping bootstrap.", output_dir)
        return

    if missing_before:
        logger.info("Bootstrapping missing models in %s: %s", output_dir, ", ".join(missing_before))
    else:
        logger.info("Refreshing baseline models in %s", output_dir)

    trained = _train_and_save_models(output_dir, num_games=args.num_games)
    missing_after = _missing_models(output_dir)

    if missing_after:
        logger.warning("Model bootstrap incomplete. Missing: %s", ", ".join(missing_after))
    if not trained:
        raise RuntimeError("Model bootstrap did not produce any model artifacts")

    logger.info("Bootstrap complete. Trained models: %s", ", ".join(trained))


if __name__ == "__main__":
    main()
