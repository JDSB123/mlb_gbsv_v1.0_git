"""Training entry point — supports RF, LR, XGBoost, LightGBM, ensemble, and tuning."""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from mlbv1.config import AppConfig
from mlbv1.data.loader import (
    ActionNetworkLoader,
    BetsAPILoader,
    CSVLoader,
    JSONLoader,
    MLBStatsAPILoader,
    OddsAPILoader,
    SyntheticDataLoader,
)
from mlbv1.data.preprocessor import preprocess, train_test_split_time
from mlbv1.features.engineer import engineer_features
from mlbv1.metrics import evaluate
from mlbv1.models.predictor import load_model
from mlbv1.models.trainer import ModelTrainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

ALL_MODELS = ("random_forest", "ridge_regression", "xgboost", "lightgbm")


def _save_ensemble(model: Any, output_dir: Path) -> Path:
    """Save an EnsembleModel to disk."""
    path = output_dir / f"{model.name}.pkl"
    with open(path, "wb") as handle:
        pickle.dump(model, handle)
    return path


def _evaluate_ensemble(model: Any, X: pd.DataFrame, y: pd.DataFrame) -> float:
    """Evaluate an EnsembleModel's accuracy."""
    preds = model.predict(X)
    return evaluate(y, preds).accuracy


def build_loader(config: AppConfig):  # noqa: ANN201
    """Construct the appropriate loader based on config."""
    loader = config.data.loader
    if loader == "synthetic":
        return SyntheticDataLoader()
    if loader == "csv":
        if not config.data.input_path:
            raise ValueError("CSV loader requires input_path")
        return CSVLoader(config.data.input_path)
    if loader == "json":
        if not config.data.input_path:
            raise ValueError("JSON loader requires input_path")
        return JSONLoader(config.data.input_path)
    if loader == "odds_api":
        return OddsAPILoader(config.data.api_base_url, config.data.api_key or "")
    if loader == "action_network":
        return ActionNetworkLoader(
            config.data.api_base_url, config.data.api_key or "", config.data.email or ""
        )
    if loader == "bets_api":
        return BetsAPILoader(config.data.api_base_url, config.data.api_key or "")
    if loader == "mlb_stats":
        return MLBStatsAPILoader()
    raise ValueError(f"Unsupported loader: {loader}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLB spread prediction models")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--loader", type=str, default=None, help="Override loader")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model type: random_forest, ridge_regression, xgboost, lightgbm, all",
    )
    parser.add_argument(
        "--tune", action="store_true", help="Enable GridSearchCV tuning"
    )
    parser.add_argument(
        "--ensemble", action="store_true", help="Build ensemble after training"
    )
    args = parser.parse_args()

    config = AppConfig.load(args.config)
    if args.loader:
        config = config.override(data={"loader": args.loader})
    if args.model:
        config = config.override(model={"type": args.model})

    loader = build_loader(config)
    logger.info("Loading data with %s …", type(loader).__name__)
    df = loader.load()
    logger.info("Loaded %d rows", len(df))

    processed = preprocess(df)
    features = engineer_features(
        processed.features,
        short_window=config.features.rolling_window_short,
        long_window=config.features.rolling_window_long,
    )
    train_df, test_df = train_test_split_time(processed.features)
    train_target = processed.target.loc[train_df.index]
    test_target = processed.target.loc[test_df.index]

    train_features = features.X.loc[train_df.index]
    test_features = features.X.loc[test_df.index]

    trainer = ModelTrainer()
    model_type = config.model.type
    trained_models: dict[str, Any] = {}

    # Standard training
    if model_type in {"random_forest", "all"}:
        rf = trainer.train_random_forest(
            train_features, train_target, config.model.random_forest
        )
        trained_models[rf.name] = rf
        trainer.save(rf)
        acc = trainer.evaluate(rf, test_features, test_target)
        preds = rf.predict(test_features)
        m = evaluate(test_target, preds)
        logger.info("RF  acc=%.3f  ROI=%.3f  Sharpe=%.3f", acc, m.roi, m.sharpe_ratio)

    if model_type in {"ridge_regression", "all"}:
        lr = trainer.train_ridge_regression(
            train_features, train_target, config.model.ridge_regression
        )
        trained_models[lr.name] = lr
        trainer.save(lr)
        acc = trainer.evaluate(lr, test_features, test_target)
        preds = lr.predict(test_features)
        m = evaluate(test_target, preds)
        logger.info("Ridge acc=%.3f  ROI=%.3f  Sharpe=%.3f", acc, m.roi, m.sharpe_ratio)

    if model_type in {"xgboost", "all"}:
        try:
            xgb = trainer.train_xgboost(
                train_features, train_target, config.model.xgboost
            )
            trained_models[xgb.name] = xgb
            trainer.save(xgb)
            acc = trainer.evaluate(xgb, test_features, test_target)
            preds = xgb.predict(test_features)
            m = evaluate(test_target, preds)
            logger.info(
                "XGB acc=%.3f  ROI=%.3f  Sharpe=%.3f", acc, m.roi, m.sharpe_ratio
            )
        except ImportError:
            logger.warning("xgboost not installed — skipping")

    if model_type in {"lightgbm", "all"}:
        try:
            lgbm = trainer.train_lightgbm(
                train_features, train_target, config.model.lightgbm
            )
            trained_models[lgbm.name] = lgbm
            trainer.save(lgbm)
            acc = trainer.evaluate(lgbm, test_features, test_target)
            preds = lgbm.predict(test_features)
            m = evaluate(test_target, preds)
            logger.info(
                "LGBM acc=%.3f  ROI=%.3f  Sharpe=%.3f", acc, m.roi, m.sharpe_ratio
            )
        except ImportError:
            logger.warning("lightgbm not installed — skipping")

    # Optionally build ensemble
    if args.ensemble:
        import contextlib

        from mlbv1.models.ensemble import EnsembleTrainer

        loaded_models: list[Any] = list(trained_models.values())
        for name in ALL_MODELS:
            if name in trained_models:
                continue
            with contextlib.suppress(FileNotFoundError):
                loaded_model = load_model(f"artifacts/models/{name}.pkl")
                if hasattr(loaded_model, "model"):
                    loaded_models.append(loaded_model)
        if len(loaded_models) >= 2:
            et = EnsembleTrainer()
            base_pairs: list[tuple[str, Any]] = [
                (m.name, m.model) for m in loaded_models
            ]
            feature_names = list(train_features.columns)

            voting = et.build_voting_ensemble(
                base_pairs, feature_names, trainer.target_names
            )
            _save_ensemble(voting, trainer.output_dir)
            vacc = _evaluate_ensemble(voting, test_features, test_target)
            logger.info("Voting ensemble accuracy: %.3f", vacc)

            stacking = et.build_stacking_ensemble(
                base_pairs,
                train_features,
                train_target,
                feature_names,
                trainer.target_names,
            )
            _save_ensemble(stacking, trainer.output_dir)
            sacc = _evaluate_ensemble(stacking, test_features, test_target)
            logger.info("Stacking ensemble accuracy: %.3f", sacc)
        else:
            logger.warning(
                "Need ≥2 base models for ensemble — only %d found", len(loaded_models)
            )

    logger.info("Training complete — artifacts saved to artifacts/models/")

    # Register Models
    try:
        from mlbv1.models.registry import ModelRegistry

        registry = ModelRegistry()
        for name in ALL_MODELS:
            model_path = trainer.output_dir / f"{name}.pkl"
            if model_path.exists():
                model = load_model(str(model_path))
                preds = model.predict(test_features)

                # Check accuracy
                from sklearn.metrics import mean_squared_error

                acc = -float(
                    mean_squared_error(
                        test_target[
                            [
                                "f5_home_score",
                                "f5_away_score",
                                "home_score",
                                "away_score",
                            ]
                        ].fillna(0),
                        preds,
                    )
                )

                # Register
                vid = registry.register_model(
                    model_name=name,
                    model_type=name,
                    file_path=str(model_path),
                    feature_names=model.feature_names,
                    accuracy=acc,
                )

                # Promote logic: Auto-promote if no active model exists, or if accuracy is better
                current = registry.get_production_model(name)
                if current is None or acc > current.accuracy:
                    registry.promote_to_production(vid)
                    logger.info(
                        "Promoted %s v%d to production (acc: %.3f)", name, vid, acc
                    )

    except Exception as exc:
        logger.warning("Could not register models: %s", exc)


if __name__ == "__main__":
    main()
