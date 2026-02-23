"""Training entry point — supports RF, LR, XGBoost, LightGBM, ensemble, and tuning."""

from __future__ import annotations

import argparse
import logging

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
from mlbv1.models.trainer import ModelTrainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

ALL_MODELS = ("random_forest", "logistic_regression", "xgboost", "lightgbm")


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
        return ActionNetworkLoader(config.data.api_base_url, config.data.api_key or "")
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
        help="Model type: random_forest, logistic_regression, xgboost, lightgbm, all",
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

    if args.tune:
        from mlbv1.config import TuningConfig

        tuning = TuningConfig(enabled=True)
        names = list(ALL_MODELS) if model_type == "all" else [model_type]
        for name in names:
            try:
                model = trainer.tune_hyperparameters(
                    train_features, train_target, name, tuning
                )
                trainer.save(model)
                acc = trainer.evaluate(model, test_features, test_target)
                logger.info("%s (tuned) accuracy: %.3f", name, acc)
            except Exception as exc:
                logger.error("Tuning %s failed: %s", name, exc)
    else:
        # Standard training
        if model_type in {"random_forest", "all"}:
            rf = trainer.train_random_forest(
                train_features, train_target, config.model.random_forest
            )
            trainer.save(rf)
            acc = trainer.evaluate(rf, test_features, test_target)
            preds = rf.model.predict(test_features)
            m = evaluate(test_target, preds)
            logger.info(
                "RF  acc=%.3f  ROI=%.3f  Sharpe=%.3f", acc, m.roi, m.sharpe_ratio
            )

        if model_type in {"logistic_regression", "all"}:
            lr = trainer.train_logistic_regression(
                train_features, train_target, config.model.logistic_regression
            )
            trainer.save(lr)
            acc = trainer.evaluate(lr, test_features, test_target)
            preds = lr.model.predict(lr.scaler.transform(test_features))  # type: ignore[union-attr]
            m = evaluate(test_target, preds)
            logger.info(
                "LR  acc=%.3f  ROI=%.3f  Sharpe=%.3f", acc, m.roi, m.sharpe_ratio
            )

        if model_type in {"xgboost", "all"}:
            try:
                xgb = trainer.train_xgboost(
                    train_features, train_target, config.model.xgboost
                )
                trainer.save(xgb)
                acc = trainer.evaluate(xgb, test_features, test_target)
                preds = xgb.model.predict(test_features)
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
                trainer.save(lgbm)
                acc = trainer.evaluate(lgbm, test_features, test_target)
                preds = lgbm.model.predict(test_features)
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
        from mlbv1.models.predictor import load_model

        base_models = []
        for name in ALL_MODELS:
            with contextlib.suppress(FileNotFoundError):
                base_models.append(load_model(f"artifacts/models/{name}.pkl"))
        if len(base_models) >= 2:
            et = EnsembleTrainer()
            voting = et.build_voting_ensemble(base_models, train_features, train_target)
            trainer.save(voting)
            vacc = trainer.evaluate(voting, test_features, test_target)
            logger.info("Voting ensemble accuracy: %.3f", vacc)

            stacking = et.build_stacking_ensemble(
                base_models, train_features, train_target
            )
            trainer.save(stacking)
            sacc = trainer.evaluate(stacking, test_features, test_target)
            logger.info("Stacking ensemble accuracy: %.3f", sacc)
        else:
            logger.warning(
                "Need ≥2 base models for ensemble — only %d found", len(base_models)
            )

    logger.info("Training complete — artifacts saved to artifacts/models/")


if __name__ == "__main__":
    main()
