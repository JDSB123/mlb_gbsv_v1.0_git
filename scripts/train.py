"""Training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mlbv1.config import AppConfig
from mlbv1.data.loader import (
    ActionNetworkLoader,
    BetsAPILoader,
    CSVLoader,
    JSONLoader,
    OddsAPILoader,
    SyntheticDataLoader,
)
from mlbv1.data.preprocessor import preprocess, train_test_split_time
from mlbv1.features.engineer import engineer_features
from mlbv1.metrics import evaluate
from mlbv1.models.trainer import ModelTrainer


def build_loader(config: AppConfig):
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
        return OddsAPILoader(config.data.api_base_url or "https://api.the-odds-api.com", config.data.api_key or "")
    if loader == "action_network":
        return ActionNetworkLoader(config.data.api_base_url or "https://api.actionnetwork.com", config.data.api_key or "")
    if loader == "bets_api":
        return BetsAPILoader(config.data.api_base_url or "https://api.betsapi.com", config.data.api_key or "")
    raise ValueError(f"Unsupported loader: {loader}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLB spread prediction models")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--loader", type=str, default=None, help="Override loader")
    parser.add_argument("--model", type=str, default=None, help="Model type: random_forest, logistic_regression, both")
    args = parser.parse_args()

    config = AppConfig.load(args.config)
    if args.loader:
        config = config.override(data={"loader": args.loader})
    if args.model:
        config = config.override(model={"type": args.model})

    loader = build_loader(config)
    df = loader.load()
    processed = preprocess(df)

    features = engineer_features(
        processed.features, short_window=config.features.rolling_window_short, long_window=config.features.rolling_window_long
    )
    train_df, test_df = train_test_split_time(processed.features)
    train_target = processed.target.loc[train_df.index]
    test_target = processed.target.loc[test_df.index]

    train_features = features.X.loc[train_df.index]
    test_features = features.X.loc[test_df.index]

    trainer = ModelTrainer()

    if config.model.type in {"random_forest", "both"}:
        rf_model = trainer.train_random_forest(train_features, train_target, config.model.random_forest)
        trainer.save(rf_model)
        rf_acc = trainer.evaluate(rf_model, test_features, test_target)
        print(f"RandomForest accuracy: {rf_acc:.3f}")
        rf_preds = rf_model.model.predict(test_features)
        metrics = evaluate(test_target, rf_preds)
        print(f"RandomForest ROI: {metrics.roi:.3f} Sharpe: {metrics.sharpe_ratio:.3f}")

    if config.model.type in {"logistic_regression", "both"}:
        lr_model = trainer.train_logistic_regression(train_features, train_target, config.model.logistic_regression)
        trainer.save(lr_model)
        lr_acc = trainer.evaluate(lr_model, test_features, test_target)
        print(f"LogisticRegression accuracy: {lr_acc:.3f}")
        lr_preds = lr_model.model.predict(lr_model.scaler.transform(test_features))
        metrics = evaluate(test_target, lr_preds)
        print(f"LogisticRegression ROI: {metrics.roi:.3f} Sharpe: {metrics.sharpe_ratio:.3f}")


if __name__ == "__main__":
    main()
