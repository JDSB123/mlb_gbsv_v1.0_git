"""Prediction entry point."""

from __future__ import annotations

import argparse

from mlbv1.config import AppConfig
from mlbv1.data.loader import (
    ActionNetworkLoader,
    BetsAPILoader,
    CSVLoader,
    JSONLoader,
    OddsAPILoader,
    SyntheticDataLoader,
)
from mlbv1.data.preprocessor import preprocess
from mlbv1.features.engineer import engineer_features
from mlbv1.models.predictor import load_model, predict


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
        return OddsAPILoader(
            config.data.api_base_url or "https://api.the-odds-api.com",
            config.data.api_key or "",
        )
    if loader == "action_network":
        return ActionNetworkLoader(
            config.data.api_base_url or "https://api.actionnetwork.com",
            config.data.api_key or "",
        )
    if loader == "bets_api":
        return BetsAPILoader(
            config.data.api_base_url or "https://api.betsapi.com",
            config.data.api_key or "",
        )
    raise ValueError(f"Unsupported loader: {loader}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLB spread predictions")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--loader", type=str, default=None, help="Override loader")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to model file"
    )
    args = parser.parse_args()

    config = AppConfig.load(args.config)
    if args.loader:
        config = config.override(data={"loader": args.loader})

    loader = build_loader(config)
    df = loader.load()
    processed = preprocess(df)
    features = engineer_features(processed.features)

    model = load_model(args.model_path)
    results = predict(model, features.X)

    output = processed.metadata.copy()
    output["prediction"] = results.predictions
    output["probability"] = results.probabilities
    print(output.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
