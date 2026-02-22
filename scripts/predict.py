"""Prediction entry point — supports all model types including ensembles."""

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
from mlbv1.data.preprocessor import preprocess
from mlbv1.features.engineer import engineer_features
from mlbv1.models.predictor import load_model, predict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


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
    parser = argparse.ArgumentParser(description="Run MLB spread predictions")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--loader", type=str, default=None, help="Override loader")
    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/models/random_forest.pkl",
        help="Path to model file",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Show top N predictions")
    args = parser.parse_args()

    config = AppConfig.load(args.config)
    if args.loader:
        config = config.override(data={"loader": args.loader})

    loader = build_loader(config)
    logger.info("Loading data with %s …", type(loader).__name__)
    df = loader.load()
    logger.info("Loaded %d rows", len(df))

    processed = preprocess(df)
    features = engineer_features(processed.features)

    model = load_model(args.model_path)
    results = predict(model, features.X)

    output = processed.metadata.copy()
    output["prediction"] = results.predictions
    output["probability"] = results.probabilities
    output["confidence"] = (results.probabilities - 0.5).abs() * 2

    output = output.sort_values("confidence", ascending=False)
    logger.info("Top %d predictions:", args.top_n)
    print(output.head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()
