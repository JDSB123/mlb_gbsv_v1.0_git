"""Prediction entry point — supports all model types including ensembles."""

from __future__ import annotations

import argparse
import logging

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
from mlbv1.data.preprocessor import preprocess
from mlbv1.features.engineer import engineer_features
from mlbv1.models.predictor import load_model, predict
from mlbv1.tracking.roi import BankrollManager

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
        return ActionNetworkLoader(
            config.data.api_base_url, config.data.api_key or "", config.data.email or ""
        )
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
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name of model to load from registry (e.g. random_forest). Overrides --model-path if production model is found.",
    )
    parser.add_argument(
        "--market",
        type=str,
        default="spread",
        choices=[
            "spread",
            "moneyline",
            "total",
            "f5_spread",
            "f5_moneyline",
            "f5_total",
        ],
        help="Market to predict (used for market-suffixed model files)",
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

    model_path = args.model_path
    if args.model_name:
        try:
            from mlbv1.models.registry import ModelRegistry

            registry = ModelRegistry()
            prod_model = registry.get_production_model(args.model_name)
            if prod_model:
                model_path = prod_model.file_path
                logger.info(
                    "Using production model from registry: %s (v%d)",
                    model_path,
                    prod_model.version_id,
                )
            else:
                logger.warning(
                    "No production model found for '%s', falling back to %s",
                    args.model_name,
                    model_path,
                )
        except Exception as exc:
            logger.warning("Could not check ModelRegistry: %s", exc)

    if model_path.endswith(".pkl") and "__" not in model_path:
        candidate = model_path[:-4] + f"__{args.market}.pkl"
        try:
            model = load_model(candidate)
            model_path = candidate
        except FileNotFoundError:
            model = load_model(model_path)
    else:
        model = load_model(model_path)

    results = predict(model, features.X)

    output = processed.metadata.copy()
    mp = results.market_probabilities
    col = "home_spread_cover_prob"
    if args.market == "moneyline":
        col = "home_ml_prob"
    elif args.market == "total":
        col = "over_total_prob"
    elif args.market == "f5_spread":
        col = "f5_home_spread_cover_prob"
    elif args.market == "f5_moneyline":
        col = "f5_home_ml_prob"
    elif args.market == "f5_total":
        col = "f5_over_total_prob"
    
    output["probability"] = mp[col]
    output["prediction"] = (output["probability"] > 0.5).astype(int)
    output["confidence"] = (output["probability"] - 0.5).abs() * 2

    # Kelly Criterion & Bet Sizing
    bm = BankrollManager()
    output["kelly_fraction"] = output.apply(
        lambda row: bm.kelly_size(
            row["probability"] if row["prediction"] == 1 else 1 - row["probability"],
            # Fallback odds for kelly sizing, assuming standard -110 if missing
            -110 if "home_moneyline" not in row or pd.isna(row.get("home_moneyline")) else row["home_moneyline"]
        ),
        axis=1
    )
    # Apply max bet cap
    output["suggested_bet_$"] = output["kelly_fraction"] * bm.balance
    output["suggested_bet_$"] = output["suggested_bet_$"].clip(upper=(bm.balance * bm.config.max_bet_pct))
    output["suggested_bet_$"] = output["suggested_bet_$"].round(2)

    output = output.sort_values("confidence", ascending=False)
    logger.info("Top %d predictions:", args.top_n)
    print(output.head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()
