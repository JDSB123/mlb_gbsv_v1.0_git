"""Daily automated pipeline — train, predict, track, alert.

Designed to run on a schedule (GitHub Actions cron or Azure Container Job).

Usage:
    python scripts/daily_run.py
    python scripts/daily_run.py --loader odds_api --settle
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from mlbv1.alerts.manager import AlertManager
from mlbv1.config import AppConfig
from mlbv1.data.loader import (
    BetsAPILoader,
    MLBStatsAPILoader,
    OddsAPILoader,
    SyntheticDataLoader,
    WeatherEnricher,
)
from mlbv1.data.historical_enrichment import (
    enrich_training_data_with_historical_sources,
)
from mlbv1.data.preprocessor import ProcessedData, preprocess, train_test_split_time
from mlbv1.features.engineer import FeatureSet, engineer_features
from mlbv1.metrics import evaluate
from mlbv1.models.predictor import load_model, predict
from mlbv1.models.trainer import ModelTrainer
from mlbv1.tracking.database import PredictionRecord, RunRecord, TrackingDB
from mlbv1.tracking.roi import BankrollConfig, BankrollManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MARKET_TARGET_MAP: dict[str, str] = {
    "spread": "spread_cover",
    "moneyline": "home_win",
    "total": "over_total",
    "f5_spread": "f5_spread_cover",
    "f5_moneyline": "f5_home_win",
    "f5_total": "f5_over_total",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily MLB prediction pipeline")
    parser.add_argument("--loader", type=str, default=None, help="Override loader")
    parser.add_argument(
        "--settle", action="store_true", help="Settle yesterday's predictions"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="Skip training, use existing models"
    )
    parser.add_argument("--no-alerts", action="store_true", help="Skip sending alerts")
    parser.add_argument(
        "--db", type=str, default="artifacts/tracking.db", help="Tracking DB path"
    )
    parser.add_argument("--config", type=str, default=None, help="Config JSON path")
    args = parser.parse_args()

    config = AppConfig.load(args.config)
    if args.loader:
        config = config.override(data={"loader": args.loader})

    db = TrackingDB(args.db)
    run_id = f"daily-{datetime.now(tz=UTC).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
    alerts = AlertManager() if not args.no_alerts else None

    try:
        # Step 1: Settle yesterday's predictions
        if args.settle:
            _settle_yesterday(db, config)

        # Step 2: Get training data (last 90 days for recent form + real historical seasons 2024, 2023)
        logger.info("=== Step 2: Loading training data ===")
        train_df = _load_training_data(config)

        # Enrich with real historical seasons (2024, 2023) for better model training
        try:
            logger.info("Loading real historical data from 2024 season...")
            history_2024 = MLBStatsAPILoader.load_historical_season(2024)
            logger.info(f"Loaded {len(history_2024)} games from 2024")

            logger.info("Loading real historical data from 2023 season...")
            history_2023 = MLBStatsAPILoader.load_historical_season(2023)
            logger.info(f"Loaded {len(history_2023)} games from 2023")

            # Combine: historical + current season data (reverse chronological for recency bias)
            # Recent games get ~3x weight via duplication for better 2026 generalization
            train_df = pd.concat(
                [
                    history_2023,
                    history_2024,
                    train_df,
                    train_df,
                    train_df,
                ],  # 3x weight on current season
                ignore_index=True,
            )
            train_df = train_df.sort_values("game_date").reset_index(drop=True)
            logger.info(
                f"Training data spans 2023-2026 (recency-weighted): {len(train_df)} games total"
            )
        except Exception as e:
            logger.warning(
                f"Could not load historical seasons ({e}); using current + synthetic fallback"
            )
            from mlbv1.data.synthetic_history import enrich_current_data_with_history

            if len(train_df) >= 50:
                train_df = enrich_current_data_with_history(
                    train_df, synthetic_years=[2024, 2023]
                )

        if len(train_df) < 50:
            logger.warning(
                "Insufficient training data (%d games). Using full synthetic fallback.",
                len(train_df),
            )
            train_df = SyntheticDataLoader(num_games=300).load()

        # Enrich with weather
        weather_key = os.getenv("VISUAL_CROSSING_API_KEY", "")
        if weather_key:
            enricher = WeatherEnricher(weather_key)
            train_df = enricher.enrich(train_df)

        # Enrich with authoritative historical data sources (Lahman, Statcast, etc.)
        logger.info(
            "Enriching with authoritative historical datasets (Lahman, Statcast)..."
        )
        try:
            train_df = enrich_training_data_with_historical_sources(
                train_df,
                include_lahman=True,
                include_statcast=True,  # Enabled: fetches exit velo, barrel %, xwOBA
                include_retrosheet=False,
            )
        except Exception as e:
            logger.warning(
                f"Historical data enrichment failed: {e}. Continuing with current data."
            )

        processed = preprocess(train_df)
        features = engineer_features(
            processed.features,
            short_window=config.features.rolling_window_short,
            long_window=config.features.rolling_window_long,
        )

        # Step 3: Train models
        if not args.no_train:
            logger.info("=== Step 3: Training models ===")
            _train_and_save(features, processed, config, db, run_id)
        else:
            logger.info("=== Skipping training (--no-train) ===")

        # Step 4: Get today's games & predict
        logger.info("=== Step 4: Loading today's games for prediction ===")
        today_df = _load_todays_games(config)
        if today_df.empty:
            logger.info("No games today — skipping predictions")
            return

        today_processed = preprocess(today_df)
        today_features = engineer_features(today_processed.features)

        # Step 5: Generate predictions from all saved models
        logger.info("=== Step 5: Generating predictions ===")
        all_picks = _predict_with_all_models(
            today_features, today_processed, db, run_id
        )
        model_weights = _build_model_weights(db)
        consensus_picks = _build_consensus_picks(all_picks, model_weights)
        logger.info(
            "Built %d weighted consensus picks from %d model picks",
            len(consensus_picks),
            len(all_picks),
        )

        # Step 6: Bankroll sizing
        logger.info("=== Step 6: Bankroll sizing ===")
        bankroll = BankrollManager(BankrollConfig())
        recommendations = []
        for pick in consensus_picks:
            if pick.get("market") != "spread":
                continue
            rec = bankroll.recommend_bet(
                game_date=pick["game_date"],
                home_team=pick["home_team"],
                away_team=pick["away_team"],
                spread=pick["spread"],
                probability=pick["probability"],
                home_moneyline=pick.get("home_moneyline", -110),
                away_moneyline=pick.get("away_moneyline", -110),
            )
            if rec:
                pick["edge"] = rec.edge
                pick["recommended_bet"] = rec.recommended_bet
                pick["side"] = rec.side
                recommendations.append(pick)

        logger.info("Generated %d actionable picks", len(recommendations))
        for r in recommendations:
            logger.info(
                "  [%s] %s @ %s | %s %+.1f | Conf: %.1f%% | Edge: %.1f%% | Bet: $%.0f | Models: %d (%.0f%% agree)",
                r.get("market", "spread"),
                r["away_team"],
                r["home_team"],
                r.get("side", "?"),
                r["spread"],
                r["probability"] * 100,
                r.get("edge", 0) * 100,
                r.get("recommended_bet", 0),
                int(r.get("model_count", 1)),
                float(r.get("agreement", 1.0)) * 100,
            )

        # Step 7: Send alerts
        if alerts and alerts.has_channels and recommendations:
            logger.info("=== Step 7: Sending alerts ===")
            alerts.send_predictions(
                recommendations, model_name="ensemble", run_id=run_id
            )

            # Daily summary
            roi_summary = db.get_roi_summary()
            roi_summary["current_balance"] = bankroll.balance
            alerts.send_daily_summary(roi_summary, top_picks=recommendations[:5])

        logger.info("=== Daily pipeline complete: %s ===", run_id)

    except Exception as exc:
        logger.exception("Daily pipeline failed: %s", exc)
        if alerts:
            alerts.send_alert(f"Daily pipeline FAILED: {exc}")
        raise


def _settle_yesterday(db: TrackingDB, config: AppConfig) -> None:
    """Settle predictions from yesterday using scores API."""
    logger.info("=== Step 1: Settling yesterday's predictions ===")

    try:
        odds_key = config.data.api_key or os.getenv("ODDS_API_KEY", "")
        if odds_key:
            from mlbv1.data.loader import OddsAPILoader

            loader = OddsAPILoader(
                base_url="https://api.the-odds-api.com/v4",
                api_key=odds_key,
            )
            scores_df = loader.load_scores(days_from=2)
            for _, row in scores_df.iterrows():
                db.settle_predictions(
                    game_date=str(row["game_date"])[:10],
                    home_team=str(row["home_team"]),
                    actual_home_score=int(row["home_score"]),
                    actual_away_score=int(row["away_score"]),
                )
            logger.info("Settled predictions using OddsAPI scores")
        else:
            logger.info("No ODDS_API_KEY — skipping settlement")
    except Exception as exc:
        logger.warning("Settlement failed: %s", exc)


def _load_training_data(config: AppConfig) -> pd.DataFrame:
    """Load last 90 days of data for training."""

    end = datetime.now(tz=UTC)
    start = end - timedelta(days=90)

    try:
        loader = MLBStatsAPILoader(
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
        )
        df = loader.load()
        if len(df) >= 50:
            return df
    except Exception as exc:
        logger.warning("MLB Stats API failed: %s", exc)

    # Fallback to configured loader
    logger.info("Using fallback loader: %s", config.data.loader)
    return SyntheticDataLoader(num_games=300).load()


def _train_and_save(
    features: FeatureSet,
    processed: ProcessedData,
    config: AppConfig,
    db: TrackingDB,
    run_id: str,
) -> None:
    """Train all models across all available markets and save artifacts."""

    train_df, test_df = train_test_split_time(processed.features, 0.8)
    train_X = features.X.loc[train_df.index]
    test_X = features.X.loc[test_df.index]
    available_markets = {
        market: target_col
        for market, target_col in MARKET_TARGET_MAP.items()
        if processed.targets is not None and target_col in processed.targets.columns
    }
    if not available_markets:
        available_markets = {"spread": "spread_cover"}

    trainer = ModelTrainer(output_dir="artifacts/models")
    model_types = ["random_forest", "logistic_regression", "xgboost", "lightgbm"]

    for market, target_col in available_markets.items():
        train_y = processed.targets.loc[train_df.index, target_col]  # type: ignore[index]
        test_y = processed.targets.loc[test_df.index, target_col]  # type: ignore[index]

        for model_type in model_types:
            try:
                if model_type == "random_forest":
                    trained = trainer.train_random_forest(
                        train_X, train_y, config.model.random_forest
                    )
                elif model_type == "logistic_regression":
                    trained = trainer.train_logistic_regression(
                        train_X, train_y, config.model.logistic_regression
                    )
                elif model_type == "xgboost":
                    trained = trainer.train_xgboost(
                        train_X, train_y, config.model.xgboost
                    )
                elif model_type == "lightgbm":
                    trained = trainer.train_lightgbm(
                        train_X, train_y, config.model.lightgbm
                    )
                else:
                    continue

                trained = trained.__class__(
                    name=f"{trained.name}__{market}",
                    model=trained.model,
                    scaler=trained.scaler,
                    feature_names=trained.feature_names,
                )

                acc = trainer.evaluate(trained, test_X, test_y)
                if trained.scaler:
                    scaled_test = pd.DataFrame(
                        trained.scaler.transform(test_X),
                        columns=trained.feature_names,
                        index=test_X.index,
                    )
                    preds = trained.model.predict(scaled_test)
                else:
                    preds = trained.model.predict(test_X)
                metrics = evaluate(test_y, preds)
                trainer.save(trained)

                logger.info(
                    "%s [%s]: acc=%.3f roi=%.3f sharpe=%.3f",
                    model_type,
                    market,
                    acc,
                    metrics.roi,
                    metrics.sharpe_ratio,
                )
                db.log_run(
                    RunRecord(
                        run_id=f"{run_id}-{model_type}-{market}",
                        model_name=model_type,
                        loader="daily",
                        target_market=market,
                        accuracy=acc,
                        roi=metrics.roi,
                        sharpe=metrics.sharpe_ratio,
                        config_json=json.dumps(config.to_dict()),
                    )
                )
            except Exception as exc:
                logger.warning("Failed to train %s for %s: %s", model_type, market, exc)


def _load_todays_games(config: AppConfig) -> pd.DataFrame:
    """Load today's games from odds APIs."""

    odds_key = config.data.api_key or os.getenv("ODDS_API_KEY", "")
    if odds_key:
        try:
            loader = OddsAPILoader(
                base_url="https://api.the-odds-api.com/v4",
                api_key=odds_key,
            )
            df = loader.load()
            if not df.empty:
                return df
        except Exception as exc:
            logger.warning("OddsAPI failed: %s", exc)

    bets_key = os.getenv("BETS_API_KEY", "")
    if bets_key:
        try:
            loader_b = BetsAPILoader(
                base_url="https://api.b365api.com",
                api_key=bets_key,
            )
            df = loader_b.load()
            if not df.empty:
                return df
        except Exception as exc:
            logger.warning("BetsAPI failed: %s", exc)

    logger.info("No live odds available — using synthetic for today")
    return SyntheticDataLoader(
        num_games=15, seed=int(datetime.now(tz=UTC).timestamp())
    ).load()


def _predict_with_all_models(
    features: FeatureSet,
    processed: ProcessedData,
    db: TrackingDB,
    run_id: str,
) -> list[dict[str, Any]]:
    """Load all saved models and generate predictions."""
    from pathlib import Path

    model_dir = Path("artifacts/models")
    all_picks: list[dict[str, Any]] = []

    if not model_dir.exists():
        logger.warning("No model directory found — skipping predictions")
        return all_picks

    model_paths = sorted(model_dir.glob("*.pkl"))
    suffixed_bases = {
        path.stem.split("__", 1)[0] for path in model_paths if "__" in path.stem
    }

    for model_path in model_paths:
        try:
            # Skip legacy unsuffixed base model if market-suffixed variants exist.
            if "__" not in model_path.stem and model_path.stem in suffixed_bases:
                continue

            model = load_model(str(model_path))
            model_name_raw = str(model.name)
            if "__" in model_name_raw:
                base_model_name, market = model_name_raw.split("__", 1)
            else:
                base_model_name, market = model_name_raw, "spread"

            result = predict(model, features.X)
            model_prediction_records: list[PredictionRecord] = []

            for i, idx in enumerate(features.X.index):
                row = processed.features.loc[idx]
                line_value = _market_line_for_row(row, market)
                pick = {
                    "game_date": str(row["game_date"]),
                    "home_team": str(row["home_team"]),
                    "away_team": str(row["away_team"]),
                    "line": line_value,
                    "spread": line_value,
                    "market": market,
                    "prediction": int(result.predictions.iloc[i]),
                    "probability": float(result.probabilities.iloc[i]),
                    "model_name": base_model_name,
                    "model_key": f"{base_model_name}__{market}",
                    "home_moneyline": int(row.get("home_moneyline", -110)),
                    "away_moneyline": int(row.get("away_moneyline", -110)),
                }
                all_picks.append(pick)

                model_prediction_records.append(
                    PredictionRecord(
                        run_id=f"{run_id}-{base_model_name}-{market}",
                        model_name=base_model_name,
                        game_date=pick["game_date"],
                        home_team=pick["home_team"],
                        away_team=pick["away_team"],
                        spread=pick["spread"],
                        prediction=pick["prediction"],
                        probability=pick["probability"],
                        home_moneyline=pick["home_moneyline"],
                        away_moneyline=pick["away_moneyline"],
                        market=market,
                    )
                )

            # Log model predictions in one batch
            db.log_predictions(model_prediction_records)
        except Exception as exc:
            logger.warning("Model %s failed: %s", model_path.name, exc)

    return all_picks


def _build_model_weights(
    db: TrackingDB,
    lookback_runs: int = 200,
    decay: float = 0.97,
) -> dict[str, float]:
    """Build per-model weights from recent run accuracy history.

    Uses exponential recency decay, where newer runs carry more weight.
    """
    runs = db.get_runs(limit=lookback_runs)
    weighted_sums: dict[str, float] = {}
    weight_totals: dict[str, float] = {}

    for age, run in enumerate(runs):
        model = str(run.get("model_name", "")).strip()
        market = str(run.get("target_market", "spread")).strip() or "spread"
        model_key = f"{model}__{market}"
        accuracy_raw = run.get("accuracy")
        if not model or accuracy_raw is None:
            continue
        try:
            accuracy = float(accuracy_raw)
        except (TypeError, ValueError):
            continue
        if 0.0 <= accuracy <= 1.0:
            recency_weight = decay**age
            weighted_sums[model_key] = weighted_sums.get(model_key, 0.0) + (
                accuracy * recency_weight
            )
            weight_totals[model_key] = (
                weight_totals.get(model_key, 0.0) + recency_weight
            )

    weights: dict[str, float] = {}
    for model_name, total in weighted_sums.items():
        denom = weight_totals.get(model_name, 0.0)
        if denom <= 0:
            continue
        # Keep a floor so weaker models still contribute a little.
        weights[model_name] = max(0.05, float(total / denom))

    if weights:
        logger.info(
            "Model weights from recent accuracy (decay=%.2f): %s",
            decay,
            ", ".join(f"{k}={v:.3f}" for k, v in sorted(weights.items())),
        )
    else:
        logger.info("No historical model accuracy found; using equal weights")

    return weights


def _build_consensus_picks(
    all_picks: list[dict[str, Any]],
    model_weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Collapse per-model picks into one weighted consensus pick per game."""
    if not all_picks:
        return []

    frame = pd.DataFrame(all_picks)
    weights = model_weights or {}
    frame["model_weight"] = (
        frame.get("model_key", frame["model_name"])
        .map(weights)
        .fillna(1.0)
        .astype(float)
    )
    keys = [
        "game_date",
        "home_team",
        "away_team",
        "market",
        "line",
        "home_moneyline",
        "away_moneyline",
    ]

    consensus: list[dict[str, Any]] = []
    grouped = frame.groupby(keys, dropna=False, sort=False)
    for game_key, group in grouped:
        total_weight = float(group["model_weight"].sum())
        if total_weight <= 0:
            total_weight = float(group.shape[0])
            group = group.assign(model_weight=1.0)

        weighted_home_prob = float(
            (group["probability"] * group["model_weight"]).sum() / total_weight
        )
        weighted_home_pick = float(
            (group["prediction"] * group["model_weight"]).sum() / total_weight
        )
        consensus_prediction = int(round(weighted_home_pick))
        agreement = max(weighted_home_pick, 1.0 - weighted_home_pick)
        model_names = sorted({str(name) for name in group["model_name"].tolist()})

        game_date, home_team, away_team, market, line, home_ml, away_ml = game_key
        consensus.append(
            {
                "game_date": str(game_date),
                "home_team": str(home_team),
                "away_team": str(away_team),
                "market": str(market),
                "line": float(line),
                "spread": float(line),
                "prediction": consensus_prediction,
                "probability": weighted_home_prob,
                "home_moneyline": int(home_ml),
                "away_moneyline": int(away_ml),
                "model_name": "consensus",
                "model_count": int(group.shape[0]),
                "agreement": float(agreement),
                "model_names": model_names,
                "weight_sum": total_weight,
            }
        )

    consensus.sort(key=lambda p: abs(float(p["probability"]) - 0.5), reverse=True)
    return consensus


def _market_line_for_row(row: pd.Series, market: str) -> float:
    """Get the relevant line for a market; reuses `spread` DB field for storage."""
    if market == "spread":
        return float(row.get("spread", 0.0))
    if market == "moneyline":
        return 0.0
    if market == "total":
        return float(row.get("total_runs", 0.0))
    if market == "f5_spread":
        return float(row.get("f5_spread", 0.0))
    if market == "f5_moneyline":
        return 0.0
    if market == "f5_total":
        return float(row.get("f5_total_runs", 0.0))
    return float(row.get("spread", 0.0))


if __name__ == "__main__":
    main()
