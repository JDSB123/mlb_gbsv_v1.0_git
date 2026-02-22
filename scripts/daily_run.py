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
from datetime import datetime, timedelta

from mlbv1.alerts.manager import AlertManager
from mlbv1.config import AppConfig
from mlbv1.data.loader import (
    BetsAPILoader,
    MLBStatsAPILoader,
    OddsAPILoader,
    SyntheticDataLoader,
    WeatherEnricher,
)
from mlbv1.data.preprocessor import preprocess, train_test_split_time
from mlbv1.features.engineer import engineer_features
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
    run_id = f"daily-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
    alerts = AlertManager() if not args.no_alerts else None

    try:
        # Step 1: Settle yesterday's predictions
        if args.settle:
            _settle_yesterday(db, config)

        # Step 2: Get training data (last 90 days for recent form)
        logger.info("=== Step 2: Loading training data ===")
        train_df = _load_training_data(config)
        if len(train_df) < 50:
            logger.warning(
                "Insufficient training data (%d games). Using synthetic fallback.",
                len(train_df),
            )
            train_df = SyntheticDataLoader(num_games=300).load()

        # Enrich with weather
        weather_key = os.getenv("VISUAL_CROSSING_API_KEY", "")
        if weather_key:
            enricher = WeatherEnricher(weather_key)
            train_df = enricher.enrich(train_df)

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

        # Step 6: Bankroll sizing
        logger.info("=== Step 6: Bankroll sizing ===")
        bankroll = BankrollManager(BankrollConfig())
        recommendations = []
        for pick in all_picks:
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
                "  %s @ %s | %s %+.1f | Conf: %.1f%% | Edge: %.1f%% | Bet: $%.0f",
                r["away_team"],
                r["home_team"],
                r.get("side", "?"),
                r["spread"],
                r["probability"] * 100,
                r.get("edge", 0) * 100,
                r.get("recommended_bet", 0),
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
        odds_key = os.getenv("ODDS_API_KEY", "")
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


def _load_training_data(config: AppConfig) -> object:
    """Load last 90 days of data for training."""

    end = datetime.utcnow()
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
    features: object,
    processed: object,
    config: AppConfig,
    db: TrackingDB,
    run_id: str,
) -> None:
    """Train all models and save artifacts."""
    from mlbv1.data.preprocessor import ProcessedData
    from mlbv1.features.engineer import FeatureSet

    assert isinstance(features, FeatureSet)
    assert isinstance(processed, ProcessedData)

    train_df, test_df = train_test_split_time(processed.features, 0.8)
    train_X = features.X.loc[train_df.index]
    train_y = processed.target.loc[train_df.index]
    test_X = features.X.loc[test_df.index]
    test_y = processed.target.loc[test_df.index]

    trainer = ModelTrainer(output_dir="artifacts/models")
    model_types = ["random_forest", "logistic_regression", "xgboost", "lightgbm"]

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
                trained = trainer.train_xgboost(train_X, train_y, config.model.xgboost)
            elif model_type == "lightgbm":
                trained = trainer.train_lightgbm(
                    train_X, train_y, config.model.lightgbm
                )
            else:
                continue

            acc = trainer.evaluate(trained, test_X, test_y)
            preds = trained.model.predict(
                trained.scaler.transform(test_X) if trained.scaler else test_X
            )
            metrics = evaluate(test_y, preds)
            trainer.save(trained)

            logger.info(
                "%s: acc=%.3f roi=%.3f sharpe=%.3f",
                model_type,
                acc,
                metrics.roi,
                metrics.sharpe_ratio,
            )
            db.log_run(
                RunRecord(
                    run_id=f"{run_id}-{model_type}",
                    model_name=model_type,
                    loader="daily",
                    accuracy=acc,
                    roi=metrics.roi,
                    sharpe=metrics.sharpe_ratio,
                    config_json=json.dumps(config.to_dict()),
                )
            )
        except Exception as exc:
            logger.warning("Failed to train %s: %s", model_type, exc)


def _load_todays_games(config: AppConfig) -> object:
    """Load today's games from odds APIs."""

    odds_key = os.getenv("ODDS_API_KEY", "")
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
        num_games=15, seed=int(datetime.utcnow().timestamp())
    ).load()


def _predict_with_all_models(
    features: object,
    processed: object,
    db: TrackingDB,
    run_id: str,
) -> list[dict]:
    """Load all saved models and generate predictions."""
    from pathlib import Path

    from mlbv1.data.preprocessor import ProcessedData
    from mlbv1.features.engineer import FeatureSet

    assert isinstance(features, FeatureSet)
    assert isinstance(processed, ProcessedData)

    model_dir = Path("artifacts/models")
    all_picks: list[dict] = []

    if not model_dir.exists():
        logger.warning("No model directory found — skipping predictions")
        return all_picks

    for model_path in model_dir.glob("*.pkl"):
        try:
            model = load_model(str(model_path))
            result = predict(model, features.X)

            for i, idx in enumerate(features.X.index):
                row = processed.features.loc[idx]
                pick = {
                    "game_date": str(row["game_date"]),
                    "home_team": str(row["home_team"]),
                    "away_team": str(row["away_team"]),
                    "spread": float(row["spread"]),
                    "prediction": int(result.predictions.iloc[i]),
                    "probability": float(result.probabilities.iloc[i]),
                    "model_name": model.name,
                    "home_moneyline": int(row.get("home_moneyline", -110)),
                    "away_moneyline": int(row.get("away_moneyline", -110)),
                }
                all_picks.append(pick)

                # Log to tracking DB
                db.log_predictions(
                    [
                        PredictionRecord(
                            run_id=f"{run_id}-{model.name}",
                            model_name=model.name,
                            game_date=pick["game_date"],
                            home_team=pick["home_team"],
                            away_team=pick["away_team"],
                            spread=pick["spread"],
                            prediction=pick["prediction"],
                            probability=pick["probability"],
                            home_moneyline=pick["home_moneyline"],
                            away_moneyline=pick["away_moneyline"],
                        )
                    ]
                )
        except Exception as exc:
            logger.warning("Model %s failed: %s", model_path.name, exc)

    return all_picks


if __name__ == "__main__":
    main()
