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
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from mlbv1.alerts.manager import AlertManager
from mlbv1.config import AppConfig
from mlbv1.data.historical_enrichment import enrich_training_data_with_historical_sources
from mlbv1.data.loader import (
    BetsAPILoader,
    MLBStatsAPILoader,
    OddsAPILoader,
    SyntheticDataLoader,
    WeatherEnricher,
    build_loader_from_config,
)
from mlbv1.data.preprocessor import ProcessedData, preprocess, train_test_split_time
from mlbv1.data.slate_filter import (
    DEFAULT_SLATE_TIMEZONE,
    filter_pregame_games_for_date,
)
from mlbv1.features.engineer import FeatureSet, engineer_features
from mlbv1.metrics import evaluate
from mlbv1.models.predictor import load_model, predict
from mlbv1.models.trainer import ModelTrainer
from mlbv1.models.training_helpers import get_training_jobs, train_model_safe
from mlbv1.observability import configure_telemetry, track_accuracy, track_prediction
from mlbv1.pipeline.consensus import build_consensus_picks, build_model_weights
from mlbv1.pipeline.drift import check_model_drift
from mlbv1.pipeline.feature_prep import prepare_live_features_with_history
from mlbv1.tracking.database import PredictionRecord, RunRecord, TrackingDB
from mlbv1.tracking.roi import BankrollConfig, BankrollManager

# Configure telemetry at module level
configure_telemetry()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
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

    # Alembic's fileConfig() resets root logger to WARN and disables loggers
    # not listed in alembic.ini.  Re-apply our desired configuration.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    logger.disabled = False

    run_id = f"daily-{datetime.now(tz=UTC).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
    alerts = AlertManager() if not args.no_alerts else None

    db.try_start_pipeline()

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

        # Enrich with Lahman pitcher stats + Statcast advanced metrics
        logger.info("=== Step 2b: Enriching with historical data (Lahman + Statcast) ===")
        train_df = enrich_training_data_with_historical_sources(
            train_df, include_lahman=True, include_statcast=False,
            include_probable_pitchers=False,
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
            logger.info("No games today \u2014 skipping predictions")
            return

        # Enrich today's games with probable pitcher stats + Lahman
        today_str = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        today_df = enrich_training_data_with_historical_sources(
            today_df, include_lahman=True, include_statcast=False,
            include_probable_pitchers=True, target_date=today_str,
        )

        today_processed, today_features, _hist_count = prepare_live_features_with_history(
            today_df, config
        )

        # Step 5: Generate predictions from all saved models
        logger.info("=== Step 5: Generating predictions ===")
        all_picks = _predict_with_all_models(
            today_features, today_processed, db, run_id
        )
        model_weights = build_model_weights(db)
        consensus_picks = build_consensus_picks(all_picks, model_weights)
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
            rec = bankroll.recommend_bet(
                game_date=str(pick["game_date"]),
                home_team=str(pick["home_team"]),
                away_team=str(pick["away_team"]),
                spread=float(str(pick["spread"])),
                probability=float(str(pick["probability"])),
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
                "  %s @ %s | %s %+.1f | Conf: %.1f%% | Edge: %.1f%% | Bet: $%.0f | Models: %d (%.0f%% agree)",
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

        # Step 8: Generate pick sheet & post slate card to Teams
        logger.info("=== Step 8: Pick sheet & Teams slate ===")
        today_str = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        _generate_and_post_slate(today_str)

        # Step 9: Model drift detection
        logger.info("=== Step 9: Model drift detection ===")
        check_model_drift(consensus_picks, alerts)

        db.finish_pipeline_run("success")
        logger.info("=== Daily pipeline complete: %s ===", run_id)

    except Exception as exc:
        logger.exception("Daily pipeline failed: %s", exc)
        db.finish_pipeline_run("error")
        if alerts:
            alerts.send_alert(f"Daily pipeline FAILED: {exc}")
        raise


def _settle_yesterday(db: TrackingDB, config: AppConfig) -> None:
    """Settle predictions from yesterday using scores API."""
    logger.info("=== Step 1: Settling yesterday's predictions ===")

    try:
        odds_key = os.getenv("ODDS_API_KEY", "") or (config.data.api_key or "")
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


def _generate_and_post_slate(target_date: str) -> None:
    """Generate/publish canonical slate and post the Adaptive Card to Teams."""
    import importlib.util

    # Run pick_sheet.py to generate and publish canonical slate output.
    spec = importlib.util.spec_from_file_location(
        "pick_sheet",
        Path(__file__).parent / "pick_sheet.py",
    )
    if spec and spec.loader:
        pick_module = importlib.util.module_from_spec(spec)
        old_argv = sys.argv[:]
        try:
            sys.argv = ["pick_sheet.py", "--date", target_date]
            spec.loader.exec_module(pick_module)
            pick_module.main()
        finally:
            sys.argv = old_argv

    # pick_sheet reconfigures logging — restore ours
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    db = TrackingDB(os.getenv("TRACKING_DB_PATH", "artifacts/tracking.db"))
    rows = db.get_published_slate(target_date)

    if not rows:
        logger.info("No canonical slate rows found for %s — skipping slate post", target_date)
        return

    logger.info("Canonical slate has %d rows for %s", len(rows), target_date)

    # Post to Teams if configured
    group_id = os.getenv("TEAMS_GROUP_ID", "")
    channel_id = os.getenv("TEAMS_CHANNEL_ID", "")
    webhook_url = os.getenv("TEAMS_WEBHOOK_URL", "")
    if not (group_id and channel_id) and not webhook_url:
        logger.info("Teams not configured — skipping slate post")
        return

    try:
        from mlbv1.alerts.teams import TeamsAlert

        alert = TeamsAlert(
            webhook_url=webhook_url,
            group_id=group_id,
            channel_id=channel_id,
        )
        ok = alert.send_slate(rows, target_date)
        if ok:
            logger.info("Slate card posted to Teams ✓")
        else:
            logger.warning("Slate card post to Teams returned failure")
    except Exception as exc:
        logger.warning("Failed to post slate to Teams: %s", exc)


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
    try:
        fallback_loader = build_loader_from_config(config)
        fallback_df = fallback_loader.load()
        if len(fallback_df) >= 50:
            return fallback_df
        logger.warning(
            "Configured fallback loader %s returned only %d rows",
            config.data.loader,
            len(fallback_df),
        )
    except Exception as exc:
        logger.warning("Configured fallback loader failed: %s", exc)

    allow_synth = os.getenv("ALLOW_SYNTHETIC_FALLBACK", "false").lower() == "true"
    if allow_synth:
        logger.warning(
            "Falling back to synthetic training data because ALLOW_SYNTHETIC_FALLBACK=true"
        )
        return SyntheticDataLoader(num_games=300).load()

    raise RuntimeError(
        "Unable to load sufficient real training data and synthetic fallback is disabled."
    )


def _train_and_save(
    features: FeatureSet,
    processed: ProcessedData,
    config: AppConfig,
    db: TrackingDB,
    run_id: str,
) -> None:
    """Train all models and save artifacts."""

    train_df, test_df = train_test_split_time(processed.features, 0.8)
    train_X = features.X.loc[train_df.index]
    train_y = processed.target.loc[train_df.index]
    test_X = features.X.loc[test_df.index]
    test_y = processed.target.loc[test_df.index]

    trainer = ModelTrainer(output_dir="artifacts/models")
    jobs = get_training_jobs(trainer, config)

    for job in jobs:
        trained = train_model_safe(job, train_X, train_y)
        if trained is None:
            continue
        try:
            acc = trainer.evaluate(trained, test_X, test_y)
            preds = trained.predict(test_X)
            metrics = evaluate(test_y, preds)
            trainer.save(trained)

            logger.info(
                "%s: acc=%.3f roi=%.3f sharpe=%.3f",
                job.model_type, acc, metrics.roi, metrics.sharpe_ratio,
            )
            track_accuracy(job.model_type, acc)
            db.log_run(
                RunRecord(
                    run_id=f"{run_id}-{job.model_type}",
                    model_name=job.model_type,
                    loader="daily",
                    accuracy=acc,
                    roi=metrics.roi,
                    sharpe=metrics.sharpe_ratio,
                    config_json=json.dumps(config.to_safe_dict()),
                )
            )
        except Exception as exc:
            logger.warning("Failed to evaluate/save %s: %s", job.model_type, exc)


def _load_todays_games(config: AppConfig) -> pd.DataFrame:
    """Load today's games from odds APIs."""
    slate_timezone = os.getenv("SLATE_TIMEZONE", DEFAULT_SLATE_TIMEZONE)
    try:
        target_date = datetime.now(tz=ZoneInfo(slate_timezone)).strftime("%Y-%m-%d")
    except Exception:
        logger.warning("Invalid SLATE_TIMEZONE '%s'; defaulting to UTC", slate_timezone)
        target_date = datetime.now(tz=UTC).strftime("%Y-%m-%d")

    def _filter_and_log(df: pd.DataFrame, source: str) -> pd.DataFrame:
        filtered, stats = filter_pregame_games_for_date(
            df,
            target_date=target_date,
            timezone_name=slate_timezone,
        )
        logger.info(
            "Pregame filter for %s (%s): kept %d/%d rows for %s (started=%d, off_date=%d, invalid_time=%d)",
            source,
            stats["timezone"],
            stats["kept_rows"],
            stats["input_rows"],
            stats["target_date"],
            stats["started_rows"],
            stats["off_date_rows"],
            stats["invalid_time_rows"],
        )
        return filtered

    odds_key = os.getenv("ODDS_API_KEY", "") or (config.data.api_key or "")
    if odds_key:
        try:
            loader = OddsAPILoader(
                base_url="https://api.the-odds-api.com/v4",
                api_key=odds_key,
            )
            df = _filter_and_log(loader.load(), "odds_api")
            if not df.empty:
                return df
        except Exception as exc:
            logger.warning("OddsAPI failed: %s", exc)

    bets_key = os.getenv("BETS_API_KEY", "")
    if bets_key:
        try:
            loader_b = BetsAPILoader(
                base_url="https://api.betsapi.com",
                api_key=bets_key,
            )
            df = _filter_and_log(loader_b.load(), "bets_api")
            if not df.empty:
                return df
        except Exception as exc:
            logger.warning("BetsAPI failed: %s", exc)

    # Try configured loader as a final real-data source.
    try:
        configured_loader = build_loader_from_config(config)
        df = _filter_and_log(configured_loader.load(), str(config.data.loader))
        if not df.empty:
            return df
    except Exception as exc:
        logger.warning("Configured today-loader failed: %s", exc)

    allow_synth = os.getenv("ALLOW_SYNTHETIC_FALLBACK", "false").lower() == "true"
    if allow_synth:
        logger.warning(
            "No live odds available — using synthetic because ALLOW_SYNTHETIC_FALLBACK=true"
        )
        return SyntheticDataLoader(
            num_games=15, seed=int(datetime.now(tz=UTC).timestamp())
        ).load()

    logger.error(
        "No live odds available and synthetic fallback is disabled. Returning empty frame."
    )
    return pd.DataFrame()


def _predict_with_all_models(
    features: FeatureSet,
    processed: ProcessedData,
    db: TrackingDB,
    run_id: str,
) -> list[dict[str, Any]]:
    """Load all saved models and generate predictions."""
    from pathlib import Path

    from mlbv1.models.registry import ModelRegistry

    all_picks: list[dict[str, Any]] = []

    try:
        registry = ModelRegistry()
        active_models = registry.get_all_production_models()
        model_paths = [Path(m.file_path) for m in active_models]
    except Exception as exc:
        logger.warning("Failed to query ModelRegistry: %s", exc)
        model_paths = []

    # Fallback to loose files if none
    if not model_paths:
        model_dir = Path("artifacts/models")
        if not model_dir.exists():
            logger.warning("No model directory found — skipping predictions")
            return all_picks
        model_paths = list(model_dir.glob("*.pkl"))

    for model_path in model_paths:
        try:
            model = load_model(str(model_path))
            result = predict(model, features.X)
            track_prediction(model.name, "spread", len(features.X))

            for i, idx in enumerate(features.X.index):
                row = processed.features.loc[idx]
                pick = {
                    "game_date": str(row["game_date"]),
                    "home_team": str(row["home_team"]),
                    "away_team": str(row["away_team"]),
                    "spread": float(row["spread"]),
                    "prediction": (
                        1
                        if float(
                            result.market_probabilities["home_spread_cover_prob"].iloc[
                                i
                            ]
                        )
                        > 0.5
                        else 0
                    ),
                    "probability": float(
                        result.market_probabilities["home_spread_cover_prob"].iloc[i]
                    ),
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
                            game_date=str(pick["game_date"]),
                            home_team=str(pick["home_team"]),
                            away_team=str(pick["away_team"]),
                            spread=float(str(pick["spread"])),
                            prediction=int(str(pick["prediction"])),
                            probability=float(str(pick["probability"])),
                            home_moneyline=int(str(pick.get("home_moneyline", 0))),
                            away_moneyline=int(str(pick.get("away_moneyline", 0))),
                        )
                    ]
                )
        except Exception as exc:
            logger.warning("Model %s failed: %s", model_path.name, exc)

    return all_picks


if __name__ == "__main__":
    main()
