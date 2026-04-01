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

import pandas as pd

from mlbv1.alerts.manager import AlertManager
from mlbv1.config import AppConfig
from mlbv1.data.loader import (
    ActionNetworkLoader,
    BetsAPILoader,
    CSVLoader,
    JSONLoader,
    MLBStatsAPILoader,
    OddsAPILoader,
    SyntheticDataLoader,
    WeatherEnricher,
)
from mlbv1.data.historical_enrichment import enrich_training_data_with_historical_sources
from mlbv1.data.preprocessor import ProcessedData, preprocess, train_test_split_time
from mlbv1.features.engineer import FeatureSet, engineer_features
from mlbv1.metrics import evaluate
from mlbv1.models.predictor import load_model, predict
from mlbv1.models.trainer import ModelTrainer
from mlbv1.observability import configure_telemetry, track_accuracy, track_prediction
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

        today_processed, today_features = _prepare_live_features_with_history(
            today_df, config
        )

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
        fallback_loader = _build_loader_from_config(config)
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
    model_types = ["random_forest", "ridge_regression", "xgboost", "lightgbm"]

    for model_type in model_types:
        try:
            if model_type == "random_forest":
                trained = trainer.train_random_forest(
                    train_X, train_y, config.model.random_forest
                )
            elif model_type == "ridge_regression":
                trained = trainer.train_ridge_regression(
                    train_X, train_y, config.model.ridge_regression
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
                "%s: acc=%.3f roi=%.3f sharpe=%.3f",
                model_type,
                acc,
                metrics.roi,
                metrics.sharpe_ratio,
            )
            track_accuracy(model_type, acc)
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


def _load_todays_games(config: AppConfig) -> pd.DataFrame:
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
                base_url="https://api.betsapi.com",
                api_key=bets_key,
            )
            df = loader_b.load()
            if not df.empty:
                return df
        except Exception as exc:
            logger.warning("BetsAPI failed: %s", exc)

    # Try configured loader as a final real-data source.
    try:
        configured_loader = _build_loader_from_config(config)
        df = configured_loader.load()
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


def _build_loader_from_config(config: AppConfig):  # noqa: ANN201
    """Build a loader from AppConfig for explicit fallback use."""
    loader_name = str(config.data.loader).strip()
    if loader_name == "synthetic":
        return SyntheticDataLoader(num_games=300)
    if loader_name == "odds_api":
        return OddsAPILoader(
            config.data.api_base_url or "https://api.the-odds-api.com/v4",
            config.data.api_key or os.getenv("ODDS_API_KEY", ""),
        )
    if loader_name == "bets_api":
        return BetsAPILoader(
            config.data.api_base_url or "https://api.betsapi.com",
            config.data.api_key or os.getenv("BETS_API_KEY", ""),
        )
    if loader_name == "action_network":
        return ActionNetworkLoader(
            config.data.api_base_url or "https://api.actionnetwork.com",
            config.data.api_key or os.getenv("ACTION_NETWORK_PASSWORD", ""),
            config.data.email or os.getenv("ACTION_NETWORK_EMAIL", ""),
        )
    if loader_name == "csv":
        if not config.data.input_path:
            raise ValueError("CSV loader requires data.input_path")
        return CSVLoader(config.data.input_path)
    if loader_name == "json":
        if not config.data.input_path:
            raise ValueError("JSON loader requires data.input_path")
        return JSONLoader(config.data.input_path)
    if loader_name == "mlb_stats":
        return MLBStatsAPILoader(days_back=90)
    raise ValueError(f"Unsupported loader in config: {loader_name}")


def _prepare_live_features_with_history(
    live_df: pd.DataFrame,
    config: AppConfig,
) -> tuple[ProcessedData, FeatureSet]:
    """Build live prediction features using recent historical context.

    This prevents early-season / slate-only cold starts where rolling features
    collapse to zero when only today's games are present.
    """
    history_days = int(os.getenv("LIVE_CONTEXT_DAYS", "120"))
    live = live_df.copy().reset_index(drop=True)
    live["_is_live"] = 1

    hist = pd.DataFrame()
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=history_days)
    try:
        hist_loader = MLBStatsAPILoader(
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
        )
        hist = hist_loader.load()
        logger.info("Loaded %d historical context games for live features", len(hist))
    except Exception as exc:
        logger.warning("Historical context load failed; using live-only features: %s", exc)

    if not hist.empty:
        hist = hist.copy()
        hist["_is_live"] = 0

        for col in live.columns:
            if col not in hist.columns:
                hist[col] = pd.NA
        for col in hist.columns:
            if col not in live.columns:
                live[col] = pd.NA

        combined = pd.concat([hist, live], ignore_index=True, sort=False)
    else:
        combined = live

    processed_all = preprocess(combined)
    features_all = engineer_features(
        processed_all.features,
        short_window=config.features.rolling_window_short,
        long_window=config.features.rolling_window_long,
    )

    live_mask = (
        processed_all.features["_is_live"].fillna(0).astype(int) == 1
        if "_is_live" in processed_all.features.columns
        else pd.Series([True] * len(processed_all.features), index=processed_all.features.index)
    )

    live_features_df = (
        processed_all.features.loc[live_mask]
        .drop(columns=["_is_live"], errors="ignore")
        .reset_index(drop=True)
    )
    live_target_df = processed_all.target.loc[live_mask].reset_index(drop=True)
    live_metadata_df = processed_all.metadata.loc[live_mask].reset_index(drop=True)
    live_targets_df = (
        processed_all.targets.loc[live_mask].reset_index(drop=True)
        if processed_all.targets is not None
        else None
    )
    live_X = features_all.X.loc[live_mask].reset_index(drop=True)

    processed_live = ProcessedData(
        features=live_features_df,
        target=live_target_df,
        metadata=live_metadata_df,
        targets=live_targets_df,
    )
    feature_live = FeatureSet(X=live_X, feature_names=list(live_X.columns))
    return processed_live, feature_live


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
        accuracy_raw = run.get("accuracy")
        if not model or accuracy_raw is None:
            continue
        try:
            accuracy = float(accuracy_raw)
        except (TypeError, ValueError):
            continue
        if True:  # accuracy can be negative MSE now
            recency_weight = decay**age
            weighted_sums[model] = weighted_sums.get(model, 0.0) + (
                accuracy * recency_weight
            )
            weight_totals[model] = weight_totals.get(model, 0.0) + recency_weight

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
    frame["model_weight"] = frame["model_name"].map(weights).fillna(1.0).astype(float)
    keys = [
        "game_date",
        "home_team",
        "away_team",
        "spread",
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

        game_date, home_team, away_team, spread, home_ml, away_ml = game_key
        consensus.append(
            {
                "game_date": str(game_date),
                "home_team": str(home_team),
                "away_team": str(away_team),
                "spread": float(str(spread)),
                "prediction": consensus_prediction,
                "probability": weighted_home_prob,
                "home_moneyline": int(str(home_ml)),
                "away_moneyline": int(str(away_ml)),
                "model_name": "consensus",
                "model_count": int(group.shape[0]),
                "agreement": float(agreement),
                "model_names": model_names,
                "weight_sum": total_weight,
            }
        )

    consensus.sort(key=lambda p: abs(float(p["probability"]) - 0.5), reverse=True)
    return consensus


if __name__ == "__main__":
    main()
