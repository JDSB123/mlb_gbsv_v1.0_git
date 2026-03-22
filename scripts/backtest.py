"""Historical backtest pipeline.

Fetches historical MLB game data from the Stats API, trains models,
and evaluates performance across multiple seasons/periods.

Usage:
    python scripts/backtest.py --start 2023-03-30 --end 2023-09-30
    python scripts/backtest.py --start 2024-03-20 --end 2024-09-29 --model both
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from typing import Any

from mlbv1.config import AppConfig
from mlbv1.data.loader import MLBStatsAPILoader
from mlbv1.data.preprocessor import preprocess, train_test_split_time
from mlbv1.features.engineer import engineer_features
from mlbv1.metrics import evaluate
from mlbv1.models.predictor import predict
from mlbv1.models.trainer import ModelTrainer
from mlbv1.tracking.database import PredictionRecord, RunRecord, TrackingDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest MLB models on historical data"
    )
    parser.add_argument(
        "--start", type=str, default="2024-03-20", help="Start date YYYY-MM-DD"
    )
    parser.add_argument(
        "--end", type=str, default="2024-09-29", help="End date YYYY-MM-DD"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="both",
        choices=[
            "random_forest",
            "logistic_regression",
            "xgboost",
            "lightgbm",
            "both",
            "all",
        ],
        help="Model type to backtest",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.7, help="Train/test split ratio"
    )
    parser.add_argument(
        "--db", type=str, default="artifacts/tracking.db", help="Tracking DB path"
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Use walk-forward validation instead of fixed split",
    )
    parser.add_argument(
        "--walk-window",
        type=int,
        default=60,
        help="Walk-forward training window (days)",
    )
    args = parser.parse_args()

    config = AppConfig.load()
    db = TrackingDB(args.db)
    run_id = f"backtest-{uuid.uuid4().hex[:8]}"

    # Fetch historical data
    logger.info("Fetching historical data %s → %s", args.start, args.end)
    loader = MLBStatsAPILoader(start_date=args.start, end_date=args.end)
    raw_df = loader.load()
    logger.info("Loaded %d historical games", len(raw_df))

    if len(raw_df) < 50:
        logger.error("Not enough games for backtest (need >= 50, got %d)", len(raw_df))
        return

    processed = preprocess(raw_df)
    features = engineer_features(
        processed.features,
        short_window=config.features.rolling_window_short,
        long_window=config.features.rolling_window_long,
    )

    if args.walk_forward:
        _walk_forward_backtest(features, processed, config, db, run_id, args)
    else:
        _fixed_split_backtest(features, processed, config, db, run_id, args)


def _fixed_split_backtest(
    features: Any,
    processed: Any,
    config: AppConfig,
    db: TrackingDB,
    run_id: str,
    args: argparse.Namespace,
) -> None:
    """Standard train/test split backtest."""
    from mlbv1.features.engineer import FeatureSet

    assert isinstance(features, FeatureSet)
    train_df, test_df = train_test_split_time(processed.features, args.train_ratio)  # type: ignore[union-attr]
    train_target = processed.target.loc[train_df.index]  # type: ignore[union-attr]
    test_target = processed.target.loc[test_df.index]  # type: ignore[union-attr]
    train_X = features.X.loc[train_df.index]
    test_X = features.X.loc[test_df.index]

    trainer = ModelTrainer(output_dir="artifacts/backtest_models")
    models_to_train = _resolve_model_types(args.model)

    for model_type in models_to_train:
        model_run_id = f"{run_id}-{model_type}"
        logger.info(
            "Training %s on %d samples, testing on %d",
            model_type,
            len(train_X),
            len(test_X),
        )

        trained = _train_model(trainer, model_type, train_X, train_target, config)
        if trained is None:
            continue

        acc = trainer.evaluate(trained, test_X, test_target)  # type: ignore  # type: ignore
        test_full_df = processed.features.loc[test_X.index]
        pred_result = predict(trained, test_X, test_full_df)  # type: ignore  # type: ignore
        preds_raw = pred_result.expected_runs.values
        probas = pred_result.market_probabilities["home_spread_cover_prob"]
        preds = (probas > 0.5).astype(int)
        metrics = evaluate(test_target, getattr(preds_raw, "values", preds_raw))

        logger.info(
            "%s: Accuracy=%.3f  ROI=%.3f  Sharpe=%.3f",
            model_type,
            acc,
            metrics.roi,
            metrics.sharpe_ratio,
        )

        # Log to tracking DB
        db.log_run(
            RunRecord(
                run_id=model_run_id,
                model_name=model_type,
                loader="mlb_stats_api",
                target_market="spread",
                accuracy=acc,
                roi=metrics.roi,
                sharpe=metrics.sharpe_ratio,
                config_json=json.dumps(config.to_dict()),
            )
        )

        # Log individual predictions
        pred_records = []
        for i, idx in enumerate(test_df.index):
            row = processed.features.loc[idx]  # type: ignore[union-attr]
            pred_records.append(
                PredictionRecord(
                    run_id=model_run_id,
                    model_name=model_type,
                    game_date=str(row["game_date"]),
                    home_team=str(row["home_team"]),
                    away_team=str(row["away_team"]),
                    spread=float(row["spread"]),
                    prediction=int(preds[i]),
                    probability=float(probas[i]),
                    home_moneyline=int(row.get("home_moneyline", -110)),
                    away_moneyline=int(row.get("away_moneyline", -110)),
                    market="spread",
                )
            )
        db.log_predictions(pred_records)

        # Settle with actual scores
        for idx in test_df.index:
            row = processed.features.loc[idx]  # type: ignore[union-attr]
            db.settle_predictions(
                game_date=str(row["game_date"])[:10],
                home_team=str(row["home_team"]),
                actual_home_score=int(row["home_score"]),
                actual_away_score=int(row["away_score"]),
            )

        trainer.save(trained)  # type: ignore  # type: ignore

    # Print summary
    summary = db.get_roi_summary(run_id=run_id)
    logger.info("Backtest complete: %s", summary)


def _walk_forward_backtest(
    features: Any,
    processed: Any,
    config: AppConfig,
    db: TrackingDB,
    run_id: str,
    args: argparse.Namespace,
) -> None:
    """Walk-forward validation — retrain periodically on expanding window."""
    import pandas as pd

    from mlbv1.features.engineer import FeatureSet

    assert isinstance(features, FeatureSet)
    df = processed.features  # type: ignore[union-attr]
    target = processed.target  # type: ignore[union-attr]
    X = features.X

    dates = pd.to_datetime(df["game_date"]).dt.date.unique()
    window_days = args.walk_window
    step_days = 7  # retrain weekly

    trainer = ModelTrainer(output_dir="artifacts/backtest_models")
    models_to_train = _resolve_model_types(args.model)
    all_preds: dict[str, list[tuple[int, int, float]]] = {
        m: [] for m in models_to_train
    }

    total_dates = len(dates)
    start_idx = max(window_days, int(total_dates * 0.3))

    for test_start in range(start_idx, total_dates, step_days):
        train_end = test_start
        test_end = min(test_start + step_days, total_dates)

        train_dates = set(dates[:train_end])
        test_dates = set(dates[test_start:test_end])

        train_mask = pd.to_datetime(df["game_date"]).dt.date.isin(train_dates)
        test_mask = pd.to_datetime(df["game_date"]).dt.date.isin(test_dates)

        train_X = X[train_mask]
        train_y = target[train_mask]
        test_X = X[test_mask]
        target[test_mask]

        if len(train_X) < 30 or len(test_X) == 0:
            continue

        for model_type in models_to_train:
            trained = _train_model(trainer, model_type, train_X, train_y, config)
            if trained is None:
                continue
            test_full_df = df.loc[test_X.index]
            pred_result = predict(trained, test_X, test_full_df)  # type: ignore  # type: ignore
            probas = pred_result.market_probabilities["home_spread_cover_prob"]
            preds = (probas > 0.5).astype(int)

            for i, idx in enumerate(test_X.index):
                row = df.loc[idx]
                actual_margin = row["home_score"] - row["away_score"]
                spread = row["spread"]
                actual_cover = 1 if (actual_margin + float(spread)) > 0 else 0
                all_preds[model_type].append(
                    (actual_cover, int(preds[i]), float(probas[i]))
                )

    # Report
    for model_type, results in all_preds.items():
        if not results:
            continue
        actuals = [r[0] for r in results]
        preds_list = [r[1] for r in results]
        metrics = evaluate(pd.DataFrame(actuals), preds_list)  # type: ignore
        logger.info(
            "Walk-Forward %s: Accuracy=%.3f  ROI=%.3f  Sharpe=%.3f  N=%d",
            model_type,
            metrics.accuracy,
            metrics.roi,
            metrics.sharpe_ratio,
            len(results),
        )
        db.log_run(
            RunRecord(
                run_id=f"{run_id}-wf-{model_type}",
                model_name=model_type,
                loader="mlb_stats_api",
                accuracy=metrics.accuracy,
                roi=metrics.roi,
                sharpe=metrics.sharpe_ratio,
            )
        )


def _resolve_model_types(model_arg: str) -> list[str]:
    if model_arg == "all":
        return ["random_forest", "logistic_regression", "xgboost", "lightgbm"]
    if model_arg == "both":
        return ["random_forest", "logistic_regression"]
    return [model_arg]


def _train_model(
    trainer: ModelTrainer,
    model_type: str,
    X: object,
    y: object,
    config: AppConfig,
) -> object | None:
    import pandas as pd

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, (pd.Series, pd.DataFrame))

    if model_type == "random_forest":
        return trainer.train_random_forest(X, y, config.model.random_forest)
    if model_type == "logistic_regression":
        return trainer.train_logistic_regression(X, y, config.model.logistic_regression)
    if model_type == "xgboost":
        return trainer.train_xgboost(X, y, config.model.xgboost)
    if model_type == "lightgbm":
        return trainer.train_lightgbm(X, y, config.model.lightgbm)
    logger.warning("Unknown model type: %s", model_type)
    return None


if __name__ == "__main__":
    main()
