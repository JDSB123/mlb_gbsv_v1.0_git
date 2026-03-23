# mypy: ignore-errors
"""Hyperparameter tuning script."""
import argparse
import logging

from mlbv1.config import AppConfig
from mlbv1.data.loader import MLBStatsAPILoader
from mlbv1.data.preprocessor import preprocess
from mlbv1.features.engineer import engineer_features
from mlbv1.models.trainer import ModelTrainer
from mlbv1.models.tuning import TimeSeriesTuner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser(description="Tune ML models with Optuna")
    parser.add_argument("--trials", type=int, default=20, help="Number of optuna trials per model")
    parser.add_argument("--splits", type=int, default=3, help="Time series CV splits")
    args = parser.parse_args()

    _ = AppConfig.load()
    logger.info("Loading historical training data via MLBStatsAPI (Last 2 years)...")
    loader = MLBStatsAPILoader(days_back=730)
    raw_df = loader.load()

    logger.info("Preprocessing...")
    processed = preprocess(raw_df)
    
    logger.info("Engineering features...")
    feature_set = engineer_features(processed.features)
    assert processed.targets is not None, "Targets are required for tuning"
    y = processed.targets.loc[feature_set.X.index].copy()

    trainer = ModelTrainer(output_dir="artifacts/models")
    tuner = TimeSeriesTuner(trainer=trainer, n_splits=args.splits)

    logger.info("Starting XGBoost tuning...")
    xgb_best = tuner.tune_xgboost(feature_set.X, y, n_trials=args.trials)
    logger.info("Final Best XGBoost Params: %s", xgb_best)

    logger.info("Starting LightGBM tuning...")
    lgb_best = tuner.tune_lightgbm(feature_set.X, y, n_trials=args.trials)
    logger.info("Final Best LightGBM Params: %s", lgb_best)
    
    logger.info("Starting Random Forest tuning...")
    rf_best = tuner.tune_random_forest(feature_set.X, y, n_trials=args.trials)
    logger.info("Final Best Random Forest Params: %s", rf_best)
    
    logger.info("Tuning complete. Update config.py with these best parameters.")

if __name__ == "__main__":
    main()
