"""Integration tests — end-to-end pipeline with synthetic data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlbv1.config import AppConfig
from mlbv1.data.loader import SyntheticDataLoader
from mlbv1.data.preprocessor import preprocess
from mlbv1.features.engineer import engineer_features
from mlbv1.metrics import evaluate
from mlbv1.models.predictor import predict
from mlbv1.models.trainer import ModelTrainer
from mlbv1.tracking.database import PredictionRecord, RunRecord, TrackingDB


class TestEndToEndPipeline:
    """Full train → predict → track → evaluate pipeline."""

    def test_full_pipeline_rf(self, tmp_path: Path) -> None:
        """Random forest end-to-end."""
        df = SyntheticDataLoader(num_games=200, seed=42).load()
        processed = preprocess(df)
        features = engineer_features(processed.features)

        config = AppConfig.load()
        trainer = ModelTrainer(output_dir=str(tmp_path / "models"))

        # Train
        trained = trainer.train_random_forest(
            features.X, processed.target, config.model.random_forest
        )
        assert trained.model is not None

        # Predict
        result = predict(trained, features.X)
        assert len(result.predictions) == len(processed.target)

        # Evaluate
        metrics = evaluate(processed.target, result.predictions)
        assert 0.0 <= metrics.accuracy <= 1.0

    def test_full_pipeline_lr(self, tmp_path: Path) -> None:
        """Logistic regression end-to-end."""
        df = SyntheticDataLoader(num_games=200, seed=42).load()
        processed = preprocess(df)
        features = engineer_features(processed.features)

        config = AppConfig.load()
        trainer = ModelTrainer(output_dir=str(tmp_path / "models"))

        trained = trainer.train_logistic_regression(
            features.X, processed.target, config.model.logistic_regression
        )
        result = predict(trained, features.X)
        metrics = evaluate(processed.target, result.predictions)
        assert 0.0 <= metrics.accuracy <= 1.0

    def test_pipeline_with_tracking(self, tmp_path: Path) -> None:
        """Pipeline with DB tracking."""
        db = TrackingDB(str(tmp_path / "test.db"))
        df = SyntheticDataLoader(num_games=100, seed=42).load()
        processed = preprocess(df)
        features = engineer_features(processed.features)

        config = AppConfig.load()
        trainer = ModelTrainer(output_dir=str(tmp_path / "models"))
        trained = trainer.train_random_forest(
            features.X, processed.target, config.model.random_forest
        )

        # Log run
        run_id = "test-integration-001"
        result = predict(trained, features.X)
        metrics = evaluate(processed.target, result.predictions)
        db.log_run(
            RunRecord(
                run_id=run_id,
                model_name="random_forest",
                loader="synthetic",
                accuracy=metrics.accuracy,
            )
        )

        # Log predictions
        records = []
        for idx in range(min(10, len(features.X))):
            records.append(
                PredictionRecord(
                    run_id=run_id,
                    model_name="random_forest",
                    game_date="2024-07-01",
                    home_team=f"Team{idx}",
                    away_team=f"Opponent{idx}",
                    prediction=int(result.predictions.iloc[idx]),
                    probability=float(result.probabilities.iloc[idx]),
                    spread=-1.5,
                )
            )
        count = db.log_predictions(records)
        assert count == len(records)

        # Verify in DB
        runs = db.get_runs()
        assert len(runs) >= 1
        assert runs[0]["model_name"] == "random_forest"

    def test_different_seeds_give_different_data(self) -> None:
        a = SyntheticDataLoader(num_games=20, seed=1).load()
        b = SyntheticDataLoader(num_games=20, seed=2).load()
        assert not a.equals(b)

    def test_feature_engineering_produces_numeric(self) -> None:
        df = SyntheticDataLoader(num_games=100, seed=42).load()
        processed = preprocess(df)
        features = engineer_features(processed.features)
        # All feature columns should be numeric
        for col in features.X.columns:
            assert pd.api.types.is_numeric_dtype(
                features.X[col]
            ), f"Non-numeric column: {col}"
