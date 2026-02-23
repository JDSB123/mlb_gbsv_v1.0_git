"""Tests for model training and prediction."""

from mlbv1.data.loader import SyntheticDataLoader
from mlbv1.data.preprocessor import preprocess
from mlbv1.features.engineer import engineer_features
from mlbv1.config import RandomForestConfig
from mlbv1.models.trainer import ModelTrainer


def test_train_random_forest() -> None:
    df = SyntheticDataLoader(num_games=40).load()
    processed = preprocess(df)
    features = engineer_features(processed.features)
    trainer = ModelTrainer(output_dir="artifacts/test_models")
    model = trainer.train_random_forest(features.X, processed.target, trainer_config())
    acc = trainer.evaluate(model, features.X, processed.target)
    assert 0.0 <= acc <= 1.0


def trainer_config() -> RandomForestConfig:
    return RandomForestConfig(n_estimators=10, max_depth=3)
