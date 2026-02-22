"""Tests for feature engineering."""

from mlbv1.data.loader import SyntheticDataLoader
from mlbv1.data.preprocessor import preprocess
from mlbv1.features.engineer import engineer_features


def test_engineer_features_columns():
    df = SyntheticDataLoader(num_games=20).load()
    processed = preprocess(df)
    features = engineer_features(processed.features)
    assert features.X.shape[0] == 20
    assert "home_win_rate_short" in features.X.columns
