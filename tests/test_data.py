"""Tests for data loaders and preprocessing."""

from mlbv1.data.loader import SyntheticDataLoader
from mlbv1.data.preprocessor import preprocess


def test_synthetic_loader_shape():
    loader = SyntheticDataLoader(num_games=50)
    df = loader.load()
    assert len(df) == 50
    assert "home_team" in df.columns


def test_preprocess_target():
    loader = SyntheticDataLoader(num_games=10)
    df = loader.load()
    processed = preprocess(df)
    assert processed.target.isin([0, 1]).all()
