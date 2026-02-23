"""Tests for feature engineering edge cases and new features."""

from __future__ import annotations

from mlbv1.data.loader import SyntheticDataLoader
from mlbv1.data.preprocessor import preprocess
from mlbv1.features.engineer import engineer_features


class TestFeatureEngineeringAdvanced:
    def test_no_nan_in_output(self) -> None:
        """After engineering, features used for training should have no NaN."""
        df = SyntheticDataLoader(num_games=100, seed=42).load()
        processed = preprocess(df)
        features = engineer_features(processed.features)
        nan_counts = features.X.isna().sum()
        assert nan_counts.sum() == 0, f"NaN found: {nan_counts[nan_counts > 0]}"

    def test_custom_windows(self) -> None:
        df = SyntheticDataLoader(num_games=100, seed=42).load()
        processed = preprocess(df)
        features = engineer_features(processed.features, short_window=5, long_window=15)
        assert features.X is not None
        assert len(features.X) > 0

    def test_implied_prob_range(self) -> None:
        """Implied probability features should be in [0, 1]."""
        df = SyntheticDataLoader(num_games=100, seed=42).load()
        processed = preprocess(df)
        features = engineer_features(processed.features)
        for col in features.X.columns:
            if "implied_prob" in col:
                assert (features.X[col] >= 0).all(), f"{col} has negatives"
                assert (features.X[col] <= 1).all(), f"{col} > 1"

    def test_feature_count(self) -> None:
        """Should produce a reasonable number of features."""
        df = SyntheticDataLoader(num_games=100, seed=42).load()
        processed = preprocess(df)
        features = engineer_features(processed.features)
        assert features.X.shape[1] >= 10, "Too few features"

    def test_small_dataset(self) -> None:
        """Should handle very small datasets without crashing."""
        df = SyntheticDataLoader(num_games=15, seed=42).load()
        processed = preprocess(df)
        features = engineer_features(processed.features)
        assert len(features.X) > 0
