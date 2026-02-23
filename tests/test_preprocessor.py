"""Tests for preprocessor edge cases."""

from __future__ import annotations

import pandas as pd

from mlbv1.data.loader import SyntheticDataLoader
from mlbv1.data.preprocessor import preprocess


class TestPreprocessor:
    def test_target_binary(self) -> None:
        df = SyntheticDataLoader(num_games=50, seed=42).load()
        processed = preprocess(df)
        unique = processed.target.unique()
        assert set(unique).issubset({0, 1})

    def test_preserves_row_count(self) -> None:
        df = SyntheticDataLoader(num_games=50, seed=42).load()
        processed = preprocess(df)
        assert len(processed.features) == len(df)
        assert len(processed.target) == len(df)

    def test_features_no_target_column(self) -> None:
        df = SyntheticDataLoader(num_games=50, seed=42).load()
        processed = preprocess(df)
        # Target should NOT be in features
        assert "target" not in processed.features.columns
        assert "result" not in processed.features.columns

    def test_handles_duplicates(self) -> None:
        df = SyntheticDataLoader(num_games=30, seed=42).load()
        df_dup = pd.concat([df, df], ignore_index=True)
        processed = preprocess(df_dup)
        assert len(processed.target) == len(df_dup)
