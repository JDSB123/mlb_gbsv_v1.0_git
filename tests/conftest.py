"""Shared test fixtures for MLBV1 test suite."""

from __future__ import annotations

import pandas as pd
import pytest

from mlbv1.data.loader import SyntheticDataLoader
from mlbv1.data.preprocessor import ProcessedData, preprocess
from mlbv1.features.engineer import FeatureSet, engineer_features


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Small synthetic DataFrame for unit tests."""
    return SyntheticDataLoader(num_games=50, seed=42).load()


@pytest.fixture
def large_synthetic_df() -> pd.DataFrame:
    """Larger synthetic DataFrame for model tests."""
    return SyntheticDataLoader(num_games=200, seed=42).load()


@pytest.fixture
def processed_data(synthetic_df: pd.DataFrame) -> ProcessedData:
    """Pre-processed data ready for feature engineering."""
    return preprocess(synthetic_df)


@pytest.fixture
def feature_set(processed_data: ProcessedData) -> FeatureSet:
    """Engineered features ready for training."""
    return engineer_features(processed_data.features)


@pytest.fixture
def large_feature_set(
    large_synthetic_df: pd.DataFrame,
) -> tuple[FeatureSet, ProcessedData]:
    """Larger feature set w/ processed data for model training."""
    processed = preprocess(large_synthetic_df)
    features = engineer_features(processed.features)
    return features, processed
