#!/usr/bin/env python
"""Debug script to inspect data quality."""
from mlbv1.data.loader import MLBStatsAPILoader
from mlbv1.data.preprocessor import preprocess
from mlbv1.features.engineer import engineer_features

loader = MLBStatsAPILoader()
raw = loader.load()
print(f"Raw data shape: {raw.shape}")
print(f"Columns: {raw.columns.tolist()}")
print(f'Date range: {raw["game_date"].min()} to {raw["game_date"].max()}')
print(f"\nFirst 5 rows:")
print(raw.head())

processed = preprocess(raw)
print(f"\nProcessed target distribution (spread_cover):")
print(processed.target.value_counts())
print(f"Targets available: {processed.targets.columns.tolist()}")

# Check feature quality
from mlbv1.data.preprocessor import train_test_split_time

train_df, test_df = train_test_split_time(raw, train_ratio=0.8)
features = engineer_features(train_df)
print(f"\nFeatures shape: {features.X.shape}")
print(f"Feature names: {features.feature_names}")
print(f"Feature stats:\n{features.X.describe()}")
print(f"\nMissing values: {features.X.isna().sum().sum()}")
