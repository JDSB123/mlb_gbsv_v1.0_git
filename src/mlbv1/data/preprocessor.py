"""Data cleaning and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd

REQUIRED_COLUMNS = [
    "game_date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "spread",
    "home_moneyline",
    "away_moneyline",
]


class PreprocessingError(ValueError):
    """Raised when preprocessing fails."""


@dataclass(frozen=True)
class ProcessedData:
    """Container for processed dataset."""

    features: pd.DataFrame
    target: pd.Series
    metadata: pd.DataFrame


def validate_schema(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise PreprocessingError(f"Missing required columns: {missing}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw game data."""
    validate_schema(df)
    df = df.copy()
    df = df.dropna(subset=["game_date", "home_team", "away_team", "spread"])
    df["home_score"] = (
        pd.to_numeric(df["home_score"], errors="coerce").fillna(0).astype(int)
    )
    df["away_score"] = (
        pd.to_numeric(df["away_score"], errors="coerce").fillna(0).astype(int)
    )
    df["spread"] = pd.to_numeric(df["spread"], errors="coerce").fillna(0.0)
    df["game_date"] = pd.to_datetime(df["game_date"], utc=True)
    df = df.sort_values("game_date")
    return df


def build_target(df: pd.DataFrame) -> pd.Series:
    """Create binary target for spread cover."""
    margin = df["home_score"] - df["away_score"]
    return (margin + df["spread"] > 0).astype(int)


def train_test_split_time(
    df: pd.DataFrame, train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.5 <= train_ratio <= 0.95:
        raise PreprocessingError("train_ratio must be between 0.5 and 0.95")
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


def preprocess(df: pd.DataFrame) -> ProcessedData:
    """Clean data and build target variable."""
    df = clean_data(df)
    target = build_target(df)
    metadata = df[["game_date", "home_team", "away_team", "spread"]].copy()
    return ProcessedData(features=df, target=target, metadata=metadata)
