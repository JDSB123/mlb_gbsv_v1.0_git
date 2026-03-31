"""Data cleaning and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass

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

# Supported columns moving forward
OPTIONAL_COLUMNS = [
    "home_spread_odds",
    "away_spread_odds",
    "total_runs",
    "over_odds",
    "under_odds",
    "f5_spread",
    "f5_home_spread_odds",
    "f5_away_spread_odds",
    "f5_home_moneyline",
    "f5_away_moneyline",
    "f5_total_runs",
    "f5_over_odds",
    "f5_under_odds",
    "f5_home_score",
    "f5_away_score",
    "home_tt",
    "away_tt",
    "home_tt_over_odds",
    "home_tt_under_odds",
    "away_tt_over_odds",
    "away_tt_under_odds",
    "event_id",
]


class PreprocessingError(ValueError):
    """Raised when preprocessing fails."""


@dataclass(frozen=True)
class ProcessedData:
    """Container for processed dataset."""

    features: pd.DataFrame
    target: pd.DataFrame
    metadata: pd.DataFrame
    targets: pd.DataFrame | None = None  # Holds all market targets


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

    # Optional odds/line columns (if present)
    optional_numeric = [
        "home_spread_odds",
        "away_spread_odds",
        "total_runs",
        "over_odds",
        "under_odds",
        "f5_spread",
        "f5_home_spread_odds",
        "f5_away_spread_odds",
        "f5_home_moneyline",
        "f5_away_moneyline",
        "f5_total_runs",
        "f5_over_odds",
        "f5_under_odds",
        "f5_home_score",
        "f5_away_score",
        "home_tt",
        "away_tt",
        "home_tt_over_odds",
        "home_tt_under_odds",
        "away_tt_over_odds",
        "away_tt_under_odds",
    ]
    for col in optional_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["game_date"] = pd.to_datetime(df["game_date"], utc=True)
    df = df.sort_values("game_date")
    return df


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create targets for all markets."""
    targets = pd.DataFrame(index=df.index)

    # Base targets for regression
    targets["home_score"] = pd.to_numeric(df["home_score"], errors="coerce").fillna(0)
    targets["away_score"] = pd.to_numeric(df["away_score"], errors="coerce").fillna(0)

    # Legacy Binary Targets
    margin = targets["home_score"] - targets["away_score"]
    targets["spread_cover"] = (margin + df["spread"] > 0).astype(int)

    # 2. Full Game Moneyline (Home Win)
    targets["home_win"] = (margin > 0).astype(int)

    # 3. Full Game Total (Over)
    if "total_runs" in df.columns:
        total = targets["home_score"] + targets["away_score"]
        targets["over_total"] = (
            total > pd.to_numeric(df["total_runs"], errors="coerce").fillna(0)
        ).astype(int)

    # 4. F5 Spread & ML & Total
    if "f5_home_score" in df.columns and "f5_away_score" in df.columns:
        targets["f5_home_score"] = pd.to_numeric(
            df["f5_home_score"], errors="coerce"
        ).fillna(0)
        targets["f5_away_score"] = pd.to_numeric(
            df["f5_away_score"], errors="coerce"
        ).fillna(0)

        f5_margin = targets["f5_home_score"] - targets["f5_away_score"]

        if "f5_spread" in df.columns:
            targets["f5_spread_cover"] = (
                f5_margin + pd.to_numeric(df["f5_spread"], errors="coerce").fillna(0)
                > 0
            ).astype(int)

        targets["f5_home_win"] = (f5_margin > 0).astype(int)

        if "f5_total_runs" in df.columns:
            f5_total = targets["f5_home_score"] + targets["f5_away_score"]
            targets["f5_over_total"] = (
                f5_total > pd.to_numeric(df["f5_total_runs"], errors="coerce").fillna(0)
            ).astype(int)
    else:
        # Fallback if F5 data is missing: ~58% of runs score in innings 1-5
        # (MLB historical average), not a flat 50%.
        targets["f5_home_score"] = (targets["home_score"] * 0.58).round()
        targets["f5_away_score"] = (targets["away_score"] * 0.58).round()

    return targets


def build_target(df: pd.DataFrame) -> pd.Series:
    """Legacy helper returning only spread target."""
    return build_targets(df)["spread_cover"]


def train_test_split_time(
    df: pd.DataFrame, train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.5 <= train_ratio <= 0.95:
        raise PreprocessingError("train_ratio must be between 0.5 and 0.95")
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


def preprocess(df: pd.DataFrame) -> ProcessedData:
    """Clean data and build target variables."""
    df = clean_data(df)
    targets = build_targets(df)
    target = targets
    metadata = df[["game_date", "home_team", "away_team", "spread"]].copy()
    return ProcessedData(features=df, target=target, metadata=metadata, targets=targets)
