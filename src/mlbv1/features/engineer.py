"""Feature engineering pipeline for MLB data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FeatureSet:
    """Container for engineered features."""

    X: pd.DataFrame
    feature_names: list[str]


def engineer_features(
    df: pd.DataFrame, short_window: int = 5, long_window: int = 20
) -> FeatureSet:
    """Generate model features from game data."""
    data = df.copy()
    data = data.sort_values("game_date").reset_index(drop=True)

    data["home_win"] = (data["home_score"] > data["away_score"]).astype(int)
    data["away_win"] = (data["away_score"] > data["home_score"]).astype(int)

    # Vectorised rolling stats via groupby().transform() — O(n) per group
    data["home_win_rate_short"] = _rolling_team_stat(
        data, "home_team", "home_win", window=short_window
    )
    data["away_win_rate_short"] = _rolling_team_stat(
        data, "away_team", "away_win", window=short_window
    )
    data["home_win_rate_long"] = _rolling_team_stat(
        data, "home_team", "home_win", window=long_window
    )
    data["away_win_rate_long"] = _rolling_team_stat(
        data, "away_team", "away_win", window=long_window
    )

    data["home_runs_avg_short"] = _rolling_team_stat(
        data, "home_team", "home_score", window=short_window
    )
    data["away_runs_avg_short"] = _rolling_team_stat(
        data, "away_team", "away_score", window=short_window
    )

    # Additional robust features
    data["home_runs_avg_long"] = _rolling_team_stat(
        data, "home_team", "home_score", window=long_window
    )
    data["away_runs_avg_long"] = _rolling_team_stat(
        data, "away_team", "away_score", window=long_window
    )

    # Win-rate differential (home advantage signal)
    data["win_rate_diff_short"] = (
        data["home_win_rate_short"] - data["away_win_rate_short"]
    )
    data["win_rate_diff_long"] = data["home_win_rate_long"] - data["away_win_rate_long"]

    # Runs differential
    data["runs_diff_short"] = data["home_runs_avg_short"] - data["away_runs_avg_short"]

    # Moneyline-implied probability
    data["home_implied_prob"] = data["home_moneyline"].apply(_ml_to_implied_prob)
    data["away_implied_prob"] = data["away_moneyline"].apply(_ml_to_implied_prob)

    data["rest_days_home"] = _rest_days(data, "home_team")
    data["rest_days_away"] = _rest_days(data, "away_team")

    data["temp_f"] = data.get(
        "temperature_f", pd.Series(70.0, index=data.index)
    ).fillna(70.0)
    data["wind_mph"] = data.get("wind_mph", pd.Series(5.0, index=data.index)).fillna(
        5.0
    )
    data["precipitation"] = data.get(
        "precipitation", pd.Series(0.0, index=data.index)
    ).fillna(0.0)

    data["month"] = data["game_date"].dt.month
    data["is_weekend"] = data["game_date"].dt.dayofweek.isin([5, 6]).astype(int)

    feature_cols = [
        "home_win_rate_short",
        "away_win_rate_short",
        "home_win_rate_long",
        "away_win_rate_long",
        "home_runs_avg_short",
        "away_runs_avg_short",
        "home_runs_avg_long",
        "away_runs_avg_long",
        "win_rate_diff_short",
        "win_rate_diff_long",
        "runs_diff_short",
        "home_implied_prob",
        "away_implied_prob",
        "rest_days_home",
        "rest_days_away",
        "temp_f",
        "wind_mph",
        "precipitation",
        "month",
        "is_weekend",
        "spread",
        "home_moneyline",
        "away_moneyline",
    ]

    data[feature_cols] = data[feature_cols].fillna(0.0)
    X = data[feature_cols].astype(float)
    return FeatureSet(X=X, feature_names=feature_cols)


# ---------------------------------------------------------------------------
# Vectorised helpers
# ---------------------------------------------------------------------------


def _rolling_team_stat(
    df: pd.DataFrame, team_col: str, value_col: str, window: int
) -> pd.Series:
    """Vectorised per-team rolling mean using groupby + rolling.

    ``shift(1)`` ensures we only use *prior* games (no look-ahead).
    """
    return (
        df.groupby(team_col)[value_col]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        .fillna(0.0)
    )


def _rest_days(df: pd.DataFrame, team_col: str) -> pd.Series:
    """Vectorised rest-day calculation using groupby + diff."""
    # Convert to numeric for diff
    numeric_dates = df["game_date"].astype("int64") // 10**9  # seconds
    rest = (
        df.assign(_ts=numeric_dates)
        .groupby(team_col)["_ts"]
        .transform(lambda s: s.diff() / 86400)  # seconds → days
    )
    return rest.fillna(3.0).clip(lower=0.0)


def _ml_to_implied_prob(ml: float) -> float:
    """American moneyline → implied probability."""
    if ml == 0:
        return 0.5
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    return 100 / (ml + 100)
