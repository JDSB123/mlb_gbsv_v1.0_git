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
    data = data.sort_values("game_date")

    data["home_win"] = (data["home_score"] > data["away_score"]).astype(int)
    data["away_win"] = (data["away_score"] > data["home_score"]).astype(int)

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


def _rolling_team_stat(
    df: pd.DataFrame, team_col: str, value_col: str, window: int
) -> pd.Series:
    values = []
    for idx, row in df.iterrows():
        team = row[team_col]
        history = df.loc[:idx, :]
        team_history = history[history[team_col] == team][value_col]
        values.append(team_history.tail(window).mean())
    return pd.Series(values, index=df.index).fillna(0.0)


def _rest_days(df: pd.DataFrame, team_col: str) -> pd.Series:
    last_game_date: dict[str, pd.Timestamp] = {}
    rest_days = []
    for _, row in df.iterrows():
        team = row[team_col]
        game_date = row["game_date"]
        if team not in last_game_date:
            rest_days.append(3.0)
        else:
            delta = (game_date - last_game_date[team]).days
            rest_days.append(float(max(delta, 0)))
        last_game_date[team] = game_date
    return pd.Series(rest_days, index=df.index)
