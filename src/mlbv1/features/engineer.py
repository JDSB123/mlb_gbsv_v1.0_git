# mypy: ignore-errors
"""Feature engineering pipeline for MLB data."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from mlbv1.data.mapping import get_stadium_info

logger = logging.getLogger(__name__)


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

    # Pitcher-level features (ERA, wins)
    _defaults_used: list[str] = []

    if "home_pitcher_era" in data.columns:
        data["home_pitcher_era"] = pd.to_numeric(
            data["home_pitcher_era"], errors="coerce"
        ).fillna(0.0)
    else:
        data["home_pitcher_era"] = 0.0
        _defaults_used.append("home_pitcher_era")

    if "away_pitcher_era" in data.columns:
        data["away_pitcher_era"] = pd.to_numeric(
            data["away_pitcher_era"], errors="coerce"
        ).fillna(0.0)
    else:
        data["away_pitcher_era"] = 0.0
        _defaults_used.append("away_pitcher_era")

    if "home_pitcher_wins" in data.columns:
        data["home_pitcher_wins"] = pd.to_numeric(
            data["home_pitcher_wins"], errors="coerce"
        ).fillna(0.0)
    else:
        data["home_pitcher_wins"] = 0.0
        _defaults_used.append("home_pitcher_wins")

    if "away_pitcher_wins" in data.columns:
        data["away_pitcher_wins"] = pd.to_numeric(
            data["away_pitcher_wins"], errors="coerce"
        ).fillna(0.0)
    else:
        data["away_pitcher_wins"] = 0.0
        _defaults_used.append("away_pitcher_wins")

    data["pitcher_era_diff"] = data["home_pitcher_era"] - data["away_pitcher_era"]
    data["pitcher_wins_diff"] = data["home_pitcher_wins"] - data["away_pitcher_wins"]

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

    # Moneyline-implied probability (devigged for true probability)
    home_raw = data["home_moneyline"].apply(_ml_to_implied_prob)
    away_raw = data["away_moneyline"].apply(_ml_to_implied_prob)
    # Remove bookmaker vig: normalize so home + away = 1.0
    total_imp = home_raw + away_raw
    data["home_implied_prob"] = home_raw / total_imp
    data["away_implied_prob"] = away_raw / total_imp

    data["rest_days_home"] = _rest_days(data, "home_team")
    data["rest_days_away"] = _rest_days(data, "away_team")

    # Weather normalization for indoor stadiums (vectorized)
    indoor_mask = data["home_team"].apply(lambda t: get_stadium_info(t)[2])
    if indoor_mask.any():
        data.loc[indoor_mask, "temperature_f"] = 72.0
        data.loc[indoor_mask, "wind_mph"] = 0.0
        data.loc[indoor_mask, "precipitation"] = 0.0

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

    # Statcast advanced metrics (launch speed, launch angle, estimated BA, estimated wOBA)
    if "launch_speed_mean" in data.columns:
        data["launch_speed_mean"] = pd.to_numeric(
            data["launch_speed_mean"], errors="coerce"
        ).fillna(88.8)
    else:
        data["launch_speed_mean"] = 88.8
        _defaults_used.append("launch_speed_mean=88.8")

    if "launch_speed_max" in data.columns:
        data["launch_speed_max"] = pd.to_numeric(
            data["launch_speed_max"], errors="coerce"
        ).fillna(95.0)
    else:
        data["launch_speed_max"] = 95.0
        _defaults_used.append("launch_speed_max=95.0")

    if "launch_angle_mean" in data.columns:
        data["launch_angle_mean"] = pd.to_numeric(
            data["launch_angle_mean"], errors="coerce"
        ).fillna(12.5)
    else:
        data["launch_angle_mean"] = 12.5
        _defaults_used.append("launch_angle_mean=12.5")

    if "estimated_ba_using_speedangle_mean" in data.columns:
        data["estimated_ba_using_speedangle_mean"] = pd.to_numeric(
            data["estimated_ba_using_speedangle_mean"], errors="coerce"
        ).fillna(0.265)
    else:
        data["estimated_ba_using_speedangle_mean"] = 0.265
        _defaults_used.append("est_BA=0.265")

    if "estimated_woba_using_speedangle_mean" in data.columns:
        data["estimated_woba_using_speedangle_mean"] = pd.to_numeric(
            data["estimated_woba_using_speedangle_mean"], errors="coerce"
        ).fillna(0.330)
    else:
        data["estimated_woba_using_speedangle_mean"] = 0.330
        _defaults_used.append("est_wOBA=0.330")

    if ("barrel", "sum") in data.columns:
        data["barrels_per_game"] = pd.to_numeric(
            data[("barrel", "sum")], errors="coerce"
        ).fillna(25)
    elif "barrels_per_game" not in data.columns:
        data["barrels_per_game"] = 25
        _defaults_used.append("barrels_per_game=25")

    if _defaults_used:
        logger.info("Feature defaults used (data not available): %s", ", ".join(_defaults_used))

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
        "pitcher_era_diff",
        "pitcher_wins_diff",
        "home_pitcher_era",
        "away_pitcher_era",
        "home_pitcher_wins",
        "away_pitcher_wins",
        "temp_f",
        "wind_mph",
        "precipitation",
        "month",
        "is_weekend",
        "launch_speed_mean",
        "launch_speed_max",
        "launch_angle_mean",
        "estimated_ba_using_speedangle_mean",
        "estimated_woba_using_speedangle_mean",
        "barrels_per_game",
        "spread",
        "home_moneyline",
        "away_moneyline",
    ]

    optional_line_cols = [
        "total_runs",
        "over_odds",
        "under_odds",
        "f5_spread",
        "f5_home_moneyline",
        "f5_away_moneyline",
        "f5_total_runs",
        "f5_over_odds",
        "f5_under_odds",
        "home_spread_odds",
        "away_spread_odds",
        "f5_home_spread_odds",
        "f5_away_spread_odds",
    ]
    feature_cols.extend([c for c in optional_line_cols if c in data.columns])

    # Fill missing values: odds columns get -110 (standard vig), others get 0.
    _odds_cols = {
        "home_moneyline", "away_moneyline", "over_odds", "under_odds",
        "f5_home_moneyline", "f5_away_moneyline", "f5_over_odds", "f5_under_odds",
        "home_spread_odds", "away_spread_odds",
        "f5_home_spread_odds", "f5_away_spread_odds",
    }
    for col in feature_cols:
        if col in _odds_cols:
            data[col] = data[col].fillna(-110.0)
        else:
            data[col] = data[col].fillna(0.0)
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

