"""Tests for slate pregame filtering helpers."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from mlbv1.data.slate_filter import filter_pregame_games_for_date


def test_filter_pregame_games_for_date_excludes_started_and_wrong_date() -> None:
    df = pd.DataFrame(
        [
            {"game_date": "2026-03-31T23:00:00Z", "home_team": "ATL"},  # keep
            {"game_date": "2026-03-31T18:00:00Z", "home_team": "CHC"},  # started
            {"game_date": "2026-04-01T01:00:00Z", "home_team": "LAD"},  # 3/31 in CT, keep
            {"game_date": "bad-date", "home_team": "NYY"},  # invalid
            {"game_date": "2026-04-01T18:00:00Z", "home_team": "SEA"},  # off-date
        ]
    )

    filtered, stats = filter_pregame_games_for_date(
        df,
        target_date="2026-03-31",
        timezone_name="America/Chicago",
        now_utc=datetime(2026, 3, 31, 20, 0, tzinfo=UTC),
    )

    assert len(filtered) == 2
    assert set(filtered["home_team"]) == {"ATL", "LAD"}
    assert stats["started_rows"] == 1
    assert stats["off_date_rows"] == 1
    assert stats["invalid_time_rows"] == 1
    assert stats["kept_rows"] == 2


def test_filter_pregame_games_for_date_handles_missing_game_date_column() -> None:
    df = pd.DataFrame([{"home_team": "ATL"}, {"home_team": "CHC"}])
    filtered, stats = filter_pregame_games_for_date(
        df,
        target_date="2026-03-31",
        timezone_name="America/Chicago",
    )

    assert filtered.empty
    assert stats["invalid_time_rows"] == 2
    assert stats["kept_rows"] == 0
