"""Helpers for filtering slate rows to only not-started games."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SLATE_TIMEZONE = "America/Chicago"


def _resolve_slate_timezone(timezone_name: str) -> ZoneInfo:
    """Resolve configured timezone with safe fallback."""
    try:
        return ZoneInfo(timezone_name)
    except Exception:
        logger.warning("Invalid SLATE_TIMEZONE '%s'; falling back to UTC", timezone_name)
        return ZoneInfo("UTC")


def _parse_target_date(target_date: str) -> date:
    """Parse YYYY-MM-DD safely."""
    ts = pd.Timestamp(target_date)
    if pd.isna(ts):
        raise ValueError(f"Invalid target_date: {target_date!r}")
    return ts.date()


def filter_pregame_games_for_date(
    games_df: pd.DataFrame,
    *,
    target_date: str,
    timezone_name: str = DEFAULT_SLATE_TIMEZONE,
    now_utc: datetime | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep only games that are on target_date (in local TZ) and not started yet."""
    input_rows = len(games_df)
    if games_df.empty:
        return games_df.copy(), {
            "timezone": timezone_name,
            "target_date": target_date,
            "input_rows": 0,
            "invalid_time_rows": 0,
            "off_date_rows": 0,
            "started_rows": 0,
            "kept_rows": 0,
        }

    if "game_date" not in games_df.columns:
        logger.warning("Missing required 'game_date' column; returning empty slate")
        return games_df.iloc[0:0].copy(), {
            "timezone": timezone_name,
            "target_date": target_date,
            "input_rows": input_rows,
            "invalid_time_rows": input_rows,
            "off_date_rows": 0,
            "started_rows": 0,
            "kept_rows": 0,
        }

    tz = _resolve_slate_timezone(timezone_name)
    target_day = _parse_target_date(target_date)
    now_ts = pd.Timestamp(now_utc or datetime.now(tz=UTC))

    parsed_dates = pd.to_datetime(games_df["game_date"], utc=True, errors="coerce")
    valid_mask = parsed_dates.notna()
    invalid_time_rows = int((~valid_mask).sum())

    working = games_df.loc[valid_mask].copy()
    working_dates = parsed_dates.loc[valid_mask]
    working["game_date"] = working_dates

    local_dates = working_dates.dt.tz_convert(tz).dt.date
    on_target_date = local_dates.eq(target_day)
    not_started = working_dates > now_ts

    kept_mask = on_target_date & not_started
    kept = working.loc[kept_mask].copy()
    kept = kept.sort_values("game_date").reset_index(drop=True)

    off_date_rows = int((~on_target_date).sum())
    started_rows = int((on_target_date & ~not_started).sum())

    stats: dict[str, Any] = {
        "timezone": str(tz),
        "target_date": target_date,
        "input_rows": input_rows,
        "invalid_time_rows": invalid_time_rows,
        "off_date_rows": off_date_rows,
        "started_rows": started_rows,
        "kept_rows": len(kept),
    }
    return kept, stats
