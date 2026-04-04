"""MLB data loaders for games, odds, and outcomes.

Supports real APIs (OddsAPI v4, BetsAPI, Action Network, MLB Stats API),
weather enrichment (Visual Crossing), plus CSV / JSON / synthetic sources.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import random as _random_mod
import time
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd

from mlbv1.data.mapping import STADIUM_DATA, normalize_team

logger = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class GameRecord:
    """Single MLB game record."""

    game_date: datetime
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    spread: float
    home_moneyline: int
    away_moneyline: int
    temperature_f: float | None = None
    wind_mph: float | None = None
    precipitation: float | None = None


class DataLoaderError(RuntimeError):
    """Raised when data loading fails."""


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class BaseLoader:
    """Base class for MLB game loaders."""

    MAX_RETRIES: int = 3
    BACKOFF_BASE: float = 1.0

    def load(self) -> pd.DataFrame:
        """Load game data into a normalized DataFrame."""
        raise NotImplementedError

    @staticmethod
    def _validate(df: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise DataLoaderError(f"Missing required columns: {missing}")
        # Ensure team columns use canonical abbreviations (e.g. CWS→CHW)
        for col in ("home_team", "away_team"):
            if col in df.columns:
                df[col] = df[col].map(normalize_team)
        return df

    # Shared HTTP helper with retry + exponential back-off ----------------

    @classmethod
    def _http_get_json(
        cls,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        data: bytes | None = None,
        method: str | None = None,
        timeout: int = 30,
    ) -> Any:  # noqa: ANN401
        """GET/POST *url*, decode JSON.  Retries on transient failures."""
        from mlbv1.observability import track_api_call

        last_exc: Exception | None = None
        loader_name = cls.__name__
        for attempt in range(cls.MAX_RETRIES):
            req = urllib.request.Request(
                url, headers=headers or {}, data=data, method=method
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    track_api_call(loader_name, "success")
                    return json.loads(resp.read().decode("utf-8"))
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                TimeoutError,
            ) as exc:
                last_exc = exc
                track_api_call(loader_name, "retry")
                wait = cls.BACKOFF_BASE * (2**attempt)
                logger.warning(
                    "HTTP attempt %d failed (%s), retrying in %.1fs",
                    attempt + 1,
                    exc,
                    wait,
                )
                time.sleep(wait)
        track_api_call(loader_name, "error")
        raise DataLoaderError(
            f"HTTP request failed after {cls.MAX_RETRIES} attempts: {last_exc}"
        )


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------


class CSVLoader(BaseLoader):
    """Load game data from CSV."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise DataLoaderError(f"CSV file not found: {self.path}")
        df = pd.read_csv(self.path)
        df["game_date"] = pd.to_datetime(df["game_date"], utc=True)
        return self._validate(df)


class JSONLoader(BaseLoader):
    """Load game data from JSON array."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise DataLoaderError(f"JSON file not found: {self.path}")
        df = pd.read_json(self.path)
        df["game_date"] = pd.to_datetime(df["game_date"], utc=True)
        return self._validate(df)


# ---------------------------------------------------------------------------
# OddsAPI v4 — uses apiKey *query parameter* (NOT header)
# ---------------------------------------------------------------------------


class OddsAPILoader(BaseLoader):
    """Load MLB game odds from The Odds API v4.

    Auth: ``apiKey`` query-string parameter.
    Docs: https://the-odds-api.com/liveapi/guides/v4/
    """

    BASE = "https://api.the-odds-api.com/v4"
    CORE_MARKETS = "h2h,spreads,totals"
    F5_MARKETS = "h2h_1st_5_innings,spreads_1st_5_innings,totals_1st_5_innings"
    # team_totals requires per-event endpoint (Additional Market)
    TT_MARKETS = "team_totals"

    def __init__(self, base_url: str | None = None, api_key: str = "") -> None:
        self.base_url = (base_url or self.BASE).rstrip("/")
        self.api_key = api_key

    def _get_odds_payload(
        self,
        markets: str,
        *,
        suppress_errors: bool = False,
    ) -> list[dict[str, Any]]:
        params = urlencode(
            {
                "apiKey": self.api_key,
                "regions": "us",
                "markets": markets,
                "oddsFormat": "american",
            }
        )
        url = f"{self.base_url}/sports/baseball_mlb/odds/?{params}"
        if suppress_errors:
            try:
                data = self._http_get_json(url)
            except Exception as exc:
                logger.debug(
                    "Optional Odds API markets '%s' unavailable: %s", markets, exc
                )
                return []
        else:
            data = self._http_get_json(url)

        if not isinstance(data, list):
            raise DataLoaderError("Odds API payload is not a list")
        return data

    def _get_event_odds(
        self,
        event_id: str,
        markets: str,
    ) -> dict[str, Any]:
        """Fetch odds for a single event via per-event endpoint (premium)."""
        params = urlencode(
            {
                "apiKey": self.api_key,
                "regions": "us",
                "markets": markets,
                "oddsFormat": "american",
            }
        )
        url = f"{self.base_url}/sports/baseball_mlb/events/{event_id}/odds/?{params}"
        data: dict[str, Any] = self._http_get_json(url)
        if not isinstance(data, dict):
            raise DataLoaderError(f"Event odds response is not a dict for {event_id}")
        return data

    def load(self) -> pd.DataFrame:
        """Upcoming MLB odds — premium: all markets including team totals."""
        if not self.api_key:
            raise DataLoaderError(
                "ODDS_API_KEY is missing. Set it in environment or config before using odds_api loader."
            )

        # Core + totals in a single main-endpoint call
        base_payload = self._get_odds_payload(self.CORE_MARKETS)

        # F5 game-period markets (main endpoint)
        f5_markets = os.getenv("ODDS_API_F5_MARKETS", "").strip() or self.F5_MARKETS
        f5_payload = self._get_odds_payload(f5_markets, suppress_errors=True)
        if f5_payload:
            base_payload = _merge_odds_payloads(base_payload, f5_payload)
            logger.info("Loaded F5 markets: %s", f5_markets)

        # Team totals via per-event endpoint (premium additional market)
        for item in list(base_payload):
            event_id = item.get("id", "")
            if not event_id:
                continue
            try:
                tt_data = self._get_event_odds(event_id, self.TT_MARKETS)
                base_payload = _merge_odds_payloads(base_payload, [tt_data])
            except Exception as exc:
                logger.debug(
                    "Team totals unavailable for event %s: %s", event_id, exc
                )

        return self._normalize(base_payload)

    def load_scores(self, days_from: int = 3) -> pd.DataFrame:
        """Completed games with final scores (for settling)."""
        if not self.api_key:
            raise DataLoaderError(
                "ODDS_API_KEY is missing. Cannot fetch scores for settlement."
            )

        params = urlencode(
            {
                "apiKey": self.api_key,
                "daysFrom": str(days_from),
            }
        )
        url = f"{self.base_url}/sports/baseball_mlb/scores/?{params}"
        data = self._http_get_json(url)
        if not isinstance(data, list):
            raise DataLoaderError("Odds API scores payload is not a list")
        return self._normalize(data)

    # Full set of premium MLB player prop markets
    PROP_MARKETS = (
        "batter_home_runs,batter_hits,batter_total_bases,batter_rbis,"
        "batter_runs_scored,batter_strikeouts,batter_walks,batter_stolen_bases,"
        "pitcher_strikeouts,pitcher_hits_allowed,pitcher_walks,"
        "pitcher_earned_runs,pitcher_outs"
    )

    def load_props(self, event_id: str) -> dict[str, Any]:
        """Fetch premium player props for a specific event."""
        return self._get_event_odds(event_id, self.PROP_MARKETS)

    def _normalize(self, payload: Iterable[dict[str, Any]]) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for item in payload:
            markets = _extract_all_markets(item)
            home_team = normalize_team(item.get("home_team", ""))
            away_team = normalize_team(item.get("away_team", ""))
            # Pass through all market fields as-is (NaN if missing).
            # No -110 defaults — downstream must handle NaN explicitly.
            record: dict[str, Any] = {
                "event_id": item.get("id", ""),
                "game_date": item.get("commence_time"),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": (
                    item.get("scores", [{}])[0].get("score", 0)
                    if item.get("scores")
                    else 0
                ),
                "away_score": (
                    item.get("scores", [{}])[1].get("score", 0)
                    if item.get("scores") and len(item.get("scores", [])) > 1
                    else 0
                ),
            }
            record.update(markets)
            records.append(record)
        df = pd.DataFrame.from_records(records)
        if df.empty:
            return self._validate(_empty_df())
        df["game_date"] = pd.to_datetime(df["game_date"], utc=True)
        return self._validate(df)


class OddsAPIHistoricalLoader(OddsAPILoader):
    """Premium loader for historical odds snapshots."""

    def load_historical(self, target_date: str) -> pd.DataFrame:
        """
        Fetch odds as they appeared at a specific ISO UTC timestamp.
        Costs 10 credits per call.
        """
        params = urlencode(
            {
                "apiKey": self.api_key,
                "regions": "us",
                "markets": "h2h,spreads,totals,h2h_1st_5_innings,spreads_1st_5_innings,totals_1st_5_innings",
                "oddsFormat": "american",
                "date": target_date,
            }
        )
        url = f"{self.base_url}/historical/sports/baseball_mlb/odds/?{params}"
        logger.warning(f"Premium historical call made for {target_date} (10 credits)")
        response = self._http_get_json(url)
        data = response.get("data", [])
        return self._normalize(data)


# ---------------------------------------------------------------------------
# BetsAPI — uses token *query parameter*
# ---------------------------------------------------------------------------


class BetsAPILoader(BaseLoader):
    """Load MLB game data from BetsAPI.

    Auth: ``token`` query-string parameter.
    Docs: https://betsapi.com/docs/
    """

    BASE = "https://api.betsapi.com"
    MLB_SPORT_ID = 16  # BetsAPI sport_id for baseball/MLB

    def __init__(self, base_url: str | None = None, api_key: str = "") -> None:
        self.base_url = (base_url or self.BASE).rstrip("/")
        self.api_key = api_key

    def load(self) -> pd.DataFrame:
        params = urlencode(
            {
                "token": self.api_key,
                "sport_id": self.MLB_SPORT_ID,
            }
        )
        url = f"{self.base_url}/v3/events/upcoming?{params}"
        payload = self._http_get_json(url)
        events = payload.get("results", []) if isinstance(payload, dict) else payload
        if not isinstance(events, list):
            raise DataLoaderError("BetsAPI payload is not a list")
        return self._normalize(events)

    def load_odds(self, event_id: str) -> dict[str, Any]:
        """Fetch odds for a specific event from BetsAPI."""
        params = urlencode(
            {
                "token": self.api_key,
                "event_id": event_id,
            }
        )
        url = f"{self.base_url}/v3/event/odds?{params}"
        result: dict[str, Any] = self._http_get_json(url)
        return result

    def _normalize(self, events: list[dict[str, Any]]) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for ev in events:
            home_info = ev.get("home", {})
            away_info = ev.get("away", {})

            # Extract basic odds if available in the event object
            # BetsAPI often nests odds under 'odds' or requires a separate call
            odds_data = ev.get("odds", {})
            main_odds = odds_data.get("kick_off", {})

            records.append(
                {
                    "game_date": datetime.fromtimestamp(
                        int(ev.get("time", 0)), tz=UTC
                    ).isoformat(),
                    "home_team": normalize_team(home_info.get("name", "")),
                    "away_team": normalize_team(away_info.get("name", "")),
                    "home_score": (
                        int(ev.get("ss", "0-0").split("-")[0]) if ev.get("ss") else 0
                    ),
                    "away_score": (
                        int(ev.get("ss", "0-0").split("-")[1]) if ev.get("ss") else 0
                    ),
                    "spread": float(main_odds.get("spread", {}).get("home_od", 0.0)),
                    "home_moneyline": int(
                        main_odds.get("moneyline", {}).get("home_od", -110)
                    ),
                    "away_moneyline": int(
                        main_odds.get("moneyline", {}).get("away_od", -110)
                    ),
                }
            )
        df = pd.DataFrame.from_records(records)
        if df.empty:
            return self._validate(_empty_df())
        df["game_date"] = pd.to_datetime(df["game_date"], utc=True)
        return self._validate(df)


# ---------------------------------------------------------------------------
# Action Network — session-based auth
# ---------------------------------------------------------------------------


class ActionNetworkLoader(BaseLoader):
    """Load MLB odds from Action Network (session-based auth).

    Uses email + password to establish a session bearer token, then fetches
    the MLB scoreboard endpoint.
    """

    BASE = "https://api.actionnetwork.com"

    def __init__(
        self, base_url: str | None = None, api_key: str = "", email: str = ""
    ) -> None:
        self.base_url = (base_url or self.BASE).rstrip("/")
        self.api_key = api_key  # password if email is provided
        self.email = email
        self._token: str | None = None

    def _login(self) -> str:
        """Establish session. Returns bearer token."""
        if not self.email or not self.api_key:
            return ""

        url = f"{self.base_url}/web/v1/users/login"
        payload = json.dumps({"email": self.email, "password": self.api_key}).encode(
            "utf-8"
        )
        data_resp = self._http_get_json(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        self._token = data_resp.get("token")
        return self._token or ""

    def load(self) -> pd.DataFrame:
        if not self._token:
            self._login()

        url = f"{self.base_url}/web/v1/scoreboard/mlb"
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        try:
            payload = self._http_get_json(url, headers=headers)
        except DataLoaderError:
            logger.warning("ActionNetwork request failed; returning empty frame")
            return self._validate(_empty_df())
        games = payload.get("games", payload) if isinstance(payload, dict) else payload
        if not isinstance(games, list):
            raise DataLoaderError("Action Network payload is not a list")
        df = pd.DataFrame.from_records(games)
        if df.empty:
            return self._validate(_empty_df())
        if "game_date" in df.columns:
            df["game_date"] = pd.to_datetime(df["game_date"], utc=True)
        return self._validate(df)


# ---------------------------------------------------------------------------
# MLB Stats API — FREE, no API key needed
# ---------------------------------------------------------------------------


class MLBStatsAPILoader(BaseLoader):
    """Load historical MLB game data from the free MLB Stats API.

    ``statsapi.mlb.com/api/v1/schedule`` returns games with
    linescore data for any date range.
    """

    BASE = "https://statsapi.mlb.com/api/v1"

    def __init__(
        self,
        days_back: int = 90,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> None:
        self.days_back = days_back
        self._start_date = start_date
        self._end_date = end_date

    @classmethod
    def load_historical_season(cls, year: int) -> pd.DataFrame:
        """Load a complete MLB season by year (e.g., 2024, 2023)."""
        start_date = f"{year}-03-28"  # Opening day
        end_date = f"{year}-10-02"  # End of regular season
        loader = cls(start_date=start_date, end_date=end_date)
        return loader.load()

    def load(self) -> pd.DataFrame:
        if self._start_date and self._end_date:
            start = datetime.strptime(self._start_date, "%Y-%m-%d")
            end = datetime.strptime(self._end_date, "%Y-%m-%d")
        else:
            end = datetime.now(tz=UTC)
            start = end - timedelta(days=self.days_back)
        all_records: list[dict[str, Any]] = []

        # Fetch in weekly chunks to avoid overly large responses
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=7), end)
            params = urlencode(
                {
                    "sportId": 1,
                    "startDate": chunk_start.strftime("%Y-%m-%d"),
                    "endDate": chunk_end.strftime("%Y-%m-%d"),
                    "hydrate": "linescore",
                }
            )
            url = f"{self.BASE}/schedule?{params}"
            try:
                data = self._http_get_json(url)
            except DataLoaderError:
                logger.warning(
                    "MLB Stats API chunk %s–%s failed, skipping",
                    chunk_start.date(),
                    chunk_end.date(),
                )
                chunk_start = chunk_end
                continue
            for date_entry in data.get("dates", []):
                for game in date_entry.get("games", []):
                    rec = self._parse_game(game)
                    if rec:
                        all_records.append(rec)
            chunk_start = chunk_end

        if not all_records:
            logger.warning(
                "MLBStatsAPI returned no games for last %d days", self.days_back
            )
            return self._validate(_empty_df())

        df = pd.DataFrame.from_records(all_records)
        df["game_date"] = pd.to_datetime(df["game_date"], utc=True)
        return self._validate(df)

    @staticmethod
    def _parse_game(game: dict[str, Any]) -> dict[str, Any] | None:
        """Extract a single game record from the MLB schedule response."""
        status = game.get("status", {}).get("abstractGameState", "")
        if status != "Final":
            return None
        teams = game.get("teams", {})
        home = teams.get("home", {})
        away = teams.get("away", {})
        linescore = game.get("linescore", {})
        home_runs = linescore.get("teams", {}).get("home", {}).get("runs", 0)
        away_runs = linescore.get("teams", {}).get("away", {}).get("runs", 0)

        # Extract F5 (first 5 innings) scores from linescore innings array
        f5_home, f5_away = MLBStatsAPILoader._extract_f5_scores(linescore)

        # Extract pitcher stats if available
        decisions = game.get("decisions", {})
        home_pitcher = decisions.get("winner", {}) if "winner" in decisions else {}
        away_pitcher = decisions.get("loser", {}) if "loser" in decisions else {}

        home_pitcher_id = home_pitcher.get("id", None)
        away_pitcher_id = away_pitcher.get("id", None)
        home_pitcher_name = home_pitcher.get("fullName", "")
        away_pitcher_name = away_pitcher.get("fullName", "")

        home_pitcher_era = MLBStatsAPILoader._get_pitcher_stat(home_pitcher, "era")
        away_pitcher_era = MLBStatsAPILoader._get_pitcher_stat(away_pitcher, "era")
        home_pitcher_wins = MLBStatsAPILoader._get_pitcher_stat(home_pitcher, "wins")
        away_pitcher_wins = MLBStatsAPILoader._get_pitcher_stat(away_pitcher, "wins")

        return {
            "game_date": game.get("gameDate", ""),
            "home_team": normalize_team(
                home.get("team", {}).get(
                    "abbreviation", home.get("team", {}).get("name", "")
                )
            ),
            "away_team": normalize_team(
                away.get("team", {}).get(
                    "abbreviation", away.get("team", {}).get("name", "")
                )
            ),
            "home_score": int(home_runs) if home_runs else 0,
            "away_score": int(away_runs) if away_runs else 0,
            "f5_home_score": f5_home,
            "f5_away_score": f5_away,
            "spread": 0.0,
            "home_moneyline": -110,
            "away_moneyline": -110,
            "home_pitcher_era": home_pitcher_era,
            "away_pitcher_era": away_pitcher_era,
            "home_pitcher_wins": home_pitcher_wins,
            "away_pitcher_wins": away_pitcher_wins,
            "home_pitcher_id": home_pitcher_id,
            "away_pitcher_id": away_pitcher_id,
            "home_pitcher_name": home_pitcher_name,
            "away_pitcher_name": away_pitcher_name,
        }

    @staticmethod
    def _extract_f5_scores(linescore: dict[str, Any]) -> tuple[int, int]:
        """Sum runs for the first 5 innings from linescore innings array."""
        innings = linescore.get("innings", [])
        home_f5 = 0
        away_f5 = 0
        for inning in innings[:5]:  # first 5 innings only
            home_f5 += int(inning.get("home", {}).get("runs", 0) or 0)
            away_f5 += int(inning.get("away", {}).get("runs", 0) or 0)
        return home_f5, away_f5

    @staticmethod
    def _get_pitcher_stat(pitcher: dict[str, Any], stat_name: str) -> float:
        """Safely extract pitcher statistic with default fallback."""
        try:
            stats = pitcher.get("seasonStats", {}).get("pitching", {})
            value = stats.get(stat_name, 0.0)
            return float(value) if value else 0.0
        except (TypeError, ValueError, KeyError):
            return 0.0


# ---------------------------------------------------------------------------
# Weather enrichment — Visual Crossing
# ---------------------------------------------------------------------------

# Derive (lat, lon) from the canonical STADIUM_DATA in mapping.py (single source of truth).
_STADIUM_COORDS: dict[str, tuple[float, float]] = {
    abbr: (lat, lon) for abbr, (lat, lon, _indoor) in STADIUM_DATA.items()
}


class WeatherEnricher:
    """Enrich a games DataFrame with weather data from Visual Crossing."""

    BASE = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temperature_f, wind_mph, precipitation columns via weather lookups.

        Batches requests by unique (stadium, date) to avoid redundant API calls
        when multiple bookmaker rows share the same game.
        """
        if not self.api_key:
            logger.info("No Visual Crossing API key — skipping weather enrichment")
            return df

        out = df.copy()

        # Build a cache keyed on (team, date_str) to avoid N+1 calls.
        cache: dict[tuple[str, str], dict[str, float]] = {}
        unique_lookups: set[tuple[str, str]] = set()
        for _, row in out.iterrows():
            team = str(row.get("home_team", ""))
            try:
                date_str = pd.Timestamp(row.get("game_date")).strftime("%Y-%m-%d")
            except Exception:
                continue
            if _STADIUM_COORDS.get(team):
                unique_lookups.add((team, date_str))

        logger.info("Weather enrichment: %d unique (stadium, date) lookups", len(unique_lookups))
        for team, date_str in unique_lookups:
            coords = _STADIUM_COORDS[team]
            cache[(team, date_str)] = self._fetch_weather(coords, date_str)

        # Map cached results back to each row.
        temps, winds, precips = [], [], []
        for _, row in out.iterrows():
            team = str(row.get("home_team", ""))
            try:
                date_str = pd.Timestamp(row.get("game_date")).strftime("%Y-%m-%d")
            except Exception:
                date_str = ""
            weather = cache.get((team, date_str), {})
            temps.append(weather.get("temp", 70.0))
            winds.append(weather.get("windspeed", 5.0))
            precips.append(weather.get("precip", 0.0))

        out["temperature_f"] = temps
        out["wind_mph"] = winds
        out["precipitation"] = precips
        return out

    def _fetch_weather(
        self, coords: tuple[float, float], date_str: str
    ) -> dict[str, float]:
        """Fetch weather for a single location + date (YYYY-MM-DD)."""
        lat, lon = coords
        params = urlencode(
            {
                "unitGroup": "us",
                "key": self.api_key,
                "include": "days",
                "contentType": "json",
            }
        )
        url = f"{self.BASE}/{lat},{lon}/{date_str}?{params}"
        try:
            data = BaseLoader._http_get_json(url, timeout=15)
            day = data.get("days", [{}])[0]
            return {
                "temp": day.get("temp", 70.0),
                "windspeed": day.get("windspeed", 5.0),
                "precip": day.get("precip", 0.0),
            }
        except DataLoaderError:
            logger.debug("Weather fetch failed for %s on %s", coords, date_str)
            return {}


# ---------------------------------------------------------------------------
# Synthetic data (testing)
# ---------------------------------------------------------------------------


class SyntheticDataLoader(BaseLoader):
    """Generate synthetic MLB game data for testing."""

    def __init__(self, num_games: int = 200, seed: int = 42) -> None:
        self.num_games = num_games
        self._rng = _random_mod.Random(seed)

    def load(self) -> pd.DataFrame:
        teams = [
            "BOS",
            "NYY",
            "LAD",
            "CHC",
            "ATL",
            "HOU",
            "TBR",
            "PHI",
            "SDP",
            "SFG",
            "SEA",
            "MIN",
            "CLE",
            "BAL",
            "TEX",
            "MIL",
        ]
        records: list[dict[str, Any]] = []
        start = datetime(2024, 3, 20)
        for _ in range(self.num_games):
            home = self._rng.choice(teams)
            away = self._rng.choice([t for t in teams if t != home])
            # Poisson-distributed scores (MLB avg ~4.3 R/G per team)
            home_score = min(15, int(self._rng.gauss(4.3, 2.2)))
            away_score = min(15, int(self._rng.gauss(4.3, 2.2)))
            home_score = max(0, home_score)
            away_score = max(0, away_score)
            spread = round(self._rng.uniform(-2.5, 2.5), 1)
            home_ml = self._rng.choice([-140, -120, -110, 110, 130, 150])
            away_ml = -home_ml if home_ml < 0 else self._rng.choice([-150, -130, -110])
            records.append(
                {
                    "game_date": start,
                    "home_team": home,
                    "away_team": away,
                    "home_score": home_score,
                    "away_score": away_score,
                    "spread": spread,
                    "home_moneyline": home_ml,
                    "away_moneyline": away_ml,
                    "temperature_f": self._rng.uniform(45, 90),
                    "wind_mph": self._rng.uniform(0, 18),
                    "precipitation": self._rng.uniform(0, 0.6),
                }
            )
            start = start + timedelta(days=1)
        df = pd.DataFrame.from_records(records)
        df["game_date"] = pd.to_datetime(df["game_date"], utc=True)
        return self._validate(df)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_df() -> pd.DataFrame:
    """Return an empty DataFrame with all required columns."""
    from mlbv1.data.preprocessor import OPTIONAL_COLUMNS, REQUIRED_COLUMNS

    cols = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
    return pd.DataFrame({col: pd.Series(dtype="object") for col in cols})


def _extract_all_markets(item: dict[str, Any]) -> dict[str, float]:
    """Extract full game and F5 markets: ML, spread, totals.

    Averages odds across ALL bookmakers for consensus pricing rather than
    using a single arbitrary book.  Lines (spread points, total points) use
    the mode (most common value) since these are identical across most books.
    """
    # Collect all values per field across bookmakers for averaging.
    from collections import defaultdict

    _odds_accum: dict[str, list[float]] = defaultdict(list)
    _line_accum: dict[str, list[float]] = defaultdict(list)

    home = item.get("home_team")
    away = item.get("away_team")

    for bm in item.get("bookmakers", []):
        for market in bm.get("markets", []):
            mkey = market.get("key")
            outcomes = market.get("outcomes", [])

            if mkey == "spreads":
                for o in outcomes:
                    if "price" not in o:
                        continue
                    if o.get("name") == home:
                        _line_accum["spread"].append(float(o["point"]))
                        _odds_accum["home_spread_odds"].append(float(o["price"]))
                    elif o.get("name") == away:
                        _odds_accum["away_spread_odds"].append(float(o["price"]))
            elif mkey == "h2h":
                for o in outcomes:
                    if "price" not in o:
                        continue
                    if o.get("name") == home:
                        _odds_accum["home_moneyline"].append(float(o["price"]))
                    elif o.get("name") == away:
                        _odds_accum["away_moneyline"].append(float(o["price"]))
            elif mkey == "totals":
                for o in outcomes:
                    if "price" not in o:
                        continue
                    _line_accum["total_runs"].append(float(o.get("point", 0)))
                    if str(o.get("name", "")).lower() == "over":
                        _odds_accum["over_odds"].append(float(o["price"]))
                    elif str(o.get("name", "")).lower() == "under":
                        _odds_accum["under_odds"].append(float(o["price"]))

            # F5 markets
            elif mkey in {"spreads_1st_half", "spreads_1st_5_innings"}:
                for o in outcomes:
                    if "price" not in o:
                        continue
                    if o.get("name") == home:
                        _line_accum["f5_spread"].append(float(o["point"]))
                        _odds_accum["f5_home_spread_odds"].append(float(o["price"]))
                    elif o.get("name") == away:
                        _odds_accum["f5_away_spread_odds"].append(float(o["price"]))
            elif mkey in {"h2h_1st_half", "h2h_1st_5_innings"}:
                for o in outcomes:
                    if "price" not in o:
                        continue
                    if o.get("name") == home:
                        _odds_accum["f5_home_moneyline"].append(float(o["price"]))
                    elif o.get("name") == away:
                        _odds_accum["f5_away_moneyline"].append(float(o["price"]))
            elif mkey in {"totals_1st_half", "totals_1st_5_innings"}:
                for o in outcomes:
                    if "price" not in o:
                        continue
                    _line_accum["f5_total_runs"].append(float(o.get("point", 0)))
                    if str(o.get("name", "")).lower() == "over":
                        _odds_accum["f5_over_odds"].append(float(o["price"]))
                    elif str(o.get("name", "")).lower() == "under":
                        _odds_accum["f5_under_odds"].append(float(o["price"]))

            # Team totals (premium per-event market)
            elif mkey == "team_totals":
                for o in outcomes:
                    if "price" not in o:
                        continue
                    desc = o.get("description", "")
                    side = str(o.get("name", "")).lower()
                    if desc == home:
                        _line_accum["home_tt"].append(float(o.get("point", 0)))
                        if side == "over":
                            _odds_accum["home_tt_over_odds"].append(float(o["price"]))
                        elif side == "under":
                            _odds_accum["home_tt_under_odds"].append(float(o["price"]))
                    elif desc == away:
                        _line_accum["away_tt"].append(float(o.get("point", 0)))
                        if side == "over":
                            _odds_accum["away_tt_over_odds"].append(float(o["price"]))
                        elif side == "under":
                            _odds_accum["away_tt_under_odds"].append(float(o["price"]))

    # Build result: consensus average for odds, mode for lines, NaN if no data.
    def _american_to_prob(odds: float) -> float:
        """Convert American odds to implied probability."""
        if odds < 0:
            return -odds / (-odds + 100)
        elif odds > 0:
            return 100 / (odds + 100)
        return 0.5  # even money

    def _prob_to_american(prob: float) -> int:
        """Convert implied probability back to American odds."""
        if prob <= 0 or prob >= 1:
            return -10000 if prob >= 1 else 10000
        if prob > 0.5:
            return round(-(prob / (1 - prob)) * 100)
        elif prob < 0.5:
            return round(((1 - prob) / prob) * 100)
        return 100  # even money

    def _avg(vals: list[float]) -> float:
        """Consensus odds: average in probability space, convert back."""
        if not vals:
            return float("nan")
        avg_prob = sum(_american_to_prob(v) for v in vals) / len(vals)
        return float(_prob_to_american(avg_prob))

    def _mode(vals: list[float]) -> float:
        if not vals:
            return float("nan")
        from statistics import mode as _stat_mode
        try:
            return _stat_mode(vals)
        except Exception:
            return vals[0]

    return {
        "spread": _mode(_line_accum["spread"]) if _line_accum["spread"] else 0.0,
        "home_spread_odds": _avg(_odds_accum["home_spread_odds"]),
        "away_spread_odds": _avg(_odds_accum["away_spread_odds"]),
        "home_moneyline": _avg(_odds_accum["home_moneyline"]),
        "away_moneyline": _avg(_odds_accum["away_moneyline"]),
        "total_runs": _mode(_line_accum["total_runs"]) if _line_accum["total_runs"] else 0.0,
        "over_odds": _avg(_odds_accum["over_odds"]),
        "under_odds": _avg(_odds_accum["under_odds"]),
        "f5_spread": _mode(_line_accum["f5_spread"]) if _line_accum["f5_spread"] else 0.0,
        "f5_home_spread_odds": _avg(_odds_accum["f5_home_spread_odds"]),
        "f5_away_spread_odds": _avg(_odds_accum["f5_away_spread_odds"]),
        "f5_home_moneyline": _avg(_odds_accum["f5_home_moneyline"]),
        "f5_away_moneyline": _avg(_odds_accum["f5_away_moneyline"]),
        "f5_total_runs": _mode(_line_accum["f5_total_runs"]) if _line_accum["f5_total_runs"] else 0.0,
        "f5_over_odds": _avg(_odds_accum["f5_over_odds"]),
        "f5_under_odds": _avg(_odds_accum["f5_under_odds"]),
        "home_tt": _mode(_line_accum["home_tt"]) if _line_accum["home_tt"] else float("nan"),
        "away_tt": _mode(_line_accum["away_tt"]) if _line_accum["away_tt"] else float("nan"),
        "home_tt_over_odds": _avg(_odds_accum["home_tt_over_odds"]),
        "home_tt_under_odds": _avg(_odds_accum["home_tt_under_odds"]),
        "away_tt_over_odds": _avg(_odds_accum["away_tt_over_odds"]),
        "away_tt_under_odds": _avg(_odds_accum["away_tt_under_odds"]),
    }


def _merge_odds_payloads(
    base_payload: list[dict[str, Any]],
    extra_payload: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge bookmakers/markets from *extra_payload* into *base_payload* by event id."""
    merged = {str(item.get("id", "")): dict(item) for item in base_payload}
    for extra in extra_payload:
        event_id = str(extra.get("id", ""))
        if not event_id:
            continue
        if event_id not in merged:
            merged[event_id] = dict(extra)
            continue

        base_item = merged[event_id]
        base_bms = {bm.get("key"): bm for bm in base_item.get("bookmakers", [])}
        for extra_bm in extra.get("bookmakers", []):
            bm_key = extra_bm.get("key")
            if bm_key not in base_bms:
                base_item.setdefault("bookmakers", []).append(extra_bm)
                continue
            base_market_keys = {
                m.get("key") for m in base_bms[bm_key].get("markets", [])
            }
            for market in extra_bm.get("markets", []):
                if market.get("key") not in base_market_keys:
                    base_bms[bm_key].setdefault("markets", []).append(market)

    return list(merged.values())


# ---------------------------------------------------------------------------
# Loader factory
# ---------------------------------------------------------------------------


def build_loader_from_config(config: Any) -> BaseLoader:  # noqa: ANN401
    """Construct a loader instance from an AppConfig object.

    Centralises the loader→class mapping so scripts don't duplicate it.
    """
    loader_name = str(config.data.loader).strip()
    if loader_name == "synthetic":
        return SyntheticDataLoader(num_games=300)
    if loader_name == "odds_api":
        return OddsAPILoader(
            config.data.api_base_url or "https://api.the-odds-api.com/v4",
            config.data.api_key or os.getenv("ODDS_API_KEY", ""),
        )
    if loader_name == "bets_api":
        return BetsAPILoader(
            config.data.api_base_url or "https://api.betsapi.com",
            config.data.api_key or os.getenv("BETS_API_KEY", ""),
        )
    if loader_name == "action_network":
        return ActionNetworkLoader(
            config.data.api_base_url or "https://api.actionnetwork.com",
            config.data.api_key or os.getenv("ACTION_NETWORK_PASSWORD", ""),
            config.data.email or os.getenv("ACTION_NETWORK_EMAIL", ""),
        )
    if loader_name == "csv":
        if not config.data.input_path:
            raise ValueError("CSV loader requires data.input_path")
        return CSVLoader(config.data.input_path)
    if loader_name == "json":
        if not config.data.input_path:
            raise ValueError("JSON loader requires data.input_path")
        return JSONLoader(config.data.input_path)
    if loader_name == "mlb_stats":
        return MLBStatsAPILoader()
    raise ValueError(f"Unsupported loader: {loader_name}")
