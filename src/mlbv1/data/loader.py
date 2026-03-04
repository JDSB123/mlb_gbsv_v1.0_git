"""MLB data loaders for games, odds, and outcomes.

Supports real APIs (OddsAPI v4, BetsAPI, Action Network, MLB Stats API),
weather enrichment (Visual Crossing), plus CSV / JSON / synthetic sources.
"""

from __future__ import annotations

import http.client
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, List, Optional
from urllib.parse import urlencode
import urllib.request
import random as _random_mod

import pandas as pd

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
    temperature_f: Optional[float] = None
    wind_mph: Optional[float] = None
    precipitation: Optional[float] = None


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
        return df

    # Shared HTTP helper with retry + exponential back-off ----------------

    @classmethod
    def _http_get_json(
        cls,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: int = 30,
    ) -> Any:  # noqa: ANN401
        """GET *url*, decode JSON.  Retries on transient failures."""
        last_exc: Exception | None = None
        for attempt in range(cls.MAX_RETRIES):
            req = urllib.request.Request(url, headers=headers or {})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                TimeoutError,
            ) as exc:
                last_exc = exc
                wait = cls.BACKOFF_BASE * (2**attempt)
                logger.warning(
                    "HTTP attempt %d failed (%s), retrying in %.1fs",
                    attempt + 1,
                    exc,
                    wait,
                )
                time.sleep(wait)
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

    def __init__(self, base_url: str | None = None, api_key: str = "") -> None:
        self.base_url = (base_url or self.BASE).rstrip("/")
        self.api_key = api_key

    def load(self) -> pd.DataFrame:
        """Upcoming MLB odds (spreads + h2h moneylines)."""
        params = urlencode(
            {
                "apiKey": self.api_key,
                "regions": "us",
                "markets": "spreads,h2h",
                "oddsFormat": "american",
            }
        )
        url = f"{self.base_url}/sports/baseball_mlb/odds/?{params}"
        data = self._http_get_json(url)
        if not isinstance(data, list):
            raise DataLoaderError("Odds API payload is not a list")
        return self._normalize(data)

    def load_scores(self) -> pd.DataFrame:
        """Completed games with final scores (for settling)."""
        params = urlencode(
            {
                "apiKey": self.api_key,
                "daysFrom": "3",
            }
        )
        url = f"{self.base_url}/sports/baseball_mlb/scores/?{params}"
        data = self._http_get_json(url)
        if not isinstance(data, list):
            raise DataLoaderError("Odds API scores payload is not a list")
        return self._normalize(data)

    def _normalize(self, payload: Iterable[dict[str, Any]]) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for item in payload:
            spread = _extract_spread(item)
            home_ml, away_ml = _extract_h2h_moneylines(item)
            records.append(
                {
                    "game_date": item.get("commence_time"),
                    "home_team": item.get("home_team", ""),
                    "away_team": item.get("away_team", ""),
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
                    "spread": spread,
                    "home_moneyline": home_ml,
                    "away_moneyline": away_ml,
                }
            )
        df = pd.DataFrame.from_records(records)
        if df.empty:
            return self._validate(_empty_df())
        df["game_date"] = pd.to_datetime(df["game_date"], utc=True)
        return self._validate(df)


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

    def _normalize(self, events: list[dict[str, Any]]) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for ev in events:
            home_info = ev.get("home", {})
            away_info = ev.get("away", {})
            records.append(
                {
                    "game_date": datetime.utcfromtimestamp(
                        int(ev.get("time", 0))
                    ).isoformat(),
                    "home_team": home_info.get("name", ""),
                    "away_team": away_info.get("name", ""),
                    "home_score": (
                        int(ev.get("ss", "0-0").split("-")[0]) if ev.get("ss") else 0
                    ),
                    "away_score": (
                        int(ev.get("ss", "0-0").split("-")[1]) if ev.get("ss") else 0
                    ),
                    "spread": 0.0,
                    "home_moneyline": -110,
                    "away_moneyline": -110,
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

    Uses email + password to establish a session cookie, then fetches
    the MLB scoreboard endpoint.
    """

    BASE = "https://api.actionnetwork.com"

    def __init__(self, base_url: str | None = None, api_key: str = "") -> None:
        self.base_url = (base_url or self.BASE).rstrip("/")
        self.api_key = api_key

    def load(self) -> pd.DataFrame:
        url = f"{self.base_url}/web/v1/scoreboard/mlb"
        try:
            payload = self._http_get_json(url, headers={"Authorization": self.api_key})
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
        return {
            "game_date": game.get("gameDate", ""),
            "home_team": home.get("team", {}).get(
                "abbreviation", home.get("team", {}).get("name", "")
            ),
            "away_team": away.get("team", {}).get(
                "abbreviation", away.get("team", {}).get("name", "")
            ),
            "home_score": int(home_runs) if home_runs else 0,
            "away_score": int(away_runs) if away_runs else 0,
            "spread": 0.0,
            "home_moneyline": -110,
            "away_moneyline": -110,
        }


# ---------------------------------------------------------------------------
# Weather enrichment — Visual Crossing
# ---------------------------------------------------------------------------

# Mapping of MLB team abbreviation → (lat, lon)
_STADIUM_COORDS: dict[str, tuple[float, float]] = {
    "ARI": (33.4453, -112.0667),
    "ATL": (33.8907, -84.4678),
    "BAL": (39.2838, -76.6217),
    "BOS": (42.3467, -71.0972),
    "CHC": (41.9484, -87.6553),
    "CWS": (41.8299, -87.6338),
    "CIN": (39.0976, -84.5069),
    "CLE": (41.4962, -81.6852),
    "COL": (39.7561, -104.9942),
    "DET": (42.3390, -83.0485),
    "HOU": (29.7573, -95.3555),
    "KC": (39.0517, -94.4803),
    "LAA": (33.8003, -117.8827),
    "LAD": (34.0739, -118.2400),
    "MIA": (25.7781, -80.2196),
    "MIL": (43.0280, -87.9712),
    "MIN": (44.9817, -93.2776),
    "NYM": (40.7571, -73.8458),
    "NYY": (40.8296, -73.9262),
    "OAK": (37.7516, -122.2005),
    "PHI": (39.9061, -75.1665),
    "PIT": (40.4469, -80.0058),
    "SD": (32.7076, -117.1570),
    "SF": (37.7786, -122.3893),
    "SEA": (47.5914, -122.3325),
    "STL": (38.6226, -90.1928),
    "TB": (27.7682, -82.6534),
    "TEX": (32.7512, -97.0832),
    "TOR": (43.6414, -79.3894),
    "WSH": (38.8730, -77.0074),
}


class WeatherEnricher:
    """Enrich a games DataFrame with weather data from Visual Crossing."""

    BASE = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temperature_f, wind_mph, precipitation columns via weather lookups."""
        if not self.api_key:
            logger.info("No Visual Crossing API key — skipping weather enrichment")
            return df

        out = df.copy()
        temps, winds, precips = [], [], []

        for _, row in out.iterrows():
            team = row.get("home_team", "")
            game_date = row.get("game_date")
            coords = _STADIUM_COORDS.get(str(team))
            if coords and game_date is not None:
                weather = self._fetch_weather(coords, game_date)
                temps.append(weather.get("temp", 70.0))
                winds.append(weather.get("windspeed", 5.0))
                precips.append(weather.get("precip", 0.0))
            else:
                temps.append(70.0)
                winds.append(5.0)
                precips.append(0.0)

        out["temperature_f"] = temps
        out["wind_mph"] = winds
        out["precipitation"] = precips
        return out

    def _fetch_weather(
        self, coords: tuple[float, float], date: Any
    ) -> dict[str, float]:
        """Fetch weather for a single location + date."""
        try:
            date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        except Exception:
            return {}
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
            "TB",
            "PHI",
            "SD",
            "SF",
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
            home_score = self._rng.randint(0, 12)
            away_score = self._rng.randint(0, 12)
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
    return pd.DataFrame({col: pd.Series(dtype="object") for col in REQUIRED_COLUMNS})


def _extract_spread(item: dict[str, Any]) -> float:
    """Extract the home spread from an OddsAPI v4 bookmakers payload."""
    for bm in item.get("bookmakers", []):
        for market in bm.get("markets", []):
            if market.get("key") == "spreads":
                for outcome in market.get("outcomes", []):
                    if outcome.get("name") == item.get("home_team"):
                        return float(outcome.get("point", 0.0))
    return 0.0


def _extract_h2h_moneylines(item: dict[str, Any]) -> tuple[int, int]:
    """Extract (home_moneyline, away_moneyline) from OddsAPI v4 h2h market."""
    home_ml, away_ml = -110, -110
    for bm in item.get("bookmakers", []):
        for market in bm.get("markets", []):
            if market.get("key") == "h2h":
                for outcome in market.get("outcomes", []):
                    price = int(outcome.get("price", -110))
                    if outcome.get("name") == item.get("home_team"):
                        home_ml = price
                    elif outcome.get("name") == item.get("away_team"):
                        away_ml = price
                return home_ml, away_ml
    return home_ml, away_ml


def _extract_moneyline(item: dict[str, Any], side: str) -> int:
    """Legacy helper kept for backward compat."""
    moneylines = item.get("moneyline") or {}
    return int(moneylines.get(side, -110))
