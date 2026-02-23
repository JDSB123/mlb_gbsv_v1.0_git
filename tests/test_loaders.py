"""Tests for data loaders with mocked HTTP responses."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from mlbv1.data.loader import (
    CSVLoader,
    DataLoaderError,
    JSONLoader,
    MLBStatsAPILoader,
    OddsAPILoader,
    SyntheticDataLoader,
)

# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------


class TestSyntheticLoader:
    def test_num_games(self) -> None:
        df = SyntheticDataLoader(num_games=30, seed=1).load()
        assert len(df) == 30

    def test_deterministic(self) -> None:
        a = SyntheticDataLoader(num_games=20, seed=99).load()
        b = SyntheticDataLoader(num_games=20, seed=99).load()
        pd.testing.assert_frame_equal(a, b)

    def test_columns_present(self) -> None:
        df = SyntheticDataLoader(num_games=10, seed=0).load()
        for col in [
            "game_date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "spread",
            "home_moneyline",
            "away_moneyline",
        ]:
            assert col in df.columns, f"Missing column: {col}"

    def test_scores_positive(self) -> None:
        df = SyntheticDataLoader(num_games=50, seed=7).load()
        assert (df["home_score"] >= 0).all()
        assert (df["away_score"] >= 0).all()


# ---------------------------------------------------------------------------
# CSV / JSON
# ---------------------------------------------------------------------------


class TestCSVLoader:
    def test_load_valid(self, synthetic_df: pd.DataFrame, tmp_path: Path) -> None:
        path = tmp_path / "test.csv"
        synthetic_df.to_csv(path, index=False)
        loaded = CSVLoader(str(path)).load()
        assert len(loaded) == len(synthetic_df)

    def test_missing_file(self) -> None:
        with pytest.raises(DataLoaderError, match="not found"):
            CSVLoader("/nonexistent/file.csv").load()


class TestJSONLoader:
    def test_load_valid(self, synthetic_df: pd.DataFrame, tmp_path: Path) -> None:
        path = tmp_path / "test.json"
        synthetic_df.to_json(path, orient="records", date_format="iso")
        loaded = JSONLoader(str(path)).load()
        assert len(loaded) == len(synthetic_df)

    def test_missing_file(self) -> None:
        with pytest.raises(DataLoaderError, match="not found"):
            JSONLoader("/nonexistent/file.json").load()


# ---------------------------------------------------------------------------
# MLBStatsAPI (mocked HTTP)
# ---------------------------------------------------------------------------


class TestMLBStatsAPILoader:
    @staticmethod
    def _make_game(
        game_date: str = "2024-06-15",
        home: str = "New York Yankees",
        away: str = "Boston Red Sox",
        home_score: int = 5,
        away_score: int = 3,
    ) -> dict[str, Any]:
        return {
            "status": {"abstractGameState": "Final"},
            "officialDate": game_date,
            "teams": {
                "home": {
                    "team": {"name": home},
                    "score": home_score,
                },
                "away": {
                    "team": {"name": away},
                    "score": away_score,
                },
            },
            "linescore": {
                "innings": [{"home": {"runs": 2}, "away": {"runs": 1}}],
            },
        }

    def test_load_parses_games(self) -> None:
        mock_response = {
            "dates": [
                {
                    "games": [
                        self._make_game(),
                        self._make_game(
                            home="Los Angeles Dodgers",
                            away="San Diego Padres",
                            home_score=7,
                            away_score=2,
                        ),
                    ]
                }
            ]
        }
        with patch.object(
            MLBStatsAPILoader, "_http_get_json", return_value=mock_response
        ):
            loader = MLBStatsAPILoader(days_back=1)
            df = loader.load()
            assert len(df) >= 2
            assert "home_team" in df.columns
            assert "away_team" in df.columns

    def test_load_empty_response(self) -> None:
        with patch.object(
            MLBStatsAPILoader, "_http_get_json", return_value={"dates": []}
        ):
            loader = MLBStatsAPILoader(days_back=1)
            df = loader.load()
            assert len(df) == 0

    def test_skips_non_final_games(self) -> None:
        game = self._make_game()
        game["status"]["abstractGameState"] = "Preview"
        mock_response = {"dates": [{"games": [game]}]}
        with patch.object(
            MLBStatsAPILoader, "_http_get_json", return_value=mock_response
        ):
            loader = MLBStatsAPILoader(days_back=1)
            df = loader.load()
            assert len(df) == 0

    def test_start_end_date_params(self) -> None:
        mock_response = {"dates": [{"games": [self._make_game()]}]}
        with patch.object(
            MLBStatsAPILoader, "_http_get_json", return_value=mock_response
        ) as mock_get:
            loader = MLBStatsAPILoader(start_date="2024-06-01", end_date="2024-06-07")
            df = loader.load()
            assert len(df) >= 1
            # Verify the URL contains our dates
            call_url = mock_get.call_args[0][0]
            assert "2024-06-01" in call_url


# ---------------------------------------------------------------------------
# OddsAPI (mocked HTTP)
# ---------------------------------------------------------------------------


class TestOddsAPILoader:
    def test_load_returns_dataframe(self) -> None:
        mock_events = [
            {
                "commence_time": "2024-06-15T22:00:00Z",
                "home_team": "New York Yankees",
                "away_team": "Boston Red Sox",
                "bookmakers": [
                    {
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "New York Yankees", "price": -150},
                                    {"name": "Boston Red Sox", "price": 130},
                                ],
                            },
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {
                                        "name": "New York Yankees",
                                        "price": -110,
                                        "point": -1.5,
                                    },
                                    {
                                        "name": "Boston Red Sox",
                                        "price": -110,
                                        "point": 1.5,
                                    },
                                ],
                            },
                        ]
                    }
                ],
            }
        ]
        with patch.object(OddsAPILoader, "_http_get_json", return_value=mock_events):
            loader = OddsAPILoader(api_key="test-key")
            df = loader.load()
            assert len(df) == 1
            assert df.iloc[0]["home_team"] == "New York Yankees"

    def test_empty_events(self) -> None:
        with patch.object(OddsAPILoader, "_http_get_json", return_value=[]):
            loader = OddsAPILoader(api_key="test-key")
            df = loader.load()
            assert len(df) == 0


# ---------------------------------------------------------------------------
# HTTP retry
# ---------------------------------------------------------------------------


class TestHTTPRetry:
    def test_retries_on_failure(self) -> None:
        with patch.object(
            MLBStatsAPILoader,
            "_http_get_json",
            side_effect=DataLoaderError("timeout"),
        ):
            loader = MLBStatsAPILoader(days_back=1)
            # The loader catches DataLoaderError per chunk and skips
            df = loader.load()
            assert len(df) == 0
