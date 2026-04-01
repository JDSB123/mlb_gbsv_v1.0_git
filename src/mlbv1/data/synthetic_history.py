"""Synthetic historical MLB data generator for enriching training sets."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd


class SyntheticHistoryGenerator:
    """Generate realistic synthetic MLB seasons (2023-2025) for model training enrichment."""

    # MLB teams
    TEAMS = [
        "Arizona Diamondbacks",
        "Atlanta Braves",
        "Baltimore Orioles",
        "Boston Red Sox",
        "Chicago Cubs",
        "Chicago White Sox",
        "Cincinnati Reds",
        "Cleveland Guardians",
        "Colorado Rockies",
        "Detroit Tigers",
        "Houston Astros",
        "Kansas City Royals",
        "Los Angeles Angels",
        "Los Angeles Dodgers",
        "Miami Marlins",
        "Milwaukee Brewers",
        "Minnesota Twins",
        "New York Mets",
        "New York Yankees",
        "Oakland Athletics",
        "Philadelphia Phillies",
        "Pittsburgh Pirates",
        "San Diego Padres",
        "San Francisco Giants",
        "Seattle Mariners",
        "St. Louis Cardinals",
        "Tampa Bay Rays",
        "Texas Rangers",
        "Toronto Blue Jays",
        "Washington Nationals",
    ]

    def __init__(self, seed: int = 42) -> None:
        random.seed(seed)
        self._rng = np.random.default_rng(seed)

    def generate_seasons(self, years: list[int] | None = None) -> pd.DataFrame:
        """Generate synthetic games for specified years (default: 2023-2025)."""
        if years is None:
            years = [2023, 2024, 2025]

        all_records = []

        for year in years:
            # Generate ~2430 games (81 games × 30 teams / 1 = ~2430)
            games_per_team = 162
            (len(self.TEAMS) * games_per_team) // 2

            # Create team season stats (win rates, run averages)
            team_stats = self._generate_team_stats()

            # Generate games for the season
            season_games = self._generate_season_games(year, team_stats)
            all_records.extend(season_games)

        df = pd.DataFrame(all_records)
        df["game_date"] = pd.to_datetime(df["game_date"], utc=True)
        df = df.sort_values("game_date").reset_index(drop=True)
        return df

    def _generate_team_stats(self) -> dict[str, dict[str, float]]:
        """Generate per-team statistics (win rate, ERA, average runs)."""
        stats = {}
        for team in self.TEAMS:
            # Home teams typically have ~54% win rate, away teams ~46%
            home_win_rate = random.uniform(0.420, 0.580)
            away_win_rate = random.uniform(0.380, 0.540)
            runs_per_game = random.uniform(3.5, 5.5)
            pitcher_era = random.uniform(3.0, 4.5)
            pitcher_wins = random.randint(5, 18)

            stats[team] = {
                "home_win_rate": home_win_rate,
                "away_win_rate": away_win_rate,
                "runs_per_game": runs_per_game,
                "pitcher_era": pitcher_era,
                "pitcher_wins": pitcher_wins,
            }

        return stats

    def _generate_season_games(
        self, year: int, team_stats: dict[str, dict[str, float]]
    ) -> list[dict[str, object]]:
        """Generate games for a single season."""
        records = []
        start_date = datetime(year, 3, 28, tzinfo=UTC)  # Spring Opening Day
        end_date = datetime(year, 10, 2, tzinfo=UTC)  # End of season

        current_date = start_date
        game_count = 0

        # Generate realistic schedule: ~15 games per day across 180+ days
        while current_date <= end_date and game_count < 2430:
            # Each day has ~15 games (30 teams / 2)
            daily_games = random.randint(12, 16)
            day_matchups = set()

            for _ in range(daily_games):
                if game_count >= 2430:
                    break
                # Pick teams not already playing today
                available = [t for t in self.TEAMS if t not in day_matchups]
                if len(available) < 2:
                    break
                home_team = random.choice(available)
                day_matchups.add(home_team)
                available.remove(home_team)
                away_team = random.choice(available)
                day_matchups.add(away_team)

                # Generate game outcome
                home_stats = team_stats[home_team]
                away_stats = team_stats[away_team]

                # Generate per-inning scores using Poisson (realistic baseball distribution)
                home_inning_mu = home_stats["runs_per_game"] / 9.0
                away_inning_mu = away_stats["runs_per_game"] / 9.0
                home_innings = self._rng.poisson(home_inning_mu, size=9)
                away_innings = self._rng.poisson(away_inning_mu, size=9)

                home_score = int(home_innings.sum())
                away_score = int(away_innings.sum())

                # Ensure no ties (baseball plays extras)
                while home_score == away_score:
                    if random.random() < home_stats["home_win_rate"]:
                        home_score += 1
                    else:
                        away_score += 1

                # F5 scores: first 5 innings
                f5_home_score = int(home_innings[:5].sum())
                f5_away_score = int(away_innings[:5].sum())

                # Spread (realistic MLB line: typically -1.5 to +1.5)
                if home_stats["runs_per_game"] > away_stats["runs_per_game"]:
                    spread = round(random.uniform(-2.0, -0.5), 1)
                else:
                    spread = round(random.uniform(0.5, 2.0), 1)

                # Generate total line from team strengths (NOT actual score — avoids target leakage)
                expected_total = home_stats["runs_per_game"] + away_stats["runs_per_game"]
                total_line = round(expected_total + random.uniform(-1.0, 1.0), 1)

                # Generate moneylines reflecting team strength
                strength_diff = (home_stats["runs_per_game"] - away_stats["runs_per_game"])
                home_ml = int(-110 - strength_diff * 30)
                away_ml = int(-110 + strength_diff * 30)

                records.append(
                    {
                        "game_date": current_date.isoformat(),
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_score": home_score,
                        "away_score": away_score,
                        "f5_home_score": f5_home_score,
                        "f5_away_score": f5_away_score,
                        "spread": spread,
                        "home_moneyline": home_ml,
                        "away_moneyline": away_ml,
                        "total_runs": total_line,
                        "home_pitcher_era": home_stats["pitcher_era"],
                        "away_pitcher_era": away_stats["pitcher_era"],
                        "home_pitcher_wins": home_stats["pitcher_wins"],
                        "away_pitcher_wins": away_stats["pitcher_wins"],
                    }
                )
                game_count += 1

            # Advance one day (games every day during season)
            current_date += timedelta(days=1)

        return records


def enrich_current_data_with_history(
    current_df: pd.DataFrame, synthetic_years: list[int] | None = None
) -> pd.DataFrame:
    """Combine current season data with synthetic historical years."""
    if synthetic_years is None:
        synthetic_years = [2023, 2024, 2025]

    generator = SyntheticHistoryGenerator()
    synthetic_df = generator.generate_seasons(synthetic_years)

    # Combine
    combined = pd.concat([synthetic_df, current_df], ignore_index=True)
    combined["game_date"] = pd.to_datetime(combined["game_date"], utc=True)
    combined = combined.sort_values("game_date").reset_index(drop=True)

    return combined
