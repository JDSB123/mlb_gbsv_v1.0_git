"""Synthetic historical MLB data generator for enriching training sets."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

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

    def generate_seasons(self, years: list[int] = None) -> pd.DataFrame:
        """Generate synthetic games for specified years (default: 2023-2025)."""
        if years is None:
            years = [2023, 2024, 2025]

        all_records = []

        for year in years:
            # Generate ~2430 games (81 games × 30 teams / 1 = ~2430)
            games_per_team = 162
            total_games = (len(self.TEAMS) * games_per_team) // 2

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
    ) -> list[dict]:
        """Generate games for a single season."""
        records = []
        start_date = datetime(year, 3, 28, tzinfo=UTC)  # Spring Opening Day
        end_date = datetime(year, 10, 2, tzinfo=UTC)  # End of season

        current_date = start_date
        game_count = 0

        # Simple round-robin schedule
        pairs_used = set()

        while current_date <= end_date and game_count < 2430:
            # Randomly select a matchup
            home_team = random.choice(self.TEAMS)
            away_team = random.choice([t for t in self.TEAMS if t != home_team])
            pair_key = (home_team, away_team, current_date.date())

            if pair_key not in pairs_used:
                pairs_used.add(pair_key)

                # Generate game outcome
                home_stats = team_stats[home_team]
                away_stats = team_stats[away_team]

                # Simulate home win with home team's advantage
                home_win_prob = home_stats["home_win_rate"]
                home_wins = random.random() < home_win_prob

                # Generate scores
                home_score = self._generate_score(
                    home_stats["runs_per_game"], is_winner=home_wins
                )
                away_score = self._generate_score(
                    away_stats["runs_per_game"], is_winner=not home_wins
                )

                # Spread (typically -110 both ways; home favorite bias)
                spread = random.uniform(-2.5, 1.5)

                records.append(
                    {
                        "game_date": current_date.isoformat(),
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_score": home_score,
                        "away_score": away_score,
                        "spread": spread,
                        "home_moneyline": -110,
                        "away_moneyline": -110,
                        "home_pitcher_era": home_stats["pitcher_era"],
                        "away_pitcher_era": away_stats["pitcher_era"],
                        "home_pitcher_wins": home_stats["pitcher_wins"],
                        "away_pitcher_wins": away_stats["pitcher_wins"],
                    }
                )
                game_count += 1

            # Advance date (games every 1-2 days)
            current_date += timedelta(
                days=random.choice([1, 1, 1, 2])
            )  # More games on some days

        return records

    @staticmethod
    def _generate_score(runs_per_game_avg: float, is_winner: bool = False) -> int:
        """Generate a realistic baseball score."""
        # Poisson-like distribution (MLB avg ~4.3 runs/game)
        base_runs = max(0, int(random.gauss(runs_per_game_avg, 2.0)))

        # Winners typically score slightly more
        if is_winner:
            base_runs += random.choice([0, 1, 1])

        return max(0, base_runs)


def enrich_current_data_with_history(
    current_df: pd.DataFrame, synthetic_years: list[int] = None
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
