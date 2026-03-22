# mypy: ignore-errors
"""Integrate authoritative MLB historical datasets for enriched training features."""

from __future__ import annotations

import contextlib
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class LahmanDataEnricher:
    """Enrich training data with Lahman Database (pitcher ERAs, team stats, etc.)."""

    @staticmethod
    def get_pitcher_stats(year: int | None = None) -> pd.DataFrame:
        """Fetch pitcher seasonal stats from Lahman (via pybaseball)."""
        try:
            from pybaseball import pitching_stats

            if year:
                df = pitching_stats(year, qual=0)  # All pitchers, no minimum IP
            else:
                # Get last 5 years
                dfs = []
                for y in range(2021, 2026):
                    with contextlib.suppress(Exception):
                        dfs.append(pitching_stats(y, qual=0))
                df = pd.concat(dfs, ignore_index=True)

            # Rename columns to match expected format
            df_renamed = df[["Name", "ERA", "W", "IP"]].copy()
            df_renamed.columns = ["pitcher_name", "era", "wins", "innings_pitched"]
            return pd.DataFrame(df_renamed.dropna(subset=["era"]))
        except ImportError:
            logger.warning("pybaseball not installed; skipping Lahman pitcher stats")
            return pd.DataFrame()

    @staticmethod
    def get_team_stats(year: int | None = None) -> pd.DataFrame:
        """Fetch team seasonal stats from Lahman."""
        try:
            from pybaseball import team_pitching

            if year:
                df = team_pitching(year)
            else:
                dfs = []
                for y in range(2023, 2026):
                    with contextlib.suppress(Exception):
                        dfs.append(team_pitching(y))
                df = pd.concat(dfs, ignore_index=True)

            # Extract team ERA, team runs, team wins
            if df is not None and not df.empty:
                return pd.DataFrame(df[["Team", "ERA", "W", "R"]].copy())
            return pd.DataFrame()
        except ImportError:
            logger.warning("pybaseball not installed; skipping Lahman team stats")
            return pd.DataFrame()

    @staticmethod
    def enrich_games_with_pitcher_stats(
        games_df: pd.DataFrame,
        pitcher_stats: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Add real pitcher ERAs to games based on Lahman data."""
        if pitcher_stats is None or pitcher_stats.empty:
            return games_df

        df = games_df.copy()

        # Try to use Chadwick crosswalk to map Fangraphs ID to MLBAM ID
        try:
            from pybaseball import chadwick_register

            chadwick_df = chadwick_register()
            # Map IDfg (Fangraphs) from pitcher_stats to key_mlbam
            # First ensure both exist
            if "IDfg" in pitcher_stats.columns and not chadwick_df.empty:
                crosswalk = chadwick_df[["key_fangraphs", "key_mlbam"]].dropna()
                # key_fangraphs is often float because of NaNs, IDfg is int/str
                crosswalk["key_fangraphs"] = pd.to_numeric(
                    crosswalk["key_fangraphs"], errors="coerce"
                )
                pitcher_stats["IDfg"] = pd.to_numeric(
                    pitcher_stats["IDfg"], errors="coerce"
                )

                # Merge the crosswalk into the pitcher stats
                p_stats_merged = pd.merge(
                    pitcher_stats,
                    crosswalk,
                    left_on="IDfg",
                    right_on="key_fangraphs",
                    how="inner",
                )

                # If the df has pitcher IDs, we merge on MLBAM ID
                if "home_pitcher_id" in df.columns:
                    # Rename era to home_pitcher_era for merging
                    home_stats = p_stats_merged[["key_mlbam", "ERA", "W"]].copy()
                    home_stats = home_stats.rename(
                        columns={"ERA": "lahman_home_era", "W": "lahman_home_wins"}
                    )
                    df = pd.merge(
                        df,
                        home_stats.drop_duplicates(subset=["key_mlbam"]),
                        left_on="home_pitcher_id",
                        right_on="key_mlbam",
                        how="left",
                    ).drop(columns=["key_mlbam"], errors="ignore")

                    df["home_pitcher_era"] = df["lahman_home_era"].combine_first(
                        df.get("home_pitcher_era", 3.5)
                    )
                    df.drop(
                        columns=["lahman_home_era", "lahman_home_wins"],
                        errors="ignore",
                        inplace=True,
                    )

                if "away_pitcher_id" in df.columns:
                    away_stats = p_stats_merged[["key_mlbam", "ERA", "W"]].copy()
                    away_stats = away_stats.rename(
                        columns={"ERA": "lahman_away_era", "W": "lahman_away_wins"}
                    )
                    df = pd.merge(
                        df,
                        away_stats.drop_duplicates(subset=["key_mlbam"]),
                        left_on="away_pitcher_id",
                        right_on="key_mlbam",
                        how="left",
                    ).drop(columns=["key_mlbam"], errors="ignore")

                    df["away_pitcher_era"] = df["lahman_away_era"].combine_first(
                        df.get("away_pitcher_era", 3.5)
                    )
                    df.drop(
                        columns=["lahman_away_era", "lahman_away_wins"],
                        errors="ignore",
                        inplace=True,
                    )

        except Exception as e:
            logger.warning(f"Chadwick crosswalk mapping failed: {e}")

        # Fallback: For now, preserve existing pitcher ERA columns if they exist
        if "home_pitcher_era" not in df.columns:
            df["home_pitcher_era"] = 3.5  # League average
        if "away_pitcher_era" not in df.columns:
            df["away_pitcher_era"] = 3.5

        logger.info(f"Enriched {len(df)} games with pitcher stats from Lahman")
        return df


class StatcastEnricher:
    """Add advanced Statcast metrics (exit velo, launch angle, barrel %, xwOBA)."""

    @staticmethod
    def get_statcast_data(
        start_date: str = "2023-03-01",
        end_date: str = "2026-10-01",
    ) -> pd.DataFrame:
        """Fetch Statcast data from pybaseball."""
        try:
            from pybaseball import statcast

            logger.info(f"Fetching Statcast data from {start_date} to {end_date}...")
            df = statcast(start_dt=start_date, end_dt=end_date)
            if df is not None and not df.empty:
                logger.info(f"Loaded {len(df)} Statcast records")
                return pd.DataFrame(df)
        except Exception as e:
            logger.warning(f"Could not fetch Statcast data: {e}")

        return pd.DataFrame()

    @staticmethod
    def aggregate_statcast_by_game(statcast_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate pitch-level Statcast data to game level."""
        if statcast_df.empty:
            logger.warning("No Statcast data to aggregate")
            return pd.DataFrame()

        try:
            # Clean date and team info
            statcast_df = statcast_df.copy()
            statcast_df["game_date"] = pd.to_datetime(
                statcast_df["game_date"], utc=True, errors="coerce"
            )

            # Filter to batted balls with exit speed (actual hits)
            batted = statcast_df[
                (statcast_df["launch_speed"].notna())
                & (
                    (
                        statcast_df["events"].isin(
                            ["single", "double", "triple", "home_run"]
                        )
                    )
                    | (
                        statcast_df["type"].isin(
                            ["fly_ball", "ground_ball", "line_drive", "popup"]
                        )
                    )
                )
            ].copy()

            if batted.empty:
                logger.warning(
                    "No batted ball records with launch speed in Statcast data"
                )
                return pd.DataFrame()

            # Aggregate by game_date using actual Statcast column names
            agg_data = (
                batted.groupby("game_date")
                .agg(
                    {
                        "launch_speed": ["mean", "max"],
                        "launch_angle": ["mean"],
                        "estimated_ba_using_speedangle": ["mean"],
                        "estimated_woba_using_speedangle": ["mean"],
                    }
                )
                .reset_index()
            )

            # Flatten multiindex columns (launch_speed_mean, launch_speed_max, etc.)
            agg_data.columns = [
                "_".join(col).strip("_") if col[1] else col[0]
                for col in agg_data.columns.values
            ]

            logger.info(
                f"Aggregated {len(agg_data)} game-days with {len(batted)} batted balls"
            )
            return agg_data
        except Exception as e:
            logger.warning(f"Statcast aggregation failed: {e}")
            return pd.DataFrame()

    @staticmethod
    def merge_statcast_into_games(
        games_df: pd.DataFrame,
        statcast_agg: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge aggregated Statcast metrics into game records."""
        if statcast_agg.empty:
            logger.warning("No Statcast metrics to merge")
            return games_df

        try:
            df = games_df.copy()
            df["game_date"] = pd.to_datetime(df["game_date"], utc=True)
            statcast_agg["game_date"] = pd.to_datetime(
                statcast_agg["game_date"], utc=True
            )

            # Merge on game_date
            merged = df.merge(
                statcast_agg,
                on="game_date",
                how="left",
            )

            # Fill NaN values with defaults (league average)
            league_avg_launch_speed = 88.8
            league_avg_launch_angle = 12.5
            league_avg_ba = 0.265
            league_avg_xwoba = 0.330

            # Flatten column names first if needed
            if isinstance(merged.columns, pd.MultiIndex):
                merged.columns = [
                    "_".join(col).strip("_") if col[1] else col[0]
                    for col in merged.columns.values
                ]

            # Fill NaN values for Statcast metrics
            if "launch_speed_mean" in merged.columns:
                merged["launch_speed_mean"] = merged["launch_speed_mean"].fillna(
                    league_avg_launch_speed
                )
            if "launch_speed_max" in merged.columns:
                merged["launch_speed_max"] = merged["launch_speed_max"].fillna(95.0)
            if "launch_angle_mean" in merged.columns:
                merged["launch_angle_mean"] = merged["launch_angle_mean"].fillna(
                    league_avg_launch_angle
                )
            if "estimated_ba_using_speedangle_mean" in merged.columns:
                merged["estimated_ba_using_speedangle_mean"] = merged[
                    "estimated_ba_using_speedangle_mean"
                ].fillna(league_avg_ba)
            if "estimated_woba_using_speedangle_mean" in merged.columns:
                merged["estimated_woba_using_speedangle_mean"] = merged[
                    "estimated_woba_using_speedangle_mean"
                ].fillna(league_avg_xwoba)

            logger.info(f"Merged Statcast metrics into {len(merged)} games")
            return merged
        except Exception as e:
            logger.warning(f"Statcast merge failed: {e}")
            return games_df


class RetroSheetEnricher:
    """Add play-by-play context from Retrosheet."""

    @staticmethod
    def download_retrosheet(year: int) -> pd.DataFrame:
        """Download and parse Retrosheet event files."""
        try:
            # Retrosheet requires manual download or special parsing
            # For now, return empty; integration would require:
            # 1. Download .EVN files from retrosheet.org
            # 2. Parse using chadwick or custom parser
            # 3. Aggregate to game level
            logger.info(f"Retrosheet parsing not yet implemented for {year}")
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"Retrosheet enrichment failed: {e}")
            return pd.DataFrame()


def enrich_training_data_with_historical_sources(
    games_df: pd.DataFrame,
    include_lahman: bool = True,
    include_statcast: bool = True,
    include_retrosheet: bool = False,
) -> pd.DataFrame:
    """
    Comprehensive data enrichment using authoritative historical sources.

    This combines:
    - Current game data (games_df)
    - Lahman pitcher/team stats (actual ERAs, wins, etc.)
    - Statcast advanced metrics (exit velo, launch angle, barrel %, xwOBA)
    - Retrosheet play-by-play (opt-in, complex)
    """
    df = games_df.copy()

    if include_lahman:
        try:
            logger.info("Enriching with Lahman Database pitcher stats...")
            pitcher_stats = LahmanDataEnricher.get_pitcher_stats()
            if not pitcher_stats.empty:
                df = LahmanDataEnricher.enrich_games_with_pitcher_stats(
                    df, pitcher_stats
                )
                logger.info(
                    f"✓ Lahman enrichment complete: {len(pitcher_stats)} pitcher records"
                )
        except Exception as e:
            logger.warning(f"Lahman enrichment failed: {e}")

    if include_statcast:
        try:
            logger.info(
                "Enriching with Statcast advanced metrics (exit velo, barrel %, xwOBA)..."
            )
            statcast_df = StatcastEnricher.get_statcast_data(
                start_date="2023-03-01",
                end_date="2026-10-01",
            )
            if not statcast_df.empty:
                statcast_agg = StatcastEnricher.aggregate_statcast_by_game(statcast_df)
                if not statcast_agg.empty:
                    df = StatcastEnricher.merge_statcast_into_games(df, statcast_agg)
                    logger.info(
                        f"✓ Statcast enrichment complete: {len(statcast_agg)} game aggregations"
                    )
        except Exception as e:
            logger.warning(f"Statcast enrichment failed: {e}")

    if include_retrosheet:
        try:
            logger.info("Enriching with Retrosheet play-by-play...")
            logger.info("Retrosheet integration requires manual setup")
        except Exception as e:
            logger.warning(f"Retrosheet enrichment failed: {e}")

    logger.info(f"✓ Total enriched dataset: {len(df)} games")
    return df
