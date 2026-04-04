# mypy: ignore-errors
"""Integrate authoritative MLB historical datasets for enriched training features."""

from __future__ import annotations

import json
import logging
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd

from mlbv1.data.mapping import normalize_team

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
                    try:
                        dfs.append(pitching_stats(y, qual=0))
                    except (ValueError, KeyError, OSError) as exc:
                        logger.debug("Skipping Lahman pitcher stats for %d: %s", y, exc)
                if not dfs:
                    return pd.DataFrame()
                df = pd.concat(dfs, ignore_index=True)

            keep = [col for col in ["Name", "ERA", "W", "IP", "IDfg"] if col in df.columns]
            if not keep:
                return pd.DataFrame()
            out = df[keep].copy()
            if "IDfg" not in out.columns:
                out["IDfg"] = pd.NA
            return pd.DataFrame(out.dropna(subset=["ERA"]))
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
                    try:
                        dfs.append(team_pitching(y))
                    except (ValueError, KeyError, OSError) as exc:
                        logger.debug("Skipping Lahman team stats for %d: %s", y, exc)
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
        stats = pitcher_stats.copy()
        # Accept either canonical pybaseball columns or previously renamed aliases.
        if "ERA" not in stats.columns and "era" in stats.columns:
            stats["ERA"] = pd.to_numeric(stats["era"], errors="coerce")
        if "W" not in stats.columns and "wins" in stats.columns:
            stats["W"] = pd.to_numeric(stats["wins"], errors="coerce")
        if "Name" not in stats.columns and "pitcher_name" in stats.columns:
            stats["Name"] = stats["pitcher_name"]

        # Try to use Chadwick crosswalk to map Fangraphs ID to MLBAM ID
        try:
            from pybaseball import chadwick_register

            chadwick_df = chadwick_register()
            # Map IDfg (Fangraphs) from pitcher_stats to key_mlbam
            # First ensure both exist
            if "IDfg" in stats.columns and not chadwick_df.empty:
                crosswalk = chadwick_df[["key_fangraphs", "key_mlbam"]].dropna()
                # key_fangraphs is often float because of NaNs, IDfg is int/str
                crosswalk["key_fangraphs"] = pd.to_numeric(
                    crosswalk["key_fangraphs"], errors="coerce"
                )
                stats["IDfg"] = pd.to_numeric(
                    stats["IDfg"], errors="coerce"
                )

                # Merge the crosswalk into the pitcher stats
                p_stats_merged = pd.merge(
                    stats,
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

        except (ImportError, KeyError, ValueError, TypeError) as e:
            logger.warning("Chadwick crosswalk mapping failed: %s", e)

        # Fallback name-based merge if IDs were not available.
        if "Name" in stats.columns and "ERA" in stats.columns and "W" in stats.columns:
            name_stats = stats[["Name", "ERA", "W"]].dropna(subset=["Name"]).copy()
            if "home_pitcher_name" in df.columns:
                home_stats = name_stats.rename(
                    columns={
                        "Name": "home_pitcher_name",
                        "ERA": "lahman_home_era",
                        "W": "lahman_home_wins",
                    }
                )
                df = pd.merge(df, home_stats, on="home_pitcher_name", how="left")
                if "lahman_home_era" in df.columns:
                    df["home_pitcher_era"] = pd.to_numeric(
                        df["lahman_home_era"], errors="coerce"
                    ).combine_first(pd.to_numeric(df.get("home_pitcher_era"), errors="coerce"))
                if "lahman_home_wins" in df.columns:
                    df["home_pitcher_wins"] = pd.to_numeric(
                        df["lahman_home_wins"], errors="coerce"
                    ).combine_first(pd.to_numeric(df.get("home_pitcher_wins"), errors="coerce"))
                df.drop(
                    columns=["lahman_home_era", "lahman_home_wins"],
                    errors="ignore",
                    inplace=True,
                )

            if "away_pitcher_name" in df.columns:
                away_stats = name_stats.rename(
                    columns={
                        "Name": "away_pitcher_name",
                        "ERA": "lahman_away_era",
                        "W": "lahman_away_wins",
                    }
                )
                df = pd.merge(df, away_stats, on="away_pitcher_name", how="left")
                if "lahman_away_era" in df.columns:
                    df["away_pitcher_era"] = pd.to_numeric(
                        df["lahman_away_era"], errors="coerce"
                    ).combine_first(pd.to_numeric(df.get("away_pitcher_era"), errors="coerce"))
                if "lahman_away_wins" in df.columns:
                    df["away_pitcher_wins"] = pd.to_numeric(
                        df["lahman_away_wins"], errors="coerce"
                    ).combine_first(pd.to_numeric(df.get("away_pitcher_wins"), errors="coerce"))
                df.drop(
                    columns=["lahman_away_era", "lahman_away_wins"],
                    errors="ignore",
                    inplace=True,
                )

        # Fallback: For now, preserve existing pitcher ERA columns if they exist
        if "home_pitcher_era" not in df.columns:
            df["home_pitcher_era"] = 3.5  # League average
        if "away_pitcher_era" not in df.columns:
            df["away_pitcher_era"] = 3.5

        logger.info("Enriched %d games with pitcher stats from Lahman", len(df))
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

            logger.info("Fetching Statcast data from %s to %s...", start_date, end_date)
            df = statcast(start_dt=start_date, end_dt=end_date)
            if df is not None and not df.empty:
                logger.info("Loaded %d Statcast records", len(df))
                return pd.DataFrame(df)
        except (ImportError, ValueError, OSError) as e:
            logger.warning("Could not fetch Statcast data: %s", e)

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
                "Aggregated %d game-days with %d batted balls", len(agg_data), len(batted)
            )
            return agg_data
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("Statcast aggregation failed: %s", e)
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

            logger.info("Merged Statcast metrics into %d games", len(merged))
            return merged
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("Statcast merge failed: %s", e)
            return games_df


class RetroSheetEnricher:
    """Add play-by-play context from Retrosheet."""

    @staticmethod
    def download_retrosheet(year: int) -> pd.DataFrame:
        """Download and parse Retrosheet event files."""
        try:
            logger.info("Retrosheet parsing not yet implemented for %d", year)
            return pd.DataFrame()
        except (ImportError, OSError) as e:
            logger.warning("Retrosheet enrichment failed: %s", e)
            return pd.DataFrame()


class ProbablePitcherEnricher:
    """Fetch today's probable starting pitchers from MLB Stats API and merge
    their prior-season stats (ERA, wins) into games loaded from Odds API.

    The MLB Stats API ``/schedule`` endpoint with ``hydrate=probablePitcher``
    returns each game's announced starters including season stats.
    """

    MLB_API = "https://statsapi.mlb.com/api/v1"

    @classmethod
    def get_probable_pitchers(cls, date: str) -> pd.DataFrame:
        """Return a DataFrame with one row per game for *date* (YYYY-MM-DD).

        Columns: home_team, away_team, home_pitcher_name, away_pitcher_name,
                 home_pitcher_era, away_pitcher_era, home_pitcher_wins, away_pitcher_wins,
                 home_pitcher_id, away_pitcher_id
        """
        url = (
            f"{cls.MLB_API}/schedule?sportId=1&date={date}"
            "&hydrate=probablePitcher(note)"
        )
        try:
            with urlopen(url, timeout=15) as resp:  # noqa: S310
                data = json.loads(resp.read())
        except (URLError, OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not fetch probable pitchers for %s: %s", date, exc)
            return pd.DataFrame()

        rows: list[dict] = []
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                teams = game.get("teams", {})
                home_info = teams.get("home", {})
                away_info = teams.get("away", {})

                home_abbr = normalize_team(
                    home_info.get("team", {}).get("abbreviation", "")
                    or home_info.get("team", {}).get("name", "")
                )
                away_abbr = normalize_team(
                    away_info.get("team", {}).get("abbreviation", "")
                    or away_info.get("team", {}).get("name", "")
                )

                hp = home_info.get("probablePitcher", {})
                ap = away_info.get("probablePitcher", {})

                rows.append({
                    "home_team": home_abbr,
                    "away_team": away_abbr,
                    "home_pitcher_name": hp.get("fullName", ""),
                    "away_pitcher_name": ap.get("fullName", ""),
                    "home_pitcher_id": hp.get("id"),
                    "away_pitcher_id": ap.get("id"),
                })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Now look up each pitcher's prior-season stats via pybaseball
        df = cls._attach_pitcher_season_stats(df)
        return df

    @classmethod
    def _attach_pitcher_season_stats(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Look up prior-season ERA/wins for each pitcher via pybaseball."""
        try:
            from pybaseball import pitching_stats
        except ImportError:
            logger.warning("pybaseball not installed — can't look up pitcher stats")
            df["home_pitcher_era"] = 0.0
            df["away_pitcher_era"] = 0.0
            df["home_pitcher_wins"] = 0.0
            df["away_pitcher_wins"] = 0.0
            return df

        # Fetch last 2 seasons to find most recent stats
        all_stats = pd.DataFrame()
        for year in [2025, 2024]:
            try:
                season = pitching_stats(year, qual=0)
                if season is not None and not season.empty:
                    season = season[["Name", "ERA", "W", "IP"]].copy()
                    season["year"] = year
                    all_stats = pd.concat([all_stats, season], ignore_index=True)
            except Exception as exc:
                logger.debug("Could not fetch %d pitching stats: %s", year, exc)

        if all_stats.empty:
            logger.warning("No pitching stats available from pybaseball")
            df["home_pitcher_era"] = 0.0
            df["away_pitcher_era"] = 0.0
            df["home_pitcher_wins"] = 0.0
            df["away_pitcher_wins"] = 0.0
            return df

        # Keep most recent year per pitcher, prefer most innings
        all_stats = all_stats.sort_values(
            ["year", "IP"], ascending=[False, False]
        ).drop_duplicates(subset=["Name"], keep="first")

        # Build lookup by name
        stats_lookup = {}
        for _, row in all_stats.iterrows():
            name = str(row["Name"]).strip()
            stats_lookup[name] = {
                "era": float(row["ERA"]) if pd.notna(row["ERA"]) else 0.0,
                "wins": float(row["W"]) if pd.notna(row["W"]) else 0.0,
            }

        def _lookup(pitcher_name: str, stat: str) -> float:
            if not pitcher_name:
                return 0.0
            info = stats_lookup.get(pitcher_name)
            if info:
                return info[stat]
            # Try partial match (last name)
            last = pitcher_name.rsplit(" ", 1)[-1].lower()
            for name, _info in stats_lookup.items():
                if name.rsplit(" ", 1)[-1].lower() == last:
                    # Ambiguous — skip
                    candidates = [
                        n for n in stats_lookup if n.rsplit(" ", 1)[-1].lower() == last
                    ]
                    if len(candidates) == 1:
                        return stats_lookup[candidates[0]][stat]
                    break
            return 0.0

        df["home_pitcher_era"] = df["home_pitcher_name"].apply(
            lambda n: _lookup(n, "era")
        )
        df["away_pitcher_era"] = df["away_pitcher_name"].apply(
            lambda n: _lookup(n, "era")
        )
        df["home_pitcher_wins"] = df["home_pitcher_name"].apply(
            lambda n: _lookup(n, "wins")
        )
        df["away_pitcher_wins"] = df["away_pitcher_name"].apply(
            lambda n: _lookup(n, "wins")
        )

        matched_home = (df["home_pitcher_era"] > 0).sum()
        matched_away = (df["away_pitcher_era"] > 0).sum()
        logger.info(
            "Probable pitcher stats: %d/%d home, %d/%d away matched from pybaseball",
            matched_home, len(df), matched_away, len(df),
        )
        return df

    @classmethod
    def enrich_games_with_probable_pitchers(
        cls,
        games_df: pd.DataFrame,
        date: str,
    ) -> pd.DataFrame:
        """Merge probable pitcher stats into games by matching team abbreviations."""
        pitchers = cls.get_probable_pitchers(date)
        if pitchers.empty:
            logger.info("No probable pitchers available for %s", date)
            return games_df

        df = games_df.copy()
        pitcher_cols = [
            "home_pitcher_name", "away_pitcher_name",
            "home_pitcher_era", "away_pitcher_era",
            "home_pitcher_wins", "away_pitcher_wins",
            "home_pitcher_id", "away_pitcher_id",
        ]

        # Merge on (home_team, away_team) — both sides use normalized abbreviations
        merged = df.merge(
            pitchers[["home_team", "away_team"] + pitcher_cols],
            on=["home_team", "away_team"],
            how="left",
            suffixes=("_orig", ""),
        )

        # Use new pitcher data where available, keep originals as fallback
        for col in pitcher_cols:
            orig_col = f"{col}_orig"
            if orig_col in merged.columns:
                merged[col] = merged[col].combine_first(merged[orig_col])
                merged.drop(columns=[orig_col], inplace=True)
            elif col not in merged.columns:
                merged[col] = 0.0

        logger.info(
            "Enriched %d games with probable pitcher data for %s", len(merged), date
        )
        return merged


def enrich_training_data_with_historical_sources(
    games_df: pd.DataFrame,
    include_lahman: bool = True,
    include_statcast: bool = False,
    include_retrosheet: bool = False,
    include_probable_pitchers: bool = True,
    target_date: str | None = None,
) -> pd.DataFrame:
    """
    Comprehensive data enrichment using authoritative historical sources.

    This combines:
    - Current game data (games_df)
    - Probable pitchers from MLB Stats API + pybaseball season stats (ERA, wins)
    - Lahman pitcher/team stats (actual ERAs, wins, etc.)
    - Statcast advanced metrics (exit velo, launch angle, barrel %, xwOBA)
      NOTE: Statcast is OFF by default — fetching 3 years of pitch-by-pitch
      data takes 30+ minutes. Enable for offline training only.
    - Retrosheet play-by-play (opt-in, complex)
    """
    df = games_df.copy()

    # Probable pitchers — fastest, most impactful for today's predictions
    if include_probable_pitchers and target_date:
        try:
            logger.info("Enriching with probable pitcher stats for %s...", target_date)
            df = ProbablePitcherEnricher.enrich_games_with_probable_pitchers(
                df, target_date
            )
            logger.info("✓ Probable pitcher enrichment complete")
        except (URLError, OSError, KeyError, ValueError) as e:
            logger.warning("Probable pitcher enrichment failed: %s", e)

    if include_lahman:
        try:
            logger.info("Enriching with Lahman Database pitcher stats...")
            pitcher_stats = LahmanDataEnricher.get_pitcher_stats()
            if not pitcher_stats.empty:
                df = LahmanDataEnricher.enrich_games_with_pitcher_stats(
                    df, pitcher_stats
                )
                logger.info(
                    "Lahman enrichment complete: %d pitcher records", len(pitcher_stats)
                )
        except (ImportError, KeyError, ValueError, OSError) as e:
            logger.warning("Lahman enrichment failed: %s", e)

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
                        "Statcast enrichment complete: %d game aggregations",
                        len(statcast_agg),
                    )
        except (ImportError, KeyError, ValueError, OSError) as e:
            logger.warning("Statcast enrichment failed: %s", e)

    if include_retrosheet:
        try:
            logger.info("Enriching with Retrosheet play-by-play...")
            logger.info("Retrosheet integration requires manual setup")
        except (ImportError, OSError) as e:
            logger.warning("Retrosheet enrichment failed: %s", e)

    logger.info("Total enriched dataset: %d games", len(df))
    return df
