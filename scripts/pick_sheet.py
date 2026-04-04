"""Generate a full pick sheet for today's MLB slate.

Produces one row per (game × segment × market) with model probabilities,
no-vig EV, Kelly confidence, and sharp indicators.

Output:  artifacts/picks_{DATE}.csv   (flat, Excel/CSV-ready)
Console: pretty-printed table of recommended picks.

Usage:
    python scripts/pick_sheet.py
    python scripts/pick_sheet.py --date 2026-03-31
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ── project imports ──────────────────────────────────────────────────────
from mlbv1.config import AppConfig
from mlbv1.data.historical_enrichment import enrich_training_data_with_historical_sources
from mlbv1.data.loader import MLBStatsAPILoader, OddsAPILoader
from mlbv1.data.preprocessor import ProcessedData, preprocess
from mlbv1.data.slate_filter import (
    DEFAULT_SLATE_TIMEZONE,
    filter_pregame_games_for_date,
)
from mlbv1.features.engineer import FeatureSet, engineer_features
from mlbv1.models.predictor import PredictionResult, load_model, predict
from mlbv1.picks.odds_math import (
    american_to_decimal,
    implied_prob,
    kelly_fraction,
    no_vig_ev,
)
from mlbv1.picks.quality import (
    compute_quality_metrics,
    run_quality_gates,
    write_audit_artifact,
)
from mlbv1.picks.rationale import build_rationale
from mlbv1.tracking.database import TrackingDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════
# Core: Build pick rows
# ═════════════════════════════════════════════════════════════════════════

def _build_pick_rows(
    games_df: pd.DataFrame,
    model_results: dict[str, PredictionResult],
    today: str,
    *,
    run_id: str,
    generated_at_utc: str,
    source_provider: str = "odds_api",
    cold_start: bool = False,
    features_X: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Generate one row per (game × segment × market × side)."""
    rows: list[dict[str, Any]] = []

    # De-duplicate games  (Odds API returns multiple bookmaker entries)
    games_deduped = games_df.groupby(
        ["home_team", "away_team"], sort=False
    ).first().reset_index()
    model_names = sorted(model_results.keys())

    for gi, game in games_deduped.iterrows():
        home = game["home_team"]
        away = game["away_team"]
        matchup = f"{away} @ {home}"
        game_time = str(game["game_date"])
        source_event_id = str(game.get("event_id", ""))

        # Feature lookup for rationale enrichment
        feat: dict[str, float] | None = None
        if features_X is not None:
            try:
                feat = features_X.iloc[gi].to_dict() if gi < len(features_X) else None
            except (IndexError, KeyError):
                feat = None

        # ── Collect per-model probabilities and average them ─────────
        n_models = len(model_results)
        prob_accum: dict[str, list[float]] = {}

        for _model_name, result in model_results.items():
            idx = None
            for i in range(len(result.market_probabilities)):
                if result.market_probabilities.index[i] == gi:
                    idx = i
                    break
            if idx is None:
                idx = gi if gi < len(result.market_probabilities) else None
            if idx is None:
                continue

            mp = result.market_probabilities.iloc[idx]
            er = result.expected_runs.iloc[idx]

            for col in mp.index:
                prob_accum.setdefault(col, []).append(float(mp[col]))
            for col in er.index:
                prob_accum.setdefault(f"exp_{col}", []).append(float(er[col]))

        avg: dict[str, float] = {k: np.mean(v) for k, v in prob_accum.items()}

        # ── Market Data extraction ───────────────────────────────────
        def _odds(val: Any, fallback: float = float("nan")) -> float:
            try:
                v = float(val)
                if pd.isna(v):
                    return fallback
                return v
            except (TypeError, ValueError):
                return fallback

        # FG
        fg_spread_raw = _odds(game.get("spread"))
        fg_spread = -1.5 if pd.isna(fg_spread_raw) else fg_spread_raw
        fg_spread_defaulted = pd.isna(fg_spread_raw)
        fg_home_spread_odds = _odds(game.get("home_spread_odds"))
        fg_away_spread_odds = _odds(game.get("away_spread_odds"))
        fg_home_ml = _odds(game.get("home_moneyline"))
        fg_away_ml = _odds(game.get("away_moneyline"))
        fg_total_raw = _odds(game.get("total_runs"))
        fg_total = 8.5 if pd.isna(fg_total_raw) else fg_total_raw
        fg_total_defaulted = pd.isna(fg_total_raw)
        fg_over_odds = _odds(game.get("over_odds"))
        fg_under_odds = _odds(game.get("under_odds"))

        home_tt = _odds(game.get("home_tt"))
        away_tt = _odds(game.get("away_tt"))
        home_tt_over_odds = _odds(game.get("home_tt_over_odds"))
        home_tt_under_odds = _odds(game.get("home_tt_under_odds"))
        away_tt_over_odds = _odds(game.get("away_tt_over_odds"))
        away_tt_under_odds = _odds(game.get("away_tt_under_odds"))

        fg_has_odds = not (pd.isna(fg_home_ml) and pd.isna(fg_away_ml))

        # F5
        f5_spread_raw = _odds(game.get("f5_spread"))
        f5_spread = 0.0 if pd.isna(f5_spread_raw) else f5_spread_raw
        f5_spread_defaulted = pd.isna(f5_spread_raw)
        f5_home_spread_odds = _odds(game.get("f5_home_spread_odds"))
        f5_away_spread_odds = _odds(game.get("f5_away_spread_odds"))
        f5_home_ml = _odds(game.get("f5_home_moneyline"))
        f5_away_ml = _odds(game.get("f5_away_moneyline"))
        f5_total_raw = _odds(game.get("f5_total_runs"))
        f5_total = 0.0 if pd.isna(f5_total_raw) else f5_total_raw
        f5_total_defaulted = pd.isna(f5_total_raw)
        f5_over_odds = _odds(game.get("f5_over_odds"))
        f5_under_odds = _odds(game.get("f5_under_odds"))

        f5_has_data = any(
            not pd.isna(v) for v in [
                f5_home_ml, f5_away_ml, f5_over_odds, f5_under_odds,
                f5_home_spread_odds, f5_away_spread_odds,
            ]
        )
        f5_odds_quality = "live" if f5_has_data else "synthetic"

        exp_fg_home = avg.get("exp_home_score", 4.0)
        exp_fg_away = avg.get("exp_away_score", 4.0)
        exp_f5_home = avg.get("exp_f5_home_score", 2.0)
        exp_f5_away = avg.get("exp_f5_away_score", 2.0)

        # ── Build market dicts ───────────────────────────────────────
        segments = _build_market_segments(
            home, away, avg,
            fg_spread, fg_spread_defaulted, fg_home_spread_odds, fg_away_spread_odds,
            fg_home_ml, fg_away_ml, fg_total, fg_total_defaulted,
            fg_over_odds, fg_under_odds,
            home_tt, away_tt,
            home_tt_over_odds, home_tt_under_odds,
            away_tt_over_odds, away_tt_under_odds,
            f5_spread, f5_spread_defaulted, f5_home_spread_odds, f5_away_spread_odds,
            f5_home_ml, f5_away_ml, f5_total, f5_total_defaulted,
            f5_over_odds, f5_under_odds,
        )

        # ── Emit rows ───────────────────────────────────────────────
        for seg in segments:
            for mkt in seg["markets"]:
                if mkt["market_type"] == "Spread" and mkt.get("line") is not None and abs(mkt["line"]) > 3.5:
                    continue

                odds_cur = mkt["odds_current"]
                dec_odds = american_to_decimal(odds_cur)
                model_p = min(mkt["model_prob"], 0.90)

                ev = no_vig_ev(model_p, dec_odds)
                kelly_val = kelly_fraction(model_p, dec_odds)
                confidence = min(1.0, kelly_val * 4)

                if seg["segment"] == "F5":
                    quality = f5_odds_quality
                else:
                    quality = "live" if fg_has_odds else "no_odds"

                if quality != "live":
                    is_rec = False
                elif cold_start:
                    is_rec = ev > 0.05 and model_p > 0.55
                else:
                    is_rec = ev > 0.03 and model_p > 0.52

                if cold_start:
                    confidence *= 0.5

                # Line movement proxy
                matching_books = games_df[
                    (games_df["home_team"] == home) & (games_df["away_team"] == away)
                ]
                ml_values = pd.to_numeric(
                    matching_books["home_moneyline"], errors="coerce"
                ).dropna()
                line_move = f"{int(ml_values.min())} → {int(ml_values.max())}" if len(ml_values) > 1 else "—"

                exp_h = exp_f5_home if seg["segment"] == "F5" else exp_fg_home
                exp_a = exp_f5_away if seg["segment"] == "F5" else exp_fg_away
                rationale = build_rationale(
                    model_prob=model_p,
                    implied_prob_val=implied_prob(odds_cur),
                    ev=ev,
                    kelly=kelly_val,
                    confidence=confidence,
                    cold_start=cold_start,
                    n_models=n_models,
                    line_move=line_move,
                    exp_home=exp_h,
                    exp_away=exp_a,
                    segment=seg["segment"],
                    market_type=mkt["market_type"],
                    home=home,
                    away=away,
                    feat=feat,
                )

                rows.append({
                    "date": today,
                    "run_id": run_id,
                    "generated_at_utc": generated_at_utc,
                    "source_provider": source_provider,
                    "source_event_id": source_event_id,
                    "game": matchup,
                    "home_team": home,
                    "away_team": away,
                    "game_time": game_time,
                    "segment": seg["segment"],
                    "market_type": mkt["market_type"],
                    "pick": mkt["pick"],
                    "odds_current": int(odds_cur),
                    "odds_quality": quality,
                    "line_move": line_move,
                    "steam_flag": False,
                    "rlm_flag": False,
                    "bets_pct": "N/A",
                    "money_pct": "N/A",
                    "model_prob": round(model_p, 4),
                    "implied_prob": round(implied_prob(odds_cur), 4),
                    "no_vig_ev": round(ev, 4),
                    "kelly": round(kelly_val, 4),
                    "confidence": round(confidence, 4),
                    "is_recommended": is_rec,
                    "cold_start": cold_start,
                    "model_count": n_models,
                    "model_names": ",".join(model_names),
                    "used_default_line": bool(mkt.get("line_defaulted", False)),
                    "used_default_counter_odds": bool(mkt.get("counter_defaulted", False)),
                    "exp_home_score": round(exp_h, 2),
                    "exp_away_score": round(exp_a, 2),
                    "rationale_bullets": rationale,
                })

    return rows


def _mkt(  # noqa: PLR0913 — intentionally wide to avoid intermediate dicts
    market_type: str, pick: str, odds: float, model_prob: float,
    counter_odds: float, line: float | None = None,
    line_defaulted: bool = False, counter_defaulted: bool = False,
) -> dict[str, Any]:
    """Shorthand to build one market dict."""
    return {
        "market_type": market_type,
        "pick": pick,
        "odds_current": odds,
        "model_prob": model_prob,
        "counter_odds": counter_odds if not pd.isna(counter_odds) else odds,
        "line": line,
        "line_defaulted": line_defaulted,
        "counter_defaulted": pd.isna(counter_odds) if counter_defaulted is False else counter_defaulted,
    }


def _build_market_segments(  # noqa: PLR0913
    home: str, away: str, avg: dict[str, float],
    fg_spread: float, fg_spread_defaulted: bool,
    fg_home_spread_odds: float, fg_away_spread_odds: float,
    fg_home_ml: float, fg_away_ml: float,
    fg_total: float, fg_total_defaulted: bool,
    fg_over_odds: float, fg_under_odds: float,
    home_tt: float, away_tt: float,
    home_tt_over_odds: float, home_tt_under_odds: float,
    away_tt_over_odds: float, away_tt_under_odds: float,
    f5_spread: float, f5_spread_defaulted: bool,
    f5_home_spread_odds: float, f5_away_spread_odds: float,
    f5_home_ml: float, f5_away_ml: float,
    f5_total: float, f5_total_defaulted: bool,
    f5_over_odds: float, f5_under_odds: float,
) -> list[dict[str, Any]]:
    """Build FG and F5 market segment dicts from raw odds data."""
    segments: list[dict[str, Any]] = []

    # ── FG Markets ──
    fg_markets: list[dict[str, Any]] = []

    if not pd.isna(fg_home_spread_odds):
        fg_markets.append(_mkt("Spread", f"{home} {fg_spread:+.1f}", fg_home_spread_odds,
                               avg.get("home_spread_cover_prob", 0.5), fg_away_spread_odds,
                               fg_spread, fg_spread_defaulted, pd.isna(fg_away_spread_odds)))
    if not pd.isna(fg_away_spread_odds):
        fg_markets.append(_mkt("Spread", f"{away} {-fg_spread:+.1f}", fg_away_spread_odds,
                               1.0 - avg.get("home_spread_cover_prob", 0.5), fg_home_spread_odds,
                               -fg_spread, fg_spread_defaulted, pd.isna(fg_home_spread_odds)))
    if not pd.isna(fg_home_ml):
        fg_markets.append(_mkt("ML", f"{home} ML", fg_home_ml,
                               avg.get("home_ml_prob", 0.5), fg_away_ml))
    if not pd.isna(fg_away_ml):
        fg_markets.append(_mkt("ML", f"{away} ML", fg_away_ml,
                               avg.get("away_ml_prob", 0.5), fg_home_ml))
    if not pd.isna(fg_over_odds):
        fg_markets.append(_mkt("Total", f"Over {fg_total}", fg_over_odds,
                               avg.get("over_total_prob", 0.5), fg_under_odds,
                               fg_total, fg_total_defaulted, pd.isna(fg_under_odds)))
    if not pd.isna(fg_under_odds):
        fg_markets.append(_mkt("Total", f"Under {fg_total}", fg_under_odds,
                               1.0 - avg.get("over_total_prob", 0.5), fg_over_odds,
                               fg_total, fg_total_defaulted, pd.isna(fg_over_odds)))

    # Team Totals
    if not pd.isna(home_tt_over_odds) and not pd.isna(home_tt):
        fg_markets.append(_mkt("Team Total", f"{home} Over {home_tt}", home_tt_over_odds,
                               avg.get("home_tt_over_prob", 0.5), home_tt_under_odds, home_tt))
    if not pd.isna(home_tt_under_odds) and not pd.isna(home_tt):
        fg_markets.append(_mkt("Team Total", f"{home} Under {home_tt}", home_tt_under_odds,
                               1.0 - avg.get("home_tt_over_prob", 0.5), home_tt_over_odds, home_tt))
    if not pd.isna(away_tt_over_odds) and not pd.isna(away_tt):
        fg_markets.append(_mkt("Team Total", f"{away} Over {away_tt}", away_tt_over_odds,
                               avg.get("away_tt_over_prob", 0.5), away_tt_under_odds, away_tt))
    if not pd.isna(away_tt_under_odds) and not pd.isna(away_tt):
        fg_markets.append(_mkt("Team Total", f"{away} Under {away_tt}", away_tt_under_odds,
                               1.0 - avg.get("away_tt_over_prob", 0.5), away_tt_over_odds, away_tt))

    if fg_markets:
        segments.append({"segment": "FG", "markets": fg_markets})

    # ── F5 Markets ──
    f5_markets: list[dict[str, Any]] = []
    f5_spread_display = f5_spread if f5_spread != 0 else -0.5

    if not pd.isna(f5_home_spread_odds):
        f5_markets.append(_mkt("Spread", f"{home} {f5_spread_display:+.1f}", f5_home_spread_odds,
                               avg.get("f5_home_spread_cover_prob", 0.5), f5_away_spread_odds,
                               f5_spread_display, f5_spread_defaulted or f5_spread == 0.0,
                               pd.isna(f5_away_spread_odds)))
    if not pd.isna(f5_away_spread_odds):
        f5_markets.append(_mkt("Spread", f"{away} {-f5_spread_display:+.1f}", f5_away_spread_odds,
                               1.0 - avg.get("f5_home_spread_cover_prob", 0.5), f5_home_spread_odds,
                               -f5_spread_display, f5_spread_defaulted or f5_spread == 0.0,
                               pd.isna(f5_home_spread_odds)))
    if not pd.isna(f5_home_ml):
        f5_markets.append(_mkt("ML", f"{home} F5 ML", f5_home_ml,
                               avg.get("f5_home_ml_prob", 0.5), f5_away_ml))
    if not pd.isna(f5_away_ml):
        f5_markets.append(_mkt("ML", f"{away} F5 ML", f5_away_ml,
                               avg.get("f5_away_ml_prob", 0.5), f5_home_ml))

    f5_total_line = f5_total if f5_total > 0 else 4.5
    if not pd.isna(f5_over_odds):
        f5_markets.append(_mkt("Total", f"F5 Over {f5_total_line}", f5_over_odds,
                               avg.get("f5_over_total_prob", 0.5), f5_under_odds,
                               f5_total_line, f5_total_defaulted or f5_total <= 0,
                               pd.isna(f5_under_odds)))
    if not pd.isna(f5_under_odds):
        f5_markets.append(_mkt("Total", f"F5 Under {f5_total_line}", f5_under_odds,
                               1.0 - avg.get("f5_over_total_prob", 0.5), f5_over_odds,
                               f5_total_line, f5_total_defaulted or f5_total <= 0,
                               pd.isna(f5_over_odds)))

    if f5_markets:
        segments.append({"segment": "F5", "markets": f5_markets})

    return segments


def _prepare_live_features_with_history(
    model_input: pd.DataFrame,
    config: AppConfig,
) -> tuple[ProcessedData, FeatureSet, int]:
    """Build live inference features with recent historical context rows."""
    history_days = int(os.getenv("LIVE_CONTEXT_DAYS", "120"))
    live = model_input.copy().reset_index(drop=True)
    live["_is_live"] = 1

    hist = pd.DataFrame()
    end = datetime.now(tz=UTC)
    start = end - pd.Timedelta(days=history_days)
    try:
        hist_loader = MLBStatsAPILoader(
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
        )
        hist = hist_loader.load()
    except Exception as exc:
        logger.warning("Historical context load failed; using live-only features: %s", exc)

    hist_rows = 0
    if not hist.empty:
        hist = hist.copy()
        hist_rows = len(hist)
        hist["_is_live"] = 0
        for col in live.columns:
            if col not in hist.columns:
                hist[col] = pd.NA
        for col in hist.columns:
            if col not in live.columns:
                live[col] = pd.NA
        combined = pd.concat([hist, live], ignore_index=True, sort=False)
    else:
        combined = live

    processed_all = preprocess(combined)
    features_all = engineer_features(
        processed_all.features,
        short_window=config.features.rolling_window_short,
        long_window=config.features.rolling_window_long,
    )

    live_mask = (
        processed_all.features["_is_live"].fillna(0).astype(int) == 1
        if "_is_live" in processed_all.features.columns
        else pd.Series([True] * len(processed_all.features), index=processed_all.features.index)
    )

    live_features_df = (
        processed_all.features.loc[live_mask]
        .drop(columns=["_is_live"], errors="ignore")
        .reset_index(drop=True)
    )
    live_target_df = processed_all.target.loc[live_mask].reset_index(drop=True)
    live_metadata_df = processed_all.metadata.loc[live_mask].reset_index(drop=True)
    live_targets_df = (
        processed_all.targets.loc[live_mask].reset_index(drop=True)
        if processed_all.targets is not None
        else None
    )
    live_X = features_all.X.loc[live_mask].reset_index(drop=True)

    processed_live = ProcessedData(
        features=live_features_df,
        target=live_target_df,
        metadata=live_metadata_df,
        targets=live_targets_df,
    )
    feature_live = FeatureSet(X=live_X, feature_names=list(live_X.columns))
    return processed_live, feature_live, hist_rows


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="MLB Pick Sheet Generator")
    parser.add_argument("--date", type=str, default=None, help="Date (YYYY-MM-DD)")
    args = parser.parse_args()

    slate_timezone = os.getenv("SLATE_TIMEZONE", DEFAULT_SLATE_TIMEZONE)
    try:
        now_local = datetime.now(tz=ZoneInfo(slate_timezone))
    except Exception:
        logger.warning("Invalid SLATE_TIMEZONE '%s'; defaulting to UTC", slate_timezone)
        now_local = datetime.now(tz=UTC)
    today = args.date or now_local.strftime("%Y-%m-%d")
    run_id = f"slate-{today}-{uuid.uuid4().hex[:8]}"
    generated_at_utc = datetime.now(tz=UTC).isoformat()
    logger.info("Generating MLB pick sheet for %s", today)

    # ── 1. Load config & today's odds ────────────────────────────────
    config = AppConfig.load()
    config = config.override(data={"loader": "odds_api"})
    loader = OddsAPILoader(api_key=config.data.api_key, base_url=config.data.api_base_url)
    raw_games_df = loader.load()
    logger.info("Loaded %d raw game/bookmaker rows", len(raw_games_df))

    games_df, filter_stats = filter_pregame_games_for_date(
        raw_games_df, target_date=today, timezone_name=slate_timezone,
    )
    logger.info(
        "Pregame filter (%s): kept %d/%d rows for %s (started=%d, off_date=%d, invalid_time=%d)",
        filter_stats["timezone"], filter_stats["kept_rows"], filter_stats["input_rows"],
        filter_stats["target_date"], filter_stats["started_rows"],
        filter_stats["off_date_rows"], filter_stats["invalid_time_rows"],
    )

    if games_df.empty:
        logger.warning("No not-started games found for %s", today)
        return

    db = TrackingDB("artifacts/tracking.db")
    logging.basicConfig(level=logging.INFO, force=True)
    logger.disabled = False

    # ── 2. Preprocess & engineer features ────────────────────────────
    model_input = games_df.groupby(
        ["home_team", "away_team"], sort=False
    ).first().reset_index()

    logger.info("Enriching with historical data (probable pitchers + Lahman)...")
    model_input = enrich_training_data_with_historical_sources(
        model_input, include_lahman=True, include_statcast=False,
        include_probable_pitchers=True, target_date=today,
    )

    processed, features, hist_rows = _prepare_live_features_with_history(model_input, config)
    logger.info("Engineered live features using %d contextual historical games", hist_rows)

    # Cold-start detection
    n_total = features.X.shape[1]
    n_constant = sum(1 for c in features.X.columns if features.X[c].nunique() <= 1)
    pct_constant = n_constant / n_total if n_total else 0
    cold_start = pct_constant > 0.5

    if cold_start:
        logger.warning(
            "COLD START: %d of %d features (%.0f%%) are constant — "
            "odds-driven predictions only, stricter thresholds applied.",
            n_constant, n_total, pct_constant * 100,
        )

    # ── 3. Load all models & predict ─────────────────────────────────
    model_dir = Path("artifacts/models")
    model_paths = sorted(model_dir.glob("*.pkl"))
    skip_stems = {"stacking_ensemble", "voting_ensemble", "logistic_regression"}
    model_paths = [p for p in model_paths if p.stem not in skip_stems]
    logger.info("Found %d models: %s", len(model_paths), [p.stem for p in model_paths])

    model_results: dict[str, PredictionResult] = {}
    for mp in model_paths:
        try:
            model = load_model(str(mp))
            result = predict(model, features.X, lines=model_input)
            model_results[model.name] = result
            logger.info("  %s: predicted %d games", model.name, len(result.expected_runs))
        except Exception as exc:
            logger.warning("  %s FAILED: %s", mp.stem, exc)

    if not model_results:
        logger.error("No models produced predictions — aborting")
        return

    # ── 4. Build pick rows ───────────────────────────────────────────
    pick_rows = _build_pick_rows(
        model_input, model_results, today,
        run_id=run_id, generated_at_utc=generated_at_utc,
        source_provider="odds_api", cold_start=cold_start, features_X=features.X,
    )
    logger.info("Generated %d pick rows across all games/segments/markets", len(pick_rows))

    # ── 5. Quality gates + audit + canonical publish ─────────────────
    quality_metrics = compute_quality_metrics(pick_rows)
    quality_passed, gates = run_quality_gates(quality_metrics)
    audit = {
        "slate_date": today,
        "run_id": run_id,
        "generated_at_utc": generated_at_utc,
        "source_provider": "odds_api",
        "slate_timezone": filter_stats.get("timezone"),
        "pregame_filter": filter_stats,
        "quality_passed": quality_passed,
        "quality_metrics": quality_metrics,
        "quality_gates": gates,
        "row_count": len(pick_rows),
        "game_count": len({row.get("game") for row in pick_rows}),
        "model_names": sorted(model_results.keys()),
        "cold_start": cold_start,
    }
    out_dir = Path("artifacts")
    audit_path = out_dir / f"data_audit_{today}.json"
    write_audit_artifact(audit_path, audit)

    if not quality_passed:
        failed = [g["name"] for g in gates if not g["passed"]]
        raise RuntimeError(
            "Slate quality gates failed: " + ", ".join(failed) + f" (see {audit_path})"
        )

    checksum = db.publish_slate(today, run_id, pick_rows, audit, status="published")
    audit["checksum"] = checksum
    write_audit_artifact(audit_path, audit)
    logger.info("Canonical slate published to DB for %s (checksum=%s)", today, checksum[:12])

    # ── 6. Write CSV ─────────────────────────────────────────────────
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"picks_{today}.csv"
    fieldnames = [
        "date", "run_id", "generated_at_utc", "source_provider", "source_event_id",
        "game", "home_team", "away_team", "game_time",
        "segment", "market_type", "pick",
        "odds_current", "odds_quality", "line_move",
        "steam_flag", "rlm_flag", "bets_pct", "money_pct",
        "model_prob", "implied_prob", "no_vig_ev", "kelly", "confidence",
        "is_recommended", "cold_start", "model_count", "model_names",
        "used_default_line", "used_default_counter_odds",
        "exp_home_score", "exp_away_score",
        "rationale_bullets",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pick_rows)
    logger.info("Wrote %s (%d rows)", csv_path, len(pick_rows))

    # ── 7. Console summary ───────────────────────────────────────────
    recs = [r for r in pick_rows if r["is_recommended"]]
    logger.info("")
    logger.info("═" * 100)
    if cold_start and recs:
        logger.info("  ⚠ COLD START — %d/%d features constant. %d picks passed stricter thresholds.", n_constant, n_total, len(recs))
        logger.info("  Odds-driven only. Confidence discounted 50%%.")
    elif cold_start:
        logger.info("  COLD START — %d/%d features constant. No picks passed stricter thresholds (EV>5%%, prob>55%%).", n_constant, n_total)
    else:
        logger.info("  RECOMMENDED PICKS FOR %s  (%d of %d)", today, len(recs), len(pick_rows))
    logger.info("═" * 100)

    if not recs:
        logger.info("  No picks met threshold (EV > 3%% AND model_prob > 52%%)")
        logger.info("  Review full sheet: %s", csv_path)
    else:
        header = f"{'Game':<22} {'Seg':>3} {'Market':<12} {'Pick':<20} {'Odds':>6} {'Qual':<5} {'Model%':>7} {'Impl%':>7} {'EV':>7} {'Kelly':>7} {'Conf':>6}"
        logger.info(header)
        logger.info("-" * len(header))
        for r in sorted(recs, key=lambda x: -x["no_vig_ev"]):
            logger.info(
                f"{r['game']:<22} {r['segment']:>3} {r['market_type']:<12} "
                f"{r['pick']:<20} {r['odds_current']:>+6d} "
                f"{r['odds_quality']:<5} "
                f"{r['model_prob']:>6.1%} {r['implied_prob']:>6.1%} "
                f"{r['no_vig_ev']:>+6.1%} {r['kelly']:>6.2%} {r['confidence']:>5.2f}"
            )
        logger.info("")
        for r in sorted(recs, key=lambda x: -x["no_vig_ev"])[:10]:
            logger.info("  ▸ %s — %s", r["pick"], r["rationale_bullets"])

    if quality_metrics.get("diversification_warning"):
        direction = quality_metrics.get("dominant_direction", "unknown")
        breakdown = quality_metrics.get("rec_direction_breakdown", {})
        logger.warning("")
        logger.warning(
            "  ⚠ DIVERSIFICATION WARNING: %d of %d total/TT recs are %s-biased. "
            "Check for systematic model bias.",
            breakdown.get("under_recs", 0) if direction == "under" else breakdown.get("over_recs", 0),
            breakdown.get("over_recs", 0) + breakdown.get("under_recs", 0),
            direction,
        )

    logger.info("")
    logger.info("Full sheet: %s", csv_path)


if __name__ == "__main__":
    main()
