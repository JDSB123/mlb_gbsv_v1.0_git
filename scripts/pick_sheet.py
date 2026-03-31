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
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam

# ── project imports ──────────────────────────────────────────────────────
from mlbv1.config import AppConfig
from mlbv1.data.loader import OddsAPILoader
from mlbv1.data.preprocessor import preprocess
from mlbv1.features.engineer import engineer_features
from mlbv1.models.market_deriver import MarketDeriver
from mlbv1.models.predictor import PredictionResult, load_model, predict
from mlbv1.tracking.database import TrackingDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════

def _american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds."""
    if odds >= 100:
        return 1 + odds / 100
    elif odds <= -100:
        return 1 + 100 / abs(odds)
    return 2.0  # fallback for zeros


def _implied_prob(odds: float) -> float:
    """Implied probability from American odds (includes vig)."""
    if odds >= 100:
        return 100 / (odds + 100)
    elif odds <= -100:
        return abs(odds) / (abs(odds) + 100)
    return 0.5


def _no_vig_prob(side_odds: float, other_odds: float) -> float:
    """Remove the vig and return the 'true' implied probability for the side bet."""
    p_side = _implied_prob(side_odds)
    p_other = _implied_prob(other_odds)
    total = p_side + p_other
    if total == 0:
        return 0.5
    return p_side / total


def _no_vig_ev(model_prob: float, decimal_odds: float) -> float:
    """Expected value given model probability and decimal odds."""
    return model_prob * decimal_odds - 1.0


def _kelly_fraction(model_prob: float, decimal_odds: float) -> float:
    """Full Kelly criterion fraction (capped at 0)."""
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    f = (model_prob * b - (1 - model_prob)) / b
    return max(0.0, f)


# ═════════════════════════════════════════════════════════════════════════
# Core: Build pick rows
# ═════════════════════════════════════════════════════════════════════════

def _build_pick_rows(
    games_df: pd.DataFrame,
    model_results: dict[str, PredictionResult],
    today: str,
    *,
    cold_start: bool = False,
) -> list[dict[str, Any]]:
    """Generate one row per (game × segment × market × side)."""
    rows: list[dict[str, Any]] = []

    # De-duplicate games  (Odds API returns multiple bookmaker entries)
    games_deduped = games_df.groupby(
        ["home_team", "away_team"], sort=False
    ).first().reset_index()

    for gi, game in games_deduped.iterrows():
        home = game["home_team"]
        away = game["away_team"]
        matchup = f"{away} @ {home}"
        game_time = str(game["game_date"])

        # ── Collect per-model probabilities and average them ─────────
        n_models = len(model_results)
        prob_accum: dict[str, list[float]] = {}

        for model_name, result in model_results.items():
            # find matching index
            idx = None
            for i in range(len(result.market_probabilities)):
                if result.market_probabilities.index[i] == gi:
                    idx = i
                    break
            if idx is None:
                # Try matching by same positional order
                idx = gi if gi < len(result.market_probabilities) else None
            if idx is None:
                continue

            mp = result.market_probabilities.iloc[idx]
            er = result.expected_runs.iloc[idx]

            for col in mp.index:
                prob_accum.setdefault(col, []).append(float(mp[col]))

            # Also store expected runs
            for col in er.index:
                prob_accum.setdefault(f"exp_{col}", []).append(float(er[col]))

        # Average across models
        avg: dict[str, float] = {k: np.mean(v) for k, v in prob_accum.items()}

        # ── FG Market Data ───────────────────────────────────────────
        # Use NaN-aware reads: if API didn't return data, the field is NaN.
        def _odds(val: Any, fallback: float = float("nan")) -> float:
            try:
                v = float(val)
                if pd.isna(v):
                    return fallback
                return v
            except (TypeError, ValueError):
                return fallback

        fg_spread = _odds(game.get("spread"), -1.5)
        fg_home_spread_odds = _odds(game.get("home_spread_odds"))
        fg_away_spread_odds = _odds(game.get("away_spread_odds"))
        fg_home_ml = _odds(game.get("home_moneyline"))
        fg_away_ml = _odds(game.get("away_moneyline"))
        fg_total = _odds(game.get("total_runs"), 8.5)
        fg_over_odds = _odds(game.get("over_odds"))
        fg_under_odds = _odds(game.get("under_odds"))

        # Team Total data (premium)
        home_tt = _odds(game.get("home_tt"))
        away_tt = _odds(game.get("away_tt"))
        home_tt_over_odds = _odds(game.get("home_tt_over_odds"))
        home_tt_under_odds = _odds(game.get("home_tt_under_odds"))
        away_tt_over_odds = _odds(game.get("away_tt_over_odds"))
        away_tt_under_odds = _odds(game.get("away_tt_under_odds"))

        # FG odds quality: "live" if we have real ML odds, else "no_odds"
        fg_has_odds = not (pd.isna(fg_home_ml) and pd.isna(fg_away_ml))

        # F5 Market Data
        f5_spread = _odds(game.get("f5_spread"), 0.0)
        f5_home_spread_odds = _odds(game.get("f5_home_spread_odds"))
        f5_away_spread_odds = _odds(game.get("f5_away_spread_odds"))
        f5_home_ml = _odds(game.get("f5_home_moneyline"))
        f5_away_ml = _odds(game.get("f5_away_moneyline"))
        f5_total = _odds(game.get("f5_total_runs"), 0.0)
        f5_over_odds = _odds(game.get("f5_over_odds"))
        f5_under_odds = _odds(game.get("f5_under_odds"))

        # F5 quality: "live" if ANY F5 odds field is a real number
        f5_has_data = any(
            not pd.isna(v) for v in [
                f5_home_ml, f5_away_ml, f5_over_odds, f5_under_odds,
                f5_home_spread_odds, f5_away_spread_odds,
            ]
        )
        f5_odds_quality = "live" if f5_has_data else "synthetic"

        # Expected runs from models
        exp_fg_home = avg.get("exp_home_score", 4.0)
        exp_fg_away = avg.get("exp_away_score", 4.0)
        exp_f5_home = avg.get("exp_f5_home_score", 2.0)
        exp_f5_away = avg.get("exp_f5_away_score", 2.0)

        # ── Define all markets ───────────────────────────────────────
        # Only include markets where we have REAL odds from a sportsbook.
        # No Team Totals — no API provides TT odds/lines.
        segments = []

        # ── FG (Full Game) ───────────────────────────────────────────
        fg_markets: list[dict[str, Any]] = []

        if not pd.isna(fg_home_spread_odds):
            fg_markets.append({
                "market_type": "Spread",
                "pick": f"{home} {fg_spread:+.1f}",
                "odds_current": fg_home_spread_odds,
                "model_prob": avg.get("home_spread_cover_prob", 0.5),
                "counter_odds": fg_away_spread_odds if not pd.isna(fg_away_spread_odds) else fg_home_spread_odds,
                "line": fg_spread,
            })

        if not pd.isna(fg_home_ml):
            fg_markets.append({
                "market_type": "ML",
                "pick": f"{home} ML",
                "odds_current": fg_home_ml,
                "model_prob": avg.get("home_ml_prob", 0.5),
                "counter_odds": fg_away_ml if not pd.isna(fg_away_ml) else fg_home_ml,
                "line": None,
            })
        if not pd.isna(fg_away_ml):
            fg_markets.append({
                "market_type": "ML",
                "pick": f"{away} ML",
                "odds_current": fg_away_ml,
                "model_prob": avg.get("away_ml_prob", 0.5),
                "counter_odds": fg_home_ml if not pd.isna(fg_home_ml) else fg_away_ml,
                "line": None,
            })

        if not pd.isna(fg_over_odds):
            fg_markets.append({
                "market_type": "Total",
                "pick": f"Over {fg_total}",
                "odds_current": fg_over_odds,
                "model_prob": avg.get("over_total_prob", 0.5),
                "counter_odds": fg_under_odds if not pd.isna(fg_under_odds) else fg_over_odds,
                "line": fg_total,
            })
        if not pd.isna(fg_under_odds):
            fg_markets.append({
                "market_type": "Total",
                "pick": f"Under {fg_total}",
                "odds_current": fg_under_odds,
                "model_prob": 1.0 - avg.get("over_total_prob", 0.5),
                "counter_odds": fg_over_odds if not pd.isna(fg_over_odds) else fg_under_odds,
                "line": fg_total,
            })

        # Team Totals (premium — only when API provides real lines)
        if not pd.isna(home_tt_over_odds):
            fg_markets.append({
                "market_type": "Team Total",
                "pick": f"{home} Over {home_tt}",
                "odds_current": home_tt_over_odds,
                "model_prob": avg.get("home_tt_over_prob", 0.5),
                "counter_odds": home_tt_under_odds if not pd.isna(home_tt_under_odds) else home_tt_over_odds,
                "line": home_tt,
            })
        if not pd.isna(home_tt_under_odds):
            fg_markets.append({
                "market_type": "Team Total",
                "pick": f"{home} Under {home_tt}",
                "odds_current": home_tt_under_odds,
                "model_prob": 1.0 - avg.get("home_tt_over_prob", 0.5),
                "counter_odds": home_tt_over_odds if not pd.isna(home_tt_over_odds) else home_tt_under_odds,
                "line": home_tt,
            })
        if not pd.isna(away_tt_over_odds):
            fg_markets.append({
                "market_type": "Team Total",
                "pick": f"{away} Over {away_tt}",
                "odds_current": away_tt_over_odds,
                "model_prob": avg.get("away_tt_over_prob", 0.5),
                "counter_odds": away_tt_under_odds if not pd.isna(away_tt_under_odds) else away_tt_over_odds,
                "line": away_tt,
            })
        if not pd.isna(away_tt_under_odds):
            fg_markets.append({
                "market_type": "Team Total",
                "pick": f"{away} Under {away_tt}",
                "odds_current": away_tt_under_odds,
                "model_prob": 1.0 - avg.get("away_tt_over_prob", 0.5),
                "counter_odds": away_tt_over_odds if not pd.isna(away_tt_over_odds) else away_tt_under_odds,
                "line": away_tt,
            })

        if fg_markets:
            segments.append({"segment": "FG", "markets": fg_markets})

        # ── F5 (First 5 Innings) ────────────────────────────────────
        f5_markets: list[dict[str, Any]] = []
        f5_spread_display = f5_spread if f5_spread != 0 else -0.5

        if not pd.isna(f5_home_spread_odds):
            f5_markets.append({
                "market_type": "Spread",
                "pick": f"{home} {f5_spread_display:+.1f}",
                "odds_current": f5_home_spread_odds,
                "model_prob": avg.get("f5_home_spread_cover_prob", 0.5),
                "counter_odds": f5_away_spread_odds if not pd.isna(f5_away_spread_odds) else f5_home_spread_odds,
                "line": f5_spread_display,
            })

        if not pd.isna(f5_home_ml):
            f5_markets.append({
                "market_type": "ML",
                "pick": f"{home} F5 ML",
                "odds_current": f5_home_ml,
                "model_prob": avg.get("f5_home_ml_prob", 0.5),
                "counter_odds": f5_away_ml if not pd.isna(f5_away_ml) else f5_home_ml,
                "line": None,
            })
        if not pd.isna(f5_away_ml):
            f5_markets.append({
                "market_type": "ML",
                "pick": f"{away} F5 ML",
                "odds_current": f5_away_ml,
                "model_prob": avg.get("f5_away_ml_prob", 0.5),
                "counter_odds": f5_home_ml if not pd.isna(f5_home_ml) else f5_away_ml,
                "line": None,
            })

        if not pd.isna(f5_over_odds):
            f5_markets.append({
                "market_type": "Total",
                "pick": f"F5 Over {f5_total if f5_total > 0 else 4.5}",
                "odds_current": f5_over_odds,
                "model_prob": avg.get("f5_over_total_prob", 0.5),
                "counter_odds": f5_under_odds if not pd.isna(f5_under_odds) else f5_over_odds,
                "line": f5_total if f5_total > 0 else 4.5,
            })
        if not pd.isna(f5_under_odds):
            f5_markets.append({
                "market_type": "Total",
                "pick": f"F5 Under {f5_total if f5_total > 0 else 4.5}",
                "odds_current": f5_under_odds,
                "model_prob": 1.0 - avg.get("f5_over_total_prob", 0.5),
                "counter_odds": f5_over_odds if not pd.isna(f5_over_odds) else f5_under_odds,
                "line": f5_total if f5_total > 0 else 4.5,
            })

        if f5_markets:
            segments.append({"segment": "F5", "markets": f5_markets})

        # ── Emit rows ───────────────────────────────────────────────
        for seg in segments:
            for mkt in seg["markets"]:
                odds_cur = mkt["odds_current"]
                counter = mkt["counter_odds"]
                dec_odds = _american_to_decimal(odds_cur)
                model_p = mkt["model_prob"]

                # No-vig probability: pass the picked side's odds first
                nv_prob = _no_vig_prob(odds_cur, counter)
                ev = _no_vig_ev(model_p, dec_odds)
                kelly = _kelly_fraction(model_p, dec_odds)
                confidence = min(1.0, kelly * 4)  # scale kelly to 0-1

                # Odds quality
                if seg["segment"] == "F5":
                    quality = f5_odds_quality
                else:
                    quality = "live" if fg_has_odds else "no_odds"

                # Recommendation: require real odds for full confidence
                #   live odds:  EV > 3% AND model_prob > 52%
                #   no_odds/synthetic: never recommend (no real line to bet against)
                #   cold_start: never recommend (features are mostly defaults)
                if cold_start:
                    is_rec = False
                elif quality == "live":
                    is_rec = ev > 0.03 and model_p > 0.52
                else:  # synthetic / no_odds
                    is_rec = False

                # In cold start, discount confidence by 50%
                if cold_start:
                    confidence *= 0.5

                # Build rationale for recommended picks
                rationale = ""
                if cold_start:
                    rationale = "COLD START — predictions clustered, low confidence"
                elif is_rec:
                    bullets = []
                    bullets.append(f"Model: {model_p:.1%} vs implied {_implied_prob(odds_cur):.1%} ({n_models} models agree)")
                    bullets.append(f"EV: {ev:+.1%} | Kelly: {kelly:.2%}")
                    if abs(model_p - _implied_prob(odds_cur)) > 0.08:
                        bullets.append("Sharp edge: large model-vs-market divergence")
                    bullets.append(f"Expected score: {exp_fg_home:.1f}-{exp_fg_away:.1f} FG / {exp_f5_home:.1f}-{exp_f5_away:.1f} F5")
                    rationale = " | ".join(bullets)

                # Line movement — Since we have multiple bookmaker rows, compute
                # the spread of odds across books as a proxy for movement.
                # (True line movement requires historical snapshots.)
                matching_books = games_df[
                    (games_df["home_team"] == home) & (games_df["away_team"] == away)
                ]
                ml_values = matching_books["home_moneyline"].tolist()
                if len(ml_values) > 1:
                    line_move = f"{int(min(ml_values))} → {int(max(ml_values))}"
                else:
                    line_move = "—"

                # Steam / RLM flags (require live consensus data — mark as N/A
                # since we don't have public betting splits in current API tier)
                steam_flag = False
                rlm_flag = False
                bets_pct = "N/A"
                money_pct = "N/A"

                rows.append({
                    "date": today,
                    "game": matchup,
                    "game_time": game_time,
                    "segment": seg["segment"],
                    "market_type": mkt["market_type"],
                    "pick": mkt["pick"],
                    "odds_current": int(odds_cur),
                    "odds_quality": quality,
                    "line_move": line_move,
                    "steam_flag": steam_flag,
                    "rlm_flag": rlm_flag,
                    "bets_pct": bets_pct,
                    "money_pct": money_pct,
                    "model_prob": round(model_p, 4),
                    "implied_prob": round(_implied_prob(odds_cur), 4),
                    "no_vig_ev": round(ev, 4),
                    "kelly": round(kelly, 4),
                    "confidence": round(confidence, 4),
                    "is_recommended": is_rec,
                    "rationale_bullets": rationale,
                })

    return rows


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="MLB Pick Sheet Generator")
    parser.add_argument("--date", type=str, default=None, help="Date (YYYY-MM-DD)")
    args = parser.parse_args()

    today = args.date or datetime.now(tz=UTC).strftime("%Y-%m-%d")
    logger.info("Generating MLB pick sheet for %s", today)

    # ── 1. Load config & today's odds ────────────────────────────────
    config = AppConfig.load()
    config = config.override(data={"loader": "odds_api"})
    loader = OddsAPILoader(api_key=config.data.api_key, base_url=config.data.api_base_url)
    games_df = loader.load()
    logger.info("Loaded %d game/bookmaker rows", len(games_df))

    if games_df.empty:
        logger.warning("No games found for %s", today)
        return

    # TrackingDB (so alembic runs)
    db = TrackingDB("artifacts/tracking.db")
    logging.basicConfig(level=logging.INFO, force=True)
    logger.disabled = False

    # ── 2. Preprocess & engineer features ────────────────────────────
    # De-dup for model input (one row per game)
    model_input = games_df.groupby(
        ["home_team", "away_team"], sort=False
    ).first().reset_index()

    processed = preprocess(model_input)
    features = engineer_features(
        processed.features,
        short_window=config.features.rolling_window_short,
        long_window=config.features.rolling_window_long,
    )

    # ── Cold-start detection ─────────────────────────────────────────
    # Count how many features are constant (zero/default) across all games.
    # When most features are constant, the model can only differentiate
    # games by market odds → predictions cluster and are unreliable.
    n_total = features.X.shape[1]
    n_constant = sum(1 for c in features.X.columns if features.X[c].nunique() <= 1)
    pct_constant = n_constant / n_total if n_total else 0
    cold_start = pct_constant > 0.5

    if cold_start:
        logger.warning(
            "COLD START: %d of %d features (%.0f%%) are constant — "
            "predictions will cluster and are LOW CONFIDENCE. "
            "This is expected early in the season.",
            n_constant, n_total, pct_constant * 100,
        )

    # ── 3. Load all models & predict ─────────────────────────────────
    model_dir = Path("artifacts/models")
    model_paths = sorted(model_dir.glob("*.pkl"))
    # Skip ensembles (they wrap base models) and stale logistic_regression alias
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
    pick_rows = _build_pick_rows(model_input, model_results, today, cold_start=cold_start)
    logger.info("Generated %d pick rows across all games/segments/markets", len(pick_rows))

    # ── 5. Write CSV ─────────────────────────────────────────────────
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"picks_{today}.csv"

    fieldnames = [
        "date", "game", "game_time", "segment", "market_type", "pick",
        "odds_current", "odds_quality", "line_move",
        "steam_flag", "rlm_flag", "bets_pct", "money_pct",
        "model_prob", "implied_prob", "no_vig_ev", "kelly", "confidence",
        "is_recommended", "rationale_bullets",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pick_rows)

    logger.info("Wrote %s (%d rows)", csv_path, len(pick_rows))

    # ── 6. Console summary of recommended picks ─────────────────────
    recs = [r for r in pick_rows if r["is_recommended"]]
    logger.info("")
    logger.info("═" * 100)
    if cold_start:
        logger.info("  COLD START — %d/%d features constant. No recommendations.", n_constant, n_total)
        logger.info("  Showing top EV picks for reference only (NOT actionable).")
    else:
        logger.info("  RECOMMENDED PICKS FOR %s  (%d of %d)", today, len(recs), len(pick_rows))
    logger.info("═" * 100)

    if cold_start:
        # Show top 15 LIVE-odds picks by EV for reference
        live_picks = [r for r in pick_rows if r["odds_quality"] == "live"]
        top = sorted(live_picks, key=lambda x: -x["no_vig_ev"])[:15]
        header = f"{'Game':<22} {'Seg':>3} {'Market':<12} {'Pick':<20} {'Odds':>6} {'Qual':<5} {'Model%':>7} {'Impl%':>7} {'EV':>7} {'Kelly':>7} {'Conf':>6}"
        logger.info(header)
        logger.info("-" * len(header))
        for r in top:
            logger.info(
                f"{r['game']:<22} {r['segment']:>3} {r['market_type']:<12} "
                f"{r['pick']:<20} {r['odds_current']:>+6d} "
                f"{r['odds_quality']:<5} "
                f"{r['model_prob']:>6.1%} {r['implied_prob']:>6.1%} "
                f"{r['no_vig_ev']:>+6.1%} {r['kelly']:>6.2%} {r['confidence']:>5.2f}"
            )
    elif not recs:
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

    logger.info("")
    logger.info("Full sheet: %s", csv_path)


if __name__ == "__main__":
    main()
