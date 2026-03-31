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


def _no_vig_prob(home_odds: float, away_odds: float, side: str) -> float:
    """Remove the vig and return the 'true' implied probability for *side*."""
    p_home = _implied_prob(home_odds)
    p_away = _implied_prob(away_odds)
    total = p_home + p_away
    if total == 0:
        return 0.5
    if side == "home":
        return p_home / total
    return p_away / total


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


def _best_odds_for_game(rows: list[dict], key: str) -> dict[str, float]:
    """Return the best (most favorable) odds across multiple bookmaker rows."""
    best: dict[str, float] = {}
    for r in rows:
        val = r.get(key, -110)
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if key not in best or _american_to_decimal(v) > _american_to_decimal(best[key]):
            best[key] = v
    return best


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
        fg_spread = float(game.get("spread", -1.5))
        fg_home_spread_odds = float(game.get("home_spread_odds", -110))
        fg_away_spread_odds = float(game.get("away_spread_odds", -110))
        fg_home_ml = float(game.get("home_moneyline", -110))
        fg_away_ml = float(game.get("away_moneyline", -110))
        fg_total = float(game.get("total_runs", 8.5))
        fg_over_odds = float(game.get("over_odds", -110))
        fg_under_odds = float(game.get("under_odds", -110))

        # F5 Market Data
        f5_spread = float(game.get("f5_spread", 0.0))
        f5_home_spread_odds = float(game.get("f5_home_spread_odds", -110))
        f5_away_spread_odds = float(game.get("f5_away_spread_odds", -110))
        f5_home_ml = float(game.get("f5_home_moneyline", -110))
        f5_away_ml = float(game.get("f5_away_moneyline", -110))
        f5_total = float(game.get("f5_total_runs", 0.0))
        f5_over_odds = float(game.get("f5_over_odds", -110))
        f5_under_odds = float(game.get("f5_under_odds", -110))

        # Expected runs from models
        exp_fg_home = avg.get("exp_home_score", 4.0)
        exp_fg_away = avg.get("exp_away_score", 4.0)
        exp_f5_home = avg.get("exp_f5_home_score", 2.0)
        exp_f5_away = avg.get("exp_f5_away_score", 2.0)

        # ── Define all markets ───────────────────────────────────────
        segments = []

        # ── FG (Full Game) ───────────────────────────────────────────
        segments.append({
            "segment": "FG",
            "markets": [
                {
                    "market_type": "Spread",
                    "pick": f"{home} {fg_spread:+.1f}",
                    "odds_current": fg_home_spread_odds,
                    "model_prob": avg.get("home_spread_cover_prob", 0.5),
                    "counter_odds": fg_away_spread_odds,
                    "line": fg_spread,
                },
                {
                    "market_type": "ML",
                    "pick": f"{home} ML",
                    "odds_current": fg_home_ml,
                    "model_prob": avg.get("home_ml_prob", 0.5),
                    "counter_odds": fg_away_ml,
                    "line": None,
                },
                {
                    "market_type": "ML",
                    "pick": f"{away} ML",
                    "odds_current": fg_away_ml,
                    "model_prob": avg.get("away_ml_prob", 0.5),
                    "counter_odds": fg_home_ml,
                    "line": None,
                },
                {
                    "market_type": "Total",
                    "pick": f"Over {fg_total}",
                    "odds_current": fg_over_odds,
                    "model_prob": avg.get("over_total_prob", 0.5),
                    "counter_odds": fg_under_odds,
                    "line": fg_total,
                },
                {
                    "market_type": "Total",
                    "pick": f"Under {fg_total}",
                    "odds_current": fg_under_odds,
                    "model_prob": 1.0 - avg.get("over_total_prob", 0.5),
                    "counter_odds": fg_over_odds,
                    "line": fg_total,
                },
                {
                    "market_type": "Team Total",
                    "pick": f"{home} Over {exp_fg_home:.1f}",
                    "odds_current": -110,  # standard TT pricing
                    "model_prob": avg.get("home_tt_over_prob", 0.5),
                    "counter_odds": -110,
                    "line": round(exp_fg_home * 2) / 2,  # nearest 0.5
                },
                {
                    "market_type": "Team Total",
                    "pick": f"{away} Over {exp_fg_away:.1f}",
                    "odds_current": -110,
                    "model_prob": avg.get("away_tt_over_prob", 0.5),
                    "counter_odds": -110,
                    "line": round(exp_fg_away * 2) / 2,
                },
            ],
        })

        # ── F5 (First 5 Innings) ────────────────────────────────────
        f5_has_data = f5_total > 0 or f5_spread != 0 or f5_home_ml != -110
        f5_spread_display = f5_spread if f5_spread != 0 else -0.5  # default F5 spread
        f5_odds_quality = "live" if f5_has_data else "synthetic"

        segments.append({
            "segment": "F5",
            "markets": [
                {
                    "market_type": "Spread",
                    "pick": f"{home} {f5_spread_display:+.1f}",
                    "odds_current": f5_home_spread_odds,
                    "model_prob": avg.get("f5_home_spread_cover_prob", 0.5),
                    "counter_odds": f5_away_spread_odds,
                    "line": f5_spread_display,
                },
                {
                    "market_type": "ML",
                    "pick": f"{home} F5 ML",
                    "odds_current": f5_home_ml,
                    "model_prob": avg.get("f5_home_ml_prob", 0.5),
                    "counter_odds": f5_away_ml,
                    "line": None,
                },
                {
                    "market_type": "ML",
                    "pick": f"{away} F5 ML",
                    "odds_current": f5_away_ml,
                    "model_prob": avg.get("f5_away_ml_prob", 0.5),
                    "counter_odds": f5_home_ml,
                    "line": None,
                },
                {
                    "market_type": "Total",
                    "pick": f"F5 Over {f5_total if f5_total > 0 else 4.5}",
                    "odds_current": f5_over_odds,
                    "model_prob": avg.get("f5_over_total_prob", 0.5),
                    "counter_odds": f5_under_odds,
                    "line": f5_total if f5_total > 0 else 4.5,
                },
                {
                    "market_type": "Total",
                    "pick": f"F5 Under {f5_total if f5_total > 0 else 4.5}",
                    "odds_current": f5_under_odds,
                    "model_prob": 1.0 - avg.get("f5_over_total_prob", 0.5),
                    "counter_odds": f5_over_odds,
                    "line": f5_total if f5_total > 0 else 4.5,
                },
                {
                    "market_type": "Team Total",
                    "pick": f"{home} F5 Over {exp_f5_home:.1f}",
                    "odds_current": -110,
                    "model_prob": avg.get("f5_home_tt_over_prob", 0.5),
                    "counter_odds": -110,
                    "line": round(exp_f5_home * 2) / 2,
                },
                {
                    "market_type": "Team Total",
                    "pick": f"{away} F5 Over {exp_f5_away:.1f}",
                    "odds_current": -110,
                    "model_prob": avg.get("f5_away_tt_over_prob", 0.5),
                    "counter_odds": -110,
                    "line": round(exp_f5_away * 2) / 2,
                },
            ],
        })

        # ── Emit rows ───────────────────────────────────────────────
        for seg in segments:
            for mkt in seg["markets"]:
                odds_cur = mkt["odds_current"]
                counter = mkt["counter_odds"]
                dec_odds = _american_to_decimal(odds_cur)
                model_p = mkt["model_prob"]

                nv_prob = _no_vig_prob(
                    odds_cur if "home" in mkt["pick"].lower() or "over" in mkt["pick"].lower() else counter,
                    counter if "home" in mkt["pick"].lower() or "over" in mkt["pick"].lower() else odds_cur,
                    "home",
                )
                ev = _no_vig_ev(model_p, dec_odds)
                kelly = _kelly_fraction(model_p, dec_odds)
                confidence = min(1.0, kelly * 4)  # scale kelly to 0-1

                # Odds quality: live (real API data), synthetic (defaults/model-derived)
                if seg["segment"] == "F5":
                    quality = f5_odds_quality
                elif mkt["market_type"] == "Team Total":
                    quality = "model"  # team total lines are model-derived
                else:
                    quality = "live"

                # Recommendation: require real odds for full confidence
                #   live odds:  EV > 3% AND model_prob > 52%
                #   model odds: EV > 5% AND model_prob > 55%
                #   synthetic:  never recommend (no real line to bet against)
                #   cold_start: never recommend (features are mostly defaults)
                if cold_start:
                    is_rec = False
                elif quality == "live":
                    is_rec = ev > 0.03 and model_p > 0.52
                elif quality == "model":
                    is_rec = ev > 0.05 and model_p > 0.55
                else:  # synthetic
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
