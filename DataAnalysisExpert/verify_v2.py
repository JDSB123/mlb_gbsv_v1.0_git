"""Verify debiased models produce correct pick distribution by running locally
(bypasses the pregame time filter which blocks when games have already started)."""
import json
import sys
sys.path.insert(0, ".")

from scripts.pick_sheet import (
    _build_pick_rows,
    _american_to_decimal,
    _implied_prob,
    _kelly_fraction,
    _no_vig_ev,
)
from mlbv1.config import AppConfig
from mlbv1.data.loader import OddsAPILoader
from mlbv1.data.preprocessor import preprocess
from mlbv1.features.engineer import engineer_features
from mlbv1.models.predictor import load_model, predict
from mlbv1.data.historical_enrichment import enrich_training_data_with_historical_sources
import os, uuid
from datetime import UTC, datetime

config = AppConfig.load()
api_key = os.getenv("ODDS_API_KEY", config.data.api_key or "")
loader = OddsAPILoader("https://api.the-odds-api.com/v4", api_key)
raw_df = loader.load()
print(f"Loaded {len(raw_df)} raw rows")

# Skip pregame filter — use all games
games_df = raw_df.copy()
print(f"Using {len(games_df)} rows (all games, no time filter)")

# Enrich
games_df = enrich_training_data_with_historical_sources(
    games_df, include_lahman=True, include_statcast=False,
    include_probable_pitchers=False,
)

processed = preprocess(games_df)
features = engineer_features(
    processed.features,
    short_window=config.features.rolling_window_short,
    long_window=config.features.rolling_window_long,
)

# Load all models and predict
model_names = ["random_forest", "ridge_regression", "xgboost", "lightgbm"]
model_results = {}
for name in model_names:
    try:
        model = load_model(f"artifacts/models/{name}.pkl")
        result = predict(model, features.X, games_df)
        model_results[name] = result
    except Exception as e:
        print(f"  Skip {name}: {e}")

print(f"Loaded {len(model_results)} models")

# Build picks
run_id = f"verify-{uuid.uuid4().hex[:8]}"
now = datetime.now(tz=UTC).isoformat()
rows = _build_pick_rows(
    games_df, model_results, "2026-04-01",
    run_id=run_id, generated_at_utc=now,
    features_X=features.X,
)

# Analyze
rec = [r for r in rows if r["is_recommended"]]
print(f"\nTotal picks: {len(rows)} | Recommended: {len(rec)}")

# Home/Away bias
sides = [r for r in rec if "Over" not in r["pick"] and "Under" not in r["pick"]]
if sides:
    home = [r for r in sides if r["home_team"] in r["pick"]]
    away = [r for r in sides if r["away_team"] in r["pick"]]
    print(f"Sides: {len(home)} Home / {len(away)} Away ({len(home)/(len(sides))*100:.0f}% home)")

# Over/Under
overs = [r for r in rec if "Over" in r["pick"]]
unders = [r for r in rec if "Under" in r["pick"]]
print(f"O/U: {len(overs)} Over / {len(unders)} Under")

# Scores
games = {}
for p in rows:
    g = p["game"]
    if g not in games:
        games[g] = {"exp_home": p["exp_home_score"], "exp_away": p["exp_away_score"]}
totals = [v["exp_home"] + v["exp_away"] for v in games.values()]
avg_total = sum(totals) / len(totals) if totals else 0
print(f"\nAvg predicted total: {avg_total:.2f} (MLB avg ~8.6)")
for g, v in games.items():
    t = v["exp_home"] + v["exp_away"]
    print(f"  {g:15s}: {v['exp_home']:.1f} - {v['exp_away']:.1f} = {t:.1f}")

# Kelly
kellys = [r["kelly"] for r in rec]
if kellys:
    print(f"\nKelly — Avg: {sum(kellys)/len(kellys)*100:.1f}% | Max: {max(kellys)*100:.1f}%")
    extreme = sum(1 for k in kellys if k > 0.5)
    print(f"Kelly > 50%: {extreme}/{len(rec)}")

# model_prob
probs = [r["model_prob"] for r in rec]
if probs:
    high85 = sum(1 for p in probs if p > 0.85)
    high90 = sum(1 for p in probs if p > 0.90)
    print(f"model_prob > 85%: {high85}/{len(rec)} | > 90%: {high90}/{len(rec)}")

# Gaps
gaps = [r["model_prob"] - r["implied_prob"] for r in rec]
if gaps:
    print(f"Avg gap: {sum(gaps)/len(gaps)*100:+.1f}pp")

# Save for comparison
with open("DataAnalysisExpert/picks_debiased_v2_2026-04-01.json", "w") as f:
    json.dump(rows, f, indent=2, default=str)
print(f"\nSaved {len(rows)} picks to DataAnalysisExpert/picks_debiased_v2_2026-04-01.json")
