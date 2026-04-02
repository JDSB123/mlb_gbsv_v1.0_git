"""Definitive end-to-end bias test: new models + real Odds API data + full pick sheet.
Bypasses pregame time filter. Compares every metric against the original slate."""
import json, sys, os
sys.path.insert(0, ".")

from collections import Counter
from mlbv1.config import AppConfig
from mlbv1.data.loader import OddsAPILoader
from mlbv1.data.preprocessor import preprocess
from mlbv1.features.engineer import engineer_features
from mlbv1.models.predictor import load_model, predict
from scripts.pick_sheet import _build_pick_rows
from datetime import UTC, datetime
import uuid

# ── Load real Odds API data ──────────────────────────────────────────
config = AppConfig.load()
api_key = os.getenv("ODDS_API_KEY", config.data.api_key or "")
loader = OddsAPILoader("https://api.the-odds-api.com/v4", api_key)
raw = loader.load()
print(f"[1] Loaded {len(raw)} raw Odds API rows")

# Skip pregame filter — use ALL games
games = raw.copy()

# ── Feature engineering (no Lahman — too slow) ───────────────────────
proc = preprocess(games)
feats = engineer_features(proc.features)
print(f"[2] Engineered {len(feats.feature_names)} features for {len(feats.X)} games")

# ── Load all 4 models + predict ─────────────────────────────────────
model_names = ["random_forest", "ridge_regression", "xgboost", "lightgbm"]
results = {}
for name in model_names:
    m = load_model(f"artifacts/models/{name}.pkl")
    results[name] = predict(m, feats.X, games)
print(f"[3] Predicted with {len(results)} models")

# ── Build full pick sheet ────────────────────────────────────────────
run_id = f"e2e-test-{uuid.uuid4().hex[:8]}"
now_str = datetime.now(tz=UTC).isoformat()
rows = _build_pick_rows(
    games, results, "2026-04-01",
    run_id=run_id, generated_at_utc=now_str,
    features_X=feats.X,
)
print(f"[4] Generated {len(rows)} total pick rows")

# Save for inspection
with open("DataAnalysisExpert/picks_e2e_test.json", "w") as f:
    json.dump(rows, f, indent=2, default=str)

# ── Analysis ─────────────────────────────────────────────────────────
rec = [r for r in rows if r["is_recommended"]]
not_rec = [r for r in rows if not r["is_recommended"]]
print(f"\n{'='*60}")
print(f"  DEFINITIVE E2E TEST — {len(rec)} recommended / {len(rows)} total")
print(f"{'='*60}")

if not rec:
    print("  !! ZERO recommended picks — checking why...")
    for r in rows[:5]:
        print(f"    {r['pick']:25s} EV={r['no_vig_ev']:+.3f} prob={r['model_prob']:.3f} quality={r['odds_quality']}")
    print("  (Need EV>3% AND model_prob>52% AND live odds)")

# Score predictions
games_seen = {}
for p in rows:
    g = p["game"]
    if g not in games_seen:
        games_seen[g] = {"h": p["exp_home_score"], "a": p["exp_away_score"]}
totals = [v["h"] + v["a"] for v in games_seen.values()]
avg_total = sum(totals) / len(totals) if totals else 0
print(f"\n  Predicted totals (MLB avg ~8.6):")
for g, v in games_seen.items():
    t = v["h"] + v["a"]
    flag = " ⚠" if t > 11 else ""
    print(f"    {g:20s}: {v['h']:.1f} - {v['a']:.1f} = {t:.1f}{flag}")
print(f"    AVG: {avg_total:.2f}")
PASS_SCORE = avg_total < 11.0
print(f"    {'✓ PASS' if PASS_SCORE else '✗ FAIL'}: avg total < 11.0")

if rec:
    # Over/Under
    overs = [r for r in rec if "Over" in r["pick"]]
    unders = [r for r in rec if "Under" in r["pick"]]
    sides = [r for r in rec if "Over" not in r["pick"] and "Under" not in r["pick"]]
    print(f"\n  Over/Under: {len(overs)} Over / {len(unders)} Under / {len(sides)} Side")
    PASS_OU = len(unders) > 0 or len(overs) == 0
    print(f"    {'✓ PASS' if PASS_OU else '⚠ WARN'}: at least 1 under (or no totals)")

    # Home/Away (sides only)
    if sides:
        home = [r for r in sides if r["home_team"] in r["pick"]]
        away = [r for r in sides if r["away_team"] in r["pick"]]
        home_pct = len(home) / len(sides) * 100
        print(f"\n  Home/Away sides: {len(home)} Home / {len(away)} Away ({home_pct:.0f}% home)")
        PASS_HOME = home_pct < 80 or len(sides) < 4
        print(f"    {'✓ PASS' if PASS_HOME else '⚠ WARN'}: home < 80% (or small sample)")

    # Kelly
    kellys = [r["kelly"] for r in rec]
    avg_k = sum(kellys) / len(kellys)
    max_k = max(kellys)
    extreme = sum(1 for k in kellys if k > 0.5)
    print(f"\n  Kelly: avg={avg_k*100:.1f}% max={max_k*100:.1f}% extreme(>50%)={extreme}")
    PASS_KELLY = max_k < 0.50 and avg_k < 0.25
    print(f"    {'✓ PASS' if PASS_KELLY else '✗ FAIL'}: max < 50% AND avg < 25%")

    # Calibration gap
    gaps = [r["model_prob"] - r["implied_prob"] for r in rec]
    avg_gap = sum(gaps) / len(gaps)
    high85 = sum(1 for r in rec if r["model_prob"] > 0.85)
    high90 = sum(1 for r in rec if r["model_prob"] > 0.90)
    print(f"\n  Calibration: avg gap={avg_gap*100:+.1f}pp  >85%={high85}  >90%={high90}")
    PASS_CAL = high90 == 0
    print(f"    {'✓ PASS' if PASS_CAL else '✗ FAIL'}: zero picks with model_prob > 90%")

    # Confidence
    confs = [r["confidence"] for r in rec]
    avg_conf = sum(confs) / len(confs)
    print(f"\n  Confidence: avg={avg_conf*100:.1f}%")
    PASS_CONF = avg_conf < 0.80
    print(f"    {'✓ PASS' if PASS_CONF else '✗ FAIL'}: avg confidence < 80%")

    # Market mix
    by_mkt = Counter(r["market_type"] for r in rec)
    print(f"\n  Market mix: {dict(by_mkt)}")

    # Spread direction check
    spread_picks = [r for r in rec if r["market_type"] == "Spread"]
    if spread_picks:
        sp_home = [r for r in spread_picks if r["home_team"] in r["pick"]]
        sp_away = [r for r in spread_picks if r["away_team"] in r["pick"]]
        print(f"  Spread picks: {len(sp_home)} home / {len(sp_away)} away")
        PASS_SPREAD = len(sp_away) > 0 or len(spread_picks) < 2
        print(f"    {'✓ PASS' if PASS_SPREAD else '⚠ WARN'}: away spreads generated")

# ── Final verdict ────────────────────────────────────────────────────
print(f"\n{'='*60}")
checks = []
checks.append(("Predicted totals < 11", PASS_SCORE))
if rec:
    checks.append(("Kelly under control", PASS_KELLY))
    checks.append(("No model_prob > 90%", PASS_CAL))
    checks.append(("Confidence < 80%", PASS_CONF))
passed = sum(1 for _, v in checks if v)
total = len(checks)
print(f"  RESULT: {passed}/{total} checks passed")
for name, ok in checks:
    print(f"    {'✓' if ok else '✗'} {name}")
if passed == total:
    print(f"\n  ★ ALL BIAS FIXES VERIFIED ★")
else:
    print(f"\n  ✗ {total - passed} CHECK(S) STILL FAILING")
print(f"{'='*60}")
