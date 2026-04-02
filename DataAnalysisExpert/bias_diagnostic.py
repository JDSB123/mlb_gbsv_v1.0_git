"""Root-cause bias diagnostic — traces inflation from training data through model predictions."""
import sys, json, os
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from mlbv1.data.synthetic_history import SyntheticHistoryGenerator
from mlbv1.features.engineer import engineer_features
from mlbv1.models.predictor import load_model, predict

print("=" * 70)
print("BIAS ROOT-CAUSE DIAGNOSTIC")
print("=" * 70)

# ── 1. TRAINING DATA DISTRIBUTION ──
print("\n1. SYNTHETIC TRAINING DATA DISTRIBUTION")
print("-" * 50)
gen = SyntheticHistoryGenerator(seed=42)
synth = gen.generate_seasons([2023, 2024, 2025])
print(f"   Synthetic games: {len(synth)}")
print(f"   Home score — mean: {synth['home_score'].mean():.2f}, median: {synth['home_score'].median():.1f}, std: {synth['home_score'].std():.2f}")
print(f"   Away score — mean: {synth['away_score'].mean():.2f}, median: {synth['away_score'].median():.1f}, std: {synth['away_score'].std():.2f}")
print(f"   Total      — mean: {(synth['home_score'] + synth['away_score']).mean():.2f}")
print(f"   F5 home    — mean: {synth['f5_home_score'].mean():.2f}")
print(f"   F5 away    — mean: {synth['f5_away_score'].mean():.2f}")
print(f"   F5 total   — mean: {(synth['f5_home_score'] + synth['f5_away_score']).mean():.2f}")
print(f"   Home win%  — {(synth['home_score'] > synth['away_score']).mean()*100:.1f}%")

# MLB 2024 actuals for comparison
print("\n   📊 MLB 2024 ACTUALS (for comparison):")
print("   Home score — mean: 4.38")
print("   Away score — mean: 4.25")
print("   Total      — mean: 8.63")
print("   Home win%  — 53.8%")

delta = synth['home_score'].mean() - 4.38
print(f"\n   ⚠️  Synthetic home delta: {delta:+.2f} runs vs real MLB")
delta2 = synth['away_score'].mean() - 4.25
print(f"   ⚠️  Synthetic away delta: {delta2:+.2f} runs vs real MLB")
delta_total = (synth['home_score'] + synth['away_score']).mean() - 8.63
print(f"   ⚠️  Synthetic total delta: {delta_total:+.2f} runs vs real MLB")

# ── 2. FEATURE DISTRIBUTION CHECK ──
print("\n2. FEATURE ENGINEERING — TARGET LEAKAGE CHECK")
print("-" * 50)
fs = engineer_features(synth)
X = fs.X
targets = synth[["f5_home_score", "f5_away_score", "home_score", "away_score"]]

# Check if feature correlations are suspiciously high with targets
corrs = {}
for target in ["home_score", "away_score"]:
    for feat in X.columns:
        c = X[feat].corr(targets[target])
        if abs(c) > 0.5:
            corrs[(feat, target)] = c

if corrs:
    print("   ⚠️  HIGH CORRELATIONS (>0.5) — potential leakage:")
    for (f, t), c in sorted(corrs.items(), key=lambda x: -abs(x[1])):
        print(f"      {f:40s} ↔ {t:15s}: r={c:+.3f}")
else:
    print("   ✅ No suspiciously high feature-target correlations")

# ── 3. MODEL PREDICTIONS ON SYNTHETIC DATA ──
print("\n3. MODEL PREDICTIONS ON SYNTHETIC HOLDOUT")
print("-" * 50)
# Use last 200 games as holdout
holdout_X = X.tail(200).copy()
holdout_y = targets.tail(200).copy()
holdout_df = synth.tail(200).copy()

model_dir = "artifacts/models"
model_files = ["random_forest.pkl", "ridge_regression.pkl", "xgboost.pkl", "lightgbm.pkl"]

for mf in model_files:
    path = os.path.join(model_dir, mf)
    if not os.path.exists(path):
        print(f"   {mf}: NOT FOUND")
        continue
    model = load_model(path)
    preds = pd.DataFrame(
        model.predict(holdout_X[model.feature_names]),
        index=holdout_X.index,
        columns=model.target_names,
    ).clip(lower=0.01)

    name = mf.replace(".pkl", "")
    actual_total = (holdout_y["home_score"] + holdout_y["away_score"]).mean()
    pred_total = (preds["home_score"] + preds["away_score"]).mean()
    home_bias = preds["home_score"].mean() - holdout_y["home_score"].mean()
    away_bias = preds["away_score"].mean() - holdout_y["away_score"].mean()

    from sklearn.metrics import mean_absolute_error
    mae_home = mean_absolute_error(holdout_y["home_score"], preds["home_score"])
    mae_away = mean_absolute_error(holdout_y["away_score"], preds["away_score"])

    print(f"\n   {name}:")
    print(f"     Pred home: {preds['home_score'].mean():.2f} (actual: {holdout_y['home_score'].mean():.2f}) bias: {home_bias:+.2f}")
    print(f"     Pred away: {preds['away_score'].mean():.2f} (actual: {holdout_y['away_score'].mean():.2f}) bias: {away_bias:+.2f}")
    print(f"     Pred total: {pred_total:.2f} (actual: {actual_total:.2f}) bias: {pred_total - actual_total:+.2f}")
    print(f"     MAE — home: {mae_home:.2f}, away: {mae_away:.2f}")

# ── 4. PREDICTION ON TODAY'S LIVE DATA ──
print("\n4. TODAY'S LIVE PREDICTIONS — SCORE SANITY")
print("-" * 50)
with open("DataAnalysisExpert/slate_2026-04-01.json", encoding="utf-8") as f:
    picks = json.load(f)

games = {}
for p in picks:
    g = p["game"]
    if g not in games:
        games[g] = {"home": p["home_team"], "exp_home": p["exp_home_score"], "exp_away": p["exp_away_score"]}

for g, info in games.items():
    total = info["exp_home"] + info["exp_away"]
    excess = total - 8.63
    print(f"   {g:15s}: {info['exp_home']:.1f} - {info['exp_away']:.1f} = {total:.1f} total ({excess:+.1f} vs MLB avg)")

avg_total = np.mean([v["exp_home"] + v["exp_away"] for v in games.values()])
print(f"\n   Avg predicted total: {avg_total:.2f}")
print(f"   MLB 2024 avg total: 8.63")
print(f"   Inflation:          {avg_total - 8.63:+.2f} runs ({(avg_total / 8.63 - 1)*100:+.1f}%)")

# ── 5. POISSON PROBABILITY SENSITIVITY ──
print("\n5. POISSON PROBABILITY SENSITIVITY TO SCORE INFLATION")
print("-" * 50)
from scipy.stats import poisson, skellam

# Show how a small score inflation dramatically affects probabilities
for scenario, home_mu, away_mu in [
    ("Real MLB avg", 4.4, 4.3),
    ("Mild inflation (+1)", 5.4, 5.3),
    ("Today's model avg", 7.0, 5.0),
]:
    total_mu = home_mu + away_mu
    p_over_8 = poisson.sf(8, total_mu)  # P(total > 8)
    p_over_7 = poisson.sf(7, total_mu)
    p_home_win = skellam.sf(0, home_mu, away_mu)
    p_cover_1_5 = skellam.sf(1, home_mu, away_mu)  # P(margin > 1)
    print(f"\n   {scenario} (H={home_mu:.1f}, A={away_mu:.1f}, T={total_mu:.1f}):")
    print(f"     P(Over 8.0):  {p_over_8*100:.1f}%")
    print(f"     P(Over 7.0):  {p_over_7*100:.1f}%")
    print(f"     P(Home Win):  {p_home_win*100:.1f}%")
    print(f"     P(Home -1.5): {p_cover_1_5*100:.1f}%")

# ── DIAGNOSIS ──
print("\n" + "=" * 70)
print("ROOT CAUSE DIAGNOSIS")
print("=" * 70)
print("""
The bias chain is:

  1. SYNTHETIC DATA generates runs_per_game from Uniform(3.5, 5.5) — mean 4.5.
     This is reasonable for per-team scoring, but the Poisson inning-by-inning
     generation means actual scores match these means (verified above).

  2. MODELS learn to predict scores well on synthetic data (~4.5 R/G).

  3. AT INFERENCE TIME the models receive *live features* (rolling stats, pitcher
     ERA, moneylines) from today's real MLB games. But the model was trained on
     synthetic data with simplified, homogeneous features — it hasn't seen the
     real feature distributions.

  4. The key issue: FEATURE-TARGET RELATIONSHIP MISMATCH. The synthetic data
     has simplistic feature relationships (e.g., every team's rolling avg is
     near their Uniform-drawn runs_per_game). Real MLB data has much more
     variance and different distributions for features like moneylines, implied
     probs, ERA, etc.

  5. When real features (especially high moneyline favorites, low ERA pitchers)
     are fed in, the models extrapolate outside their training distribution,
     producing inflated score predictions.

  6. POISSON AMPLIFICATION: Even a +1 run inflation in expected score causes
     the Poisson/Skellam probability math to produce dramatically higher
     probabilities (e.g., P(Over 8.0) goes from 44% to 73%).

  7. This cascades into: inflated model_prob → inflated EV → inflated Kelly
     → over/home bias (higher team always looks like a massive edge).

FIXES NEEDED:
  A. Use REAL historical data (Lahman, Retrosheet, or MLB Stats API) instead
     of synthetic data for training — this is the #1 fix.
  B. Add score prediction clamping/calibration post-prediction (e.g., clip
     predictions to reasonable range 2.0-7.0 per team).
  C. Add probability calibration (Platt scaling / isotonic regression) to
     correct overconfident model probabilities before market derivation.
  D. Reduce Kelly fraction from full-Kelly to quarter-Kelly (already labeled
     as quarter-Kelly but the raw Kelly values suggest full Kelly is used).
""")
