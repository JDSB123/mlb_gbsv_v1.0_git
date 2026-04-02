"""Bias analysis on today's picks slate."""
import json, re, sys
from collections import Counter, defaultdict

with open("DataAnalysisExpert/slate_2026-04-01.json", encoding="utf-8") as f:
    picks = json.load(f)

print(f"Total picks: {len(picks)}")
recommended = [p for p in picks if p["is_recommended"] == "True"]
not_rec = [p for p in picks if p["is_recommended"] != "True"]
print(f"Recommended: {len(recommended)} | Not recommended: {len(not_rec)}")
print()

# ── 1. HOME vs AWAY BIAS ──
print("=" * 60)
print("1. HOME vs AWAY BIAS")
print("=" * 60)
home_picks = [p for p in recommended if p["home_team"] in p["pick"]]
away_picks = [p for p in recommended if p["away_team"] in p["pick"]]
neutral = [p for p in recommended if p["home_team"] not in p["pick"] and p["away_team"] not in p["pick"]]
print(f"  Home team picks: {len(home_picks)} ({len(home_picks)/len(recommended)*100:.0f}%)")
print(f"  Away team picks: {len(away_picks)} ({len(away_picks)/len(recommended)*100:.0f}%)")
print(f"  Neutral (O/U):   {len(neutral)} ({len(neutral)/len(recommended)*100:.0f}%)")

home_ev = sum(p["no_vig_ev"] for p in home_picks) / len(home_picks) if home_picks else 0
away_ev = sum(p["no_vig_ev"] for p in away_picks) / len(away_picks) if away_picks else 0
print(f"  Avg EV — Home: {home_ev*100:.1f}% | Away: {away_ev*100:.1f}%")
print()

# ── 2. OVER vs UNDER BIAS ──
print("=" * 60)
print("2. OVER vs UNDER BIAS")
print("=" * 60)
overs = [p for p in recommended if "Over" in p["pick"]]
unders = [p for p in recommended if "Under" in p["pick"]]
sides = [p for p in recommended if "Over" not in p["pick"] and "Under" not in p["pick"]]
print(f"  Overs:  {len(overs)} ({len(overs)/len(recommended)*100:.0f}%)")
print(f"  Unders: {len(unders)} ({len(unders)/len(recommended)*100:.0f}%)")
print(f"  Sides:  {len(sides)} ({len(sides)/len(recommended)*100:.0f}%)")
if overs:
    print(f"  Avg Over model_prob:  {sum(p['model_prob'] for p in overs)/len(overs)*100:.1f}%")
if unders:
    print(f"  Avg Under model_prob: {sum(p['model_prob'] for p in unders)/len(unders)*100:.1f}%")
print()

# ── 3. MARKET TYPE DISTRIBUTION ──
print("=" * 60)
print("3. MARKET TYPE DISTRIBUTION")
print("=" * 60)
by_market = Counter(p["market_type"] for p in recommended)
for mkt, cnt in by_market.most_common():
    subset = [p for p in recommended if p["market_type"] == mkt]
    avg_ev = sum(p["no_vig_ev"] for p in subset) / len(subset)
    avg_mp = sum(p["model_prob"] for p in subset) / len(subset)
    avg_ip = sum(p["implied_prob"] for p in subset) / len(subset)
    print(f"  {mkt:15s}: {cnt:2d} picks | Avg EV {avg_ev*100:+6.1f}% | Model {avg_mp*100:.1f}% vs Implied {avg_ip*100:.1f}% | Gap {(avg_mp-avg_ip)*100:+.1f}pp")
print()

# ── 4. MODEL PROBABILITY CALIBRATION ──
print("=" * 60)
print("4. MODEL PROB vs IMPLIED PROB (CALIBRATION GAP)")
print("=" * 60)
all_gaps = [(p["model_prob"] - p["implied_prob"]) for p in recommended]
avg_gap = sum(all_gaps) / len(all_gaps)
min_gap = min(all_gaps)
max_gap = max(all_gaps)
print(f"  Avg gap (model - implied): {avg_gap*100:+.1f}pp")
print(f"  Min gap: {min_gap*100:+.1f}pp | Max gap: {max_gap*100:+.1f}pp")
print(f"  All picks have model > implied: {all(g > 0 for g in all_gaps)}")

# Check for unrealistically high model probs
high_prob = [p for p in recommended if p["model_prob"] > 0.85]
print(f"  Picks with model_prob > 85%: {len(high_prob)}/{len(recommended)} ({len(high_prob)/len(recommended)*100:.0f}%)")
very_high = [p for p in recommended if p["model_prob"] > 0.90]
print(f"  Picks with model_prob > 90%: {len(very_high)}/{len(recommended)} ({len(very_high)/len(recommended)*100:.0f}%)")
print()

# ── 5. EXPECTED SCORE BIAS ──
print("=" * 60)
print("5. EXPECTED SCORE ANALYSIS")
print("=" * 60)
games = {}
for p in picks:
    g = p["game"]
    if g not in games:
        games[g] = {"home": p["home_team"], "away": p["away_team"],
                     "exp_home": p["exp_home_score"], "exp_away": p["exp_away_score"]}

for g, info in games.items():
    total = info["exp_home"] + info["exp_away"]
    diff = info["exp_home"] - info["exp_away"]
    print(f"  {g:15s}: Home {info['exp_home']:.1f} - Away {info['exp_away']:.1f} | Total {total:.1f} | Margin {diff:+.1f}")

all_home = [v["exp_home"] for v in games.values()]
all_away = [v["exp_away"] for v in games.values()]
all_totals = [v["exp_home"] + v["exp_away"] for v in games.values()]
print(f"\n  Avg home score: {sum(all_home)/len(all_home):.2f}")
print(f"  Avg away score: {sum(all_away)/len(all_away):.2f}")
print(f"  Avg total:      {sum(all_totals)/len(all_totals):.2f}")
print(f"  Home advantage: {(sum(all_home)/len(all_home)) - (sum(all_away)/len(all_away)):+.2f} runs")
print()

# ── 6. KELLY / EV SANITY CHECK ──
print("=" * 60)
print("6. KELLY & EV SANITY CHECK")
print("=" * 60)
kelly_values = [p["kelly"] for p in recommended]
ev_values = [p["no_vig_ev"] for p in recommended]
print(f"  Kelly — Min: {min(kelly_values)*100:.1f}% | Max: {max(kelly_values)*100:.1f}% | Avg: {sum(kelly_values)/len(kelly_values)*100:.1f}%")
print(f"  EV    — Min: {min(ev_values)*100:.1f}% | Max: {max(ev_values)*100:.1f}% | Avg: {sum(ev_values)/len(ev_values)*100:.1f}%")
extreme_kelly = [p for p in recommended if p["kelly"] > 0.5]
print(f"  Kelly > 50% (over-aggressive): {len(extreme_kelly)}/{len(recommended)}")
for p in extreme_kelly:
    print(f"    {p['pick']:25s} Kelly={p['kelly']*100:.1f}% EV={p['no_vig_ev']*100:.1f}% ModelProb={p['model_prob']*100:.1f}%")
print()

# ── 7. FAVORITE vs UNDERDOG BIAS ──
print("=" * 60)
print("7. FAVORITE vs UNDERDOG BIAS")
print("=" * 60)
fav_picks = [p for p in recommended if p["odds_current"] < 0]
dog_picks = [p for p in recommended if p["odds_current"] > 0]
print(f"  Favorites (neg odds): {len(fav_picks)} ({len(fav_picks)/len(recommended)*100:.0f}%)")
print(f"  Underdogs (pos odds): {len(dog_picks)} ({len(dog_picks)/len(recommended)*100:.0f}%)")
if fav_picks:
    print(f"  Avg fav EV: {sum(p['no_vig_ev'] for p in fav_picks)/len(fav_picks)*100:.1f}%")
if dog_picks:
    print(f"  Avg dog EV: {sum(p['no_vig_ev'] for p in dog_picks)/len(dog_picks)*100:.1f}%")
print()

# ── 8. PER-GAME RECOMMENDATION COUNT ──
print("=" * 60)
print("8. PER-GAME PICK DENSITY")
print("=" * 60)
by_game = Counter(p["game"] for p in recommended)
for g, cnt in by_game.most_common():
    print(f"  {g:15s}: {cnt} recommended picks")
print(f"\n  Avg picks per game: {len(recommended)/len(by_game):.1f}")

# ── VERDICT ──
print()
print("=" * 60)
print("BIAS VERDICT")
print("=" * 60)
issues = []
if len(overs) > 0 and len(unders) == 0:
    issues.append(f"OVER BIAS: {len(overs)} overs, 0 unders — model never predicts under")
if avg_gap > 0.25:
    issues.append(f"OVERCONFIDENCE: Avg model-implied gap is {avg_gap*100:.1f}pp (>25pp)")
if len(high_prob) / len(recommended) > 0.6:
    issues.append(f"CALIBRATION: {len(high_prob)}/{len(recommended)} picks have model_prob >85%")
if len(extreme_kelly) / len(recommended) > 0.4:
    issues.append(f"KELLY INFLATION: {len(extreme_kelly)}/{len(recommended)} picks have Kelly >50%")
home_ratio = len(home_picks) / (len(home_picks) + len(away_picks)) if (len(home_picks) + len(away_picks)) > 0 else 0
if home_ratio > 0.75:
    issues.append(f"HOME BIAS: {home_ratio*100:.0f}% of side picks are on home team")
avg_total = sum(all_totals) / len(all_totals)
if avg_total > 11.0:
    issues.append(f"SCORE INFLATION: Avg predicted total is {avg_total:.1f} (MLB avg ~8.5)")

if issues:
    print("  ⚠️  ISSUES FOUND:")
    for iss in issues:
        print(f"    • {iss}")
else:
    print("  ✅ No major systematic biases detected")
