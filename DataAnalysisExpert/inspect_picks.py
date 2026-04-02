import json

rows = json.load(open("DataAnalysisExpert/picks_e2e_test.json"))
rec = [r for r in rows if r["is_recommended"]]
print(f"=== {len(rec)} RECOMMENDED PICKS ===\n")

for r in sorted(rec, key=lambda x: -x["model_prob"]):
    gap = r["model_prob"] - r["implied_prob"]
    print(f"  {r['pick']:30s} mkt={r['market_type']:10s} "
          f"model_prob={r['model_prob']:.3f} implied={r['implied_prob']:.3f} "
          f"gap={gap:+.3f} kelly={r['kelly']*100:.1f}% ev={r['no_vig_ev']:+.3f}")

# Check the >90% ones
print(f"\n=== model_prob > 0.90 DETAIL ===")
for r in rows:
    if r["model_prob"] > 0.90:
        print(f"  Game: {r['game']}")
        print(f"  Pick: {r['pick']}")
        print(f"  model_prob={r['model_prob']:.4f}, implied_prob={r['implied_prob']:.4f}")
        print(f"  exp_home={r['exp_home_score']:.2f}, exp_away={r['exp_away_score']:.2f}")
        print(f"  market_type={r['market_type']}, segment={r.get('segment','?')}")
        print()

# Side picks breakdown
sides = [r for r in rec if r["market_type"] in ("ML", "Spread")]
print(f"=== SIDE PICKS DETAIL ({len(sides)}) ===")
for r in sides:
    is_home = r["home_team"] in r["pick"]
    direction = "HOME" if is_home else "AWAY"
    print(f"  {direction}: {r['pick']:30s} prob={r['model_prob']:.3f} implied={r['implied_prob']:.3f}")

# Check ALL spread picks (not just recommended)
print(f"\n=== ALL SPREAD PICKS ===")
for r in rows:
    if r["market_type"] == "Spread":
        is_home = r["home_team"] in r["pick"]
        direction = "HOME" if is_home else "AWAY"
        print(f"  {direction}: {r['pick']:30s} rec={r['is_recommended']}  prob={r['model_prob']:.3f} ev={r['no_vig_ev']:+.3f}")
