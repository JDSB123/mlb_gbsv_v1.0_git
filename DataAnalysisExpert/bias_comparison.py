"""Compare bias metrics: original picks vs debiased picks."""
import json, sys
from collections import Counter

def load_picks(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "picks" in data:
        return data["picks"] if data["picks"] else data.get("rows", [])
    return data

# Load both datasets
original = load_picks("DataAnalysisExpert/slate_2026-04-01.json")
debiased = load_picks("DataAnalysisExpert/picks_debiased_2026-04-01.json")

def analyze(picks, label):
    rec = [p for p in picks if str(p.get("is_recommended", "")).lower() == "true"]
    all_p = picks
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Total picks: {len(all_p)} | Recommended: {len(rec)}")

    if not rec:
        print("  No recommended picks to analyze.")
        return {}

    # Over/Under
    overs = [p for p in rec if "Over" in p["pick"]]
    unders = [p for p in rec if "Under" in p["pick"]]
    sides = [p for p in rec if "Over" not in p["pick"] and "Under" not in p["pick"]]
    print(f"\n  Over/Under: {len(overs)} Over / {len(unders)} Under / {len(sides)} Side")

    # Home/Away (sides only)
    if sides:
        home = [p for p in sides if p["home_team"] in p["pick"]]
        away = [p for p in sides if p["away_team"] in p["pick"]]
        home_pct = len(home) / len(sides) * 100 if sides else 0
        print(f"  Home/Away:  {len(home)} Home ({home_pct:.0f}%) / {len(away)} Away ({100-home_pct:.0f}%)")

    # Score predictions
    games = {}
    for p in all_p:
        g = p["game"]
        if g not in games:
            games[g] = {"exp_home": p["exp_home_score"], "exp_away": p["exp_away_score"]}
    totals = [v["exp_home"] + v["exp_away"] for v in games.values()]
    avg_total = sum(totals) / len(totals) if totals else 0
    print(f"\n  Avg predicted total: {avg_total:.2f} (MLB avg ~8.63)")
    for g, v in games.items():
        t = v["exp_home"] + v["exp_away"]
        print(f"    {g:15s}: {v['exp_home']:.1f} - {v['exp_away']:.1f} = {t:.1f}")

    # Calibration
    gaps = [p["model_prob"] - p["implied_prob"] for p in rec]
    avg_gap = sum(gaps) / len(gaps)
    print(f"\n  Avg model-implied gap: {avg_gap*100:+.1f}pp")
    high85 = len([p for p in rec if p["model_prob"] > 0.85])
    high90 = len([p for p in rec if p["model_prob"] > 0.90])
    print(f"  model_prob > 85%: {high85}/{len(rec)} ({high85/len(rec)*100:.0f}%)")
    print(f"  model_prob > 90%: {high90}/{len(rec)} ({high90/len(rec)*100:.0f}%)")

    # Kelly
    kellys = [p["kelly"] for p in rec]
    avg_kelly = sum(kellys) / len(kellys)
    max_kelly = max(kellys)
    extreme = len([k for k in kellys if k > 0.5])
    print(f"\n  Kelly — Avg: {avg_kelly*100:.1f}% | Max: {max_kelly*100:.1f}%")
    print(f"  Kelly > 50%: {extreme}/{len(rec)}")

    # EV
    evs = [p["no_vig_ev"] for p in rec]
    avg_ev = sum(evs) / len(evs)
    print(f"  EV — Avg: {avg_ev*100:+.1f}% | Min: {min(evs)*100:+.1f}% | Max: {max(evs)*100:+.1f}%")

    # Confidence
    confs = [p["confidence"] for p in rec]
    avg_conf = sum(confs) / len(confs)
    print(f"  Confidence — Avg: {avg_conf*100:.1f}%")

    # Market distribution
    by_mkt = Counter(p["market_type"] for p in rec)
    print(f"\n  Market mix: {dict(by_mkt)}")

    return {
        "total": len(all_p),
        "recommended": len(rec),
        "overs": len(overs),
        "unders": len(unders),
        "avg_total": avg_total,
        "avg_gap": avg_gap,
        "high85": high85,
        "high90": high90,
        "avg_kelly": avg_kelly,
        "max_kelly": max_kelly,
        "extreme_kelly": extreme,
        "avg_ev": avg_ev,
        "avg_conf": avg_conf,
    }

orig_metrics = analyze(original, "ORIGINAL (Pre-Fix)")
debi_metrics = analyze(debiased, "DEBIASED (Post-Fix)")

if orig_metrics and debi_metrics:
    print(f"\n{'='*70}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*70}")
    comparisons = [
        ("Recommended picks", "recommended", "{}", "lower is more selective"),
        ("Overs", "overs", "{}", "was 13/0"),
        ("Unders", "unders", "{}", "was 0"),
        ("Avg predicted total", "avg_total", "{:.2f}", "MLB avg 8.63"),
        ("Avg model-implied gap", "avg_gap", "{:+.1f}pp", "was +30.8pp"),
        ("model_prob > 85%", "high85", "{}", "was 57%"),
        ("model_prob > 90%", "high90", "{}", "was excessive"),
        ("Avg Kelly", "avg_kelly", "{:.1%}", "was 61.4%"),
        ("Max Kelly", "max_kelly", "{:.1%}", "was near 100%"),
        ("Kelly > 50%", "extreme_kelly", "{}", "was 15/23"),
        ("Avg EV", "avg_ev", "{:+.1%}", "context"),
        ("Avg confidence", "avg_conf", "{:.1%}", "context"),
    ]
    for label, key, fmt, note in comparisons:
        o = orig_metrics[key]
        d = debi_metrics[key]
        if "pp" in fmt:
            o_str = fmt.format(o * 100)
            d_str = fmt.format(d * 100)
        else:
            o_str = fmt.format(o)
            d_str = fmt.format(d)
        arrow = "↓" if d < o else "↑" if d > o else "="
        print(f"  {label:25s}: {o_str:>10s} → {d_str:>10s}  {arrow}  ({note})")
