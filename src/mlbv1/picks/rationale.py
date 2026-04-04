"""Build categorized rationale bullets for pick justifications."""

from __future__ import annotations


def build_rationale(
    *,
    model_prob: float,
    implied_prob_val: float,
    ev: float,
    kelly: float,
    confidence: float,
    cold_start: bool,
    n_models: int,
    line_move: str,
    exp_home: float,
    exp_away: float,
    segment: str,
    market_type: str,
    home: str,
    away: str,
    feat: dict[str, float] | None = None,
) -> str:
    """Build categorized rationale — always >=3 of 6 categories."""
    cats: list[str] = []

    if cold_start:
        cats.append("\u26a0 COLD START \u2014 limited historical data, odds-driven")

    # 1. Market Context
    div = abs(model_prob - implied_prob_val)
    mkt = f"Model-market spread {div:.1%}"
    if div > 0.08:
        mkt = f"Large divergence ({div:.1%}) \u2014 sharp-side value"
    elif div > 0.04:
        mkt = f"Moderate divergence ({div:.1%})"
    if line_move and line_move != "\u2014":
        mkt += f" \u00b7 Line range: {line_move}"
    cats.append(f"\U0001f4ca Market: {mkt}")

    # 2. Team Fundamentals
    fund = f"Projected: {home} {exp_home:.1f} \u2013 {away} {exp_away:.1f}"
    if feat:
        hwr = feat.get("home_win_rate_long", 0)
        awr = feat.get("away_win_rate_long", 0)
        if hwr > 0 and awr > 0:
            fund = f"{home} L20 {hwr:.0%} vs {away} {awr:.0%} \u00b7 " + fund
        hrs = feat.get("home_runs_avg_short", 0)
        ars = feat.get("away_runs_avg_short", 0)
        if hrs > 0:
            fund += f" \u00b7 Scoring L5: {home} {hrs:.1f}, {away} {ars:.1f} R/G"
    cats.append(f"\u26be Fundamentals: {fund}")

    # 3. Model Confidence
    cats.append(
        f"\U0001f916 Model: {model_prob:.1%} vs implied {implied_prob_val:.1%} "
        f"({n_models} models) \u00b7 EV {ev:+.1%} \u00b7 Kelly {kelly:.2%} \u00b7 "
        f"Conf {confidence:.0%}"
    )

    # 4. Situational Factors
    sit: list[str] = []
    if feat:
        rh = feat.get("rest_days_home", 1)
        ra = feat.get("rest_days_away", 1)
        if rh >= 2 and rh > ra:
            sit.append(f"{home} rested ({rh:.0f}d off)")
        elif ra >= 2 and ra > rh:
            sit.append(f"{away} rested ({ra:.0f}d off)")
        if feat.get("is_weekend", 0):
            sit.append("Weekend slate")
    if segment == "F5":
        sit.append("First 5 \u2014 starter-driven")
    if sit:
        cats.append("\U0001f4cb Situational: " + " \u00b7 ".join(sit))

    # 5. Sentiment / Sharp Action
    if div > 0.06:
        sharp = "Model sees sharp-side value vs consensus"
        if line_move and line_move != "\u2014":
            sharp += " \u00b7 Book-to-book variance suggests instability"
        cats.append(f"\U0001f4b0 Sentiment: {sharp}")

    # 6. Historical Trends
    if feat and not cold_start:
        trends: list[str] = []
        hws = feat.get("home_win_rate_short", 0)
        hwl = feat.get("home_win_rate_long", 0)
        aws = feat.get("away_win_rate_short", 0)
        awl = feat.get("away_win_rate_long", 0)
        if hws > 0 and hwl > 0:
            if hws > hwl + 0.05:
                trends.append(f"{home} hot (L5 {hws:.0%} vs L20 {hwl:.0%})")
            elif hws < hwl - 0.05:
                trends.append(f"{home} cold (L5 {hws:.0%} vs L20 {hwl:.0%})")
        if aws > 0 and awl > 0:
            if aws > awl + 0.05:
                trends.append(f"{away} hot (L5 {aws:.0%} vs L20 {awl:.0%})")
            elif aws < awl - 0.05:
                trends.append(f"{away} cold (L5 {aws:.0%} vs L20 {awl:.0%})")
        if trends:
            cats.append("\U0001f4c8 Trends: " + " \u00b7 ".join(trends))

    return " | ".join(cats)
