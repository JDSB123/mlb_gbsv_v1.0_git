"""Pure math utilities for odds conversion, EV, and Kelly sizing."""

from __future__ import annotations


def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds."""
    if odds >= 100:
        return 1 + odds / 100
    elif odds <= -100:
        return 1 + 100 / abs(odds)
    return 2.0  # fallback for zeros


def implied_prob(odds: float) -> float:
    """Implied probability from American odds (includes vig)."""
    if odds >= 100:
        return 100 / (odds + 100)
    elif odds <= -100:
        return abs(odds) / (abs(odds) + 100)
    return 0.5


def no_vig_prob(side_odds: float, other_odds: float) -> float:
    """Remove the vig and return the 'true' implied probability for the side bet."""
    p_side = implied_prob(side_odds)
    p_other = implied_prob(other_odds)
    total = p_side + p_other
    if total == 0:
        return 0.5
    return p_side / total


def no_vig_ev(model_prob: float, decimal_odds: float) -> float:
    """Expected value given model probability and decimal odds."""
    return model_prob * decimal_odds - 1.0


def kelly_fraction(model_prob: float, decimal_odds: float) -> float:
    """Quarter-Kelly criterion fraction (capped at 0)."""
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    f = (model_prob * b - (1 - model_prob)) / b
    return max(0.0, f * 0.25)
