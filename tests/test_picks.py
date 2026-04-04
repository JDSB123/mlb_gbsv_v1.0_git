"""Tests for mlbv1.picks package — odds math, rationale, quality gates."""

from __future__ import annotations

import pandas as pd

from mlbv1.picks.odds_math import (
    american_to_decimal,
    implied_prob,
    kelly_fraction,
    no_vig_ev,
    no_vig_prob,
)
from mlbv1.picks.quality import (
    compute_diversification_metrics,
    compute_quality_metrics,
    run_quality_gates,
)
from mlbv1.picks.rationale import build_rationale

# ── odds_math ────────────────────────────────────────────────────────────

def test_american_to_decimal_positive() -> None:
    assert american_to_decimal(200) == 3.0


def test_american_to_decimal_negative() -> None:
    assert american_to_decimal(-200) == 1.5


def test_american_to_decimal_zero_fallback() -> None:
    assert american_to_decimal(0) == 2.0


def test_implied_prob_favorite() -> None:
    p = implied_prob(-150)
    assert 0.59 < p < 0.61


def test_implied_prob_underdog() -> None:
    p = implied_prob(150)
    assert 0.39 < p < 0.41


def test_no_vig_prob_sums_to_one() -> None:
    p1 = no_vig_prob(-110, -110)
    p2 = no_vig_prob(-110, -110)
    assert abs(p1 + p2 - 1.0) < 0.01


def test_no_vig_ev_positive() -> None:
    # 60% model prob at +100 (decimal 2.0) → EV = 0.6 * 2.0 - 1 = 0.2
    assert abs(no_vig_ev(0.6, 2.0) - 0.2) < 0.001


def test_kelly_fraction_no_edge() -> None:
    # 50% at -110 (decimal ~1.909) → no edge → kelly should be 0
    assert kelly_fraction(0.5, 1.909) == 0.0


def test_kelly_fraction_strong_edge() -> None:
    k = kelly_fraction(0.65, 2.0)
    assert k > 0


# ── rationale ────────────────────────────────────────────────────────────

def test_build_rationale_returns_string() -> None:
    result = build_rationale(
        model_prob=0.58,
        implied_prob_val=0.52,
        ev=0.05,
        kelly=0.02,
        confidence=0.6,
        cold_start=False,
        n_models=4,
        line_move="—",
        exp_home=4.2,
        exp_away=3.8,
        segment="FG",
        market_type="ML",
        home="NYY",
        away="BOS",
    )
    assert isinstance(result, str)
    assert "Model:" in result
    assert "Fundamentals:" in result


def test_build_rationale_cold_start() -> None:
    result = build_rationale(
        model_prob=0.55,
        implied_prob_val=0.50,
        ev=0.03,
        kelly=0.01,
        confidence=0.3,
        cold_start=True,
        n_models=2,
        line_move="—",
        exp_home=4.0,
        exp_away=4.0,
        segment="FG",
        market_type="Spread",
        home="LAD",
        away="ATL",
    )
    assert "COLD START" in result


# ── quality ──────────────────────────────────────────────────────────────

def _make_rows(n: int = 5, **overrides: object) -> list[dict]:
    base = {
        "game": "BOS @ NYY",
        "home_team": "NYY",
        "away_team": "BOS",
        "segment": "FG",
        "market_type": "ML",
        "pick": "NYY ML",
        "odds_current": -130,
        "odds_quality": "live",
        "model_prob": 0.58,
        "used_default_line": False,
        "used_default_counter_odds": False,
        "model_count": 4,
        "is_recommended": True,
    }
    base.update(overrides)
    return [dict(base) for _ in range(n)]


def test_compute_quality_metrics_empty() -> None:
    m = compute_quality_metrics([])
    assert m["row_count"] == 0


def test_compute_quality_metrics_basic() -> None:
    m = compute_quality_metrics(_make_rows(10))
    assert m["row_count"] == 10
    assert m["live_odds_rate"] == 1.0
    assert m["unk_team_rows"] == 0


def test_run_quality_gates_pass() -> None:
    # Each row must be unique on (game, segment, market_type, pick) to avoid duplicate gate.
    rows = []
    for i in range(10):
        rows.append({
            **_make_rows(1)[0],
            "game": f"Team{i} @ NYY",
            "pick": f"NYY ML {i}",
        })
    m = compute_quality_metrics(rows)
    passed, gates = run_quality_gates(m)
    assert passed
    assert all(g["passed"] for g in gates)


def test_run_quality_gates_fail_on_unknown_teams() -> None:
    rows = _make_rows(10, home_team="UNK")
    m = compute_quality_metrics(rows)
    passed, gates = run_quality_gates(m)
    assert not passed


def test_diversification_warning() -> None:
    rows = _make_rows(4, pick="Under 8.5") + _make_rows(1, pick="Over 8.5")
    df = pd.DataFrame(rows)
    result = compute_diversification_metrics(df)
    assert result["diversification_warning"] is True
    assert result["dominant_direction"] == "under"
