"""Bankroll and ROI tracking utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BankrollConfig:
    """Bankroll management configuration."""

    initial_balance: float = 10_000.0
    base_unit: float = 100.0
    max_bet_pct: float = 0.05  # max 5% of bankroll per bet
    kelly_fraction: float = 0.25  # quarter-Kelly for safety
    min_edge: float = 0.02  # minimum predicted edge to bet
    min_confidence: float = 0.55  # minimum probability to bet


@dataclass
class BetRecommendation:
    """A recommended bet with sizing."""

    game_date: str
    home_team: str
    away_team: str
    side: str  # "home" or "away"
    spread: float
    probability: float
    edge: float
    kelly_bet: float
    recommended_bet: float
    moneyline: int


class BankrollManager:
    """Manage betting bankroll with Kelly criterion sizing."""

    def __init__(self, config: BankrollConfig | None = None) -> None:
        self.config = config or BankrollConfig()
        self.balance = self.config.initial_balance
        self.history: list[dict[str, Any]] = []

    def kelly_size(
        self,
        prob_win: float,
        odds: int = -110,
    ) -> float:
        """Calculate Kelly criterion bet size.

        Args:
            prob_win: Estimated probability of winning the bet.
            odds: American odds (e.g., -110, +150).

        Returns:
            Optimal fraction of bankroll to wager.
        """
        decimal_odds = 1 + 100.0 / abs(odds) if odds < 0 else 1 + odds / 100.0

        b = decimal_odds - 1  # net odds
        q = 1 - prob_win

        if b <= 0:
            return 0.0

        kelly = (b * prob_win - q) / b
        return max(kelly * self.config.kelly_fraction, 0.0)

    def recommend_bet(
        self,
        game_date: str,
        home_team: str,
        away_team: str,
        spread: float,
        probability: float,
        home_moneyline: int = -110,
        away_moneyline: int = -110,
    ) -> BetRecommendation | None:
        """Generate a bet recommendation with Kelly sizing.

        Returns None if the edge is below threshold.
        """
        # Determine which side to bet
        implied_prob = self._implied_probability(home_moneyline)
        edge = probability - implied_prob

        if probability >= self.config.min_confidence and edge >= self.config.min_edge:
            side = "home"
            odds = home_moneyline
        elif (1 - probability) >= self.config.min_confidence and (
            -edge >= self.config.min_edge
        ):
            side = "away"
            odds = away_moneyline
            probability = 1 - probability
            edge = probability - self._implied_probability(away_moneyline)
        else:
            return None  # No edge

        kelly_frac = self.kelly_size(probability, odds)
        kelly_bet = kelly_frac * self.balance
        max_bet = self.balance * self.config.max_bet_pct
        recommended = min(kelly_bet, max_bet)
        recommended = max(recommended, self.config.base_unit)

        if recommended > self.balance:
            return None  # Can't afford

        return BetRecommendation(
            game_date=game_date,
            home_team=home_team,
            away_team=away_team,
            side=side,
            spread=spread,
            probability=probability,
            edge=edge,
            kelly_bet=kelly_bet,
            recommended_bet=round(recommended, 2),
            moneyline=odds,
        )

    def record_result(self, bet: BetRecommendation, won: bool) -> float:
        """Record a bet result and update balance."""
        if won:
            if bet.moneyline < 0:
                payout = bet.recommended_bet * (100.0 / abs(bet.moneyline))
            else:
                payout = bet.recommended_bet * (bet.moneyline / 100.0)
            self.balance += payout
        else:
            payout = -bet.recommended_bet
            self.balance -= bet.recommended_bet

        self.history.append(
            {
                "date": bet.game_date,
                "bet": float(bet.recommended_bet),
                "payout": float(payout),
                "balance": float(self.balance),
            }
        )
        return self.balance

    def get_stats(self) -> dict[str, float]:
        """Calculate bankroll statistics."""
        if not self.history:
            return {
                "total_bets": 0,
                "total_wagered": 0.0,
                "net_profit": 0.0,
                "roi": 0.0,
                "max_drawdown": 0.0,
                "sharpe": 0.0,
                "current_balance": self.balance,
            }

        payouts = [h["payout"] for h in self.history]
        bets = [h["bet"] for h in self.history]
        balances = [h["balance"] for h in self.history]
        total_wagered = sum(bets)
        net = sum(payouts)

        # Max drawdown
        peak = self.config.initial_balance
        max_dd = 0.0
        for bal in balances:
            peak = max(peak, bal)
            dd = (peak - bal) / peak
            max_dd = max(max_dd, dd)

        # Sharpe ratio
        returns_arr = np.array(payouts)
        std = float(returns_arr.std(ddof=1)) if len(returns_arr) > 1 else 0.0
        sharpe = float(returns_arr.mean() / std) if std > 0 else 0.0

        return {
            "total_bets": len(self.history),
            "total_wagered": total_wagered,
            "net_profit": net,
            "roi": net / total_wagered if total_wagered > 0 else 0.0,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "current_balance": self.balance,
        }

    @staticmethod
    def _implied_probability(odds: int) -> float:
        """Convert American odds to implied probability."""
        if odds < 0:
            return abs(odds) / (abs(odds) + 100.0)
        return 100.0 / (odds + 100.0)
