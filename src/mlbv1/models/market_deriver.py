# mypy: ignore-errors

import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam


class MarketDeriver:
    """Derives exact probabilities for moneyline, spread, and totals using Poisson & Skellam distributions."""

    @staticmethod
    def derive_markets(
        f5_home: pd.Series,
        f5_away: pd.Series,
        fg_home: pd.Series,
        fg_away: pd.Series,
        lines: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Takes Series of expected runs and a DataFrame of lines (e.g. spread, total_runs) and computes all probabilities.
        """
        results = pd.DataFrame(index=fg_home.index)

        # Helper for a single half (F5 or Full Game)
        def compute_half(home_mu: pd.Series, away_mu: pd.Series, prefix: str):
            # Team Totals — use model expected runs as fallback when no market
            # line exists, so over/under stays ~50/50 instead of biasing.
            home_tt_line = lines.get(f"{prefix}home_tt")
            away_tt_line = lines.get(f"{prefix}away_tt")
            if isinstance(home_tt_line, pd.Series):
                home_tt_line = home_tt_line.fillna(home_mu)
            elif home_tt_line is None:
                home_tt_line = home_mu
            if isinstance(away_tt_line, pd.Series):
                away_tt_line = away_tt_line.fillna(away_mu)
            elif away_tt_line is None:
                away_tt_line = away_mu

            # Poisson SF: P(X > k) = 1 - cdf(k).
            # E.g. TT = 3.5 -> P(X >= 4) -> sf(3)
            # P(X > 3.5) = 1 - cdf(floor(3.5)) = sf(3)
            def prob_over_total(mu, line):
                k = np.floor(line) if isinstance(line, (int, float)) else np.floor(line.fillna(0))
                return poisson.sf(k, mu)

            results[f"{prefix}home_tt_over_prob"] = prob_over_total(
                home_mu, home_tt_line
            )
            results[f"{prefix}away_tt_over_prob"] = prob_over_total(
                away_mu, away_tt_line
            )

            # Game Totals — when no market line is available, use the model's
            # own expected total so over/under probabilities stay ~50/50
            # rather than biasing toward under with a fixed 8.5 fallback.
            sum_mu = home_mu + away_mu
            total_line = lines.get(f"{prefix}total_runs")
            if isinstance(total_line, pd.Series):
                # Use model total as fallback where market line is missing
                total_line = total_line.fillna(sum_mu)
            elif total_line is None:
                total_line = sum_mu
            results[f"{prefix}over_total_prob"] = prob_over_total(sum_mu, total_line)

            # Spread Cover
            # Margin = Home - Away
            # Margin + Spread > 0 => Margin > -Spread => Margin >= floor(-Spread + 1)
            spread_line = lines.get(f"{prefix}spread", -1.5)
            # Probability Home covers spread
            # We want P(Margin > -Spread).
            # For each game: sum of skellam pmf from k=ceil(-spread + 0.1) to infinity
            # Or 1 - skellam.cdf(k, mu1, mu2) where k is floor(-spread)

            def skellam_sf(margin_threshold, mu1, mu2):
                # P(Margin > threshold) = 1 - P(Margin <= threshold) = sf(threshold)
                return skellam.sf(margin_threshold, mu1, mu2)

            spread_threshold = np.floor(-spread_line)
            results[f"{prefix}home_spread_cover_prob"] = skellam_sf(
                spread_threshold, home_mu, away_mu
            )

            # Moneyline
            # P(Home Win) = P(Margin > 0), P(Away Win) = P(Margin < 0), P(Tie) = P(Margin == 0)
            p_home_win = skellam.sf(0, home_mu, away_mu)
            p_away_win = skellam.cdf(-1, home_mu, away_mu)
            p_tie = skellam.pmf(0, home_mu, away_mu)

            # Moneyline is typically Tie No Bet for F5 or baseball has extra innings.
            # Adjust prob assuming tie implies a push/refund, so we want prob of winning GIVEN there is a winner.
            # Base probability of home winning given no tie: p_win / (1 - p_tie)
            p_no_tie = 1.0 - p_tie
            # To avoid division by zero (impossible with our models, but still):
            results[f"{prefix}home_ml_prob"] = np.where(
                p_no_tie > 0, p_home_win / p_no_tie, 0.5
            )
            results[f"{prefix}away_ml_prob"] = np.where(
                p_no_tie > 0, p_away_win / p_no_tie, 0.5
            )
            results[f"{prefix}tie_prob"] = p_tie

        # Run for Full Game
        compute_half(fg_home, fg_away, prefix="")

        # Run for F5
        compute_half(f5_home, f5_away, prefix="f5_")

        return results

