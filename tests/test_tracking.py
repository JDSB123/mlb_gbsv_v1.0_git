"""Tests for tracking database and ROI."""

from __future__ import annotations

import os
import tempfile

from mlbv1.tracking.database import PredictionRecord, RunRecord, TrackingDB
from mlbv1.tracking.roi import BankrollConfig, BankrollManager, BetRecommendation


class TestTrackingDB:
    """Tests for SQLite tracking database."""

    def setup_method(self) -> None:
        self.tmp_fd, self.tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(self.tmp_fd)
        self.db = TrackingDB(self.tmp_path)

    def teardown_method(self) -> None:
        os.unlink(self.tmp_path)

    def test_log_and_get_run(self) -> None:
        run = RunRecord(
            run_id="test-run-1",
            model_name="random_forest",
            loader="synthetic",
            accuracy=0.65,
            roi=0.05,
            sharpe=1.2,
        )
        self.db.log_run(run)
        runs = self.db.get_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == "test-run-1"
        assert runs[0]["accuracy"] == 0.65

    def test_log_and_get_predictions(self) -> None:
        preds = [
            PredictionRecord(
                run_id="test-run-1",
                model_name="rf",
                game_date="2024-06-15",
                home_team="NYY",
                away_team="BOS",
                spread=-1.5,
                prediction=1,
                probability=0.72,
            ),
            PredictionRecord(
                run_id="test-run-1",
                model_name="rf",
                game_date="2024-06-15",
                home_team="LAD",
                away_team="CHC",
                spread=1.0,
                prediction=0,
                probability=0.45,
            ),
        ]
        count = self.db.log_predictions(preds)
        assert count == 2
        results = self.db.get_predictions(run_id="test-run-1")
        assert len(results) == 2

    def test_settle_predictions(self) -> None:
        preds = [
            PredictionRecord(
                run_id="test-run-1",
                model_name="rf",
                game_date="2024-06-15",
                home_team="NYY",
                away_team="BOS",
                spread=-1.5,
                prediction=1,
                probability=0.72,
            ),
        ]
        self.db.log_predictions(preds)
        settled = self.db.settle_predictions(
            game_date="2024-06-15",
            home_team="NYY",
            actual_home_score=5,
            actual_away_score=3,
        )
        assert settled == 1

        results = self.db.get_predictions(settled=True)
        assert len(results) == 1
        assert results[0]["actual_home_score"] == 5

    def test_roi_summary(self) -> None:
        preds = [
            PredictionRecord(
                run_id="test-run-1",
                model_name="rf",
                game_date="2024-06-15",
                home_team="NYY",
                away_team="BOS",
                spread=-1.5,
                prediction=1,
                probability=0.72,
            ),
        ]
        self.db.log_predictions(preds)
        self.db.settle_predictions("2024-06-15", "NYY", 5, 3)

        summary = self.db.get_roi_summary()
        assert summary["total_bets"] == 1
        assert summary["wins"] == 1
        assert summary["win_rate"] == 1.0

    def test_model_comparison(self) -> None:
        for model in ["rf", "lr"]:
            preds = [
                PredictionRecord(
                    run_id=f"run-{model}",
                    model_name=model,
                    game_date="2024-06-15",
                    home_team="NYY",
                    away_team="BOS",
                    spread=-1.5,
                    prediction=1,
                    probability=0.65,
                ),
            ]
            self.db.log_predictions(preds)

        comparison = self.db.get_model_comparison()
        assert len(comparison) == 2


class TestBankrollManager:
    """Tests for Kelly criterion bankroll management."""

    def test_kelly_size_positive_edge(self) -> None:
        mgr = BankrollManager()
        size = mgr.kelly_size(prob_win=0.60, odds=-110)
        assert size > 0
        assert size < 0.25  # quarter kelly

    def test_kelly_size_no_edge(self) -> None:
        mgr = BankrollManager()
        size = mgr.kelly_size(prob_win=0.45, odds=-110)
        assert size == 0.0

    def test_recommend_bet_with_edge(self) -> None:
        mgr = BankrollManager(BankrollConfig(initial_balance=10000))
        rec = mgr.recommend_bet(
            game_date="2024-06-15",
            home_team="NYY",
            away_team="BOS",
            spread=-1.5,
            probability=0.65,
            home_moneyline=-120,
        )
        assert rec is not None
        assert rec.side == "home"
        assert rec.recommended_bet > 0
        assert rec.edge > 0

    def test_recommend_bet_no_edge(self) -> None:
        mgr = BankrollManager()
        rec = mgr.recommend_bet(
            game_date="2024-06-15",
            home_team="NYY",
            away_team="BOS",
            spread=-1.5,
            probability=0.50,
        )
        assert rec is None

    def test_record_result(self) -> None:
        mgr = BankrollManager(BankrollConfig(initial_balance=10000))
        bet = BetRecommendation(
            game_date="2024-06-15",
            home_team="NYY",
            away_team="BOS",
            side="home",
            spread=-1.5,
            probability=0.65,
            edge=0.10,
            kelly_bet=200,
            recommended_bet=100,
            moneyline=-110,
        )
        balance = mgr.record_result(bet, won=True)
        assert balance > 10000

    def test_get_stats(self) -> None:
        mgr = BankrollManager(BankrollConfig(initial_balance=10000))
        bet = BetRecommendation(
            game_date="2024-06-15",
            home_team="NYY",
            away_team="BOS",
            side="home",
            spread=-1.5,
            probability=0.65,
            edge=0.10,
            kelly_bet=200,
            recommended_bet=100,
            moneyline=-110,
        )
        mgr.record_result(bet, won=True)
        mgr.record_result(bet, won=False)
        stats = mgr.get_stats()
        assert stats["total_bets"] == 2
        assert "max_drawdown" in stats
        assert "sharpe" in stats

    def test_implied_probability(self) -> None:
        assert abs(BankrollManager._implied_probability(-110) - 0.5238) < 0.01
        assert abs(BankrollManager._implied_probability(150) - 0.40) < 0.01
        assert abs(BankrollManager._implied_probability(-200) - 0.6667) < 0.01
