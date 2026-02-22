"""SQLite-backed prediction tracking database.

Stores every prediction alongside actual outcomes for historical
analysis, ROI tracking, and model comparison.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "artifacts/tracking.db"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT    NOT NULL,
    model_name  TEXT    NOT NULL,
    game_date   TEXT    NOT NULL,
    home_team   TEXT    NOT NULL,
    away_team   TEXT    NOT NULL,
    spread      REAL    NOT NULL DEFAULT 0.0,
    prediction  INTEGER NOT NULL,
    probability REAL    NOT NULL DEFAULT 0.5,
    home_moneyline INTEGER DEFAULT -110,
    away_moneyline INTEGER DEFAULT -110,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    -- outcome fields (filled after game)
    actual_home_score INTEGER,
    actual_away_score INTEGER,
    actual_result     INTEGER,  -- 1 = spread covered, 0 = not
    settled           INTEGER NOT NULL DEFAULT 0,
    settled_at        TEXT
);

CREATE INDEX IF NOT EXISTS idx_pred_run ON predictions(run_id);
CREATE INDEX IF NOT EXISTS idx_pred_game ON predictions(game_date, home_team);
CREATE INDEX IF NOT EXISTS idx_pred_settled ON predictions(settled);

CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    loader     TEXT NOT NULL DEFAULT 'synthetic',
    accuracy   REAL,
    roi        REAL,
    sharpe     REAL,
    config_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS bankroll (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT    NOT NULL,
    game_date  TEXT    NOT NULL,
    bet_amount REAL    NOT NULL DEFAULT 100.0,
    payout     REAL    NOT NULL DEFAULT 0.0,
    balance    REAL    NOT NULL DEFAULT 0.0,
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_bankroll_date ON bankroll(game_date);
"""


@dataclass
class PredictionRecord:
    """Single prediction to track."""

    run_id: str
    model_name: str
    game_date: str
    home_team: str
    away_team: str
    spread: float
    prediction: int
    probability: float
    home_moneyline: int = -110
    away_moneyline: int = -110


@dataclass
class RunRecord:
    """Training or prediction run metadata."""

    run_id: str
    model_name: str
    loader: str = "synthetic"
    accuracy: float | None = None
    roi: float | None = None
    sharpe: float | None = None
    config_json: str = "{}"


class TrackingDB:
    """SQLite database for prediction tracking."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)
        logger.info("Tracking DB initialized at %s", self.db_path)

    # -- writes ---------------------------------------------------------------

    def log_run(self, run: RunRecord) -> None:
        """Record a training/prediction run."""
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO runs
                   (run_id, model_name, loader, accuracy, roi, sharpe, config_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    run.run_id,
                    run.model_name,
                    run.loader,
                    run.accuracy,
                    run.roi,
                    run.sharpe,
                    run.config_json,
                ),
            )

    def log_predictions(self, predictions: list[PredictionRecord]) -> int:
        """Batch-insert predictions. Returns count inserted."""
        with self._connect() as conn:
            conn.executemany(
                """INSERT INTO predictions
                   (run_id, model_name, game_date, home_team, away_team,
                    spread, prediction, probability, home_moneyline, away_moneyline)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        p.run_id,
                        p.model_name,
                        p.game_date,
                        p.home_team,
                        p.away_team,
                        p.spread,
                        p.prediction,
                        p.probability,
                        p.home_moneyline,
                        p.away_moneyline,
                    )
                    for p in predictions
                ],
            )
        logger.info(
            "Logged %d predictions for run %s", len(predictions), predictions[0].run_id
        )
        return len(predictions)

    def settle_predictions(
        self,
        game_date: str,
        home_team: str,
        actual_home_score: int,
        actual_away_score: int,
    ) -> int:
        """Settle predictions for a specific game with actual scores."""
        margin = actual_home_score - actual_away_score
        now = datetime.now(tz=UTC).isoformat()
        with self._connect() as conn:
            # Fetch matching unsettled predictions
            rows = conn.execute(
                """SELECT id, spread FROM predictions
                   WHERE game_date LIKE ? AND home_team = ? AND settled = 0""",
                (f"{game_date}%", home_team),
            ).fetchall()
            count = 0
            for row in rows:
                actual_result = 1 if (margin + row["spread"]) > 0 else 0
                conn.execute(
                    """UPDATE predictions
                       SET actual_home_score = ?, actual_away_score = ?,
                           actual_result = ?, settled = 1, settled_at = ?
                       WHERE id = ?""",
                    (
                        actual_home_score,
                        actual_away_score,
                        actual_result,
                        now,
                        row["id"],
                    ),
                )
                count += 1
        logger.info("Settled %d predictions for %s @ %s", count, home_team, game_date)
        return count

    def log_bankroll(
        self,
        run_id: str,
        game_date: str,
        bet_amount: float,
        payout: float,
        balance: float,
    ) -> None:
        """Record a bankroll entry."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO bankroll (run_id, game_date, bet_amount, payout, balance)
                   VALUES (?, ?, ?, ?, ?)""",
                (run_id, game_date, bet_amount, payout, balance),
            )

    # -- reads ----------------------------------------------------------------

    def get_predictions(
        self,
        run_id: str | None = None,
        settled: bool | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Query predictions with optional filters."""
        query = "SELECT * FROM predictions WHERE 1=1"
        params: list[Any] = []
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        if settled is not None:
            query += " AND settled = ?"
            params.append(1 if settled else 0)
        query += " ORDER BY game_date DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent runs."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_roi_summary(self, run_id: str | None = None) -> dict[str, Any]:
        """Calculate ROI summary from settled predictions."""
        query = """
            SELECT
                COUNT(*) as total_bets,
                SUM(CASE WHEN actual_result = prediction THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN actual_result != prediction THEN 1 ELSE 0 END) as losses,
                AVG(CASE WHEN actual_result = prediction THEN 1.0 ELSE 0.0 END) as win_rate
            FROM predictions WHERE settled = 1
        """
        params: list[Any] = []
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)

        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()

        if not row or row["total_bets"] == 0:
            return {
                "total_bets": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "roi": 0.0,
            }

        total = row["total_bets"]
        wins = row["wins"]
        stake = 100.0
        vig = 110.0
        win_payout = stake * (100.0 / vig)
        total_return = (wins * win_payout) - ((total - wins) * stake)
        roi = total_return / (total * stake)

        return {
            "total_bets": total,
            "wins": wins,
            "losses": row["losses"],
            "win_rate": row["win_rate"],
            "roi": roi,
            "total_wagered": total * stake,
            "net_profit": total_return,
        }

    def get_bankroll_history(self, run_id: str | None = None) -> list[dict[str, Any]]:
        """Get bankroll progression."""
        query = "SELECT * FROM bankroll"
        params: list[Any] = []
        if run_id:
            query += " WHERE run_id = ?"
            params.append(run_id)
        query += " ORDER BY game_date"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_model_comparison(self) -> list[dict[str, Any]]:
        """Compare model performance across runs."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT model_name,
                          COUNT(*) as total_predictions,
                          AVG(CASE WHEN settled=1 AND actual_result=prediction
                              THEN 1.0 ELSE 0.0 END) as accuracy,
                          COUNT(CASE WHEN settled=1 THEN 1 END) as settled_count
                   FROM predictions
                   GROUP BY model_name
                   ORDER BY accuracy DESC"""
            ).fetchall()
        return [dict(r) for r in rows]
