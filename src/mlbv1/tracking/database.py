"""SQLite-backed prediction tracking database.

Stores every prediction alongside actual outcomes for historical
analysis, ROI tracking, and model comparison.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "artifacts/tracking.db"


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
    market: str = "spread"


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
    target_market: str = "spread"


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
        alembic_ini_path = Path(__file__).resolve().parents[3] / "alembic.ini"
        if not alembic_ini_path.exists():
            logger.warning(
                "alembic.ini not found at %s. Skipping auto-migration.",
                alembic_ini_path,
            )
            return

        alembic_cfg = Config(str(alembic_ini_path))
        alembic_cfg.set_main_option(
            "sqlalchemy.url", f"sqlite:///{self.db_path.absolute()}"
        )
        command.upgrade(alembic_cfg, "head")
        logger.info("Tracking DB auto-migrated at %s", self.db_path)

    # -- writes ---------------------------------------------------------------

    def log_run(self, run: RunRecord) -> None:
        """Record a training/prediction run."""
        with self._connect() as conn:
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO runs
                       (run_id, model_name, loader, accuracy, roi, sharpe, config_json, target_market)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run.run_id,
                        run.model_name,
                        run.loader,
                        run.accuracy,
                        run.roi,
                        run.sharpe,
                        run.config_json,
                        run.target_market,
                    ),
                )
            except sqlite3.OperationalError:
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
        if not predictions:
            return 0

        with self._connect() as conn:
            try:
                conn.executemany(
                    """INSERT INTO predictions
                       (run_id, model_name, game_date, home_team, away_team,
                        spread, prediction, probability, home_moneyline, away_moneyline, market)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                            p.market,
                        )
                        for p in predictions
                    ],
                )
            except sqlite3.OperationalError:
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
        *,
        f5_home_score: int | None = None,
        f5_away_score: int | None = None,
    ) -> int:
        """Settle predictions for a specific game with actual scores.

        If *f5_home_score* and *f5_away_score* are provided, F5 markets
        will be settled as well.
        """
        margin = actual_home_score - actual_away_score
        total_runs = actual_home_score + actual_away_score
        now = datetime.now(tz=UTC).isoformat()
        with self._connect() as conn:
            # Fetch matching unsettled predictions
            rows = conn.execute(
                """SELECT id, spread, COALESCE(market, 'spread') AS market FROM predictions
                   WHERE game_date LIKE ? AND home_team = ? AND settled = 0""",
                (f"{game_date}%", home_team),
            ).fetchall()
            count = 0
            for row in rows:
                market = str(row["market"])
                actual_result: int | None = None

                if market == "spread":
                    actual_result = 1 if (margin + float(row["spread"])) > 0 else 0
                elif market == "moneyline":
                    actual_result = 1 if margin > 0 else 0
                elif market == "total":
                    # The spread column stores the total line for total-market predictions.
                    actual_result = 1 if total_runs > float(row["spread"]) else 0
                elif market in {"f5_spread", "f5_moneyline", "f5_total"}:
                    if f5_home_score is None or f5_away_score is None:
                        continue  # Cannot settle F5 without inning-level scores
                    f5_margin = f5_home_score - f5_away_score
                    f5_total = f5_home_score + f5_away_score
                    if market == "f5_spread":
                        actual_result = 1 if (f5_margin + float(row["spread"])) > 0 else 0
                    elif market == "f5_moneyline":
                        actual_result = 1 if f5_margin > 0 else 0
                    elif market == "f5_total":
                        actual_result = 1 if f5_total > float(row["spread"]) else 0
                else:
                    actual_result = 1 if (margin + float(row["spread"])) > 0 else 0

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

    def publish_slate(
        self,
        slate_date: str,
        run_id: str,
        rows: list[dict[str, Any]],
        manifest: dict[str, Any],
        *,
        status: str = "published",
    ) -> str:
        """Atomically replace the published slate for a date and persist manifest.

        Returns the SHA256 checksum of serialized rows.
        """
        if not rows:
            raise ValueError("Cannot publish an empty slate")

        serialized_rows: list[str] = [
            json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)
            for row in rows
        ]
        checksum = hashlib.sha256("".join(serialized_rows).encode("utf-8")).hexdigest()

        with self._connect() as conn:
            conn.execute(
                "DELETE FROM published_slate WHERE slate_date = ?",
                (slate_date,),
            )
            conn.executemany(
                """INSERT INTO published_slate
                   (slate_date, run_id, row_order, game, segment, market_type, pick, row_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        slate_date,
                        run_id,
                        idx,
                        str(row.get("game", "")),
                        str(row.get("segment", "")),
                        str(row.get("market_type", "")),
                        str(row.get("pick", "")),
                        payload,
                    )
                    for idx, (row, payload) in enumerate(zip(rows, serialized_rows, strict=False))
                ],
            )
            conn.execute(
                """INSERT INTO slate_manifests
                   (slate_date, run_id, status, checksum, manifest_json, created_at)
                   VALUES (?, ?, ?, ?, ?, datetime('now'))
                   ON CONFLICT(slate_date)
                   DO UPDATE SET
                       run_id = excluded.run_id,
                       status = excluded.status,
                       checksum = excluded.checksum,
                       manifest_json = excluded.manifest_json,
                       created_at = datetime('now')""",
                (
                    slate_date,
                    run_id,
                    status,
                    checksum,
                    json.dumps(manifest, sort_keys=True, default=str),
                ),
            )
        logger.info(
            "Published slate SSOT for %s (%d rows, checksum=%s)",
            slate_date,
            len(rows),
            checksum[:12],
        )
        return checksum

    def try_start_pipeline(self) -> bool:
        """Atomically mark pipeline as running if no run is currently active."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT last_run_status FROM pipeline_status WHERE id = 1"
            ).fetchone()
            if row and row["last_run_status"] == "running":
                return False

            now = datetime.now(tz=UTC).isoformat()
            conn.execute(
                """UPDATE pipeline_status
                   SET last_run_status = 'running', updated_at = ?
                   WHERE id = 1""",
                (now,),
            )
            return True

    def finish_pipeline_run(self, status: str) -> None:
        """Mark pipeline run completion with final status."""
        final_status = status if status in {"success", "error"} else "error"
        now = datetime.now(tz=UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """UPDATE pipeline_status
                   SET last_run = ?, last_run_status = ?, updated_at = ?
                   WHERE id = 1""",
                (now, final_status, now),
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

    def get_published_slate(self, slate_date: str) -> list[dict[str, Any]]:
        """Return the canonical published slate rows for a date."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT row_json
                   FROM published_slate
                   WHERE slate_date = ?
                   ORDER BY row_order ASC""",
                (slate_date,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                out.append(json.loads(str(row["row_json"])))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed published_slate row for %s", slate_date)
        return out

    def get_slate_manifest(self, slate_date: str) -> dict[str, Any] | None:
        """Fetch manifest metadata for the published slate on a date."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT slate_date, run_id, status, checksum, manifest_json, created_at
                   FROM slate_manifests
                   WHERE slate_date = ?""",
                (slate_date,),
            ).fetchone()
        if not row:
            return None
        manifest_json = row["manifest_json"]
        parsed: dict[str, Any]
        try:
            parsed = json.loads(str(manifest_json))
        except json.JSONDecodeError:
            parsed = {}
        parsed.update(
            {
                "slate_date": row["slate_date"],
                "run_id": row["run_id"],
                "status": row["status"],
                "checksum": row["checksum"],
                "created_at": row["created_at"],
            }
        )
        return parsed

    def get_pipeline_status(self) -> dict[str, Any]:
        """Get the shared pipeline status used by health checks."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT last_run, last_run_status, updated_at FROM pipeline_status WHERE id = 1"
            ).fetchone()

        if not row:
            return {
                "last_run": None,
                "last_run_status": "never_run",
                "updated_at": datetime.now(tz=UTC).isoformat(),
            }

        return dict(row)
