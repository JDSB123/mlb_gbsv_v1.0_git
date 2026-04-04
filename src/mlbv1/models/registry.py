"""Local SQLite Model Registry."""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "artifacts/tracking.db"

_REGISTRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_registry (
    version_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name      TEXT NOT NULL,
    model_type      TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    feature_names   TEXT NOT NULL,
    accuracy        REAL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    is_active       INTEGER NOT NULL DEFAULT 0
);
"""


@dataclass
class ModelVersion:
    """Registered Model."""

    version_id: int
    model_name: str
    model_type: str
    file_path: str
    feature_names: list[str]
    accuracy: float
    created_at: str
    is_active: bool


def _parse_feature_names(raw: str) -> list[str]:
    """Deserialize feature names stored as JSON array or legacy comma-separated."""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return raw.split(",")


class ModelRegistry:
    """SQLite-backed registry to track ML model versions and metrics."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
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
            conn.executescript(_REGISTRY_SCHEMA)

    def register_model(
        self,
        model_name: str,
        model_type: str,
        file_path: str,
        feature_names: list[str],
        accuracy: float,
    ) -> int:
        """Register a new model version."""
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO model_registry (model_name, model_type, file_path, feature_names, accuracy)
                   VALUES (?, ?, ?, ?, ?)""",
                (model_name, model_type, file_path, json.dumps(feature_names), accuracy),
            )
            return cur.lastrowid  # type: ignore

    def promote_to_production(self, version_id: int) -> None:
        """Mark a specific version as active and deactivate others of same type."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT model_name FROM model_registry WHERE version_id = ?",
                (version_id,),
            )
            row = cur.fetchone()
            if not row:
                raise ValueError("Model version not found.")
            name = row["model_name"]

            conn.execute(
                "UPDATE model_registry SET is_active = 0 WHERE model_name = ?", (name,)
            )
            conn.execute(
                "UPDATE model_registry SET is_active = 1 WHERE version_id = ?",
                (version_id,),
            )
            logger.info("Promoted %s version %d to production.", name, version_id)

    def get_production_model(self, model_name: str) -> ModelVersion | None:
        """Get the current active model version metadata."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM model_registry WHERE model_name = ? AND is_active = 1 LIMIT 1",
                (model_name,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return ModelVersion(
                version_id=row["version_id"],
                model_name=row["model_name"],
                model_type=row["model_type"],
                file_path=row["file_path"],
                feature_names=_parse_feature_names(row["feature_names"]),
                accuracy=row["accuracy"],
                created_at=row["created_at"],
                is_active=bool(row["is_active"]),
            )

    def get_all_production_models(self) -> list[ModelVersion]:
        """Get all currently active models."""
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM model_registry WHERE is_active = 1")
            rows = cur.fetchall()
            return [
                ModelVersion(
                    version_id=row["version_id"],
                    model_name=row["model_name"],
                    model_type=row["model_type"],
                    file_path=row["file_path"],
                    feature_names=_parse_feature_names(row["feature_names"]),
                    accuracy=row["accuracy"],
                    created_at=row["created_at"],
                    is_active=bool(row["is_active"]),
                )
                for row in rows
            ]
