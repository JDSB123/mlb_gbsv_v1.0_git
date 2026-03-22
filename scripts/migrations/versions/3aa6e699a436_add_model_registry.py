"""add model registry

Revision ID: 3aa6e699a436
Revises: 2b81c85f23ea
Create Date: 2026-03-22 19:36:23.419747

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3aa6e699a436"
down_revision: str | None = "2b81c85f23ea"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
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
    )


def downgrade() -> None:
    op.drop_table("model_registry")
