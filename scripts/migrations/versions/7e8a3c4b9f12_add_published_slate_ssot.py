"""add published slate ssot tables

Revision ID: 7e8a3c4b9f12
Revises: 3aa6e699a436
Create Date: 2026-03-31 18:05:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7e8a3c4b9f12"
down_revision: str | None = "3aa6e699a436"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
    CREATE TABLE IF NOT EXISTS published_slate (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        slate_date  TEXT NOT NULL,
        run_id      TEXT NOT NULL,
        row_order   INTEGER NOT NULL,
        game        TEXT NOT NULL,
        segment     TEXT NOT NULL,
        market_type TEXT NOT NULL,
        pick        TEXT NOT NULL,
        row_json    TEXT NOT NULL,
        created_at  TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_published_slate_date_order ON published_slate(slate_date, row_order);"
    )
    op.execute(
        """
    CREATE TABLE IF NOT EXISTS slate_manifests (
        slate_date    TEXT PRIMARY KEY,
        run_id        TEXT NOT NULL,
        status        TEXT NOT NULL,
        checksum      TEXT NOT NULL,
        manifest_json TEXT NOT NULL,
        created_at    TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """
    )


def downgrade() -> None:
    op.drop_table("slate_manifests")
    op.drop_table("published_slate")
