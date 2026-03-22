"""init schema

Revision ID: 2b81c85f23ea
Revises: 
Create Date: 2026-03-22 19:32:41.614254

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '2b81c85f23ea'
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id      TEXT    NOT NULL,
        model_name  TEXT    NOT NULL,
        market      TEXT    NOT NULL DEFAULT 'spread',
        game_date   TEXT    NOT NULL,
        home_team   TEXT    NOT NULL,
        away_team   TEXT    NOT NULL,
        spread      REAL    NOT NULL DEFAULT 0.0,
        prediction  INTEGER NOT NULL,
        probability REAL    NOT NULL DEFAULT 0.5,
        home_moneyline INTEGER DEFAULT -110,
        away_moneyline INTEGER DEFAULT -110,
        created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
        actual_home_score INTEGER,
        actual_away_score INTEGER,
        actual_result     INTEGER,
        settled           INTEGER NOT NULL DEFAULT 0,
        settled_at        TEXT
    );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_pred_run ON predictions(run_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_pred_game ON predictions(game_date, home_team, market);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_pred_settled ON predictions(settled);")
    
    op.execute("""
    CREATE TABLE IF NOT EXISTS runs (
        run_id        TEXT PRIMARY KEY,
        model_name    TEXT NOT NULL,
        target_market TEXT NOT NULL DEFAULT 'spread',
        loader        TEXT NOT NULL DEFAULT 'synthetic',
        accuracy      REAL,
        roi           REAL,
        sharpe        REAL,
        config_json   TEXT,
        created_at    TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """)
    
    op.execute("""
    CREATE TABLE IF NOT EXISTS bankroll (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id     TEXT    NOT NULL,
        game_date  TEXT    NOT NULL,
        bet_amount REAL    NOT NULL DEFAULT 100.0,
        payout     REAL    NOT NULL DEFAULT 0.0,
        balance    REAL    NOT NULL DEFAULT 0.0,
        created_at TEXT    NOT NULL DEFAULT (datetime('now'))
    );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_bankroll_date ON bankroll(game_date);")
    
    op.execute("""
    CREATE TABLE IF NOT EXISTS pipeline_status (
        id               INTEGER PRIMARY KEY CHECK (id = 1),
        last_run         TEXT,
        last_run_status  TEXT NOT NULL DEFAULT 'never_run',
        updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """)
    op.execute("""
    INSERT OR IGNORE INTO pipeline_status (id, last_run, last_run_status)
    VALUES (1, NULL, 'never_run');
    """)


def downgrade() -> None:
    op.drop_table('pipeline_status')
    op.drop_table('bankroll')
    op.drop_table('runs')
    op.drop_table('predictions')
