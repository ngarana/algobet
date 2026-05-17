"""add odds columns to matches table

Revision ID: 678575054c96
Revises:
Create Date: 2026-05-17 17:49:52.440549

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "678575054c96"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "matches", sa.Column("odds_asian_handicap", sa.Float(), nullable=True)
    )
    op.add_column(
        "matches", sa.Column("odds_asian_handicap_line", sa.Float(), nullable=True)
    )
    op.add_column("matches", sa.Column("odds_over_under_25", sa.Float(), nullable=True))
    op.add_column(
        "matches", sa.Column("odds_over_under_line", sa.Float(), nullable=True)
    )
    op.add_column("matches", sa.Column("avg_home_odds", sa.Float(), nullable=True))
    op.add_column("matches", sa.Column("avg_draw_odds", sa.Float(), nullable=True))
    op.add_column("matches", sa.Column("avg_away_odds", sa.Float(), nullable=True))
    op.add_column("matches", sa.Column("max_home_odds", sa.Float(), nullable=True))
    op.add_column("matches", sa.Column("max_draw_odds", sa.Float(), nullable=True))
    op.add_column("matches", sa.Column("max_away_odds", sa.Float(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("matches", "max_away_odds")
    op.drop_column("matches", "max_draw_odds")
    op.drop_column("matches", "max_home_odds")
    op.drop_column("matches", "avg_away_odds")
    op.drop_column("matches", "avg_draw_odds")
    op.drop_column("matches", "avg_home_odds")
    op.drop_column("matches", "odds_over_under_line")
    op.drop_column("matches", "odds_over_under_25")
    op.drop_column("matches", "odds_asian_handicap_line")
    op.drop_column("matches", "odds_asian_handicap")
