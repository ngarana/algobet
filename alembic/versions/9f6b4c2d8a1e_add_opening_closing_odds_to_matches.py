"""add opening and closing odds snapshots to matches

Revision ID: 9f6b4c2d8a1e
Revises: 678575054c96
Create Date: 2026-05-17 20:45:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9f6b4c2d8a1e"
down_revision: str | Sequence[str] | None = "678575054c96"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


ODDS_COLUMNS = (
    "opening_odds_home",
    "opening_odds_draw",
    "opening_odds_away",
    "closing_odds_home",
    "closing_odds_draw",
    "closing_odds_away",
    "opening_avg_home_odds",
    "opening_avg_draw_odds",
    "opening_avg_away_odds",
    "closing_avg_home_odds",
    "closing_avg_draw_odds",
    "closing_avg_away_odds",
    "opening_max_home_odds",
    "opening_max_draw_odds",
    "opening_max_away_odds",
    "closing_max_home_odds",
    "closing_max_draw_odds",
    "closing_max_away_odds",
    "opening_asian_handicap",
    "opening_asian_handicap_away",
    "opening_asian_handicap_line",
    "closing_asian_handicap",
    "closing_asian_handicap_away",
    "closing_asian_handicap_line",
    "opening_over_under_25",
    "opening_over_under_25_under",
    "opening_over_under_line",
    "closing_over_under_25",
    "closing_over_under_25_under",
    "closing_over_under_line",
)


def upgrade() -> None:
    """Upgrade schema."""
    for column in ODDS_COLUMNS:
        op.add_column("matches", sa.Column(column, sa.Float(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    for column in reversed(ODDS_COLUMNS):
        op.drop_column("matches", column)
