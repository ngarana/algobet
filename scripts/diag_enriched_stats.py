#!/usr/bin/env python3
"""Diagnostic: check enriched stats (Understat/ESPN) coverage in the database."""

import os
import sys

sys.path.insert(0, "/home/arch/Coding/algobet")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_USER", "algobet")
os.environ.setdefault("POSTGRES_PASSWORD", "password")
os.environ.setdefault("POSTGRES_DB", "football")

# Import all models to initialize mappers before querying
from sqlalchemy import func, select

# Force import of all model modules so SQLAlchemy mappers resolve correctly
from algobet.infrastructure.database import get_session
from algobet.matches.models import Match, MatchStatistics, PlayerMatchStats


def main():
    with get_session() as session:
        total_finished = session.scalar(
            select(func.count(Match.id)).where(
                Match.status == "FINISHED",
                Match.home_score.is_not(None),
            )
        )

        with_understat = session.scalar(
            select(func.count(Match.id))
            .join(MatchStatistics)
            .where(
                Match.status == "FINISHED",
                Match.home_score.is_not(None),
                MatchStatistics.home_xg.is_not(None),
                MatchStatistics.away_xg.is_not(None),
            )
        )

        with_player_stats = session.scalar(
            select(func.count(func.distinct(PlayerMatchStats.match_id))).where(
                PlayerMatchStats.match_id.is_not(None)
            )
        )

        with_both = session.scalar(
            select(func.count(Match.id))
            .join(MatchStatistics)
            .where(
                Match.status == "FINISHED",
                Match.home_score.is_not(None),
                MatchStatistics.home_xg.is_not(None),
                MatchStatistics.away_xg.is_not(None),
                MatchStatistics.match_id.in_(
                    select(func.distinct(PlayerMatchStats.match_id))
                ),
            )
        )

        print(f"Total finished matches:        {total_finished}")
        print(
            f"With Understat xG:             {with_understat} ({with_understat / total_finished * 100:.1f}%)"
        )
        print(
            f"With player stats:             {with_player_stats} ({with_player_stats / total_finished * 100:.1f}%)"
        )
        print(
            f"With BOTH (enriched_stats):    {with_both} ({with_both / total_finished * 100:.1f}%)"
        )

        # Breakdown by tournament for matches with odds
        print("\n--- Coverage by tournament (matches with odds) ---")
        rows = session.execute(
            select(
                Match.tournament_id,
                func.count(Match.id).label("total"),
            )
            .where(
                Match.status == "FINISHED",
                Match.home_score.is_not(None),
                Match.odds_home.is_not(None),
            )
            .group_by(Match.tournament_id)
            .order_by(func.count(Match.id).desc())
        ).all()

        for tid, total in rows[:5]:
            enriched = session.scalar(
                select(func.count(Match.id))
                .join(MatchStatistics)
                .where(
                    Match.tournament_id == tid,
                    Match.status == "FINISHED",
                    Match.home_score.is_not(None),
                    Match.odds_home.is_not(None),
                    MatchStatistics.home_xg.is_not(None),
                    MatchStatistics.match_id.in_(
                        select(func.distinct(PlayerMatchStats.match_id))
                    ),
                )
            )
            print(
                f"  Tournament {tid}: {enriched}/{total} ({enriched / total * 100:.1f}%)"
            )


if __name__ == "__main__":
    main()
