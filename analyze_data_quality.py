#!/usr/bin/env python3
"""Analyze why model only predicts home wins - check data quality and class balance"""

import os
import sys

sys.path.insert(0, "/home/arch/Coding/algobet")

os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_USER", "algobet")
os.environ.setdefault("POSTGRES_PASSWORD", "password")
os.environ.setdefault("POSTGRES_DB", "football")

from algobet.infrastructure.database import get_session
from algobet.models import Match


def main():
    with get_session() as session:
        # Get all matches with scores
        matches = (
            session.query(Match)
            .filter(Match.home_score != None, Match.away_score != None)
            .all()
        )

        if not matches:
            print("No matches found with scores")
            return

        print(f"=== Match Analysis ({len(matches)} matches) ===")

        # Outcome distribution
        home_wins = sum(1 for m in matches if m.home_score > m.away_score)
        draws = sum(1 for m in matches if m.home_score == m.away_score)
        away_wins = sum(1 for m in matches if m.home_score < m.away_score)

        print("\nOutcome distribution:")
        print(f"  Home wins: {home_wins} ({100 * home_wins / len(matches):.1f}%)")
        print(f"  Draws: {draws} ({100 * draws / len(matches):.1f}%)")
        print(f"  Away wins: {away_wins} ({100 * away_wins / len(matches):.1f}%)")

        # Check sample of matches for enriched_stats availability
        print("\n=== Sample Match Statistics Check ===")
        for i, m in enumerate(matches[:10]):
            stats = m.statistics
            has_stats = stats is not None
            if has_stats:
                xg_home = getattr(stats, "home_xg", None) if stats else None
                xg_away = getattr(stats, "away_xg", None) if stats else None
            print(
                f"Match {m.id}: stats={has_stats}"
                + (f", xg_home={xg_home}" if has_stats else "")
            )

        # Check player_stats
        print("\n=== Player Stats Check ===")
        total_matches = len(matches)
        with_player_stats = sum(
            1 for m in matches if m.player_stats and len(m.player_stats) > 0
        )
        with_match_stats = sum(1 for m in matches if m.statistics is not None)

        print(
            f"Matches with player_stats: {with_player_stats}/{total_matches} ({100 * with_player_stats / total_matches:.1f}%)"
        )
        print(
            f"Matches with match statistics: {with_match_stats}/{total_matches} ({100 * with_match_stats / total_matches:.1f}%)"
        )

        # Check if data is the issue
        percentage_with_stats = (with_player_stats + with_match_stats) / (
            2 * total_matches
        )
        if percentage_with_stats < 0.5:
            print("\n*** CRITICAL: Most matches lack enriched statistics! ***")
            print(
                "This explains why enriched_stats features are not being used - they're mostly zeros."
            )


if __name__ == "__main__":
    main()
