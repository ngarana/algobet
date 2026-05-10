#!/usr/bin/env python3
"""Diagnose WHICH generator is the bottleneck."""

import os
import sys
import time

sys.path.insert(0, "/home/arch/Coding/algobet")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_USER", "algobet")
os.environ.setdefault("POSTGRES_PASSWORD", "password")
os.environ.setdefault("POSTGRES_DB", "football")

import warnings

warnings.filterwarnings("ignore")

from algobet.infrastructure.database import get_session
from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.generators import (
    EnrichedStatsFeatureGenerator,
    HeadToHeadGenerator,
    StandingsFeatureGenerator,
    TeamFormGenerator,
    TemporalFeatureGenerator,
)
from algobet.predictions.features.pipeline import prepare_match_dataframe


def main():
    with get_session() as session:
        from sqlalchemy import and_

        from algobet.matches.models import Match

        matches = (
            session.query(Match)
            .filter(
                and_(
                    Match.status == "FINISHED",
                    Match.home_score.is_not(None),
                    Match.away_score.is_not(None),
                    Match.tournament_id == 359,
                )
            )
            .order_by(Match.match_date)
            .all()
        )
        print(f"Loaded {len(matches)} matches", flush=True)

        matches_df = prepare_match_dataframe(matches)

        all_team_ids = list(
            set(
                matches_df["home_team_id"].tolist()
                + matches_df["away_team_id"].tolist()
            )
        )
        max_match_date = matches_df["match_date"].max()

        repo = MatchRepository(session)
        repo.preload_team_matches(all_team_ids, before_date=max_match_date)
        team_pairs = list(
            zip(
                matches_df["home_team_id"].tolist(),
                matches_df["away_team_id"].tolist(),
                strict=False,
            )
        )
        repo.preload_h2h_matches(team_pairs, before_date=max_match_date)
        tournament_season_pairs = list(
            set(
                zip(
                    matches_df["tournament_id"].tolist(),
                    matches_df["season_id"].tolist(),
                    strict=False,
                )
            )
        )
        repo.preload_season_standings(
            tournament_season_pairs, before_date=max_match_date
        )
        print("Preloaded caches", flush=True)

        generators = [
            (
                "team_form",
                TeamFormGenerator(window_sizes=[3, 5, 10], include_venue_specific=True),
            ),
            ("head_to_head", HeadToHeadGenerator(max_h2h_matches=5, max_years_back=3)),
            (
                "temporal",
                TemporalFeatureGenerator(
                    include_rest_days=True,
                    include_fixture_density=True,
                    include_season_period=True,
                ),
            ),
            (
                "standings",
                StandingsFeatureGenerator(
                    relegation_threshold=3, euro_spot_start=4, euro_spot_end=7
                ),
            ),
            (
                "enriched_stats",
                EnrichedStatsFeatureGenerator(window_sizes=[3, 5], include_diffs=False),
            ),
        ]

        for name, gen in generators:
            t0 = time.time()
            df = gen.generate(matches_df, repo)
            elapsed = time.time() - t0
            print(
                f"  {name:20s}: {len(df)} rows x {len(df.columns)} cols in {elapsed:6.2f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
