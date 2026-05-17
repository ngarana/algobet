#!/usr/bin/env python3
"""Train Dixon-Coles model for draw probability correction."""

from pathlib import Path

import numpy as np

# Import all models first to avoid SQLAlchemy relationship errors
from algobet import models  # noqa: F401
from algobet.infrastructure.database import session_scope
from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.pipeline import (
    FeaturePipeline,
    prepare_match_dataframe,
)
from algobet.predictions.training.classifiers import DixonColesPredictor, ModelConfig


def main():
    output_path = Path("data/models/dixon_coles_epl.joblib")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with session_scope() as session:
        repo = MatchRepository(session)

        # Fetch EPL matches with scores (tournament_id=359)
        matches = repo.get_historical_matches(
            tournament_ids=[359],
            max_date="2025-05-31",
            require_results=True,
        )

        print(f"Loaded {len(matches)} finished matches")

        # Convert to DataFrame and build real features
        matches_df = prepare_match_dataframe(matches)
        pipeline = FeaturePipeline.create_default()
        X = pipeline.fit_transform(matches_df, repo)

        # Extract scores and encode outcomes
        home_goals = matches_df["home_score"].values.astype(np.float64)
        away_goals = matches_df["away_score"].values.astype(np.float64)
        y = np.where(
            matches_df["home_score"] > matches_df["away_score"],
            0,
            np.where(
                matches_df["home_score"] < matches_df["away_score"],
                2,
                1,
            ),
        ).astype(np.int64)

        print(f"Training on {len(home_goals)} matches with scores")
        print(f"Feature matrix shape: {X.shape}")

        # Train Dixon-Coles with real features
        config = ModelConfig(model_type="dixon_coles")
        dc = DixonColesPredictor(config)
        dc.fit_with_scores(
            X=X,
            y=y,
            home_goals=home_goals,
            away_goals=away_goals,
        )

        # Save
        dc.save(output_path)
        print(f"Saved Dixon-Coles model to {output_path}")


if __name__ == "__main__":
    main()
