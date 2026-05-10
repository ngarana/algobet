#!/usr/bin/env python3
"""Diagnose why model 62 predicts Home for every backtest match."""

import os
import sys

sys.path.insert(0, "/home/arch/Coding/algobet")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_USER", "algobet")
os.environ.setdefault("POSTGRES_PASSWORD", "password")
os.environ.setdefault("POSTGRES_DB", "football")

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
from sqlalchemy import and_

from algobet.infrastructure.database import get_session
from algobet.matches.models import Match
from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.pipeline import (
    FeaturePipeline,
    prepare_match_dataframe,
)
from algobet.predictions.models.registry import ModelRegistry


def main():
    with get_session() as session:
        registry = ModelRegistry(storage_path=Path("data/models"), session=session)
        model = registry.load_model("xgboost_20260508_190916")
        model_meta = next(
            (
                m
                for m in registry.list_models()
                if m.version == "xgboost_20260508_190916"
            ),
            None,
        )

        print(f"Model loaded: {type(model).__name__}")
        print(
            f"Is calibrated: {hasattr(model, 'calibrator') and model.calibrator is not None}"
        )

        # Load feature pipeline
        pipeline_path = None
        if model_meta and model_meta.hyperparameters:
            pipeline_path = model_meta.hyperparameters.get("feature_pipeline_path")
            print(f"Pipeline path: {pipeline_path}")
            print(
                f"Selected features: {len(model_meta.hyperparameters.get('selected_feature_names', []))}"
            )

        if pipeline_path:
            fp = FeaturePipeline.load(Path(pipeline_path))
            print(
                f"Pipeline loaded: fitted={fp.is_fitted}, features={len(fp.feature_names)}"
            )
            print(f"First 5 feature names: {fp.feature_names[:5]}")
        else:
            print("NO PIPELINE PATH FOUND")
            return

        # Get a few recent matches
        matches = (
            session.query(Match)
            .filter(
                and_(
                    Match.status == "FINISHED",
                    Match.home_score.is_not(None),
                    Match.away_score.is_not(None),
                    Match.odds_home.is_not(None),
                    Match.odds_draw.is_not(None),
                    Match.odds_away.is_not(None),
                )
            )
            .order_by(Match.match_date.desc())
            .limit(10)
            .all()
        )

        print(f"\nLoaded {len(matches)} recent matches for spot-check")

        matches_df = prepare_match_dataframe(matches)
        matches_df["result"] = matches_df.apply(
            lambda m: "H"
            if m["home_score"] > m["away_score"]
            else ("A" if m["home_score"] < m["away_score"] else "D"),
            axis=1,
        )

        repo = MatchRepository(session)
        all_team_ids = list(
            set(
                matches_df["home_team_id"].tolist()
                + matches_df["away_team_id"].tolist()
            )
        )
        max_match_date = matches_df["match_date"].max()
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

        X = fp.transform(matches_df, repo)
        print(f"Feature matrix shape: {X.shape}")
        print(
            f"Feature matrix variance per column: min={np.var(X, axis=0).min():.6f} max={np.var(X, axis=0).max():.6f}"
        )
        print(
            f"Feature matrix mean per column range: [{np.mean(X, axis=0).min():.4f}, {np.mean(X, axis=0).max():.4f}]"
        )

        # Check for constant columns
        var_per_col = np.var(X, axis=0)
        constant_cols = np.where(var_per_col < 1e-10)[0]
        print(f"Constant (zero-variance) columns: {len(constant_cols)} / {X.shape[1]}")
        if len(constant_cols) > 0:
            print(f"  Constant feature indices: {constant_cols[:10].tolist()}...")

        # Predictions
        probas = model.predict_proba(X)
        preds = np.argmax(probas, axis=1)

        print("\nPredictions on 10 recent matches:")
        outcomes = ["H", "D", "A"]
        for i, (_, row) in enumerate(matches_df.iterrows()):
            pred_label = outcomes[preds[i]]
            true_label = row["result"]
            probs_str = " / ".join(
                [f"{outcomes[j]}={probas[i, j]:.3f}" for j in range(3)]
            )
            match_str = (
                f"{row['home_team_id']} vs {row['away_team_id']} on {row['match_date']}"
            )
            print(
                f"  Match {row['id']:>6}: true={true_label} pred={pred_label} | {probs_str} | {match_str}"
            )

        # Check if all predictions are identical
        unique_preds = np.unique(preds)
        print(
            f"\nUnique predictions across {len(preds)} matches: {unique_preds} (labels: {[outcomes[u] for u in unique_preds]})"
        )

        # Raw feature variance check on a LARGER sample
        print("\n--- Testing on larger sample (100 matches) ---")
        matches_big = (
            session.query(Match)
            .filter(
                and_(
                    Match.status == "FINISHED",
                    Match.home_score.is_not(None),
                    Match.odds_home.is_not(None),
                )
            )
            .order_by(Match.match_date.desc())
            .limit(100)
            .all()
        )
        matches_df_big = prepare_match_dataframe(matches_big)
        all_team_ids_big = list(
            set(
                matches_df_big["home_team_id"].tolist()
                + matches_df_big["away_team_id"].tolist()
            )
        )
        repo2 = MatchRepository(session)
        repo2.preload_team_matches(
            all_team_ids_big, before_date=matches_df_big["match_date"].max()
        )
        team_pairs_big = list(
            zip(
                matches_df_big["home_team_id"].tolist(),
                matches_df_big["away_team_id"].tolist(),
                strict=False,
            )
        )
        repo2.preload_h2h_matches(
            team_pairs_big, before_date=matches_df_big["match_date"].max()
        )
        ts_pairs_big = list(
            set(
                zip(
                    matches_df_big["tournament_id"].tolist(),
                    matches_df_big["season_id"].tolist(),
                    strict=False,
                )
            )
        )
        repo2.preload_season_standings(
            ts_pairs_big, before_date=matches_df_big["match_date"].max()
        )

        X_big = fp.transform(matches_df_big, repo2)
        preds_big = np.argmax(model.predict_proba(X_big), axis=1)
        unique_preds_big = np.unique(preds_big)
        print(
            f"Unique predictions across 100 matches: {unique_preds_big} (labels: {[outcomes[u] for u in unique_preds_big]})"
        )
        print(
            f"Pred distribution: H={np.sum(preds_big == 0)}, D={np.sum(preds_big == 1)}, A={np.sum(preds_big == 2)}"
        )

        # Check feature variance
        var_big = np.var(X_big, axis=0)
        print(
            f"Feature variance summary: mean={var_big.mean():.6f}, min={var_big.min():.6f}, max={var_big.max():.6f}"
        )
        print(f"Zero-variance columns: {np.sum(var_big < 1e-10)} / {X_big.shape[1]}")


if __name__ == "__main__":
    main()
