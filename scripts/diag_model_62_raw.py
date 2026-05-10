#!/usr/bin/env python3
"""Check raw vs calibrated probabilities for model 62."""

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

        print(f"Model type: {type(model).__name__}")
        base_predictor = model.predictor
        calibrator = model.calibrator

        print(f"Base predictor type: {type(base_predictor).__name__}")
        print(f"Calibrator: {calibrator is not None}")

        # Load feature pipeline
        model_meta = next(
            (
                m
                for m in registry.list_models()
                if m.version == "xgboost_20260508_190916"
            ),
            None,
        )
        pipeline_path = model_meta.hyperparameters.get("feature_pipeline_path")
        fp = FeaturePipeline.load(Path(pipeline_path))
        print(f"Pipeline: fitted={fp.is_fitted}, features={len(fp.feature_names)}")

        # Get 20 diverse matches
        matches = (
            session.query(Match)
            .filter(
                and_(
                    Match.status == "FINISHED",
                    Match.home_score.is_not(None),
                    Match.odds_home.is_not(None),
                )
            )
            .order_by(Match.match_date.desc())
            .limit(20)
            .all()
        )

        matches_df = prepare_match_dataframe(matches)
        repo = MatchRepository(session)
        all_team_ids = list(
            set(
                matches_df["home_team_id"].tolist()
                + matches_df["away_team_id"].tolist()
            )
        )
        repo.preload_team_matches(
            all_team_ids, before_date=matches_df["match_date"].max()
        )
        team_pairs = list(
            zip(
                matches_df["home_team_id"].tolist(),
                matches_df["away_team_id"].tolist(),
                strict=False,
            )
        )
        repo.preload_h2h_matches(team_pairs, before_date=matches_df["match_date"].max())
        ts_pairs = list(
            set(
                zip(
                    matches_df["tournament_id"].tolist(),
                    matches_df["season_id"].tolist(),
                    strict=False,
                )
            )
        )
        repo.preload_season_standings(
            ts_pairs, before_date=matches_df["match_date"].max()
        )

        X = fp.transform(matches_df, repo)

        raw_probas = base_predictor.predict_proba(X)
        cal_probas = model.predict_proba(X)

        print("\nRaw vs Calibrated probabilities (first 10 matches):")
        outcomes = ["H", "D", "A"]
        for i in range(min(10, len(matches))):
            raw = raw_probas[i]
            cal = cal_probas[i]
            raw_pred = outcomes[np.argmax(raw)]
            cal_pred = outcomes[np.argmax(cal)]
            raw_str = " ".join([f"{outcomes[j]}={raw[j]:.3f}" for j in range(3)])
            cal_str = " ".join([f"{outcomes[j]}={cal[j]:.3f}" for j in range(3)])
            match = matches[i]
            result = (
                "H"
                if match.home_score > match.away_score
                else ("A" if match.home_score < match.away_score else "D")
            )
            print(
                f"Match {match.id}: true={result} | RAW pred={raw_pred} [{raw_str}] | CAL pred={cal_pred} [{cal_str}]"
            )

        # Stats
        raw_preds = np.argmax(raw_probas, axis=1)
        cal_preds = np.argmax(cal_probas, axis=1)
        print(
            f"\nRaw predictions: H={np.sum(raw_preds == 0)} D={np.sum(raw_preds == 1)} A={np.sum(raw_preds == 2)}"
        )
        print(
            f"Cal predictions: H={np.sum(cal_preds == 0)} D={np.sum(cal_preds == 1)} A={np.sum(cal_preds == 2)}"
        )

        # Check raw probability variance
        print(
            f"\nRaw prob variance: H={np.var(raw_probas[:, 0]):.6f} D={np.var(raw_probas[:, 1]):.6f} A={np.var(raw_probas[:, 2]):.6f}"
        )
        print(
            f"Cal prob variance: H={np.var(cal_probas[:, 0]):.6f} D={np.var(cal_probas[:, 1]):.6f} A={np.var(cal_probas[:, 2]):.6f}"
        )

        # Are raw probs almost identical?
        raw_diffs = np.ptp(raw_probas, axis=0)  # max - min per class
        cal_diffs = np.ptp(cal_probas, axis=0)
        print(
            f"Raw prob range (max-min): H={raw_diffs[0]:.6f} D={raw_diffs[1]:.6f} A={raw_diffs[2]:.6f}"
        )
        print(
            f"Cal prob range (max-min): H={cal_diffs[0]:.6f} D={cal_diffs[1]:.6f} A={cal_diffs[2]:.6f}"
        )

        # Compare model 59
        print("\n--- Comparing to model 59 ---")
        model59 = registry.load_model("xgboost_20260508_120707")
        meta59 = next(
            (
                m
                for m in registry.list_models()
                if m.version == "xgboost_20260508_120707"
            ),
            None,
        )
        fp59 = FeaturePipeline.load(
            Path(meta59.hyperparameters.get("feature_pipeline_path"))
        )
        X59 = fp59.transform(matches_df, repo)
        raw59 = model59.predictor.predict_proba(X59)
        cal59 = model59.predict_proba(X59)
        raw59_preds = np.argmax(raw59, axis=1)
        cal59_preds = np.argmax(cal59, axis=1)
        print(
            f"Model 59 raw: H={np.sum(raw59_preds == 0)} D={np.sum(raw59_preds == 1)} A={np.sum(raw59_preds == 2)}"
        )
        print(
            f"Model 59 cal: H={np.sum(cal59_preds == 0)} D={np.sum(cal59_preds == 1)} A={np.sum(cal59_preds == 2)}"
        )

        # Side-by-side for first 5 matches
        print("\nSide-by-side (model 62 vs 59):")
        for i in range(5):
            r62 = " ".join([f"{outcomes[j]}={raw_probas[i, j]:.3f}" for j in range(3)])
            r59 = " ".join([f"{outcomes[j]}={raw59[i, j]:.3f}" for j in range(3)])
            match = matches[i]
            result = (
                "H"
                if match.home_score > match.away_score
                else ("A" if match.home_score < match.away_score else "D")
            )
            print(f"Match {match.id} true={result}:")
            print(f"  62: {r62}")
            print(f"  59: {r59}")


if __name__ == "__main__":
    main()
