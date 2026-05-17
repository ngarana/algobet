"""Audit Dixon-Coles / Hybrid Poisson goal regressors on the held-out test season.

Prints λ_home / λ_away distributions vs actual goals, so we can see whether
the home-bias hypothesis (RC-2 in the improvement plan) is the actual cause
of the home-dominant argmax predictions.

Usage:
    uv run python scripts/audit_score_models.py \
        --dc-version dixon_coles_20260513_142504 \
        --hp-version hybrid_poisson_20260513_141050 \
        --tournament-id 359
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import algobet.models  # noqa: F401  -- registers all SQLAlchemy mappers
from algobet.infrastructure.database import session_scope
from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.pipeline import (
    FeaturePipeline,
    prepare_match_dataframe,
)
from algobet.predictions.training.split import SeasonAwareSplitter


MODELS_ROOT = Path("data/models")


def _load(model_type: str, version: str) -> tuple[object, FeaturePipeline]:
    model_dir = MODELS_ROOT / model_type / version
    with open(model_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)
    pipeline = FeaturePipeline.load(model_dir / "feature_pipeline")
    return model, pipeline


def _unwrap(predictor: object) -> object:
    """Strip CalibratedPredictor / DrawAwarePredictor wrappers."""
    inner = predictor
    while hasattr(inner, "base_predictor") or hasattr(inner, "predictor"):
        inner = getattr(inner, "base_predictor", None) or getattr(inner, "predictor")
    return inner


def _summarize(name: str, values: np.ndarray) -> None:
    q = np.percentile(values, [5, 25, 50, 75, 95])
    print(
        f"  {name:>14}: "
        f"mean={values.mean():.3f}  std={values.std():.3f}  "
        f"p5={q[0]:.2f}  p25={q[1]:.2f}  p50={q[2]:.2f}  p75={q[3]:.2f}  p95={q[4]:.2f}"
    )


def audit(model_type: str, version: str, tournament_id: int) -> None:
    print(f"\n=== {model_type} / {version} ===")
    model, pipeline = _load(model_type, version)
    inner = _unwrap(model)

    if not hasattr(inner, "_home_model") or not hasattr(inner, "_away_model"):
        print(f"  Skipping: predictor {type(inner).__name__} lacks _home_model/_away_model")
        return

    with session_scope() as session:
        repo = MatchRepository(session)
        matches = repo.get_historical_matches(
            tournament_ids=[tournament_id],
            require_results=True,
        )
        if not matches:
            print("  No matches found")
            return
        df = prepare_match_dataframe(matches)

        # Mirror data_preparation.py: preload caches before feature gen
        team_ids = list(set(df["home_team_id"]) | set(df["away_team_id"]))
        max_date = df["match_date"].max()
        repo.preload_team_matches(team_ids, before_date=max_date)
        repo.preload_h2h_matches(
            list(zip(df["home_team_id"], df["away_team_id"], strict=False)),
            before_date=max_date,
        )
        repo.preload_season_standings(
            list(set(zip(df["tournament_id"], df["season_id"], strict=False))),
            before_date=max_date,
        )

        splitter = SeasonAwareSplitter(train_seasons=8, val_seasons=1, test_seasons=1)
        split = list(splitter.split(df))[0]
        test_df = df.iloc[split.test_indices].reset_index(drop=True)
        print(f"  Test rows: {len(test_df)}  date range: "
              f"{test_df['match_date'].min().date()} → {test_df['match_date'].max().date()}")

        X_test = pipeline.transform(test_df, repo)

    home_lambda = inner._home_model.predict(X_test)
    away_lambda = inner._away_model.predict(X_test)
    home_actual = test_df["home_score"].to_numpy(dtype=np.float64)
    away_actual = test_df["away_score"].to_numpy(dtype=np.float64)
    rho = getattr(inner, "_best_rho", None)

    print(f"  ρ (DC correlation): {rho}")
    print("  Predicted goal expectations vs actuals:")
    _summarize("λ_home", home_lambda)
    _summarize("home_actual", home_actual)
    _summarize("λ_away", away_lambda)
    _summarize("away_actual", away_actual)

    diff = home_lambda - away_lambda
    actual_diff = home_actual - away_actual
    print("  Goal-difference (home − away):")
    _summarize("λ_diff", diff)
    _summarize("actual_diff", actual_diff)

    # Calibration: mean predicted vs mean actual
    print(
        f"  Mean bias: λ_home − home_actual = {home_lambda.mean() - home_actual.mean():+.3f}  "
        f"λ_away − away_actual = {away_lambda.mean() - away_actual.mean():+.3f}"
    )

    # Argmax distribution from full predict_proba
    probs = inner.predict_proba(X_test)
    argmax = np.argmax(probs, axis=1)
    counts = np.bincount(argmax, minlength=3)
    print(f"  Argmax counts:  H={counts[0]}  D={counts[1]}  A={counts[2]}")
    print(f"  Mean P(H/D/A): {probs.mean(axis=0).round(3).tolist()}")
    print(f"  Max  P(D) observed: {probs[:, 1].max():.3f}   "
          f"P(D)>=0.34 count: {int((probs[:, 1] >= 0.34).sum())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dc-version", default="dixon_coles_20260513_142504")
    parser.add_argument("--hp-version", default="hybrid_poisson_20260513_141050")
    parser.add_argument("--tournament-id", type=int, default=359)
    args = parser.parse_args()

    pd.set_option("display.width", 140)
    audit("dixon_coles", args.dc_version, args.tournament_id)
    audit("hybrid_poisson", args.hp_version, args.tournament_id)


if __name__ == "__main__":
    main()
