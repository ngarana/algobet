#!/usr/bin/env python3
"""Evaluate a saved model on EPL matches with comprehensive diagnostics.

Loads the model-specific feature pipeline, filters to FINISHED matches with
valid scores/season/tournament, and prints/saves a full evaluation report
including log loss, market log loss, confusion matrix, per-class metrics,
predicted-class distribution, model-vs-market favorite agreement, and
feature selection info.
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/home/arch/Coding/algobet")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sqlalchemy import and_
from sqlalchemy.orm import Session, joinedload

import algobet.matches.models  # noqa: F401
import algobet.models  # noqa: F401
import algobet.predictions.models  # noqa: F401
import algobet.teams.models  # noqa: F401
from algobet.infrastructure.database import get_session
from algobet.models import Match
from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_outcome_accuracy,
)
from algobet.predictions.features.pipeline import (
    FeaturePipeline,
    prepare_match_dataframe,
)
from algobet.predictions.models.registry import ModelRegistry
from algobet.predictions.training.calibration import calculate_calibration_metrics


def compute_market_diagnostics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    matches_df: pd.DataFrame,
) -> dict:
    required = ["odds_home", "odds_draw", "odds_away"]
    if not all(c in matches_df.columns for c in required):
        return {"market_samples": 0}
    odds = matches_df[required].astype(float).to_numpy(dtype=np.float64)
    valid = np.isfinite(odds).all(axis=1) & (odds > 0).all(axis=1)
    if not np.any(valid):
        return {"market_samples": 0}
    valid_odds = odds[valid]
    implied = 1.0 / valid_odds
    implied = implied / implied.sum(axis=1, keepdims=True)
    y_v = y_true[valid]
    m_v = y_proba[valid]
    ri = np.arange(len(y_v))
    market_ll = -np.log(np.clip(implied[ri, y_v], 1e-12, 1.0)).mean()
    market_fav = np.argmax(implied, axis=1)
    model_fav = np.argmax(m_v, axis=1)
    return {
        "market_samples": int(len(y_v)),
        "market_log_loss": float(market_ll),
        "market_favorite_accuracy": float(np.mean(market_fav == y_v)),
        "market_model_probability_mae": float(np.abs(m_v - implied).mean()),
        "market_favorite_agreement": float(np.mean(model_fav == market_fav)),
    }


def class_distribution(y_pred: np.ndarray, n_classes: int = 3) -> dict:
    counts = np.bincount(y_pred, minlength=n_classes)
    total = len(y_pred)
    labels = {0: "H", 1: "D", 2: "A"}
    return {
        "counts": {labels[i]: int(counts[i]) for i in range(n_classes)},
        "shares": {labels[i]: round(counts[i] / total, 4) for i in range(n_classes)},
        "num_classes": int(np.count_nonzero(counts)),
        "max_share": float(counts.max() / total) if total else 0.0,
    }


def grouped_feature_counts(feature_names: list[str]) -> dict[str, int]:
    families = {
        "draw": ["draw", "Draw"],
        "away": ["away_win", "away_away", "away_draw", "h2h_away"],
        "low_scoring": ["low_scoring", "clean_sheet", "failed_to_score", "btts"],
        "enriched": ["xg", "npxg", "shot", "corner", "ppda", "deep", "player_"],
        "coverage": ["coverage", "has_enriched", "has_player"],
        "standings": [
            "league_position",
            "points_total",
            "points_per_game",
            "win_rate_season",
            "in_relegation",
            "in_euro",
            "is_leader",
            "position_normalized",
            "draw_rate_season",
            "loss_rate_season",
            "top_six",
            "bottom_six",
        ],
        "form": [
            "points_last",
            "win_rate",
            "goals_for",
            "goals_against",
            "goal_diff",
            "form_trend",
            "form_diff",
            "home_home",
            "away_away",
        ],
        "temporal": [
            "day_of_week",
            "month",
            "weekend",
            "season",
            "rest_days",
            "fixture",
            "days_from",
        ],
        "h2h": ["h2h_"],
    }
    result = {}
    for family, patterns in families.items():
        count = sum(1 for f in features if any(p in f for p in patterns))
        if count:
            result[family] = count
    return result


def run_evaluation(
    session: Session,
    model_version: str,
    tournament_id: int,
    start_date: str | None = None,
    end_date: str | None = None,
    min_matches: int = 50,
) -> dict:
    registry = ModelRegistry(storage_path=Path("data/models"), session=session)
    model = registry.load_model(model_version)
    meta = None
    for m in registry.list_models():
        if m.version == model_version:
            meta = m
            break

    query = session.query(Match).options(
        joinedload(Match.home_team), joinedload(Match.away_team)
    )
    filters = [
        Match.status == "FINISHED",
        Match.home_score.is_not(None),
        Match.away_score.is_not(None),
        Match.season_id.is_not(None),
        Match.tournament_id == tournament_id,
    ]
    if start_date:
        filters.append(Match.match_date >= datetime.fromisoformat(start_date))
    if end_date:
        filters.append(Match.match_date <= datetime.fromisoformat(end_date))

    matches = query.filter(and_(*filters)).order_by(Match.match_date).all()
    if len(matches) < min_matches:
        raise ValueError(f"Insufficient matches: {len(matches)} < {min_matches}")

    matches_df = prepare_match_dataframe(matches)
    matches_df["result"] = matches_df.apply(
        lambda m: "H"
        if m["home_score"] > m["away_score"]
        else ("A" if m["home_score"] < m["away_score"] else "D"),
        axis=1,
    )

    repo = MatchRepository(session)
    feature_pipeline = None
    pipeline_path = None
    if meta and meta.hyperparameters:
        pipeline_path = meta.hyperparameters.get("feature_pipeline_path")
    if pipeline_path:
        pp = Path(pipeline_path)
        if pp.exists() and (pp / "config.json").exists():
            try:
                feature_pipeline = FeaturePipeline.load(pp)
            except Exception:
                pass
    if feature_pipeline is None:
        feature_pipeline = FeaturePipeline.create_default()

    train_size = int(len(matches) * 0.3)
    train_df = matches_df.iloc[:train_size]
    test_df = matches_df.iloc[train_size:]

    if not feature_pipeline.is_fitted:
        feature_pipeline.fit(train_df, repo)
    X_test = feature_pipeline.transform(test_df, repo)

    y_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_proba, axis=1)
    result_map = {"H": 0, "D": 1, "A": 2}
    y_true = test_df["result"].map(result_map).values

    class_metrics = calculate_classification_metrics(y_true, y_pred, y_proba)
    cal_metrics = calculate_calibration_metrics(y_true, y_proba)
    outcome_acc = calculate_outcome_accuracy(y_true, y_pred)
    market_diag = compute_market_diagnostics(y_true, y_proba, test_df)
    dist = class_distribution(y_pred)
    feature_names = feature_pipeline.feature_names
    grouped = grouped_feature_counts(feature_names)

    selected_features = None
    feature_selection_report = None
    ensemble_weights = None
    base_model_metrics = None
    calibration_method = None
    if meta and meta.hyperparameters:
        selected_features = meta.hyperparameters.get("selected_feature_names")
        feature_selection_report = meta.hyperparameters.get("feature_selection")
        ensemble_weights = meta.hyperparameters.get("ensemble_weights")
        base_model_metrics = meta.hyperparameters.get("base_model_metrics")
        calibration_method = meta.hyperparameters.get("calibration_method")

    report = {
        "model_version": model_version,
        "tournament_id": tournament_id,
        "evaluated_at": datetime.now().isoformat(),
        "num_samples": len(y_true),
        "date_range": [
            str(test_df["match_date"].min().date()),
            str(test_df["match_date"].max().date()),
        ],
        "log_loss": float(class_metrics.log_loss),
        "accuracy": float(class_metrics.accuracy),
        "market_log_loss": market_diag.get("market_log_loss"),
        "confusion_matrix": class_metrics.confusion_matrix,
        "per_class_precision": class_metrics.per_class_precision,
        "per_class_recall": class_metrics.per_class_recall,
        "per_class_f1": class_metrics.per_class_f1,
        "predicted_class_distribution": dist,
        "model_vs_market_favorite_agreement": market_diag.get(
            "market_favorite_agreement"
        ),
        "selected_features": selected_features,
        "all_features": feature_names,
        "grouped_feature_counts": grouped,
        "feature_selection_report": feature_selection_report,
        "ensemble_weights": ensemble_weights,
        "base_model_metrics": base_model_metrics,
        "calibration_method": calibration_method,
        "expected_calibration_error": cal_metrics["expected_calibration_error"],
        "outcome_accuracy": outcome_acc,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate an EPL model")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--tournament-id", type=int, default=359)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--min-matches", type=int, default=50)
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with get_session() as session:
        report = run_evaluation(
            session=session,
            model_version=args.model_version,
            tournament_id=args.tournament_id,
            start_date=args.start_date,
            end_date=args.end_date,
            min_matches=args.min_matches,
        )

    print(f"\n{'=' * 60}")
    print(f"EPL Model Evaluation: {report['model_version']}")
    print(f"{'=' * 60}")
    print(f"Samples: {report['num_samples']}")
    print(f"Date range: {report['date_range']}")
    print("\n--- Core Metrics ---")
    print(f"  Log loss:  {report['log_loss']:.4f}")
    if report["market_log_loss"] is not None:
        print(f"  Market LL: {report['market_log_loss']:.4f}")
    print(f"  Accuracy:  {report['accuracy']:.4f}")
    print(f"  ECE:       {report['expected_calibration_error']:.4f}")

    print("\n--- Per-Class F1 ---")
    for cls in ["H", "D", "A"]:
        print(f"  {cls}: {report['per_class_f1'].get(cls, 0):.4f}")

    print("\n--- Predicted Class Distribution ---")
    dist = report["predicted_class_distribution"]
    for cls in ["H", "D", "A"]:
        print(f"  {cls}: {dist['counts'][cls]} ({dist['shares'][cls]:.1%})")
    print(f"  Classes predicted: {dist['num_classes']}")
    print(f"  Max class share:   {dist['max_share']:.1%}")

    if report.get("model_vs_market_favorite_agreement") is not None:
        print("\n--- Market Diagnostics ---")
        print(
            f"  Model-market fav agreement: "
            f"{report['model_vs_market_favorite_agreement']:.4f}"
        )

    print("\n--- Features ---")
    print(f"  Total features: {len(report['all_features'])}")
    if report.get("selected_features"):
        print(f"  Selected: {len(report['selected_features'])}")
    print(f"  Grouped: {report['grouped_feature_counts']}")

    if report.get("ensemble_weights"):
        print("\n--- Ensemble Weights ---")
        for k, v in report["ensemble_weights"].items():
            print(f"  {k}: {v}")

    out_path = output_dir / f"eval_{args.model_version}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
