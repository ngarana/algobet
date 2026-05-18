# docker exec -i algobet-api python - <<'PY'
import json
from datetime import datetime
from pathlib import Path

import algobet.models  # noqa: F401 - ensures SQLAlchemy relationship targets are registered
from algobet.infrastructure.database import session_scope
from algobet.predictions.training.pipeline import TrainingConfig, TrainingPipeline

TOP5_TOURNAMENT_IDS = [359, 545, 98, 28, 123]
DEFAULT_GROUPS = [
    "team_form",
    "head_to_head",
    "temporal",
    "standings",
    "enriched_stats",
    "draw_signals",
    "matchup_interaction",
]
DETAILED_ODDS_FEATURES = {
    "avg_implied_prob_home",
    "avg_implied_prob_draw",
    "avg_implied_prob_away",
    "avg_bookmaker_margin",
    "max_implied_prob_home",
    "max_implied_prob_draw",
    "max_implied_prob_away",
    "odds_disagreement_home",
    "odds_disagreement_draw",
    "odds_disagreement_away",
    "asian_handicap_line",
    "asian_handicap_implied_margin",
    "over_under_25_over_odds",
    "over_under_25_under_odds",
    "over_under_implied_total",
    "over_under_implied_margin",
    "ah_vs_1x2_spread_diff",
    "ou_vs_1x2_total_diff",
}
RUNS = [
    ("top5_default_features", DEFAULT_GROUPS),
    ("top5_default_plus_detailed_odds", DEFAULT_GROUPS + ["detailed_odds"]),
]
common = dict(
    model_type="xgboost",
    tune_hyperparameters=False,
    start_date=datetime(2012, 7, 1),
    end_date=datetime(2025, 6, 30, 23, 59, 59),
    min_matches=100,
    tournament_ids=TOP5_TOURNAMENT_IDS,
    split_strategy="walk_forward",
    train_seasons=8,
    val_seasons=1,
    test_seasons=1,
    feature_selection=True,
    feature_selection_threshold=0.005,
    min_samples_per_feature=40,
    calibrate_probabilities=False,
    use_feature_cache=False,
    random_seed=42,
    early_stopping_rounds=50,
    tags={
        "comparison": "top5_detailed_odds_20260517",
        "activated": "false",
    },
)
results = []
for run_name, groups in RUNS:
    print(f"START {run_name}", flush=True)
    config = TrainingConfig(
        **common,
        feature_groups=groups,
        model_name=run_name,
        description=(
            "Top-5 2012/13-2024/25 comparison run: "
            f"{run_name}; activate=false; calibrate_probabilities=false"
        ),
    )
    with session_scope() as session:
        pipeline = TrainingPipeline(
            config=config,
            session=session,
            models_path=Path("data/models"),
        )
        result = pipeline.run()
        selected = (
            pipeline._selected_feature_names or pipeline.feature_pipeline.feature_names
        )
        selected_detailed = sorted(set(selected) & DETAILED_ODDS_FEATURES)
    row = {
        "run_name": run_name,
        "model_version": result.model_version,
        "model_type": result.model_type,
        "model_path": str(result.model_path),
        "duration_seconds": result.training_duration_seconds,
        "num_features": result.num_features,
        "feature_groups": groups,
        "selected_feature_count": len(selected),
        "selected_detailed_odds_features": selected_detailed,
        "train_metrics": result.train_metrics,
        "val_metrics": result.val_metrics,
        "test_metrics": result.test_metrics,
    }
    results.append(row)
    print("RESULT " + json.dumps(row, sort_keys=True, default=float), flush=True)
baseline = results[0]
new = results[1]
comparison = {
    "generated_at": datetime.now().isoformat(),
    "tournament_ids": TOP5_TOURNAMENT_IDS,
    "date_range": [common["start_date"].isoformat(), common["end_date"].isoformat()],
    "runs": results,
    "delta_new_minus_baseline": {
        "test_accuracy": new["test_metrics"].get("accuracy", 0.0)
        - baseline["test_metrics"].get("accuracy", 0.0),
        "test_log_loss": new["test_metrics"].get("log_loss", 0.0)
        - baseline["test_metrics"].get("log_loss", 0.0),
        "test_brier_score": new["test_metrics"].get("brier_score", 0.0)
        - baseline["test_metrics"].get("brier_score", 0.0),
        "test_ece": new["test_metrics"].get("ece", 0.0)
        - baseline["test_metrics"].get("ece", 0.0),
        "test_market_model_probability_mae": new["test_metrics"].get(
            "market_model_probability_mae", 0.0
        )
        - baseline["test_metrics"].get("market_model_probability_mae", 0.0),
    },
}
path = Path("/tmp/top5_detailed_odds_comparison.json")
path.write_text(json.dumps(comparison, indent=2, sort_keys=True, default=float))
print(f"SUMMARY_PATH {path}", flush=True)
print(
    "SUMMARY " + json.dumps(comparison["delta_new_minus_baseline"], sort_keys=True),
    flush=True,
)
# PY
