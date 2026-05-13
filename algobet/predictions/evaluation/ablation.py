"""Feature ablation and permutation importance for model interpretability.

Provides two methods to assess feature family contributions:

1. **Permutation importance** - Shuffles feature columns for each family
   on a trained model and measures the performance drop. Fast, no retraining.

2. **Leave-one-out ablation** - Retrains the model excluding each feature
   group and compares metrics. Slow but shows training-time impact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, log_loss

from algobet.predictions.training.feature_selection import classify_feature


def group_features_by_family(feature_names: list[str]) -> dict[str, list[str]]:
    """Group feature names into families using classify_feature."""
    families: dict[str, list[str]] = {}
    for name in feature_names:
        family = classify_feature(name)
        families.setdefault(family, []).append(name)
    return families


def group_features_by_generator(
    feature_names: list[str],
    generator_names: list[str],
) -> dict[str, list[str]]:
    """Map feature names to their generator group.

    Uses a prefix-based heuristic: features whose name starts with a
    generator-specific prefix are assigned to that group. This is a
    best-effort mapping; features that cannot be mapped go to 'other'.
    """
    prefix_map = {
        "elo_rating": [
            "elo_diff",
            "elo_expected",
            "elo_change",
        ],
        "expected_points": [
            "xpts_diff",
            "points_vs_xpts",
            "xpts_coverage",
        ],
        "head_to_head": ["h2h_"],
        "temporal": [
            "day_of_week",
            "month_",
            "is_weekend",
            "weekend",
            "season_",
            "rest_days",
            "fixture_density",
            "days_from",
            "season_progress",
        ],
        "standings": [
            "league_position",
            "points_total",
            "points_per_game",
            "win_rate_season",
            "in_relegation",
            "in_euro",
            "is_leader",
            "position_norm",
            "draw_rate_season",
            "loss_rate_season",
            "top_six",
            "bottom_six",
            "points_per_game_diff",
            "draw_rate_diff",
            "loss_rate_diff",
        ],
        "enriched_stats": [
            "xg_for",
            "xg_against",
            "npxg",
            "shots_for",
            "shots_against",
            "corners_for",
            "corners_against",
            "ppda",
            "deep_complete",
            "player_",
            "starter_",
            "saves_",
            "fouls_",
            "yellow_card",
            "red_card",
            "offsides",
            "shot_quality",
            "xg_conversion",
            "shots_on_target_rate",
            "has_enriched",
            "has_player",
            "enriched_match_coverage",
            "player_stats_coverage",
        ],
        "team_form": [
            "home_",
            "away_",
            "form_",
            "points_last",
            "win_rate",
            "draw_rate",
            "loss_rate",
            "goals_for_",
            "goals_against_",
            "goal_diff",
            "goal_variance",
            "points_volatility",
            "streak",
            "home_home",
            "away_away",
            "home_record",
            "away_record",
            "home_clean",
            "away_clean",
            "home_scored",
            "away_scored",
        ],
    }
    groups: dict[str, list[str]] = {}
    assigned: set[str] = set()

    for name in feature_names:
        matched = False
        name_lower = name.lower()
        for gen_name, prefixes in prefix_map.items():
            if any(name_lower.startswith(p) or p in name_lower for p in prefixes):
                groups.setdefault(gen_name, []).append(name)
                assigned.add(name)
                matched = True
                break
        if not matched:
            groups.setdefault("other", []).append(name)

    return groups


@dataclass
class PermutationFamilyResult:
    family: str
    features_in_family: list[str]
    features_found: list[str]
    baseline_log_loss: float
    permuted_log_loss: float
    log_loss_increase: float
    baseline_accuracy: float
    permuted_accuracy: float
    accuracy_decrease: float
    importance_score: float = 0.0
    importance_rank: int = 0


@dataclass
class PermutationImportanceResult:
    model_version: str
    num_samples: int
    n_repeats: int
    baseline_log_loss: float
    baseline_accuracy: float
    families: list[PermutationFamilyResult] = field(default_factory=list)
    raw_feature_importance: dict[str, float] | None = None


def compute_permutation_importance(
    y_true: NDArray[np.int64],
    y_proba_baseline: NDArray[np.float64],
    X_test: NDArray[np.float64],
    feature_names: list[str],
    model: Any,
    n_repeats: int = 10,
    families: dict[str, list[str]] | None = None,
    random_state: int = 42,
) -> PermutationImportanceResult:
    """Compute permutation importance by feature family.

    For each feature family, shuffles those columns in X_test and
    measures the degradation in model performance (log loss increase).

    Args:
        y_true: True labels encoded as 0/1/2.
        y_proba_baseline: Baseline predicted probabilities from the model.
        X_test: Test feature matrix (must align with feature_names).
        feature_names: Column names for X_test.
        model: Trained model with predict_proba().
        n_repeats: Number of permutation repeats per family.
        families: Optional family-to-feature mapping.
            If None, grouped via classify_feature.
        random_state: Random seed for reproducibility.

    Returns:
        PermutationImportanceResult with per-family importance scores.
    """
    rng = np.random.RandomState(random_state)

    baseline_pred = np.argmax(y_proba_baseline, axis=1)
    baseline_ll = float(log_loss(y_true, y_proba_baseline, labels=[0, 1, 2]))
    baseline_acc = float(accuracy_score(y_true, baseline_pred))

    if families is None:
        families = group_features_by_family(feature_names)

    name_to_idx: dict[str, int] = {name: i for i, name in enumerate(feature_names)}

    results: list[PermutationFamilyResult] = []

    for family, family_features in families.items():
        found = [f for f in family_features if f in name_to_idx]
        if not found:
            results.append(
                PermutationFamilyResult(
                    family=family,
                    features_in_family=family_features,
                    features_found=[],
                    baseline_log_loss=baseline_ll,
                    permuted_log_loss=baseline_ll,
                    log_loss_increase=0.0,
                    baseline_accuracy=baseline_acc,
                    permuted_accuracy=baseline_acc,
                    accuracy_decrease=0.0,
                )
            )
            continue

        indices = [name_to_idx[f] for f in found]

        perm_lls: list[float] = []
        perm_accs: list[float] = []

        for _ in range(n_repeats):
            X_perm = X_test.copy()
            for idx in indices:
                rng.shuffle(X_perm[:, idx])

            proba = model.predict_proba(X_perm)
            proba = np.asarray(proba, dtype=np.float64)
            pred = np.argmax(proba, axis=1)

            perm_lls.append(float(log_loss(y_true, proba, labels=[0, 1, 2])))
            perm_accs.append(float(accuracy_score(y_true, pred)))

        mean_ll = float(np.mean(perm_lls))
        mean_acc = float(np.mean(perm_accs))

        results.append(
            PermutationFamilyResult(
                family=family,
                features_in_family=family_features,
                features_found=found,
                baseline_log_loss=baseline_ll,
                permuted_log_loss=mean_ll,
                log_loss_increase=mean_ll - baseline_ll,
                baseline_accuracy=baseline_acc,
                permuted_accuracy=mean_acc,
                accuracy_decrease=baseline_acc - mean_acc,
            )
        )

    total = sum(abs(r.log_loss_increase) for r in results)
    for r in results:
        r.importance_score = abs(r.log_loss_increase) / total if total > 0 else 0.0

    ranked = sorted(results, key=lambda r: r.log_loss_increase, reverse=True)
    for rank, r in enumerate(ranked, 1):
        r.importance_rank = rank

    results.sort(key=lambda r: r.family)

    return PermutationImportanceResult(
        model_version="",
        num_samples=int(len(y_true)),
        n_repeats=n_repeats,
        baseline_log_loss=baseline_ll,
        baseline_accuracy=baseline_acc,
        families=results,
    )
