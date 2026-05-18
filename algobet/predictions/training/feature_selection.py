"""Group-aware feature selection with correlation pruning and retention guards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FEATURE_FAMILIES: dict[str, list[str]] = {
    "elo": [
        "elo_diff",
        "elo_expected",
        "elo_change",
    ],
    "xpts": [
        "xpts_diff",
        "points_vs_xpts",
        "xpts_coverage",
    ],
    "draw": ["draw_rate", "draws", "home_draw", "away_draw", "h2h_draw"],
    "draw_signal": [
        "defensive_balance",
        "low_scoring_probability",
        "clean_sheet_interaction",
        "goal_convergence",
        "volatility_sum",
        "xg_parity",
        "strength_parity",
        "h2h_draw_boost",
    ],
    "away": [
        "away_win",
        "away_away",
        "h2h_away_win",
        "away_away_win",
        "away_away_clean",
        "away_clean_sheet",
    ],
    "low_scoring": [
        "low_scoring",
        "clean_sheet",
        "failed_to_score",
        "btts",
    ],
    "enriched": [
        "xg_for",
        "xg_against",
        "npxg",
        "shots_for",
        "shots_against",
        "corners_for",
        "corners_against",
        "ppda",
        "deep_completions",
        "player_",
        "saves",
        "fouls",
        "yellow_card",
        "red_card",
        "offsides",
        "starter_",
        "shot_quality",
        "xg_conversion",
        "shots_on_target_rate",
        "has_enriched",
        "has_player",
        "finishing_rate",
    ],
    "coverage": [
        "enriched_match_coverage",
        "player_stats_coverage",
        "has_enriched_match_stats",
        "has_player_stats",
    ],
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
        "points_per_game_diff",
        "draw_rate_diff",
        "loss_rate_diff",
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
        "goal_variance",
        "points_volatility",
    ],
    "temporal": [
        "day_of_week",
        "month",
        "weekend",
        "season",
        "rest_days",
        "fixture",
        "days_from",
        "season_progress",
    ],
    "h2h": ["h2h_"],
    "odds": [
        "implied_prob",
        "bookmaker_margin",
        "odds_home_away_ratio",
        "favorite_outcome",
        "favorite_implied_prob",
        "odds_quality_score",
    ],
    "odds_residual": [
        "form_surprise",
        "venue_surprise",
        "home_advantage_net",
    ],
    "market_mediation": [
        "mediation_",
    ],
}


def classify_feature(name: str) -> str:
    for family, patterns in FEATURE_FAMILIES.items():
        if any(p in name for p in patterns):
            return family
    return "other"


@dataclass
class FeatureSelectionReport:
    selected_features: list[str]
    dropped_features: list[dict[str, Any]]
    group_counts_before: dict[str, int]
    group_counts_after: dict[str, int]
    correlation_clusters: dict[str, list[str]]
    retained_protected: list[str]


def prune_correlation(
    feature_names: list[str],
    X: NDArray[np.float64],
    max_correlation: float = 0.94,
) -> tuple[list[str], list[dict[str, str]]]:
    """Remove highly correlated features, keeping the one with higher variance."""
    if len(feature_names) <= 1:
        return list(feature_names), []

    corr = np.corrcoef(X.T)
    n = len(feature_names)
    variances = np.var(X, axis=0)
    to_remove: set[int] = set()
    drop_log: list[dict[str, str]] = []

    for i in range(n):
        if i in to_remove:
            continue
        for j in range(i + 1, n):
            if j in to_remove:
                continue
            if abs(corr[i, j]) >= max_correlation:
                if variances[i] >= variances[j]:
                    to_remove.add(j)
                    drop_log.append(
                        {
                            "feature": feature_names[j],
                            "reason": f"correlation_{corr[i, j]:.3f}",
                            "kept": feature_names[i],
                        }
                    )
                else:
                    to_remove.add(i)
                    drop_log.append(
                        {
                            "feature": feature_names[i],
                            "reason": f"correlation_{corr[i, j]:.3f}",
                            "kept": feature_names[j],
                        }
                    )
                    break

    kept = [name for idx, name in enumerate(feature_names) if idx not in to_remove]
    return kept, drop_log


def group_aware_selection(
    feature_names: list[str],
    X: NDArray[np.float64],
    importance: dict[str, float] | None,
    n_samples: int,
    threshold: float = 0.01,
    min_samples_per_feature: int | None = None,
    max_correlation: float = 0.94,
    min_draw_features: int = 3,
    min_away_features: int = 3,
    min_enriched_or_coverage: int = 5,
    min_low_scoring_features: int = 2,
    min_market_mediation_features: int = 0,
    enriched_coverage_threshold: float = 0.0,
) -> tuple[list[str], FeatureSelectionReport]:
    group_map = {name: classify_feature(name) for name in feature_names}
    group_counts_before: dict[str, int] = {}
    for name in feature_names:
        g = group_map[name]
        group_counts_before[g] = group_counts_before.get(g, 0) + 1

    drop_log: list[dict[str, str]] = []
    corr_clusters: dict[str, list[str]] = {}

    # Pre-select features that must survive correlation pruning.
    # For market_mediation, structural features (overround, entropy, disagreement,
    # AH/OU context) are required by the CLV head. They are often pruned because
    # avg/max implied probs correlate with opening implied probs, and entropy/
    # favorite_prob correlate with home_prob. Force-protect the top-N non-constant
    # mediation features by temporarily removing them from the correlation pool,
    # then reinserting after pruning.
    forced_features: list[str] = []
    prunable_names = list(feature_names)
    if min_market_mediation_features > 0:
        mm_candidates = [
            n
            for n in feature_names
            if group_map.get(n) == "market_mediation"
            and not _is_constant(X, feature_names, n)
        ]
        forced_features = mm_candidates[:min_market_mediation_features]
        prunable_names = [n for n in feature_names if n not in set(forced_features)]

    all_name_to_idx = {n: i for i, n in enumerate(feature_names)}
    prunable_col_indices = [all_name_to_idx[n] for n in prunable_names]
    X_prunable = X[:, prunable_col_indices]
    pruned_names, corr_drops = prune_correlation(
        prunable_names, X_prunable, max_correlation
    )
    pruned_names = pruned_names + forced_features
    drop_log.extend(corr_drops)
    removed_by_corr = set(feature_names) - set(pruned_names)
    for name in removed_by_corr:
        for entry in corr_drops:
            if entry["feature"] == name:
                kept = entry.get("kept", "")
                if kept:
                    corr_clusters.setdefault(kept, []).append(name)

    total_importance = sum(
        max(float((importance or {}).get(n, 0.0)), 0.0) for n in pruned_names
    )
    if total_importance <= 0.0:
        normalized = {n: 1.0 / len(pruned_names) for n in pruned_names}
    else:
        normalized = {
            n: max(float((importance or {}).get(n, 0.0)), 0.0) / total_importance
            for n in pruned_names
        }

    selected = [n for n in pruned_names if normalized[n] >= threshold]
    if not selected:
        selected = [max(pruned_names, key=lambda n: normalized[n])]

    for n in pruned_names:
        if n not in selected:
            drop_log.append({"feature": n, "reason": "below_threshold", "kept": ""})

    if min_samples_per_feature:
        max_features = max(1, n_samples // min_samples_per_feature)
        if len(selected) > max_features:
            sorted_sel = sorted(selected, key=lambda n: normalized[n], reverse=True)
            dropped = sorted_sel[max_features:]
            for d in dropped:
                drop_log.append(
                    {
                        "feature": d,
                        "reason": "min_samples_per_feature",
                        "kept": "",
                    }
                )
            selected = sorted_sel[:max_features]

    retained_protected: list[str] = []
    selected_set = set(selected)

    def _ensure_family(
        family: str,
        minimum: int,
        label: str,
        check_coverage: bool = False,
    ) -> None:
        family_features = [n for n in pruned_names if group_map.get(n) == family]
        already = [n for n in family_features if n in selected_set]
        if len(already) >= minimum:
            return
        if check_coverage and enriched_coverage_threshold <= 0.0:
            return
        candidates = [
            n
            for n in family_features
            if n not in selected_set and not _is_constant(X, feature_names, n)
        ]
        candidates.sort(key=lambda n: normalized.get(n, 0.0), reverse=True)
        needed = minimum - len(already)
        for c in candidates[:needed]:
            selected.append(c)
            selected_set.add(c)
            retained_protected.append(c)

    _ensure_family("draw", min_draw_features, "draw")
    _ensure_family("away", min_away_features, "away")
    _ensure_family("low_scoring", min_low_scoring_features, "low_scoring")

    enriched_features = [
        n for n in pruned_names if group_map.get(n) in ("enriched", "coverage")
    ]
    enriched_already = [n for n in enriched_features if n in selected_set]
    if len(enriched_already) < min_enriched_or_coverage:
        candidates = [
            n
            for n in enriched_features
            if n not in selected_set and not _is_constant(X, feature_names, n)
        ]
        candidates.sort(key=lambda n: normalized.get(n, 0.0), reverse=True)
        needed = min_enriched_or_coverage - len(enriched_already)
        for c in candidates[:needed]:
            selected.append(c)
            selected_set.add(c)
            retained_protected.append(c)

    group_counts_after: dict[str, int] = {}
    for name in selected:
        g = group_map.get(name, "other")
        group_counts_after[g] = group_counts_after.get(g, 0) + 1

    report = FeatureSelectionReport(
        selected_features=selected,
        dropped_features=drop_log,
        group_counts_before=group_counts_before,
        group_counts_after=group_counts_after,
        correlation_clusters=corr_clusters,
        retained_protected=retained_protected,
    )
    return selected, report


def _is_constant(
    X: NDArray[np.float64],
    feature_names: list[str],
    feature_name: str,
    tol: float = 1e-10,
) -> bool:
    if feature_name not in feature_names:
        return True
    idx = feature_names.index(feature_name)
    col = X[:, idx]
    return bool(np.nanvar(col) < tol)
