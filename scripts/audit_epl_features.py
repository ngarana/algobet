#!/usr/bin/env python3
"""Audit EPL feature quality before training.

Generates raw features once for tournament 359, then exports per-feature
statistics (null rate, zero rate, variance, correlation cluster, family label,
univariate signal, selection status) and grouped summaries.
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
from algobet.predictions.features.generators import create_generators_by_names
from algobet.predictions.features.pipeline import prepare_match_dataframe
from algobet.predictions.training.split import encode_targets

FEATURE_FAMILIES = {
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
        "points_per_game_diff",
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


def classify_feature(name: str) -> str:
    for family, patterns in FEATURE_FAMILIES.items():
        if any(p in name for p in patterns):
            return family
    return "other"


def compute_univariate_signal(
    feature_values: np.ndarray, y: np.ndarray, n_bins: int = 10
) -> float:
    try:
        from sklearn.feature_selection import mutual_info_classif

        mi = mutual_info_classif(
            feature_values.reshape(-1, 1), y, random_state=42, n_neighbors=5
        )
        return float(mi[0])
    except Exception:
        return 0.0


def build_correlation_clusters(
    df: pd.DataFrame, threshold: float = 0.94
) -> dict[str, list[str]]:
    corr = df.corr().abs()
    clusters: dict[str, list[str]] = {}
    visited = set()
    for col in corr.columns:
        if col in visited:
            continue
        group = [col]
        visited.add(col)
        for other in corr.columns:
            if other not in visited and corr.loc[col, other] >= threshold:
                group.append(other)
                visited.add(other)
        if len(group) > 1:
            for member in group:
                clusters[member] = [g for g in group if g != member]
    return clusters


def run_audit(
    session: Session,
    tournament_id: int,
    feature_groups: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    min_matches: int = 200,
    max_feature_correlation: float = 0.94,
) -> dict:
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

    matches = (
        session.query(Match)
        .options(joinedload(Match.home_team), joinedload(Match.away_team))
        .filter(and_(*filters))
        .order_by(Match.match_date)
        .all()
    )
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
    all_team_ids = list(
        set(matches_df["home_team_id"].tolist() + matches_df["away_team_id"].tolist())
    )
    max_date = matches_df["match_date"].max()
    repo.preload_team_matches(all_team_ids, before_date=max_date)
    team_pairs = list(
        zip(
            matches_df["home_team_id"].tolist(),
            matches_df["away_team_id"].tolist(),
            strict=False,
        )
    )
    repo.preload_h2h_matches(team_pairs, before_date=max_date)
    ts_pairs = list(
        set(
            zip(
                matches_df["tournament_id"].tolist(),
                matches_df["season_id"].tolist(),
                strict=False,
            )
        )
    )
    repo.preload_season_standings(ts_pairs, before_date=max_date)

    generators = create_generators_by_names(feature_groups)
    raw_features = generators.generate(matches_df, repo)
    y = encode_targets(matches_df["result"].values)

    feature_names = list(raw_features.columns)
    n = len(raw_features)

    corr_clusters = build_correlation_clusters(raw_features, max_feature_correlation)

    feature_records = []
    for name in feature_names:
        values = raw_features[name].values
        null_rate = float(np.isnan(values).mean()) if values.dtype.kind == "f" else 0.0
        zero_rate = float((values == 0).mean())
        variance = float(np.nanvar(values)) if values.dtype.kind == "f" else 0.0
        family = classify_feature(name)
        mi_signal = compute_univariate_signal(np.nan_to_num(values, nan=0.0), y)
        cluster = corr_clusters.get(name, [])
        feature_records.append(
            {
                "feature": name,
                "null_rate": round(null_rate, 4),
                "zero_rate": round(zero_rate, 4),
                "variance": round(variance, 6),
                "family": family,
                "correlation_cluster": cluster,
                "univariate_mi": round(mi_signal, 6),
            }
        )

    family_summary = {}
    for rec in feature_records:
        fam = rec["family"]
        if fam not in family_summary:
            family_summary[fam] = {"count": 0, "selected": 0, "features": []}
        family_summary[fam]["count"] += 1
        family_summary[fam]["features"].append(rec["feature"])

    high_corr_pairs = []
    seen = set()
    for feat, partners in corr_clusters.items():
        for partner in partners:
            pair = tuple(sorted([feat, partner]))
            if pair not in seen:
                seen.add(pair)
                high_corr_pairs.append(list(pair))

    report = {
        "tournament_id": tournament_id,
        "feature_groups": feature_groups,
        "num_matches": n,
        "num_features": len(feature_names),
        "evaluated_at": datetime.now().isoformat(),
        "max_feature_correlation": max_feature_correlation,
        "features": feature_records,
        "family_summary": family_summary,
        "high_correlation_pairs": high_corr_pairs,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Audit EPL features")
    parser.add_argument("--tournament-id", type=int, default=359)
    parser.add_argument(
        "--feature-groups",
        nargs="+",
        default=[
            "team_form",
            "head_to_head",
            "temporal",
            "standings",
            "enriched_stats",
        ],
    )
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--min-matches", type=int, default=200)
    parser.add_argument("--max-feature-correlation", type=float, default=0.94)
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with get_session() as session:
        report = run_audit(
            session=session,
            tournament_id=args.tournament_id,
            feature_groups=args.feature_groups,
            start_date=args.start_date,
            end_date=args.end_date,
            min_matches=args.min_matches,
            max_feature_correlation=args.max_feature_correlation,
        )

    json_path = output_dir / "epl_feature_audit.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    csv_path = output_dir / "epl_feature_audit.csv"
    pd.DataFrame(report["features"]).to_csv(csv_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"EPL Feature Audit: tournament={args.tournament_id}")
    print(f"{'=' * 60}")
    print(f"Matches: {report['num_matches']}")
    print(f"Features: {report['num_features']}")
    print("\n--- Family Summary ---")
    for fam, info in sorted(report["family_summary"].items()):
        print(f"  {fam}: {info['count']} features")
    print(
        f"\nHigh correlation pairs (>{args.max_feature_correlation}): "
        f"{len(report['high_correlation_pairs'])}"
    )
    print(f"\nJSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
