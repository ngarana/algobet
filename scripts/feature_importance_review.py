"""Per-model feature importance + cross-model orthogonality review.

Steps:
  1. Load each session model and extract gain importance via the existing
     XGBoostPredictor.feature_importance property.
  2. Build a per-model top-N table.
  3. Build a survival matrix (rank in each model, blank if not selected).
  4. Run correlation analysis on the winner's pipeline against real EPL data.
  5. Classify each feature: edge / collinear / recipe-dependent / filler / noise.

Run with:
    uv run python scripts/throw-away/feature_importance_review.py
"""

from __future__ import annotations

import json
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

REPO_ROOT = Path("/home/arch/Coding/algobet")
sys.path.insert(0, str(REPO_ROOT))

MODELS_ROOT = REPO_ROOT / "data" / "models" / "xgboost"
OUT_DIR = REPO_ROOT / "scripts"
SESSION_IDS = [
    "180433",  # baseline (no player_quality, no balance)
    "184948",  # +player_quality, pre-temperature
    "185942",  # single-T temp
    "192746",  # sigmoid + tight search
    "194212",  # per-class T + 8x draw penalty
    "195921",  # per-class T + 3x draw penalty
    "200943",  # no cal, 180433 recipe
    "201906",  # WINNER: no cal + mild balance 0.3
]


# ---------------------------------------------------------------------------
# Importance extraction
# ---------------------------------------------------------------------------

def load_model(model_id: str):
    d = MODELS_ROOT / f"xgboost_20260512_{model_id}"
    with open(d / "model.pkl", "rb") as fh:
        obj = pickle.load(fh)
    with open(d / "feature_pipeline" / "config.json") as fh:
        cfg = json.load(fh)
    return obj, cfg


def importance_for(model_id: str) -> pd.DataFrame:
    obj, cfg = load_model(model_id)
    imp = obj.feature_importance  # property, returns dict {name: gain}
    if not imp:
        raise RuntimeError(f"No importance for {model_id}")
    df = pd.DataFrame(
        [{"feature": k, "gain": v} for k, v in imp.items()]
    )
    df["model"] = model_id
    total = df["gain"].sum()
    df["gain_norm"] = df["gain"] / total if total > 0 else 0.0
    df = df.sort_values("gain_norm", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["selected"] = df["gain"] > 0  # XGBoost only sees selected features
    return df


# ---------------------------------------------------------------------------
# Per-model + cross-model tables
# ---------------------------------------------------------------------------

def per_model_topN(all_imps: dict[str, pd.DataFrame], n: int = 15) -> dict[str, pd.DataFrame]:
    return {mid: df.head(n).copy() for mid, df in all_imps.items()}


def survival_matrix(all_imps: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Each cell = rank in that model, NaN if feature absent / zero gain."""
    rows = {}
    for mid, df in all_imps.items():
        ranks = df.set_index("feature")["rank"]
        # treat zero-gain features as "not used"
        zero = df[df["gain"] == 0]["feature"].tolist()
        ranks = ranks.drop(zero, errors="ignore")
        rows[mid] = ranks
    mat = pd.DataFrame(rows)
    mat["n_models"] = mat.notna().sum(axis=1)
    mat = mat.sort_values(["n_models", "201906"], ascending=[False, True])
    return mat


def aggregate_gain(all_imps: dict[str, pd.DataFrame]) -> pd.DataFrame:
    long = pd.concat(all_imps.values(), ignore_index=True)
    long = long[long["gain"] > 0]  # ignore unselected
    agg = (
        long.groupby("feature")["gain_norm"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_gain", "std": "std_gain", "count": "n_models"})
    )
    agg["cv"] = (agg["std_gain"] / agg["mean_gain"]).fillna(0)
    return agg.sort_values("mean_gain", ascending=False)


# ---------------------------------------------------------------------------
# Correlation analysis on the winning pipeline
# ---------------------------------------------------------------------------

def correlation_on_winner() -> pd.DataFrame | None:
    """Push real EPL matches through the winner's pipeline; compute correlations."""
    import algobet.models  # noqa: F401  — register all SQLAlchemy mappers
    from algobet.infrastructure.database import session_scope
    from algobet.predictions.data.queries import MatchRepository
    from algobet.predictions.features.pipeline import (
        FeaturePipeline,
        prepare_match_dataframe,
    )

    pipeline_path = MODELS_ROOT / "xgboost_20260512_201906" / "feature_pipeline"
    pipeline = FeaturePipeline.load(pipeline_path)

    selected = pipeline.selected_feature_names
    print(f"[correlation] winner pipeline has {len(selected) if selected else '?'} selected features")

    with session_scope() as session:
        repo = MatchRepository(session)

        # EPL training-era matches: pre-2024 (avoid test season for correlation)
        max_date = datetime(2024, 8, 1)
        min_date = datetime(2014, 1, 1)
        matches = repo.get_historical_matches(
            tournament_ids=[359],
            min_date=min_date,
            max_date=max_date,
            require_results=True,
        )
        print(f"[correlation] fetched {len(matches)} EPL matches for correlation panel")

        if not matches:
            return None

        matches_df = prepare_match_dataframe(matches)
        # Preload caches required by feature generators
        team_ids = list(set(matches_df["home_team_id"].tolist() + matches_df["away_team_id"].tolist()))
        repo.preload_team_matches(team_ids, before_date=max_date)
        repo.preload_h2h_matches(
            [(m.home_team_id, m.away_team_id) for m in matches],
            before_date=max_date,
        )
        season_pairs = list(set(zip(matches_df["tournament_id"], matches_df["season_id"], strict=True)))
        repo.preload_season_standings(season_pairs, before_date=max_date)

        X = pipeline.transform(matches_df, repo)
        names = pipeline.selected_feature_names
        if names is None or len(names) != X.shape[1]:
            names = [f"f{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=names)
        # Raw pre-selection features (all generator outputs)
        raw_df = pipeline.generate_raw(matches_df, repo)

    return X_df, raw_df


def correlation_clusters(X_df: pd.DataFrame, hard: float = 0.85, soft: float = 0.70):
    corr = X_df.corr().abs()
    pairs = []
    cols = corr.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = corr.loc[a, b]
            if r >= soft:
                pairs.append({"a": a, "b": b, "r": r, "hard": r >= hard})
    return pd.DataFrame(pairs).sort_values("r", ascending=False) if pairs else pd.DataFrame()


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(
    surv: pd.DataFrame,
    agg: pd.DataFrame,
    pairs: pd.DataFrame,
    winner_set: set,
) -> pd.DataFrame:
    rows = []
    q75 = agg["mean_gain"].quantile(0.75) if len(agg) else 0
    q25 = agg["mean_gain"].quantile(0.25) if len(agg) else 0

    # Build collinear lookup: for each feature, which higher-gain partner has |r|>=0.85?
    hard_pairs = pairs[pairs["hard"]] if not pairs.empty else pd.DataFrame()
    collinear_to = {}
    for _, p in hard_pairs.iterrows():
        a, b = p["a"], p["b"]
        gain_a = agg["mean_gain"].get(a, 0)
        gain_b = agg["mean_gain"].get(b, 0)
        # mark the weaker as collinear duplicate of the stronger
        if gain_a >= gain_b:
            collinear_to.setdefault(b, []).append((a, p["r"]))
        else:
            collinear_to.setdefault(a, []).append((b, p["r"]))

    for feat in surv.index:
        n = int(surv.loc[feat, "n_models"])
        mean_g = float(agg["mean_gain"].get(feat, 0))
        cv = float(agg["cv"].get(feat, 0))
        in_winner = feat in winner_set
        partner = collinear_to.get(feat)

        if partner:
            label = "Collinear"
            note = f"|r|≥0.85 with {partner[0][0]} (r={partner[0][1]:.2f})"
        elif mean_g >= q75 and n >= 6:
            label = "Edge"
            note = f"top-quartile gain, survives {n}/8"
        elif n <= 2 and not in_winner:
            label = "Noise"
            note = f"survives only {n}/8, absent from winner"
        elif cv > 1.0:
            label = "Recipe-dependent"
            note = f"CV={cv:.2f} → unstable across models"
        elif mean_g <= q25:
            label = "Filler"
            note = f"bottom-quartile gain ({mean_g:.4f})"
        else:
            label = "Mid"
            note = ""

        rows.append({
            "feature": feat,
            "label": label,
            "n_models": n,
            "mean_gain": mean_g,
            "cv": cv,
            "in_winner": in_winner,
            "note": note,
        })
    return pd.DataFrame(rows).sort_values(
        ["label", "mean_gain"], ascending=[True, False]
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("PHASE 1: Per-model importance extraction")
    print("=" * 80)

    all_imps = {}
    for mid in SESSION_IDS:
        try:
            df = importance_for(mid)
            nonzero = df[df["gain"] > 0]
            all_imps[mid] = df
            print(f"\n--- {mid} --- ({len(nonzero)} non-zero of {len(df)} features) ---")
            print(nonzero.head(15).to_string(
                index=False,
                columns=["rank", "feature", "gain_norm"],
                float_format=lambda x: f"{x:.4f}",
            ))
        except Exception as e:
            print(f"FAIL {mid}: {e}")

    if not all_imps:
        print("No models loaded — aborting.")
        return

    print("\n\n" + "=" * 80)
    print("PHASE 2: Cross-model survival matrix (rank per model)")
    print("=" * 80)
    surv = survival_matrix(all_imps)
    print(surv.head(40).to_string(float_format=lambda x: f"{int(x)}" if pd.notna(x) else "-"))
    surv.to_csv(OUT_DIR / "_survival.csv")

    print("\n\n" + "=" * 80)
    print("PHASE 2b: Aggregate gain across 8 models")
    print("=" * 80)
    agg = aggregate_gain(all_imps)
    print(agg.head(30).to_string(float_format=lambda x: f"{x:.4f}"))
    agg.to_csv(OUT_DIR / "_aggregate_gain.csv")

    print("\n\n" + "=" * 80)
    print("PHASE 3: Correlation on winner's pipeline")
    print("=" * 80)
    try:
        X_df, raw_df = correlation_on_winner()
    except Exception as e:
        print(f"Correlation phase failed: {e}")
        X_df, raw_df = None, None

    pairs = pd.DataFrame()
    raw_pairs = pd.DataFrame()
    if X_df is not None:
        pairs = correlation_clusters(X_df, hard=0.85, soft=0.70)
        print(f"\n[selected-14] {len(pairs)} pairs |r|≥0.70 "
              f"({pairs['hard'].sum() if not pairs.empty else 0} hard ≥0.85):\n")
        if not pairs.empty:
            print(pairs.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        pairs.to_csv(OUT_DIR / "_correlation_pairs.csv", index=False)
        X_df.to_csv(OUT_DIR / "_winner_features.csv", index=False)

    if raw_df is not None:
        # Filter to numeric columns only, drop near-constant columns
        num = raw_df.select_dtypes(include="number")
        num = num.loc[:, num.std() > 1e-9]
        raw_pairs = correlation_clusters(num, hard=0.85, soft=0.70)
        print(f"\n[raw {num.shape[1]}-feature panel] {len(raw_pairs)} pairs |r|≥0.70 "
              f"({raw_pairs['hard'].sum() if not raw_pairs.empty else 0} hard ≥0.85)")
        # Show top 40 hard pairs
        if not raw_pairs.empty:
            hard = raw_pairs[raw_pairs["hard"]].head(60)
            print("\nTop hard-collinear pairs (|r|≥0.85):")
            print(hard.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        raw_pairs.to_csv(OUT_DIR / "_raw_correlation_pairs.csv", index=False)

        # For each Edge feature, find the strongest correlate among raw features
        edge_feats = [
            "away_ppda_against_avg_3", "draw_rate_diff_season",
            "home_draw_rate_season", "is_season_mid", "home_xg_for_avg_3",
            "home_deep_completions_for_avg_3", "h2h_away_win_rate",
            "low_scoring_matchup_5", "away_win_rate_season",
            "home_clean_sheet_rate_10", "away_draw_rate_season",
            "away_clean_sheet_rate_10",
        ]
        print("\n\nFor each Edge feature → its strongest raw-feature partner (|r|≥0.70):")
        edge_partners = []
        corr_full = num.corr().abs()
        for ef in edge_feats:
            if ef not in corr_full.columns:
                continue
            col = corr_full[ef].drop(ef).sort_values(ascending=False)
            top = col.head(5)
            for partner, r in top.items():
                if r >= 0.70:
                    edge_partners.append({"edge": ef, "partner": partner, "r": r})
        if edge_partners:
            ep_df = pd.DataFrame(edge_partners)
            print(ep_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            ep_df.to_csv(OUT_DIR / "_edge_partners.csv", index=False)
        else:
            print("(no partners ≥0.70)")

    print("\n\n" + "=" * 80)
    print("PHASE 4: Feature classification")
    print("=" * 80)
    winner_set = set(all_imps["201906"][all_imps["201906"]["gain"] > 0]["feature"])
    cls = classify(surv, agg, pairs, winner_set)
    print(cls.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    cls.to_csv(OUT_DIR / "_classification.csv", index=False)

    print("\nDone. Artifacts written to scripts/throw-away/_*.csv")


if __name__ == "__main__":
    main()
