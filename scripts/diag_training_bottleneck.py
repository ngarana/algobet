#!/usr/bin/env python3
"""Diagnose training bottleneck by timing each step."""

import os
import sys
import time

sys.path.insert(0, "/home/arch/Coding/algobet")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_USER", "algobet")
os.environ.setdefault("POSTGRES_PASSWORD", "password")
os.environ.setdefault("POSTGRES_DB", "football")

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

from algobet.infrastructure.database import get_session
from algobet.predictions.training.pipeline import TrainingConfig, TrainingPipeline


def timed(label, fn):
    print(f"[TIME] {label} ...", flush=True)
    t0 = time.time()
    result = fn()
    elapsed = time.time() - t0
    print(f"[TIME] {label} => {elapsed:.2f}s", flush=True)
    return result


def main():
    config = TrainingConfig(
        model_type="xgboost",
        description="Diagnostic training run",
        tournament_ids=[359],
        feature_groups=[
            "team_form",
            "head_to_head",
            "temporal",
            "standings",
            "enriched_stats",
        ],
        feature_selection=True,
        feature_selection_threshold=0.005,
        min_samples_per_feature=40,
        min_matches=150,
        outcome_balance=False,
        tune_hyperparameters=False,
        calibrate_probabilities=True,
        calibration_method="sigmoid",
        hyperparameters={
            "max_depth": 3,
            "learning_rate": 0.03,
            "n_estimators": 1200,
            "min_child_weight": 10,
            "gamma": 1.0,
            "reg_alpha": 2.0,
            "reg_lambda": 10.0,
            "subsample": 0.7,
            "colsample_bytree": 0.5,
        },
        tags={"diagnostic": "true"},
    )

    with get_session() as session:
        pipeline = TrainingPipeline(
            config=config,
            session=session,
            models_path=Path("data/models"),
        )
        print("Pipeline initialized.", flush=True)

        # Time _prepare_data step-by-step
        print("\n=== TIMING _prepare_data ===", flush=True)

        from algobet.predictions.features.pipeline import prepare_match_dataframe

        t0 = time.time()
        matches = pipeline.repo.get_historical_matches(
            min_date=config.start_date,
            max_date=config.end_date,
            tournament_ids=config.tournament_ids,
            team_ids=config.team_ids,
            require_results=True,
            min_total_goals=config.min_total_goals,
            max_total_goals=config.max_total_goals,
            venue_filter=config.venue_filter,
        )
        print(
            f"  get_historical_matches: {len(matches)} matches in {time.time() - t0:.2f}s",
            flush=True,
        )

        t0 = time.time()
        matches_df = prepare_match_dataframe(matches)
        print(f"  prepare_match_dataframe: {time.time() - t0:.2f}s", flush=True)

        all_team_ids = list(
            set(
                matches_df["home_team_id"].tolist()
                + matches_df["away_team_id"].tolist()
            )
        )
        max_match_date = matches_df["match_date"].max()

        t0 = time.time()
        pipeline.repo.preload_team_matches(all_team_ids, before_date=max_match_date)
        print(
            f"  preload_team_matches ({len(all_team_ids)} teams): {time.time() - t0:.2f}s",
            flush=True,
        )

        t0 = time.time()
        team_pairs = list(
            zip(
                matches_df["home_team_id"].tolist(),
                matches_df["away_team_id"].tolist(),
                strict=False,
            )
        )
        pipeline.repo.preload_h2h_matches(team_pairs, before_date=max_match_date)
        print(f"  preload_h2h_matches: {time.time() - t0:.2f}s", flush=True)

        t0 = time.time()
        tournament_season_pairs = list(
            set(
                zip(
                    matches_df["tournament_id"].tolist(),
                    matches_df["season_id"].tolist(),
                    strict=False,
                )
            )
        )
        pipeline.repo.preload_season_standings(
            tournament_season_pairs, before_date=max_match_date
        )
        print(f"  preload_season_standings: {time.time() - t0:.2f}s", flush=True)

        matches_df["result"] = matches_df.apply(
            lambda m: "H"
            if m["home_score"] > m["away_score"]
            else ("A" if m["home_score"] < m["away_score"] else "D"),
            axis=1,
        )

        from algobet.predictions.training.split import TemporalSplitter, encode_targets

        splitter = TemporalSplitter(
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            gap_days=config.gap_days,
        )
        splits = list(splitter.split(matches_df))
        split = splits[0]
        y = encode_targets(matches_df["result"].values)
        y_train = y[split.train_indices]
        y_val = y[split.val_indices]
        y_test = y[split.test_indices]

        train_df = matches_df.iloc[split.train_indices]
        val_df = matches_df.iloc[split.val_indices]
        test_df = matches_df.iloc[split.test_indices]
        pipeline._train_df = train_df
        pipeline._val_df = val_df
        pipeline._test_df = test_df
        print(
            f"  Data split: train={len(train_df)} val={len(val_df)} test={len(test_df)}",
            flush=True,
        )

        print("\n=== TIMING feature generation ===", flush=True)
        t0 = time.time()
        X_train = pipeline.feature_pipeline.fit_transform(train_df, pipeline.repo)
        print(
            f"  fit_transform(train): shape={X_train.shape} in {time.time() - t0:.2f}s",
            flush=True,
        )

        t0 = time.time()
        X_val = pipeline.feature_pipeline.transform(val_df, pipeline.repo)
        print(
            f"  transform(val): shape={X_val.shape} in {time.time() - t0:.2f}s",
            flush=True,
        )

        t0 = time.time()
        X_test = pipeline.feature_pipeline.transform(test_df, pipeline.repo)
        print(
            f"  transform(test): shape={X_test.shape} in {time.time() - t0:.2f}s",
            flush=True,
        )

        pipeline._X_train = X_train
        pipeline._X_val = X_val
        pipeline._X_test = X_test
        pipeline._y_train = y_train
        pipeline._y_val = y_val
        pipeline._y_test = y_test

        print("\n=== TIMING feature selection ===", flush=True)
        if config.feature_selection:
            t0 = time.time()
            X_train, X_val, X_test = pipeline._apply_feature_selection(
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=y_train,
                y_val=y_val,
                class_weights=None,
                hyperparameters=config.hyperparameters.copy(),
            )
            print(
                f"  _apply_feature_selection: shape={X_train.shape} in {time.time() - t0:.2f}s",
                flush=True,
            )
        else:
            print("  feature_selection disabled", flush=True)

        print("\n=== TIMING model training ===", flush=True)
        t0 = time.time()
        predictor = pipeline._train_model(
            X_train, y_train, X_val, y_val, config.hyperparameters.copy(), None
        )
        print(f"  _train_model: {time.time() - t0:.2f}s", flush=True)

        print("\n=== TIMING calibration ===", flush=True)
        if config.calibrate_probabilities:
            from algobet.predictions.training.calibration import ProbabilityCalibrator

            t0 = time.time()
            calibrator = ProbabilityCalibrator(method=config.calibration_method)
            val_probas = predictor.predict_proba(X_val)
            calibrator.fit(val_probas, y_val)
            print(f"  calibrator.fit: {time.time() - t0:.2f}s", flush=True)
        else:
            print("  calibration disabled", flush=True)

        print("\n=== TIMING evaluation ===", flush=True)
        t0 = time.time()
        train_m = pipeline._evaluate(
            predictor, X_train, y_train, train_df, apply_calibration=False
        )
        print(f"  _evaluate(train): {time.time() - t0:.2f}s", flush=True)

        t0 = time.time()
        val_m = pipeline._evaluate(
            predictor, X_val, y_val, val_df, apply_calibration=False
        )
        print(f"  _evaluate(val): {time.time() - t0:.2f}s", flush=True)

        t0 = time.time()
        test_m = pipeline._evaluate(
            predictor, X_test, y_test, test_df, apply_calibration=True
        )
        print(f"  _evaluate(test): {time.time() - t0:.2f}s", flush=True)

        print("\n=== RESULTS ===", flush=True)
        print(
            f"Train acc: {train_m['accuracy']:.4f}  log_loss: {train_m['log_loss']:.4f}",
            flush=True,
        )
        print(
            f"Val   acc: {val_m['accuracy']:.4f}  log_loss: {val_m['log_loss']:.4f}",
            flush=True,
        )
        print(
            f"Test  acc: {test_m['accuracy']:.4f}  log_loss: {test_m['log_loss']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
