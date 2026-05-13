"""Ablation / permutation importance runner for ML operations."""

from __future__ import annotations

import contextlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException
from sqlalchemy import and_
from sqlalchemy.orm import Session, joinedload

from algobet.api.schemas.ablation import (
    AblationFamilyResult,
    AblationModelConfig,
    AblationRequest,
    AblationStudyResponse,
    PermutationFamilyResultSchema,
    PermutationImportanceResponse,
)
from algobet.infrastructure.logging_config import get_logger
from algobet.models import Match
from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.evaluation.ablation import (
    group_features_by_family,
    group_features_by_generator,
)
from algobet.predictions.features.pipeline import (
    FeaturePipeline,
    prepare_match_dataframe,
)
from algobet.predictions.models.registry import ModelRegistry
from algobet.predictions.training.config import ALLOWED_FEATURE_GROUPS, TrainingConfig
from algobet.predictions.training.pipeline import (
    MODEL_FEATURE_SCHEMA_VERSION,
    TrainingPipeline,
)
from algobet.predictions.training.split import encode_targets

logger = get_logger("ml_operations")


class AblationRunner:
    """Runs permutation importance and leave-one-out ablation studies."""

    def __init__(
        self,
        models_path: Path = Path("data/models"),
    ) -> None:
        self.models_path = models_path

    def run(
        self,
        request: AblationRequest,
        db: Session,
    ) -> PermutationImportanceResponse | AblationStudyResponse:
        """Dispatch to permutation or ablation based on request method."""
        if request.method == "permutation":
            return self._run_permutation(request, db)
        return self._run_ablation(request, db)

    # ------------------------------------------------------------------
    # Permutation importance
    # ------------------------------------------------------------------

    def _run_permutation(
        self,
        request: AblationRequest,
        db: Session,
    ) -> PermutationImportanceResponse:
        """Compute permutation feature importance on a trained model."""
        from algobet.predictions.evaluation.ablation import (
            compute_permutation_importance,
        )

        registry = ModelRegistry(storage_path=self.models_path, session=db)
        model, model_meta, feature_pipeline = self._load_model_and_pipeline(
            request.model_version,
            registry,
            db,
        )

        matches_df, y_true = self._prepare_evaluation_data(request, db)
        X_test, test_slice = self._transform_test_data(
            feature_pipeline,
            model_meta,
            matches_df,
            y_true,
            db,
        )
        y_test = y_true[test_slice]
        y_proba = np.asarray(model.predict_proba(X_test), dtype=np.float64)

        # Determine families
        families = self._resolve_families(
            request,
            feature_pipeline.feature_names,
        )

        result = compute_permutation_importance(
            y_true=y_test,
            y_proba_baseline=y_proba,
            X_test=X_test,
            feature_names=feature_pipeline.feature_names,
            model=model,
            n_repeats=request.n_repeats,
            families=families,
            random_state=request.random_state,
        )
        result.model_version = model_meta.version

        raw_importance = None
        if hasattr(model, "feature_importance"):
            raw_importance = model.feature_importance

        return PermutationImportanceResponse(
            method="permutation",
            model_version=result.model_version,
            num_samples=result.num_samples,
            n_repeats=result.n_repeats,
            baseline_log_loss=result.baseline_log_loss,
            baseline_accuracy=result.baseline_accuracy,
            families=[
                PermutationFamilyResultSchema(
                    family=f.family,
                    features_in_family=f.features_in_family,
                    features_found=f.features_found,
                    baseline_log_loss=f.baseline_log_loss,
                    permuted_log_loss=f.permuted_log_loss,
                    log_loss_increase=f.log_loss_increase,
                    baseline_accuracy=f.baseline_accuracy,
                    permuted_accuracy=f.permuted_accuracy,
                    accuracy_decrease=f.accuracy_decrease,
                    importance_score=f.importance_score,
                    importance_rank=f.importance_rank,
                )
                for f in result.families
            ],
            raw_feature_importance=raw_importance,
        )

    # ------------------------------------------------------------------
    # Leave-one-out ablation
    # ------------------------------------------------------------------

    def _run_ablation(
        self,
        request: AblationRequest,
        db: Session,
    ) -> AblationStudyResponse:
        """Retrain models, each excluding one feature group."""
        groups = self._resolve_ablation_groups(request.feature_families)

        # Train baseline with all groups
        baseline_config = self._build_training_config(request, feature_groups=None)
        baseline_result = self._train_model(baseline_config, db)

        ablation_results: list[AblationFamilyResult] = []

        for group in groups:
            excluded = [g for g in groups if g != group]
            cfg = self._build_training_config(request, feature_groups=excluded)
            try:
                result = self._train_model(cfg, db)
            except Exception as exc:
                logger.warning("Ablation training without %s failed: %s", group, exc)
                continue

            all_feature_names = baseline_result.feature_importance
            if all_feature_names is None:
                all_feature_names = {}

            ablation_results.append(
                AblationFamilyResult(
                    family=group,
                    features_excluded=self._features_for_group(
                        group,
                        baseline_config,
                    ),
                    num_features_used=result.num_features,
                    model_version=result.model_version,
                    train_metrics={
                        k: float(v) for k, v in result.train_metrics.items()
                    },
                    val_metrics={k: float(v) for k, v in result.val_metrics.items()},
                    test_metrics={k: float(v) for k, v in result.test_metrics.items()},
                    log_loss_delta=float(
                        result.test_metrics.get("log_loss", 0.0)
                        - baseline_result.test_metrics.get("log_loss", 0.0)
                    ),
                    accuracy_delta=float(
                        result.test_metrics.get("accuracy", 0.0)
                        - baseline_result.test_metrics.get("accuracy", 0.0)
                    ),
                )
            )

        return AblationStudyResponse(
            method="ablation",
            baseline_model_version=baseline_result.model_version,
            baseline_num_features=baseline_result.num_features,
            baseline_train_metrics={
                k: float(v) for k, v in baseline_result.train_metrics.items()
            },
            baseline_val_metrics={
                k: float(v) for k, v in baseline_result.val_metrics.items()
            },
            baseline_test_metrics={
                k: float(v) for k, v in baseline_result.test_metrics.items()
            },
            families=ablation_results,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_model_and_pipeline(
        self,
        model_version: str | None,
        registry: ModelRegistry,
        db: Session,
    ) -> tuple[Any, Any, FeaturePipeline]:
        """Load a model and its feature pipeline from the registry."""
        try:
            if model_version:
                model = registry.load_model(model_version)
                model_meta = next(
                    (m for m in registry.list_models() if m.version == model_version),
                    None,
                )
                if model_meta is None:
                    raise ValueError(f"Model metadata not found for {model_version}")
            else:
                model, model_meta = registry.get_active_model()
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {exc}",
            ) from exc

        feature_pipeline = None
        if model_meta and model_meta.hyperparameters:
            pipeline_path = model_meta.hyperparameters.get("feature_pipeline_path")
            if pipeline_path:
                pipeline_path = Path(pipeline_path)
                if pipeline_path.exists() and (pipeline_path / "config.json").exists():
                    with contextlib.suppress(Exception):
                        feature_pipeline = FeaturePipeline.load(pipeline_path)

        if feature_pipeline is None:
            feature_pipeline = FeaturePipeline.create_default()

        return model, model_meta, feature_pipeline

    def _prepare_evaluation_data(
        self,
        request: AblationRequest,
        db: Session,
    ) -> tuple[Any, np.ndarray]:
        """Load and prepare match data for evaluation."""
        end_date = request.end_date or datetime.now()
        start_date = request.start_date or (end_date - timedelta(days=365))

        query = db.query(Match).options(
            joinedload(Match.home_team),
            joinedload(Match.away_team),
        )
        filters = [
            Match.status == "FINISHED",
            Match.home_score.is_not(None),
            Match.away_score.is_not(None),
            Match.match_date >= start_date,
            Match.match_date <= end_date,
            Match.odds_home.is_not(None),
            Match.odds_draw.is_not(None),
            Match.odds_away.is_not(None),
        ]
        if request.tournament_ids:
            filters.append(Match.tournament_id.in_(request.tournament_ids))

        matches = query.filter(and_(*filters)).order_by(Match.match_date).all()

        if len(matches) < request.min_matches:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient matches: {len(matches)} < {request.min_matches}",
            )

        matches_df = prepare_match_dataframe(matches)
        matches_df["result"] = matches_df.apply(
            lambda m: "H"
            if m["home_score"] > m["away_score"]
            else ("A" if m["home_score"] < m["away_score"] else "D"),
            axis=1,
        )
        y = encode_targets(matches_df["result"].values)
        return matches_df, y

    def _transform_test_data(
        self,
        feature_pipeline: FeaturePipeline,
        model_meta: Any,
        matches_df: Any,
        y: np.ndarray,
        db: Session,
    ) -> tuple[np.ndarray, slice]:
        """Fit pipeline on train split, transform test split.

        Returns (X_test, test_slice) where test_slice selects rows from y.
        """
        train_ratio = 0.7
        n = len(matches_df)
        train_end = int(n * train_ratio)

        train_df = matches_df.iloc[:train_end]
        test_df = matches_df.iloc[train_end:]

        repo = MatchRepository(db)

        train_matches_df = train_df.copy()
        test_matches_df = test_df.copy()

        all_team_ids = list(
            set(
                train_matches_df["home_team_id"].tolist()
                + train_matches_df["away_team_id"].tolist()
            )
        )
        max_date = train_matches_df["match_date"].max()
        repo.preload_team_matches(all_team_ids, before_date=max_date)
        team_pairs = list(
            zip(
                train_matches_df["home_team_id"].tolist(),
                train_matches_df["away_team_id"].tolist(),
                strict=False,
            )
        )
        repo.preload_h2h_matches(team_pairs, before_date=max_date)
        tournament_season_pairs = list(
            set(
                zip(
                    train_matches_df["tournament_id"].tolist(),
                    train_matches_df["season_id"].tolist(),
                    strict=False,
                )
            )
        )
        repo.preload_season_standings(tournament_season_pairs, before_date=max_date)

        if not feature_pipeline.is_fitted:
            feature_pipeline.fit(train_matches_df, repo)
        X_test = feature_pipeline.transform(test_matches_df, repo)

        test_start = train_end
        test_slice = slice(test_start, None)

        return X_test, test_slice

    def _resolve_families(
        self,
        request: AblationRequest,
        feature_names: list[str],
    ) -> dict[str, list[str]]:
        """Build the family-to-features mapping based on request params."""
        if request.group_by == "generator":
            groups = group_features_by_generator(
                feature_names,
                list(ALLOWED_FEATURE_GROUPS),
            )
        else:
            groups = group_features_by_family(feature_names)

        if request.feature_families:
            filtered = {
                k: v for k, v in groups.items() if k in request.feature_families
            }
            if not filtered:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"None of the requested families "
                        f"{request.feature_families} were found. "
                        f"Available: {sorted(groups.keys())}"
                    ),
                )
            return filtered

        return groups

    def _resolve_ablation_groups(
        self,
        feature_families: list[str] | None,
    ) -> list[str]:
        """Determine feature groups for leave-one-out ablation."""
        groups = list(ALLOWED_FEATURE_GROUPS)
        if feature_families:
            invalid = set(feature_families) - set(ALLOWED_FEATURE_GROUPS)
            if invalid:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid feature groups: {sorted(invalid)}. "
                        f"Allowed: {sorted(ALLOWED_FEATURE_GROUPS)}"
                    ),
                )
            return [g for g in groups if g in feature_families]
        return groups

    def _build_training_config(
        self,
        request: AblationRequest,
        feature_groups: list[str] | None,
    ) -> TrainingConfig:
        """Build a TrainingConfig from an AblationRequest."""
        cfg = request.ablation_model_config or AblationModelConfig()

        config = TrainingConfig(
            model_type=cfg.model_type,
            tune_hyperparameters=cfg.tune_hyperparameters,
            description=f"Ablation study – groups: {feature_groups or 'all'}",
            start_date=request.start_date,
            end_date=request.end_date,
            min_matches=request.min_matches,
            tournament_ids=request.tournament_ids,
            train_ratio=request.train_ratio,
            val_ratio=request.val_ratio,
            test_ratio=request.test_ratio,
            gap_days=request.gap_days,
            random_seed=cfg.random_seed,
            early_stopping_rounds=cfg.early_stopping_rounds,
            calibrate_probabilities=cfg.calibrate_probabilities,
            calibration_method=cfg.calibration_method,
            feature_groups=feature_groups,
            feature_schema_version=MODEL_FEATURE_SCHEMA_VERSION,
        )
        return config

    def _train_model(
        self,
        config: TrainingConfig,
        db: Session,
    ) -> Any:
        """Run a single training pipeline and return the result."""
        pipeline = TrainingPipeline(
            config=config,
            session=db,
            models_path=self.models_path,
        )
        result = pipeline.run()
        db.commit()
        return result

    def _features_for_group(
        self,
        group: str,
        config: TrainingConfig,
    ) -> list[str]:
        """Return feature name list for a generator group (best-effort)."""
        from algobet.predictions.features.composite import (
            create_generators_by_names,
        )

        try:
            gen = create_generators_by_names([group])
            return gen.feature_names
        except (ValueError, KeyError):
            return []
