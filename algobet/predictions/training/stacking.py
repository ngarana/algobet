"""Stacking ensemble meta-learner for match prediction.

Combines base model outputs through a lightweight calibrated classifier
(LogisticRegression) for more expressive blending than weighted averaging.

Time-aware OOF: base predictions for the meta-learner are generated via
expanding-window folds so that each prediction is produced by a model
trained only on earlier matches. This prevents future-to-past leakage.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray

from algobet.predictions.training.calibration import ProbabilityCalibrator
from algobet.predictions.training.classifiers import MatchPredictor, ModelConfig
from algobet.predictions.training.split import decode_targets


class StackingEnsemble(MatchPredictor):
    """Stacking ensemble with logistic meta-learner and isotonic calibration.

    Base models train on X_train and predict on X_val. The meta-learner
    trains on the concatenated base probabilities from X_val. Isotonic
    calibration is applied to the meta-learner output for well-calibrated
    final probabilities.

    Time-aware OOF: when OOF mode is enabled, base predictions for the
    meta-learner are generated via expanding-window folds so that each
    prediction is produced by a model trained only on earlier seasons.
    """

    def __init__(
        self,
        base_predictors: list[MatchPredictor],
        config: ModelConfig | None = None,
        meta_learner_type: str = "logistic",
        oof_folds: int | None = None,
        matches_df: Any = None,
    ) -> None:
        """Initialize stacking ensemble.

        Args:
            base_predictors: List of UNFITTED base predictor instances
            config: Optional model configuration
            meta_learner_type: "logistic" (default) or "mlp"
            oof_folds: Number of OOF folds for time-aware stacking.
                None = use validation set directly (legacy mode).
            matches_df: DataFrame with season_id and match_date columns.
                Required when oof_folds is set.
        """
        if config is None:
            config = ModelConfig(model_type="stacking_ensemble")

        super().__init__(config)
        self.base_predictors = base_predictors
        self._meta_learner: Any = None
        self._meta_calibrator: ProbabilityCalibrator | None = None
        self.meta_learner_type = meta_learner_type
        self.oof_folds = oof_folds
        self._matches_df = matches_df
        self._oof_fold_metrics: list[dict[str, float]] = []

    @property
    def model_type(self) -> str:
        return "stacking_ensemble"

    def fit(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.int64] | None = None,
    ) -> "StackingEnsemble":
        """Fit base models and meta-learner.

        Uses validation set for meta-learner training. The validation set
        is chronologically after the training set, so this is time-safe
        (no future-to-past leakage).

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features for meta-learner training
            y_val: Validation labels

        Returns:
            self
        """

        if X_val is None or y_val is None:
            raise ValueError(
                "StackingEnsemble requires X_val and y_val for meta-learner training"
            )

        # Step 1: Fit base models on training data (skip if already fitted)
        for predictor in self.base_predictors:
            if not predictor._is_fitted:
                predictor.fit(X_train, y_train, X_val, y_val)

        # Step 2: Collect base predictions for meta-learner training
        if self.oof_folds is not None and self._matches_df is not None:
            meta_train_probas, meta_train_labels = self._collect_oof_base_probas(
                X_train, y_train, X_val, y_val
            )
        else:
            meta_train_probas = self._collect_base_probas(X_val)
            meta_train_labels = y_val

        # Step 3: Train meta-learner on base predictions
        self._meta_learner = self._create_meta_learner()
        self._meta_learner.fit(meta_train_probas, meta_train_labels)

        # Step 4: Calibration on meta-learner output
        val_probas_for_calib = self._collect_base_probas(X_val)
        meta_val_probas = self._meta_learner.predict_proba(val_probas_for_calib)
        val_labels = y_val

        # Sample gate: isotonic needs ~1000+ samples; fall back to temperature
        n_samples = len(val_labels)
        if self.meta_learner_type == "logistic" and n_samples >= 1000:
            calib_method = "isotonic"
        else:
            calib_method = "temperature"

        self._meta_calibrator = ProbabilityCalibrator(method=calib_method)
        self._meta_calibrator.fit(meta_val_probas, val_labels)

        # Calibration gate: check if calibration improves log loss
        from sklearn.metrics import log_loss as sk_log_loss

        raw_ll = sk_log_loss(val_labels, meta_val_probas, labels=[0, 1, 2])
        calib_probas = self._meta_calibrator.calibrate(meta_val_probas)
        calib_ll = sk_log_loss(val_labels, calib_probas, labels=[0, 1, 2])

        if calib_ll > raw_ll + 1e-6:
            # Calibration worsened log loss — disable it
            self._meta_calibrator = None

        # Mark as fitted before collapse guard (predict_proba checks _is_fitted)
        self._is_fitted = True

        # Collapse guard: check predicted class distribution
        final_probas = self.predict_proba(X_val)
        final_preds = np.argmax(final_probas, axis=1)
        unique_classes = len(np.unique(final_preds))
        if unique_classes < 3:
            # Log warning but don't fail - stacking may still be useful
            import logging

            logging.getLogger(__name__).warning(
                "Stacking ensemble predicts only %d classes on validation set. "
                "This may indicate class imbalance or meta-learner overfitting.",
                unique_classes,
            )

        return self

    def _create_meta_learner(self) -> Any:
        """Create the meta-learner based on configuration."""
        from sklearn.linear_model import LogisticRegression

        if self.meta_learner_type == "mlp":
            from sklearn.neural_network import MLPClassifier

            return MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.001,
                learning_rate="adaptive",
                max_iter=1000,
                random_state=self.config.random_seed,
                early_stopping=True,
                validation_fraction=0.15,
            )
        else:
            return LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=self.config.random_seed,
            )

    def _collect_oof_base_probas(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        X_val: NDArray[np.float64] | None,
        y_val: NDArray[np.int64] | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Collect base predictions using time-aware OOF folds."""
        from algobet.predictions.training.split import OOFTimeAwareSplitter

        if self._matches_df is None:
            raise ValueError("matches_df is required for time-aware OOF stacking")

        n_folds = self.oof_folds or 5
        splitter = OOFTimeAwareSplitter(
            n_folds=n_folds,
            season_column="season_id",
            date_column="match_date",
        )

        # Build index mapping from matches_df to X_train positions
        # self._matches_df corresponds row-by-row to X_train.
        index_to_pos = {idx: i for i, idx in enumerate(self._matches_df.index)}

        n_train = len(X_train)
        n_classes = 3
        n_base = len(self.base_predictors)

        # Accumulators for OOF predictions
        oof_base_probas = np.zeros((n_train, n_base * n_classes))
        oof_labels = np.full(n_train, -1, dtype=np.int64)
        oof_assigned = np.zeros(n_train, dtype=bool)

        folds_used = 0
        for _fold_idx, (fold_train_idx, fold_oof_idx) in enumerate(
            splitter.split(self._matches_df)
        ):
            # Map to X_train positions
            fold_train_pos = [
                index_to_pos[idx] for idx in fold_train_idx if idx in index_to_pos
            ]
            fold_oof_pos = [
                index_to_pos[idx] for idx in fold_oof_idx if idx in index_to_pos
            ]

            if not fold_train_pos or not fold_oof_pos:
                continue

            X_fold_train = (
                X_train.values[fold_train_pos]
                if hasattr(X_train, "values")
                else X_train[fold_train_pos]
            )
            y_fold_train = y_train[fold_train_pos]
            X_fold_oof = (
                X_train.values[fold_oof_pos]
                if hasattr(X_train, "values")
                else X_train[fold_oof_pos]
            )
            y_fold_oof = y_train[fold_oof_pos]

            # Train base models on this fold's training data
            fold_predictors = []
            for predictor in self.base_predictors:
                # Clone the predictor for this fold
                from algobet.predictions.training.classifiers import (
                    DixonColesPredictor,
                    HybridPoissonPredictor,
                )

                if isinstance(predictor, DixonColesPredictor):
                    fold_pred = DixonColesPredictor(
                        self._clone_base_config(predictor, "dixon_coles")
                    )
                    # Need goal data for score-based models
                    home_goals = self._matches_df.loc[
                        fold_train_idx, "home_score"
                    ].values.astype(np.float64)
                    away_goals = self._matches_df.loc[
                        fold_train_idx, "away_score"
                    ].values.astype(np.float64)
                    fold_pred.fit_with_scores(
                        X_fold_train,
                        y_fold_train,
                        home_goals,
                        away_goals,
                    )
                elif isinstance(predictor, HybridPoissonPredictor):
                    fold_pred = HybridPoissonPredictor(
                        self._clone_base_config(predictor, "hybrid_poisson")
                    )
                    home_goals = self._matches_df.loc[
                        fold_train_idx, "home_score"
                    ].values.astype(np.float64)
                    away_goals = self._matches_df.loc[
                        fold_train_idx, "away_score"
                    ].values.astype(np.float64)
                    fold_pred.fit_with_scores(
                        X_fold_train,
                        y_fold_train,
                        home_goals,
                        away_goals,
                    )
                else:
                    # For tree models, create a fresh instance
                    from algobet.predictions.training.classifiers import (
                        create_predictor,
                    )

                    fold_pred = create_predictor(
                        predictor.model_type,
                        self._clone_base_config(predictor),
                    )
                    fold_pred.fit(X_fold_train, y_fold_train)

                fold_predictors.append(fold_pred)

            # Collect predictions on OOF portion
            for base_idx, fold_pred in enumerate(fold_predictors):
                base_proba = fold_pred.predict_proba(X_fold_oof)
                start_col = base_idx * n_classes
                oof_base_probas[fold_oof_pos, start_col : start_col + n_classes] = (
                    base_proba
                )

            oof_labels[fold_oof_pos] = y_fold_oof
            oof_assigned[fold_oof_pos] = True
            folds_used += 1

        if folds_used == 0:
            raise ValueError(
                "No valid OOF folds generated for stacking. "
                f"Ensure enough seasons for {n_folds} folds."
            )

        # Filter to assigned rows
        valid_mask = oof_assigned
        return oof_base_probas[valid_mask], oof_labels[valid_mask]

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict probabilities through the full stacking pipeline."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        base_probas = self._collect_base_probas(X)
        meta_probas = self._meta_learner.predict_proba(base_probas)

        if self._meta_calibrator is not None:
            calibrated = self._meta_calibrator.calibrate(meta_probas)
        else:
            calibrated = meta_probas

        # Collapse guard: ensure at least 2% per class floor
        calibrated = np.maximum(calibrated, 0.02)
        row_sums = calibrated.sum(axis=1, keepdims=True)
        calibrated = calibrated / np.maximum(row_sums, 1e-10)

        return calibrated

    def predict(self, X: NDArray[np.float64]) -> list[str]:
        """Predict outcomes."""
        proba = self.predict_proba(X)
        encoded_preds = np.argmax(proba, axis=1)
        return decode_targets(encoded_preds)

    @property
    def feature_importance(self) -> dict[str, float] | None:
        """Return meta-learner coefficient importance by base model."""
        if not self._is_fitted or self._meta_learner is None:
            return None

        # LogisticRegression with multinomial has coef_ of shape (n_classes, n_features)
        coeffs = np.abs(self._meta_learner.coef_).mean(axis=0)
        labels = []
        for predictor in self.base_predictors:
            for cls in range(3):
                labels.append(f"{predictor.model_type}_class_{cls}")
        return dict(zip(labels, coeffs.tolist(), strict=False))

    def _collect_base_probas(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Concatenate base model probabilities."""
        probas = [predictor.predict_proba(X) for predictor in self.base_predictors]
        return np.hstack(probas)

    def _clone_base_config(
        self,
        predictor: MatchPredictor,
        model_type: str | None = None,
    ) -> ModelConfig:
        """Copy the fitted pipeline's base-model configuration for OOF folds."""
        class_weights = (
            dict(predictor.config.class_weights)
            if predictor.config.class_weights is not None
            else None
        )
        return ModelConfig(
            model_type=model_type or predictor.model_type,
            hyperparameters=dict(predictor.config.hyperparameters),
            random_seed=predictor.config.random_seed,
            class_weights=class_weights,
            early_stopping_rounds=predictor.config.early_stopping_rounds,
            eval_metric=predictor.config.eval_metric,
        )

    def save(self, path: Path) -> None:
        """Save stacking ensemble to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save base predictors
        for i, predictor in enumerate(self.base_predictors):
            predictor.save(path / f"base_{i}_{predictor.model_type}.joblib")

        joblib.dump(
            {
                "config": self.config,
                "meta_learner": self._meta_learner,
                "meta_calibrator": self._meta_calibrator,
                "base_predictor_types": [p.model_type for p in self.base_predictors],
                "is_fitted": self._is_fitted,
                "meta_learner_type": self.meta_learner_type,
                "oof_folds": self.oof_folds,
                "oof_fold_metrics": self._oof_fold_metrics,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "StackingEnsemble":
        """Load stacking ensemble from disk."""
        path = Path(path)

        meta = joblib.load(path)
        predictor_types = meta["base_predictor_types"]

        from algobet.predictions.training.classifiers import (
            DixonColesPredictor,
            HybridPoissonPredictor,
            LightGBMPredictor,
            XGBoostPredictor,
        )

        type_map: dict[str, type[MatchPredictor]] = {
            "xgboost": XGBoostPredictor,
            "lightgbm": LightGBMPredictor,
            "dixon_coles": DixonColesPredictor,
            "hybrid_poisson": HybridPoissonPredictor,
        }

        base_predictors = []
        for i, pt in enumerate(predictor_types):
            predictor_class = type_map.get(pt)
            if predictor_class is None:
                raise ValueError(f"Unknown base predictor type: {pt}")
            base_predictors.append(predictor_class.load(path / f"base_{i}_{pt}.joblib"))

        ensemble = cls(
            base_predictors=base_predictors,
            config=meta["config"],
            meta_learner_type=meta.get("meta_learner_type", "logistic"),
            oof_folds=meta.get("oof_folds"),
        )
        ensemble._meta_learner = meta["meta_learner"]
        ensemble._meta_calibrator = meta["meta_calibrator"]
        ensemble._is_fitted = meta["is_fitted"]
        ensemble._oof_fold_metrics = meta.get("oof_fold_metrics", [])
        return ensemble
