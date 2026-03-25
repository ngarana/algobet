"""Unit tests for backtest evaluation metrics."""

import numpy as np
import pytest

from algobet.predictions.evaluation.metrics import (
    BettingMetrics,
    ClassificationMetrics,
    EvaluationResult,
    calculate_betting_metrics,
    calculate_classification_metrics,
    calculate_outcome_accuracy,
    compare_models,
    evaluate_predictions,
)


class TestCalculateClassificationMetrics:
    """Tests for calculate_classification_metrics function."""

    def test_perfect_predictions(self) -> None:
        """Perfect predictions should yield accuracy 1.0 and log_loss near 0."""
        y_true = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
        y_proba = np.array(
            [
                [0.99, 0.005, 0.005],
                [0.005, 0.99, 0.005],
                [0.005, 0.005, 0.99],
                [0.99, 0.005, 0.005],
                [0.005, 0.99, 0.005],
                [0.005, 0.005, 0.99],
            ],
            dtype=np.float64,
        )
        y_pred = np.argmax(y_proba, axis=1)

        metrics = calculate_classification_metrics(y_true, y_pred, y_proba)

        assert metrics.accuracy == pytest.approx(1.0, abs=1e-6)
        assert metrics.log_loss < 0.1
        assert metrics.brier_score < 0.02
        assert metrics.f1_macro == pytest.approx(1.0, abs=1e-6)
        assert metrics.cohen_kappa == pytest.approx(1.0, abs=1e-6)

    def test_random_predictions(self) -> None:
        """Random predictions should yield accuracy around 0.33."""
        np.random.seed(42)
        n = 1000
        y_true = np.random.randint(0, 3, size=n, dtype=np.int64)
        y_proba = np.random.dirichlet([1, 1, 1], size=n).astype(np.float64)
        y_pred = np.argmax(y_proba, axis=1)

        metrics = calculate_classification_metrics(y_true, y_pred, y_proba)

        assert 0.2 < metrics.accuracy < 0.5
        assert 0.5 < metrics.log_loss < 2.0
        assert metrics.precision_macro >= 0.0
        assert metrics.recall_macro >= 0.0

    def test_per_class_metrics(self) -> None:
        """Per-class metrics should be calculated correctly."""
        y_true = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int64)
        y_proba = np.array(
            [
                [0.9, 0.05, 0.05],
                [0.9, 0.05, 0.05],
                [0.9, 0.05, 0.05],
                [0.1, 0.8, 0.1],
                [0.1, 0.8, 0.1],
                [0.05, 0.05, 0.9],
                [0.05, 0.05, 0.9],
                [0.05, 0.05, 0.9],
                [0.05, 0.05, 0.9],
            ],
            dtype=np.float64,
        )
        y_pred = np.array([0, 0, 1, 1, 1, 2, 2, 2, 0], dtype=np.int64)

        metrics = calculate_classification_metrics(y_true, y_pred, y_proba)

        assert "H" in metrics.per_class_f1
        assert "D" in metrics.per_class_f1
        assert "A" in metrics.per_class_f1
        assert 0.5 <= metrics.per_class_f1["H"] <= 1.0
        assert 0.5 <= metrics.per_class_f1["D"] <= 1.0

    def test_confusion_matrix_shape(self) -> None:
        """Confusion matrix should be 3x3 for 3 classes."""
        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 3, size=n, dtype=np.int64)
        y_proba = np.random.dirichlet([1, 1, 1], size=n).astype(np.float64)
        y_pred = np.argmax(y_proba, axis=1)

        metrics = calculate_classification_metrics(y_true, y_pred, y_proba)

        assert len(metrics.confusion_matrix) == 3
        assert all(len(row) == 3 for row in metrics.confusion_matrix)

    def test_confusion_matrix_sums(self) -> None:
        """Confusion matrix rows and columns should sum correctly."""
        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 3, size=n, dtype=np.int64)
        y_proba = np.random.dirichlet([1, 1, 1], size=n).astype(np.float64)
        y_pred = np.argmax(y_proba, axis=1)

        metrics = calculate_classification_metrics(y_true, y_pred, y_proba)

        total = sum(sum(row) for row in metrics.confusion_matrix)
        assert total == n

    def test_top_2_accuracy(self) -> None:
        """Top-2 accuracy should be higher than regular accuracy."""
        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 3, size=n, dtype=np.int64)
        y_proba = np.random.dirichlet([1, 1, 1], size=n).astype(np.float64)
        y_pred = np.argmax(y_proba, axis=1)

        metrics = calculate_classification_metrics(y_true, y_pred, y_proba)

        assert metrics.top_2_accuracy >= metrics.accuracy

    def test_top_2_accuracy_perfect(self) -> None:
        """Top-2 accuracy should be 1.0 when correct answer always in top 2."""
        y_true = np.array([0, 1, 2], dtype=np.int64)
        y_proba = np.array(
            [
                [0.5, 0.4, 0.1],
                [0.4, 0.5, 0.1],
                [0.1, 0.4, 0.5],
            ],
            dtype=np.float64,
        )
        y_pred = np.argmax(y_proba, axis=1)

        metrics = calculate_classification_metrics(y_true, y_pred, y_proba)

        assert metrics.top_2_accuracy == pytest.approx(1.0, abs=1e-6)

    def test_weighted_vs_macro_metrics(self) -> None:
        """Weighted metrics should differ from macro with imbalanced classes."""
        y_true = np.array([0] * 90 + [1] * 5 + [2] * 5, dtype=np.int64)
        y_proba = np.zeros((100, 3), dtype=np.float64)
        y_proba[:, 0] = 0.8
        y_proba[:, 1] = 0.1
        y_proba[:, 2] = 0.1
        y_pred = np.zeros(100, dtype=np.int64)

        metrics = calculate_classification_metrics(y_true, y_pred, y_proba)

        assert metrics.precision_weighted != pytest.approx(
            metrics.precision_macro, abs=0.01
        )


class TestCalculateBettingMetrics:
    """Tests for calculate_betting_metrics function."""

    def test_all_winning_bets(self) -> None:
        """All winning bets should yield positive ROI."""
        y_true = np.array([0, 0, 0], dtype=np.int64)
        y_proba = np.array(
            [
                [0.6, 0.2, 0.2],
                [0.6, 0.2, 0.2],
                [0.6, 0.2, 0.2],
            ],
            dtype=np.float64,
        )
        odds = np.array(
            [
                [2.0, 3.5, 4.0],
                [2.0, 3.5, 4.0],
                [2.0, 3.5, 4.0],
            ],
            dtype=np.float64,
        )

        metrics = calculate_betting_metrics(y_true, y_proba, odds, min_edge=0.0)

        assert metrics.total_bets > 0
        assert metrics.win_rate == pytest.approx(1.0, abs=1e-6)
        assert metrics.roi_percent > 0

    def test_all_losing_bets(self) -> None:
        """All losing bets should yield negative ROI."""
        y_true = np.array([1, 1, 1], dtype=np.int64)
        y_proba = np.array(
            [
                [0.6, 0.2, 0.2],
                [0.6, 0.2, 0.2],
                [0.6, 0.2, 0.2],
            ],
            dtype=np.float64,
        )
        odds = np.array(
            [
                [2.0, 3.5, 4.0],
                [2.0, 3.5, 4.0],
                [2.0, 3.5, 4.0],
            ],
            dtype=np.float64,
        )

        metrics = calculate_betting_metrics(y_true, y_proba, odds, min_edge=0.0)

        assert metrics.total_bets > 0
        assert metrics.win_rate == pytest.approx(0.0, abs=1e-6)
        assert metrics.roi_percent < 0

    def test_min_edge_filter(self) -> None:
        """min_edge should filter out low-edge bets."""
        y_true = np.array([0, 0, 0], dtype=np.int64)
        y_proba = np.array(
            [
                [0.51, 0.25, 0.24],
                [0.51, 0.25, 0.24],
                [0.85, 0.1, 0.05],
            ],
            dtype=np.float64,
        )
        odds = np.array(
            [
                [1.9, 3.5, 4.0],
                [1.9, 3.5, 4.0],
                [2.0, 3.5, 4.0],
            ],
            dtype=np.float64,
        )

        metrics_no_filter = calculate_betting_metrics(
            y_true, y_proba, odds, min_edge=0.0
        )
        metrics_filtered = calculate_betting_metrics(
            y_true, y_proba, odds, min_edge=0.4
        )

        assert metrics_filtered.total_bets <= metrics_no_filter.total_bets

    def test_profit_loss_calculation(self) -> None:
        """Profit/loss should be return minus stake."""
        y_true = np.array([0], dtype=np.int64)
        y_proba = np.array([[0.6, 0.2, 0.2]], dtype=np.float64)
        odds = np.array([[2.0, 3.5, 4.0]], dtype=np.float64)

        metrics = calculate_betting_metrics(
            y_true, y_proba, odds, stake=10.0, kelly_fraction=1.0
        )

        assert metrics.winning_bets == 1
        assert metrics.total_return > metrics.total_stake

    def test_max_drawdown_calculation(self) -> None:
        """Maximum drawdown should track equity curve peak-to-trough."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        y_proba = np.array(
            [
                [0.6, 0.2, 0.2],
                [0.2, 0.6, 0.2],
                [0.6, 0.2, 0.2],
                [0.2, 0.6, 0.2],
                [0.6, 0.2, 0.2],
                [0.2, 0.6, 0.2],
                [0.6, 0.2, 0.2],
                [0.2, 0.6, 0.2],
            ],
            dtype=np.float64,
        )
        odds = np.full((8, 3), [2.0, 2.0, 4.0], dtype=np.float64)

        metrics = calculate_betting_metrics(y_true, y_proba, odds)

        assert 0.0 <= metrics.max_drawdown <= 1.0

    def test_sharpe_ratio(self) -> None:
        """Sharpe ratio should be calculated from equity returns."""
        y_true = np.array([0] * 20, dtype=np.int64)
        y_proba = np.full((20, 3), [0.6, 0.2, 0.2], dtype=np.float64)
        odds = np.full((20, 3), [2.0, 3.5, 4.0], dtype=np.float64)

        metrics = calculate_betting_metrics(y_true, y_proba, odds)

        assert metrics.sharpe_ratio > 0

    def test_empty_bets(self) -> None:
        """Empty results when no bets meet criteria."""
        y_true = np.array([0, 1, 2], dtype=np.int64)
        y_proba = np.array(
            [
                [0.33, 0.33, 0.34],
                [0.33, 0.33, 0.34],
                [0.33, 0.33, 0.34],
            ],
            dtype=np.float64,
        )
        odds = np.array(
            [
                [3.0, 3.0, 3.0],
                [3.0, 3.0, 3.0],
                [3.0, 3.0, 3.0],
            ],
            dtype=np.float64,
        )

        metrics = calculate_betting_metrics(y_true, y_proba, odds, min_edge=0.5)

        assert metrics.total_bets == 0


class TestCalculateOutcomeAccuracy:
    """Tests for calculate_outcome_accuracy function."""

    def test_outcome_accuracy_basic(self) -> None:
        """Outcome accuracy should be calculated per predicted outcome."""
        y_true = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        y_pred = np.array([0, 1, 1, 2, 2, 0], dtype=np.int64)

        accuracy = calculate_outcome_accuracy(y_true, y_pred)

        assert accuracy["H"] == pytest.approx(0.5, abs=1e-6)
        assert accuracy["D"] == pytest.approx(0.5, abs=1e-6)
        assert accuracy["A"] == pytest.approx(0.5, abs=1e-6)

    def test_outcome_accuracy_perfect(self) -> None:
        """Perfect predictions yield 100% accuracy for all outcomes."""
        y_true = np.array([0, 1, 2], dtype=np.int64)
        y_pred = np.array([0, 1, 2], dtype=np.int64)

        accuracy = calculate_outcome_accuracy(y_true, y_pred)

        assert accuracy["H"] == pytest.approx(1.0, abs=1e-6)
        assert accuracy["D"] == pytest.approx(1.0, abs=1e-6)
        assert accuracy["A"] == pytest.approx(1.0, abs=1e-6)

    def test_outcome_accuracy_no_predictions(self) -> None:
        """Missing outcomes should have 0% accuracy."""
        y_true = np.array([0, 0, 0], dtype=np.int64)
        y_pred = np.array([0, 0, 0], dtype=np.int64)

        accuracy = calculate_outcome_accuracy(y_true, y_pred)

        assert accuracy["H"] == pytest.approx(1.0, abs=1e-6)
        assert accuracy["D"] == pytest.approx(0.0, abs=1e-6)
        assert accuracy["A"] == pytest.approx(0.0, abs=1e-6)


class TestEvaluatePredictions:
    """Tests for evaluate_predictions function."""

    def test_evaluate_predictions_basic(self) -> None:
        """evaluate_predictions should return complete result."""
        np.random.seed(42)
        n = 50
        y_true = np.random.randint(0, 3, size=n, dtype=np.int64)
        y_proba = np.random.dirichlet([1, 1, 1], size=n).astype(np.float64)
        y_pred = np.argmax(y_proba, axis=1)
        odds = np.full((n, 3), [2.5, 3.3, 2.8], dtype=np.float64)

        result = evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            odds=odds,
            model_version="test_v1",
            date_range=("2024-01-01", "2024-12-31"),
        )

        assert result.model_version == "test_v1"
        assert result.num_samples == n
        assert result.date_range == ("2024-01-01", "2024-12-31")
        assert result.classification is not None
        assert result.betting is not None

    def test_evaluate_predictions_without_odds(self) -> None:
        """evaluate_predictions should work without odds."""
        np.random.seed(42)
        n = 30
        y_true = np.random.randint(0, 3, size=n, dtype=np.int64)
        y_proba = np.random.dirichlet([1, 1, 1], size=n).astype(np.float64)
        y_pred = np.argmax(y_proba, axis=1)

        result = evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            model_version="test_v2",
        )

        assert result.betting is None
        assert result.classification is not None

    def test_evaluate_predictions_calibration(self) -> None:
        """evaluate_predictions should include calibration metrics."""
        np.random.seed(42)
        n = 50
        y_true = np.random.randint(0, 3, size=n, dtype=np.int64)
        y_proba = np.random.dirichlet([1, 1, 1], size=n).astype(np.float64)
        y_pred = np.argmax(y_proba, axis=1)

        result = evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            model_version="test_v3",
        )

        assert result.expected_calibration_error >= 0
        assert result.maximum_calibration_error >= 0


class TestCompareModels:
    """Tests for compare_models function."""

    def test_compare_models_basic(self) -> None:
        """compare_models should identify best models per metric."""
        results = []
        for i, (acc, ll) in enumerate([(0.5, 1.0), (0.6, 0.9), (0.55, 1.1)]):
            np.random.seed(42 + i)
            n = 50
            y_true = np.random.randint(0, 3, size=n, dtype=np.int64)
            y_proba = np.random.dirichlet([1, 1, 1], size=n).astype(np.float64)
            y_pred = np.argmax(y_proba, axis=1)

            result = evaluate_predictions(
                y_true=y_true,
                y_pred=y_pred,
                y_proba=y_proba,
                model_version=f"model_{i}",
            )
            result.classification.accuracy = acc
            result.classification.log_loss = ll
            results.append(result)

        comparison = compare_models(results)

        assert "models" in comparison
        assert "accuracy" in comparison
        assert "best_accuracy" in comparison
        assert "best_log_loss" in comparison

    def test_compare_models_single(self) -> None:
        """compare_models should work with single model."""
        np.random.seed(42)
        n = 50
        y_true = np.random.randint(0, 3, size=n, dtype=np.int64)
        y_proba = np.random.dirichlet([1, 1, 1], size=n).astype(np.float64)
        y_pred = np.argmax(y_proba, axis=1)

        result = evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            model_version="single_model",
        )

        comparison = compare_models([result])

        assert len(comparison["models"]) == 1
        assert comparison["best_accuracy"] == "single_model"


class TestDataClasses:
    """Tests for data class behavior."""

    def test_classification_metrics_immutable(self) -> None:
        """ClassificationMetrics fields should be accessible."""
        metrics = ClassificationMetrics(
            accuracy=0.5,
            log_loss=1.0,
            brier_score=0.25,
            precision_macro=0.5,
            recall_macro=0.5,
            f1_macro=0.5,
            precision_weighted=0.5,
            recall_weighted=0.5,
            f1_weighted=0.5,
            per_class_precision={"H": 0.5, "D": 0.5, "A": 0.5},
            per_class_recall={"H": 0.5, "D": 0.5, "A": 0.5},
            per_class_f1={"H": 0.5, "D": 0.5, "A": 0.5},
            confusion_matrix=[[10, 5, 5], [5, 10, 5], [5, 5, 10]],
            top_2_accuracy=0.8,
            cohen_kappa=0.25,
        )

        assert metrics.accuracy == 0.5
        assert metrics.confusion_matrix[0][0] == 10

    def test_betting_metrics_defaults(self) -> None:
        """BettingMetrics should store all required fields."""
        metrics = BettingMetrics(
            total_bets=10,
            winning_bets=6,
            losing_bets=4,
            total_stake=100.0,
            total_return=150.0,
            profit_loss=50.0,
            roi_percent=50.0,
            yield_percent=50.0,
            sharpe_ratio=1.5,
            max_drawdown=0.2,
            win_rate=0.6,
            average_winning_odds=2.5,
            average_losing_odds=3.0,
            average_kelly_fraction=0.25,
            optimal_kelly_fraction=0.3,
        )

        assert metrics.total_bets == 10
        assert metrics.win_rate == pytest.approx(0.6, abs=1e-6)

    def test_evaluation_result_optional_fields(self) -> None:
        """EvaluationResult should handle optional betting metrics."""
        result = EvaluationResult(
            model_version="test",
            evaluated_at="2024-01-01T00:00:00",
            num_samples=100,
            date_range=("2024-01-01", "2024-12-31"),
            classification=ClassificationMetrics(
                accuracy=0.5,
                log_loss=1.0,
                brier_score=0.25,
                precision_macro=0.5,
                recall_macro=0.5,
                f1_macro=0.5,
                precision_weighted=0.5,
                recall_weighted=0.5,
                f1_weighted=0.5,
                per_class_precision={"H": 0.5, "D": 0.5, "A": 0.5},
                per_class_recall={"H": 0.5, "D": 0.5, "A": 0.5},
                per_class_f1={"H": 0.5, "D": 0.5, "A": 0.5},
                confusion_matrix=[[10, 5, 5], [5, 10, 5], [5, 5, 10]],
            ),
        )

        assert result.betting is None
        assert result.outcome_accuracy == {}
