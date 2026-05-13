"""Unit tests for ML training runner behavior."""

from types import SimpleNamespace

from algobet.services.ml_ops.training_runner import TrainingRunner


class TestTrainingRunner:
    """Test cases for training operation safeguards."""

    def test_activation_gate_rejects_single_class_predictions(self) -> None:
        """Collapsed predictors should not become active automatically."""
        result = SimpleNamespace(
            test_metrics={
                "predicted_classes": 1.0,
                "log_loss": 1.0,
                "market_log_loss": 1.1,
            }
        )

        assert TrainingRunner()._passes_activation_gate(result) is False

    def test_activation_gate_rejects_market_underperformance(self) -> None:
        """Models worse than the market diagnostic should not be auto-activated."""
        result = SimpleNamespace(
            test_metrics={
                "predicted_classes": 3.0,
                "log_loss": 1.1,
                "market_log_loss": 1.0,
            }
        )

        assert TrainingRunner()._passes_activation_gate(result) is False

    def test_activation_gate_accepts_diverse_model_beating_market(self) -> None:
        """A diverse model with better log loss can be activated."""
        result = SimpleNamespace(
            test_metrics={
                "predicted_classes": 3.0,
                "log_loss": 0.95,
                "market_log_loss": 1.0,
            }
        )

        assert TrainingRunner()._passes_activation_gate(result) is True
