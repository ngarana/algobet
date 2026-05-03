"""Unit tests for GPU-aware training tuners."""

from __future__ import annotations

from algobet.predictions.training.tuner import _build_model_config


def test_build_model_config_applies_xgboost_gpu_profile(
    monkeypatch,
) -> None:
    """Tuner model configs should inherit the XGBoost iGPU profile."""
    monkeypatch.setenv("ALGOBET_ACCELERATION_PROFILE", "intel_igpu")
    monkeypatch.setenv("ALGOBET_REQUIRE_GPU", "true")
    monkeypatch.setenv("ALGOBET_XGBOOST_DEVICE", "sycl:gpu:0")
    monkeypatch.setenv("ALGOBET_XGBOOST_TREE_METHOD", "hist")

    config = _build_model_config(
        model_type="xgboost",
        hyperparameters={"max_depth": 6, "n_estimators": 10},
        class_weights={0: 1.0, 1: 1.2, 2: 1.1},
    )

    assert config.hyperparameters["device"] == "sycl:gpu:0"
    assert config.hyperparameters["tree_method"] == "hist"
    assert config.hyperparameters["max_depth"] == 6
    assert config.class_weights == {0: 1.0, 1: 1.2, 2: 1.1}
