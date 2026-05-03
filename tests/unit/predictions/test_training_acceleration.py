"""Tests for GPU acceleration profile helpers."""

import pytest

from algobet.predictions.training.acceleration import (
    load_acceleration_profile,
    resolve_training_hyperparameters,
)


def test_load_profile_defaults_to_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Acceleration profile should be disabled by default."""
    monkeypatch.delenv("ALGOBET_ACCELERATION_PROFILE", raising=False)
    monkeypatch.delenv("ALGOBET_REQUIRE_GPU", raising=False)

    profile = load_acceleration_profile()

    assert profile.enabled is False
    assert profile.require_gpu is False
    assert profile.lightgbm_device is None
    assert profile.xgboost_device is None
    assert profile.xgboost_tree_method is None


def test_lightgbm_gpu_overrides_are_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LightGBM should inherit GPU settings from the Intel iGPU profile."""
    monkeypatch.setenv("ALGOBET_ACCELERATION_PROFILE", "intel_igpu")
    monkeypatch.setenv("ALGOBET_REQUIRE_GPU", "true")
    monkeypatch.setenv("ALGOBET_LIGHTGBM_DEVICE", "gpu")
    monkeypatch.setenv("ALGOBET_GPU_PLATFORM_ID", "0")
    monkeypatch.setenv("ALGOBET_GPU_DEVICE_ID", "1")

    resolved = resolve_training_hyperparameters(
        model_type="lightgbm",
        hyperparameters={"learning_rate": 0.05},
    )

    assert resolved["learning_rate"] == 0.05
    assert resolved["device"] == "gpu"
    assert resolved["gpu_platform_id"] == 0
    assert resolved["gpu_device_id"] == 1


def test_lightgbm_requires_gpu_when_profile_is_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A strict Intel iGPU worker should reject CPU LightGBM runs."""
    monkeypatch.setenv("ALGOBET_ACCELERATION_PROFILE", "intel_igpu")
    monkeypatch.setenv("ALGOBET_REQUIRE_GPU", "true")

    with pytest.raises(ValueError, match="LightGBM"):
        resolve_training_hyperparameters(
            model_type="lightgbm",
            hyperparameters={"device": "cpu"},
        )


def test_xgboost_sycl_overrides_are_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """XGBoost should inherit SYCL GPU settings from the Intel iGPU profile."""
    monkeypatch.setenv("ALGOBET_ACCELERATION_PROFILE", "intel_igpu")
    monkeypatch.setenv("ALGOBET_REQUIRE_GPU", "true")
    monkeypatch.setenv("ALGOBET_XGBOOST_DEVICE", "sycl:gpu:0")
    monkeypatch.setenv("ALGOBET_XGBOOST_TREE_METHOD", "hist")

    resolved = resolve_training_hyperparameters(
        model_type="xgboost",
        hyperparameters={"max_depth": 6},
    )

    assert resolved["max_depth"] == 6
    assert resolved["device"] == "sycl:gpu:0"
    assert resolved["tree_method"] == "hist"


def test_xgboost_requires_explicit_gpu_device_when_profile_is_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A strict Intel iGPU worker should reject CPU XGBoost runs."""
    monkeypatch.setenv("ALGOBET_ACCELERATION_PROFILE", "intel_igpu")
    monkeypatch.setenv("ALGOBET_REQUIRE_GPU", "true")

    with pytest.raises(ValueError, match="XGBoost"):
        resolve_training_hyperparameters(
            model_type="xgboost",
            hyperparameters={"device": "cpu", "tree_method": "hist"},
        )


def test_xgboost_requires_hist_tree_method_when_profile_is_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Intel iGPU path should reject non-hist XGBoost tree methods."""
    monkeypatch.setenv("ALGOBET_ACCELERATION_PROFILE", "intel_igpu")
    monkeypatch.setenv("ALGOBET_REQUIRE_GPU", "true")

    with pytest.raises(ValueError, match="tree_method='hist'"):
        resolve_training_hyperparameters(
            model_type="xgboost",
            hyperparameters={"device": "sycl:gpu:0", "tree_method": "approx"},
        )


def test_unsupported_model_types_are_rejected_when_gpu_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported model types should not silently fall back to CPU."""
    monkeypatch.setenv("ALGOBET_ACCELERATION_PROFILE", "intel_igpu")
    monkeypatch.setenv("ALGOBET_REQUIRE_GPU", "true")

    with pytest.raises(ValueError, match="model_type='random_forest'"):
        resolve_training_hyperparameters(
            model_type="random_forest",
            hyperparameters={"max_depth": 6},
        )
