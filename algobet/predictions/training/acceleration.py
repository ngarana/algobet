"""Helpers for environment-driven hardware acceleration profiles."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _read_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE_VALUES


def _read_int(name: str) -> int | None:
    """Read an integer environment variable when present."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


@dataclass(frozen=True)
class AccelerationProfile:
    """Resolved acceleration profile for the current process."""

    profile_name: str | None
    enabled: bool
    require_gpu: bool
    lightgbm_device: str | None
    xgboost_device: str | None
    xgboost_tree_method: str | None
    gpu_platform_id: int | None
    gpu_device_id: int | None


def load_acceleration_profile() -> AccelerationProfile:
    """Load the active acceleration profile from environment variables."""
    profile_name = os.getenv("ALGOBET_ACCELERATION_PROFILE", "").strip().lower() or None
    enabled = profile_name in {"intel_igpu", "intel_gpu"}

    lightgbm_device = os.getenv("ALGOBET_LIGHTGBM_DEVICE")
    if lightgbm_device:
        lightgbm_device = lightgbm_device.strip().lower()
    elif enabled:
        lightgbm_device = "gpu"

    xgboost_device = os.getenv("ALGOBET_XGBOOST_DEVICE")
    if xgboost_device:
        xgboost_device = xgboost_device.strip().lower()
    elif enabled:
        xgboost_device = "sycl:gpu:0"

    xgboost_tree_method = os.getenv("ALGOBET_XGBOOST_TREE_METHOD")
    if xgboost_tree_method:
        xgboost_tree_method = xgboost_tree_method.strip().lower()
    elif enabled:
        xgboost_tree_method = "hist"

    return AccelerationProfile(
        profile_name=profile_name,
        enabled=enabled,
        require_gpu=_read_bool("ALGOBET_REQUIRE_GPU", default=enabled),
        lightgbm_device=lightgbm_device,
        xgboost_device=xgboost_device,
        xgboost_tree_method=xgboost_tree_method,
        gpu_platform_id=_read_int("ALGOBET_GPU_PLATFORM_ID"),
        gpu_device_id=_read_int("ALGOBET_GPU_DEVICE_ID"),
    )


def resolve_training_hyperparameters(
    model_type: str,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    """Apply acceleration profile overrides for a model type."""
    profile = load_acceleration_profile()
    resolved = dict(hyperparameters)

    if not profile.enabled:
        return resolved

    if model_type == "lightgbm":
        if "device" not in resolved and profile.lightgbm_device:
            resolved["device"] = profile.lightgbm_device

        resolved_device = str(resolved.get("device", "")).strip().lower()

        if resolved_device == "gpu":
            if (
                "gpu_platform_id" not in resolved
                and profile.gpu_platform_id is not None
            ):
                resolved["gpu_platform_id"] = profile.gpu_platform_id
            if "gpu_device_id" not in resolved and profile.gpu_device_id is not None:
                resolved["gpu_device_id"] = profile.gpu_device_id

        if profile.require_gpu and resolved_device != "gpu":
            raise ValueError(
                "Intel iGPU acceleration is required for this worker. "
                "Use LightGBM with hyperparameters.device='gpu' or disable "
                "ALGOBET_REQUIRE_GPU."
            )

        return resolved

    if model_type == "xgboost":
        if "device" not in resolved and profile.xgboost_device:
            resolved["device"] = profile.xgboost_device
        if "tree_method" not in resolved and profile.xgboost_tree_method:
            resolved["tree_method"] = profile.xgboost_tree_method

        resolved_device = str(resolved.get("device", "")).strip().lower()
        resolved_tree_method = str(resolved.get("tree_method", "")).strip().lower()

        if profile.require_gpu and not resolved_device.startswith("sycl:gpu"):
            raise ValueError(
                "Intel iGPU acceleration is required for this worker. "
                "Use XGBoost with hyperparameters.device='sycl:gpu[:ordinal]' "
                "or disable ALGOBET_REQUIRE_GPU."
            )

        if profile.require_gpu and resolved_tree_method != "hist":
            raise ValueError(
                "Intel iGPU acceleration is required for this worker. "
                "Use XGBoost with hyperparameters.tree_method='hist' or disable "
                "ALGOBET_REQUIRE_GPU."
            )

        return resolved

    if profile.require_gpu:
        raise ValueError(
            f"Intel iGPU acceleration is required for this worker, but "
            f"model_type='{model_type}' is not configured for the current "
            "Intel GPU path. Use model_type='xgboost' or 'lightgbm' or disable "
            "ALGOBET_REQUIRE_GPU."
        )

    return resolved
