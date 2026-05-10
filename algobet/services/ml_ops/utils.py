"""Shared ML operation service helpers."""

from typing import Any

import numpy as np


def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy/scalar payloads to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return type(obj)(convert_numpy_types(item) for item in obj)
    if isinstance(obj, np.floating | np.integer):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
