"""Shared helpers for thin API routers."""

from collections.abc import Sequence
from typing import Any

from fastapi import HTTPException, Request


def handle_service_error(exc: Exception) -> HTTPException:
    """Normalize service-layer failures into HTTP exceptions."""
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, LookupError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def parse_pagination_params(
    request: Request,
    *,
    default_limit: int = 50,
    max_limit: int = 200,
) -> tuple[int, int]:
    """Parse common limit/offset query parameters from a request."""
    limit = int(request.query_params.get("limit", default_limit))
    offset = int(request.query_params.get("offset", 0))
    return min(max(limit, 1), max_limit), max(offset, 0)


def serialize_paginated(
    results: Sequence[Any],
    total: int,
    page: int,
    size: int,
) -> dict[str, Any]:
    """Build a generic paginated payload for router-specific schemas."""
    return {
        "items": list(results),
        "total": total,
        "limit": size,
        "offset": max(page, 0) * size,
    }
