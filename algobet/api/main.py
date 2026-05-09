"""FastAPI application entry point for AlgoBet API."""

import json
import os
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from algobet.api.routers import (
    matches_router,
    ml_operations_router,
    models_router,
    predictions_router,
    schedules_router,
    seasons_router,
    teams_router,
    tournaments_router,
    value_bets_router,
    workflow_router,
)
from algobet.api.websockets import websocket_endpoint
from algobet.infrastructure.logging_config import get_logger
from algobet.services.scheduler_service import SchedulerService

try:
    from algobet.api.routers.scraping import router as scraping_router
except ModuleNotFoundError as scraping_import_error:
    scraping_router = None  # type: ignore[assignment]
    print(f"Warning: scraping router disabled during startup: {scraping_import_error}")

logger = get_logger("api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Industrial-grade request/response logging middleware.

    Logs structured JSON for all requests including:
    - Request ID for correlation
    - Timestamp
    - Client info
    - Request body (for POST/PUT/PATCH)
    - Response status and duration
    """

    SENSITIVE_HEADERS = {"authorization", "cookie", "x-api-key"}
    SENSITIVE_FIELDS = {"password", "token", "secret", "api_key"}

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Log incoming request
        client_host = request.client.host if request.client else "unknown"
        client_port = request.client.port if request.client else 0

        # Safely read request body for logging
        body_bytes = b""
        body_logged = False
        try:
            if request.method in ("POST", "PUT", "PATCH"):
                body_bytes = await request.body()

                # Properly restore the request stream by replacing the receive callable
                async def receive() -> dict[str, Any]:
                    return {"type": "http.request", "body": body_bytes}

                request._receive = receive
                body_logged = True
        except Exception as e:
            logger.warning(f"Request ID {request_id}: Failed to read body: {e}")

        # Build safe headers dict
        safe_headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in self.SENSITIVE_HEADERS
        }

        # Redact sensitive fields from body (not full JSON for performance)
        body_preview: dict[str, Any] | list[Any] | str | None = None
        if body_logged and body_bytes:
            try:
                body_preview = self._redact_sensitive(
                    json.loads(body_bytes.decode("utf-8"))
                )
            except (json.JSONDecodeError, UnicodeDecodeError):
                body_preview = "<binary or invalid json>"

        # Log request
        logger.info(
            json.dumps(
                {
                    "event": "request_start",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "query": str(request.url.query) if request.url.query else None,
                    "client": f"{client_host}:{client_port}",
                    "headers": safe_headers,
                    "body": body_preview,
                }
            )
        )

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = round((time.time() - start_time) * 1000, 2)

        # Log response
        logger.info(
            json.dumps(
                {
                    "event": "request_complete",
                    "request_id": request_id,
                    "status": response.status_code,
                    "duration_ms": duration_ms,
                    "method": request.method,
                    "path": request.url.path,
                }
            )
        )

        # Add request ID to response headers for debugging
        response.headers["X-Request-ID"] = request_id
        return response

    def _redact_sensitive(
        self, data: dict[str, Any] | list[Any]
    ) -> dict[str, Any] | list[Any]:
        """Recursively redact sensitive fields from data."""
        if isinstance(data, dict):
            return {
                k: "***REDACTED***"
                if k.lower() in self.SENSITIVE_FIELDS
                else self._redact_sensitive(v)
                for k, v in data.items()
            }
        if isinstance(data, list):
            return [self._redact_sensitive(item) for item in data]
        return data


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown events.

    Manages scheduler initialization and cleanup along with other resources.
    """
    # Startup: Initialize connections, caches, and scheduler
    print("Starting AlgoBet API...")

    # Initialize logging with JSON format for structured output
    from algobet.infrastructure.logging_config import setup_logging

    setup_logging()
    logger.info("Logging initialized with JSON format")

    # Initialize Redis Cache
    try:
        from fastapi_cache import FastAPICache
        from fastapi_cache.backends.redis import RedisBackend
        from redis import asyncio as aioredis

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        redis = aioredis.from_url(redis_url, encoding="utf8", decode_responses=False)
        FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
        print("Redis cache initialized successfully")
    except Exception as redis_error:
        try:
            from fastapi_cache import FastAPICache
            from fastapi_cache.backends.inmemory import InMemoryBackend

            FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")
            print(
                "Redis cache unavailable; using in-memory cache fallback: "
                f"{redis_error}"
            )
        except Exception as cache_error:
            print(
                "Warning: Failed to initialize API cache backends: "
                f"redis={redis_error}; fallback={cache_error}"
            )

    # Start scheduler if enabled (default: enabled)
    enable_scheduler = os.getenv("ENABLE_SCHEDULER", "true").lower() == "true"
    if enable_scheduler:
        try:
            print("Initializing APScheduler...")
            SchedulerService.start_scheduler()
            SchedulerService.load_all_active_tasks()
            print("Scheduler started successfully")
        except Exception as e:
            print(f"Warning: Failed to start scheduler: {e}")
            print("Continuing without scheduler...")
    else:
        print("Scheduler disabled via ENABLE_SCHEDULER=false")

    yield

    # Shutdown: Clean up connections and scheduler
    print("Shutting down AlgoBet API...")

    if enable_scheduler:
        try:
            SchedulerService.shutdown_scheduler()
            print("Scheduler stopped successfully")
        except Exception as e:
            print(f"Warning: Error stopping scheduler: {e}")

    print("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="AlgoBet API",
    description="Football match database and prediction API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add request logging middleware (before CORS for accurate logging)
app.add_middleware(RequestLoggingMiddleware)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3002",
    ],  # Add production URLs as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    tournaments_router,
    prefix="/api/v1/tournaments",
    tags=["tournaments"],
)

app.include_router(
    seasons_router,
    prefix="/api/v1/seasons",
    tags=["seasons"],
)

app.include_router(
    teams_router,
    prefix="/api/v1/teams",
    tags=["teams"],
)

app.include_router(
    matches_router,
    prefix="/api/v1/matches",
    tags=["matches"],
)

app.include_router(
    predictions_router,
    prefix="/api/v1/predictions",
    tags=["predictions"],
)

app.include_router(
    models_router,
    prefix="/api/v1/models",
    tags=["models"],
)

app.include_router(
    ml_operations_router,
    prefix="/api/v1/ml",
    tags=["ml-operations"],
)

app.include_router(
    value_bets_router,
    prefix="/api/v1/value-bets",
    tags=["value-bets"],
)

app.include_router(
    workflow_router,
    prefix="/api/v1/workflow",
    tags=["workflow"],
)

if scraping_router is not None:
    app.include_router(
        scraping_router,
        prefix="/api/v1/scraping",
        tags=["scraping"],
    )

app.include_router(
    schedules_router,
    prefix="/api/v1/schedules",
    tags=["schedules"],
)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "name": "AlgoBet API",
        "version": "0.1.0",
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.websocket("/ws/scraping/{job_id}")
async def progress_websocket(websocket: WebSocket, job_id: str) -> None:
    """WebSocket endpoint for real-time progress updates."""
    await websocket_endpoint(websocket, job_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "algobet.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
