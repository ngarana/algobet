"""FastAPI application entry point for AlgoBet API."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from algobet.api.routers import (
    matches_router,
    models_router,
    predictions_router,
    scraping_router,
    seasons_router,
    teams_router,
    tournaments_router,
    value_bets_router,
)
from algobet.api.websockets import websocket_endpoint


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown events."""
    # Startup: Initialize connections, caches, etc.
    print("Starting AlgoBet API...")
    yield
    # Shutdown: Clean up connections, etc.
    print("Shutting down AlgoBet API...")


# Create FastAPI application
app = FastAPI(
    title="AlgoBet API",
    description="Football match database and prediction API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
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
    value_bets_router,
    prefix="/api/v1/value-bets",
    tags=["value-bets"],
)

app.include_router(
    scraping_router,
    prefix="/api/v1/scraping",
    tags=["scraping"],
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


@app.websocket("/ws/progress/{client_id}")
async def progress_websocket(
    websocket: WebSocket, client_id: str, job_id: str | None = None
) -> None:
    """WebSocket endpoint for real-time progress updates."""
    await websocket_endpoint(websocket, client_id, job_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "algobet.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
