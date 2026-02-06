"""API routers for AlgoBet"""

from .matches import router as matches_router
from .models import router as models_router
from .predictions import router as predictions_router
from .scraping import router as scraping_router
from .seasons import router as seasons_router
from .teams import router as teams_router
from .tournaments import router as tournaments_router
from .value_bets import router as value_bets_router

__all__ = [
    "matches_router",
    "teams_router",
    "tournaments_router",
    "seasons_router",
    "predictions_router",
    "models_router",
    "value_bets_router",
    "scraping_router",
]
