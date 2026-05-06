"""Data importers for external football data sources.

This package provides importers for various football data sources:
- Football-Data.co.uk: Free historical match data and betting odds
- SoccerData: Unified scraper for FBref, WhoScored, and more
"""

from algobet.importers.football_data import FootballDataImporter
from algobet.importers.soccerdata_importer import SoccerDataImporter

__all__ = ["FootballDataImporter", "SoccerDataImporter"]
