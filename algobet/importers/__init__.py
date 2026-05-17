"""Data importers for external football data sources.

This package provides importers for various football data sources:
- SoccerData: Unified scraper for FBref, WhoScored, and more
- FDImporter: Football-Data.co.uk historical data (25+ years, no CAPTCHA issues)
- FBref: Direct Playwright-based FBref scraper (bypasses Cloudflare blocks)
"""

from algobet.importers.fd_importer import FDImporter
from algobet.importers.soccerdata_importer import SoccerDataImporter

__all__ = ["SoccerDataImporter", "FDImporter"]
