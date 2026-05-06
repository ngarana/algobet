"""Shared team name resolution for cross-source normalization.

Provides a single source of truth for resolving team names from different
data sources (FBref, WhoScored, OddsPortal, Football-Data.co.uk) to a
canonical name, using the teamname_replacements.json config file.

Usage:
    from algobet.utils.team_resolver import TeamResolver

    resolver = TeamResolver()
    name = resolver.resolve("Man City")  # → "Manchester City"
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = os.path.join(
    os.environ.get("SOCCERDATA_DIR", os.path.expanduser("~/.soccerdata")),
    "config",
    "teamname_replacements.json",
)


class TeamResolver:
    """Resolves team names to canonical forms using teamname_replacements.json.

    Attributes:
        mappings: Dict of canonical_name → list[alias] from the config file.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.mappings = _load_mappings(config_path or _DEFAULT_CONFIG_PATH)

    def resolve(self, name: str) -> str:
        """Resolve a team name to its canonical form.

        Args:
            name: Team name from any data source

        Returns:
            Canonical team name, or the original name if no mapping found
        """
        for std_name, aliases in self.mappings.items():
            if name == std_name or name in aliases:
                return std_name
        return name

    @property
    def canonical_names(self) -> list[str]:
        """Return all canonical team names."""
        return list(self.mappings.keys())


def _load_mappings(config_path: str | Path) -> dict[str, list[str]]:
    """Load team name mappings from config file."""
    path = Path(config_path)
    if not path.exists():
        logger.warning("teamname_replacements.json not found at %s", path)
        return {}
    with open(path) as f:
        return json.load(f)
