"""Use-case collaborators for the SoccerDataImporter facade."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from algobet.importers.soccerdata_importer import ImportResult


class SoccerDataImplementation(Protocol):
    """Private implementation contract exposed during importer decomposition."""

    def _import_schedule_impl(
        self,
        league: str,
        season: str,
        no_cache: bool = False,
        headless: bool = True,
    ) -> ImportResult: ...

    def _enrich_understat_stats_impl(
        self,
        league: str,
        season: str,
    ) -> int: ...

    def _enrich_player_stats_impl(
        self,
        league: str,
        season: str,
        skip_existing: bool = True,
    ) -> dict[str, int]: ...


class ScheduleImporter:
    """Import fixture schedules through the facade implementation."""

    def __init__(self, implementation: SoccerDataImplementation) -> None:
        self.implementation = implementation

    def run(
        self,
        league: str,
        season: str,
        no_cache: bool = False,
        headless: bool = True,
    ) -> ImportResult:
        return self.implementation._import_schedule_impl(
            league,
            season,
            no_cache=no_cache,
            headless=headless,
        )


class StatsEnricher:
    """Enrich matches with external statistics."""

    def __init__(self, implementation: SoccerDataImplementation) -> None:
        self.implementation = implementation

    def enrich_understat(self, league: str, season: str) -> int:
        return self.implementation._enrich_understat_stats_impl(league, season)

    def enrich_player_stats(
        self,
        league: str,
        season: str,
        skip_existing: bool = True,
    ) -> dict[str, int]:
        return self.implementation._enrich_player_stats_impl(
            league,
            season,
            skip_existing=skip_existing,
        )
