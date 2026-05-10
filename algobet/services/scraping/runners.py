"""Scraping use-case collaborators for the ScrapingService facade."""

from __future__ import annotations

from datetime import date as date_cls
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from algobet.services.scraping_service import ScrapingProgress


class ScrapingImplementation(Protocol):
    """Private implementation contract exposed during service decomposition."""

    def _scrape_upcoming_impl(
        self,
        url: str | None = None,
        headless: bool = True,
        target_team_id: int | None = None,
    ) -> ScrapingProgress: ...

    def _scrape_matches_by_date_impl(
        self,
        target_date: date_cls,
        url: str | None = None,
        headless: bool = True,
        target_team_id: int | None = None,
    ) -> ScrapingProgress: ...

    def _scrape_results_impl(
        self,
        url: str,
        max_pages: int | None = None,
        headless: bool = True,
        target_team_id: int | None = None,
        season: str | None = None,
    ) -> ScrapingProgress: ...

    def _scrape_results_range_impl(
        self,
        url: str,
        seasons: list[str],
        max_pages: int | None = None,
        headless: bool = True,
        target_team_id: int | None = None,
    ) -> ScrapingProgress: ...


class UpcomingScraper:
    """Scrape upcoming matches through the facade implementation."""

    def __init__(self, implementation: ScrapingImplementation) -> None:
        self.implementation = implementation

    def run(
        self,
        url: str | None = None,
        headless: bool = True,
        target_team_id: int | None = None,
    ) -> ScrapingProgress:
        return self.implementation._scrape_upcoming_impl(
            url=url,
            headless=headless,
            target_team_id=target_team_id,
        )


class ByDateScraper:
    """Scrape scheduled matches for one date."""

    def __init__(self, implementation: ScrapingImplementation) -> None:
        self.implementation = implementation

    def run(
        self,
        target_date: date_cls,
        url: str | None = None,
        headless: bool = True,
        target_team_id: int | None = None,
    ) -> ScrapingProgress:
        return self.implementation._scrape_matches_by_date_impl(
            target_date,
            url=url,
            headless=headless,
            target_team_id=target_team_id,
        )


class ResultScraper:
    """Scrape historical results for one season or results URL."""

    def __init__(self, implementation: ScrapingImplementation) -> None:
        self.implementation = implementation

    def run(
        self,
        url: str,
        max_pages: int | None = None,
        headless: bool = True,
        target_team_id: int | None = None,
        season: str | None = None,
    ) -> ScrapingProgress:
        return self.implementation._scrape_results_impl(
            url,
            max_pages=max_pages,
            headless=headless,
            target_team_id=target_team_id,
            season=season,
        )


class RangeScraper:
    """Scrape historical results across many seasons."""

    def __init__(self, implementation: ScrapingImplementation) -> None:
        self.implementation = implementation

    def run(
        self,
        url: str,
        seasons: list[str],
        max_pages: int | None = None,
        headless: bool = True,
        target_team_id: int | None = None,
    ) -> ScrapingProgress:
        return self.implementation._scrape_results_range_impl(
            url,
            seasons,
            max_pages=max_pages,
            headless=headless,
            target_team_id=target_team_id,
        )
