"""Collaborators for the OddsPortalScraper facade."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from algobet.scraper import ScrapedMatch


class PageNavigator:
    """Own page navigation entry points while the scraper is decomposed."""

    def __init__(self, implementation: Any) -> None:
        self.implementation = implementation

    def navigate_to_results(self, url: str) -> None:
        self.implementation._navigate_to_results_impl(url)

    def navigate_to_upcoming(self, url: str) -> None:
        self.implementation._navigate_to_upcoming_impl(url)


class MatchExtractor:
    """Extract historical result rows from the current page."""

    def __init__(self, implementation: Any) -> None:
        self.implementation = implementation

    def scrape_current_page(self) -> list[ScrapedMatch]:
        return self.implementation._scrape_current_page_impl()  # type: ignore[no-any-return]


class UpcomingMatchExtractor:
    """Extract upcoming match rows from OddsPortal."""

    def __init__(self, implementation: Any) -> None:
        self.implementation = implementation

    def scrape_upcoming_matches(self, **kwargs: Any) -> list[dict[str, Any]]:
        return cast(
            list[dict[str, Any]],
            self.implementation._scrape_upcoming_matches_impl(**kwargs),
        )
