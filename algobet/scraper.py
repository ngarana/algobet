"""Playwright-based web scraper for OddsPortal football match data."""

import contextlib
import functools
import logging
import re
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, TypeVar, cast

from playwright.sync_api import (
    Browser,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

from algobet.scraping import MatchExtractor, PageNavigator, UpcomingMatchExtractor

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])
NETWORK_ERROR_MARKERS = (
    "err_name_not_resolved",
    "err_internet_disconnected",
    "err_connection_refused",
    "err_connection_reset",
    "err_connection_timed_out",
    "err_network_changed",
    "net::",
    "networkidle",
)


def is_retryable_network_error(error: Exception) -> bool:
    """Return True when an exception looks like a transient network failure."""
    error_message = str(error).lower()
    return isinstance(error, ConnectionError | PlaywrightTimeoutError) or any(
        marker in error_message for marker in NETWORK_ERROR_MARKERS
    )


def retry_on_network_error(
    max_retries: int = 3,
    delay: float = 5.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator to retry network operations with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exception types to catch and retry
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    is_network_error = is_retryable_network_error(e)

                    if not is_network_error:
                        raise

                    if attempt < max_retries:
                        logger.warning(
                            f"Network error in {func.__name__}: {e}. "
                            f"Retrying in {current_delay}s... (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}"
                        )
                        raise

            if last_exception:
                raise last_exception
            return None

        return cast(F, wrapper)

    return decorator


def check_dns_resolution(
    hostname: str = "www.oddsportal.com", timeout: int = 5
) -> bool:
    """Check if DNS resolution works for the target hostname.

    Args:
        hostname: Hostname to check
        timeout: Timeout in seconds

    Returns:
        True if DNS resolution succeeds, False otherwise
    """
    try:
        socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
        return True
    except socket.gaierror as e:
        logger.error(f"DNS resolution failed for {hostname}: {e}")
        return False
    except Exception as e:
        logger.warning(f"Unexpected error during DNS check for {hostname}: {e}")
        return False


@dataclass
class ScrapedMatch:
    """Data class for a scraped match."""

    match_date: datetime
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    odds_home: float | None = None
    odds_draw: float | None = None
    odds_away: float | None = None
    num_bookmakers: int | None = None


@dataclass
class SeasonInfo:
    """Information about a season."""

    name: str  # e.g., "2023/2024"
    url_suffix: str | None  # e.g., "2023-2024" or None for current
    is_current: bool


class OddsPortalScraper:
    """Scraper for OddsPortal football match results."""

    # CSS Selectors
    MATCH_ROW_SELECTOR = 'div[data-testid="game-row"]'
    SEASON_LINK_SELECTOR = "a.bg-gray-medium"
    PAGINATION_SELECTOR = "a.pagination-link"

    def __init__(self, headless: bool = True):
        """Initialize the scraper.

        Args:
            headless: Run browser in headless mode (no GUI).
        """
        self.headless = headless
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._page: Page | None = None
        self._navigator = PageNavigator(self)
        self._match_extractor = MatchExtractor(self)
        self._upcoming_extractor = UpcomingMatchExtractor(self)

    def __enter__(self) -> "OddsPortalScraper":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    def start(self) -> None:
        """Start the browser."""
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        self._page = self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
            locale="en-US",
        ).new_page()
        # Hide webdriver flag
        self._page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        # Set generous timeout for slow connections
        self._page.set_default_timeout(120000)

    def close(self) -> None:
        """Close the browser."""
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()

    def _dismiss_overlays(self) -> None:
        """Dismiss consent and survey overlays that block the content."""
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")

        dismiss_targets = [
            'button:has-text("Reject All")',
            'button:has-text("I Accept")',
            'button:has-text("Accept")',
            'button:has-text("Close")',
            'button[aria-label="Close"]',
            'text="Reject All"',
            'text="I Accept"',
        ]

        for selector in dismiss_targets:
            try:
                locator = self._page.locator(selector).first
                if locator.count() == 0:
                    continue
                if not locator.is_visible(timeout=1500):
                    continue
                locator.click(timeout=5000)
                self._page.wait_for_timeout(1500)
            except Exception:
                continue

    def _wait_for_match_rows(self, timeout: int) -> None:
        """Wait for match rows, retrying once after clearing overlays."""
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")

        try:
            self._page.wait_for_selector(self.MATCH_ROW_SELECTOR, timeout=15000)
            return
        except PlaywrightTimeoutError:
            self._dismiss_overlays()

        try:
            self._page.wait_for_selector(self.MATCH_ROW_SELECTOR, timeout=timeout)
        except PlaywrightTimeoutError:
            logger.warning("Timeout waiting for game-row selector")

    @retry_on_network_error(
        max_retries=3,
        delay=5.0,
        backoff=2.0,
        exceptions=(Exception,),
    )
    def navigate_to_results(self, url: str) -> None:
        """Navigate to a results page."""
        self._navigator.navigate_to_results(url)

    @retry_on_network_error(
        max_retries=3,
        delay=5.0,
        backoff=2.0,
        exceptions=(Exception,),
    )
    def _navigate_to_results_impl(self, url: str) -> None:
        """Navigate to a results page.

        Args:
            url: The URL of the results page.

        Raises:
            ConnectionError: If DNS resolution fails
            RuntimeError: If browser is not started
        """
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")

        # Check DNS resolution before attempting navigation
        hostname = url.replace("https://", "").replace("http://", "").split("/")[0]
        if not check_dns_resolution(hostname):
            raise ConnectionError(
                f"Cannot resolve hostname {hostname}. "
                "Please check your network connection and DNS settings."
            )

        logger.info(f"Navigating to results page: {url}")

        # Use domcontentloaded + fixed wait for JS rendering
        try:
            self._page.goto(url, wait_until="domcontentloaded", timeout=120000)
        except Exception as e:
            error_msg = str(e).lower()
            if "err_name_not_resolved" in error_msg:
                raise ConnectionError(
                    f"DNS resolution failed for {url}. "
                    "The hostname cannot be resolved. "
                    "Please check your network connection."
                ) from e
            raise

        self._page.wait_for_timeout(5000)
        self._dismiss_overlays()
        self._wait_for_match_rows(timeout=120000)

    def get_available_seasons(self) -> list[SeasonInfo]:
        """Get list of available seasons from the current page.

        Returns:
            List of SeasonInfo objects.
        """
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")
        seasons = []

        # Find all season links
        season_links = self._page.query_selector_all(self.SEASON_LINK_SELECTOR)

        for link in season_links:
            href = link.get_attribute("href") or ""
            text = link.inner_text().strip()

            # Check if it's a season link (contains year pattern)
            if not re.search(r"\d{4}/\d{4}", text):
                continue

            # Determine if current season (no year suffix in URL)
            is_current = "-20" not in href and "-19" not in href

            # Extract URL suffix for past seasons
            url_suffix = None
            match = re.search(r"-(\d{4}-\d{4})(?:/|$)", href)
            if match:
                url_suffix = match.group(1)

            seasons.append(
                SeasonInfo(name=text, url_suffix=url_suffix, is_current=is_current)
            )

        return seasons

    def get_season_url(self, base_url: str, season: SeasonInfo) -> str:
        """Build the URL for a specific season.

        Args:
            base_url: Base URL of the current season (e.g., .../premier-league/results/)
            season: SeasonInfo object.

        Returns:
            Full URL for the season.
        """
        if season.is_current or season.url_suffix is None:
            return base_url

        # Replace league slug with season-specific slug
        # e.g., /premier-league/results/ -> /premier-league-2023-2024/results/
        normalized_base_url = re.sub(r"-(\d{4}-\d{4})(?=/results/)", "", base_url)
        pattern = r"/([^/]+)/results/"
        replacement = f"/\\1-{season.url_suffix}/results/"
        return re.sub(pattern, replacement, normalized_base_url)

    def scrape_current_page(self) -> list[ScrapedMatch]:
        """Scrape all matches from the current page."""
        return self._match_extractor.scrape_current_page()

    def _scrape_current_page_impl(self) -> list[ScrapedMatch]:
        """Scrape all matches from the current page.

        Returns:
            List of ScrapedMatch objects.
        """
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")
        matches = []

        # Ensure content is loaded - wait for at least one game row
        try:
            self._page.wait_for_selector('div[data-testid="game-row"]', timeout=30000)
            # Also wait for odds to populate (they might load slightly later)
            self._page.wait_for_selector(
                'div[data-testid="odd-container-default"]', timeout=10000
            )
        except Exception:
            self._dismiss_overlays()
            logger.warning(
                "Timeout waiting for game rows or odds in scrape_current_page, attempting to scrape anyway..."
            )

        # Use JavaScript to extract all match data with proper date association
        # This iterates through all children of ALL eventRow containers
        match_data = self._page.evaluate(
            """
            () => {
                const results = [];
                let currentDate = null;

                // Find all containers with events
                const containers = Array.from(document.querySelectorAll('.eventRow'));

                // Fallback if no eventRow class found
                if (containers.length === 0) {
                    const panel = document.querySelector('div[data-testid="results-panel"]');
                    if (panel) containers.push(panel);
                }

                for (const container of containers) {
                    // Get all direct children
                    const children = Array.from(container.children);

                    for (const child of children) {
                        // Check if this is a date header
                        const dateHeader = child.querySelector('div[data-testid="date-header"]');
                        if (dateHeader) {
                            // Extract date text (format: "08 Jan 2026")
                            const dateText = dateHeader.innerText.trim();
                            if (/\\d{1,2} [A-Za-z]+ \\d{4}/.test(dateText)) {
                                currentDate = dateText;
                            }
                            continue;
                        }

                        // Check if this is a match row
                        const gameRow = child.querySelector('div[data-testid="game-row"]') ||
                                        (child.getAttribute('data-testid') === 'game-row' ? child : null);
                        if (!gameRow) continue;

                        // Extract time
                        const timeElem = gameRow.querySelector('div[data-testid="time-item"]');
                        const timeStr = timeElem ? timeElem.innerText.trim() : '00:00';

                        // Extract teams (links with title attribute)
                        const teamLinks = Array.from(gameRow.querySelectorAll('a[title]'));
                        if (teamLinks.length < 2) continue;

                        const homeTeam = teamLinks[0].getAttribute('title') || teamLinks[0].innerText.trim();
                        const awayTeam = teamLinks[1].getAttribute('title') || teamLinks[1].innerText.trim();

                        // Extract score
                        const rowText = gameRow.innerText;
                        const scoreMatch = rowText.match(/(\\d+)\\s*[–-]\\s*(\\d+)/);
                        if (!scoreMatch) continue;

                        const homeScore = parseInt(scoreMatch[1]);
                        const awayScore = parseInt(scoreMatch[2]);

                        // Extract odds (decimal numbers)
                        const oddsMatches = rowText.match(/(\\d+\\.\\d+)/g) || [];

                        results.push({
                            date: currentDate,
                            time: timeStr,
                            homeTeam: homeTeam,
                            awayTeam: awayTeam,
                            homeScore: homeScore,
                            awayScore: awayScore,
                            oddsHome: oddsMatches[0] ? parseFloat(oddsMatches[0]) : null,
                            oddsDraw: oddsMatches[1] ? parseFloat(oddsMatches[1]) : null,
                            oddsAway: oddsMatches[2] ? parseFloat(oddsMatches[2]) : null,
                            numBookmakers: oddsMatches.length >= 3 ? parseInt(rowText.trim().split(/\\s+/).pop()) || null : null
                        });
                    }
                }

                return results;
            }
        """
        )

        # Convert JavaScript results to ScrapedMatch objects
        for data in match_data:
            try:
                # Parse the date and time
                match_date = datetime.now()
                if data.get("date"):
                    try:
                        date_str = data["date"]
                        time_str = data.get("time", "00:00")
                        match_date = datetime.strptime(
                            f"{date_str} {time_str}", "%d %b %Y %H:%M"
                        )
                    except ValueError:
                        with contextlib.suppress(ValueError):
                            match_date = datetime.strptime(date_str, "%d %b %Y")

                match = ScrapedMatch(
                    match_date=match_date,
                    home_team=data["homeTeam"],
                    away_team=data["awayTeam"],
                    home_score=data["homeScore"],
                    away_score=data["awayScore"],
                    odds_home=data.get("oddsHome"),
                    odds_draw=data.get("oddsDraw"),
                    odds_away=data.get("oddsAway"),
                    num_bookmakers=data.get("numBookmakers")
                    if data.get("numBookmakers") and data["numBookmakers"] < 100
                    else None,
                )
                matches.append(match)
            except Exception as e:
                logger.error(f"Error parsing match: {e}")
                continue

        return matches

    @retry_on_network_error(
        max_retries=3,
        delay=5.0,
        backoff=2.0,
        exceptions=(Exception,),
    )
    def navigate_to_upcoming(
        self, url: str = "https://www.oddsportal.com/matches/"
    ) -> None:
        """Navigate to upcoming matches page (global or league-specific)."""
        self._navigator.navigate_to_upcoming(url)

    @retry_on_network_error(
        max_retries=3,
        delay=5.0,
        backoff=2.0,
        exceptions=(Exception,),
    )
    def _navigate_to_upcoming_impl(
        self, url: str = "https://www.oddsportal.com/matches/"
    ) -> None:
        """Navigate to upcoming matches page (global or league-specific).

        Args:
            url: The URL of the matches page. Can be:
                - Global: https://www.oddsportal.com/matches/
                - League: https://www.oddsportal.com/football/england/premier-league/

        Raises:
            ConnectionError: If DNS resolution fails
            RuntimeError: If browser is not started
        """
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")

        # Check DNS resolution before attempting navigation
        hostname = url.replace("https://", "").replace("http://", "").split("/")[0]
        if not check_dns_resolution(hostname):
            raise ConnectionError(
                f"Cannot resolve hostname {hostname}. "
                "Please check your network connection and DNS settings."
            )

        logger.info(f"Navigating to upcoming matches page: {url}")

        # networkidle ensures the React app's XHR data requests complete before we proceed
        try:
            self._page.goto(url, wait_until="networkidle", timeout=120000)
        except Exception as e:
            error_msg = str(e).lower()
            if "err_name_not_resolved" in error_msg:
                raise ConnectionError(
                    f"DNS resolution failed for {url}. "
                    "The hostname cannot be resolved. "
                    "Please check your network connection."
                ) from e
            raise

        self._dismiss_overlays()
        self._wait_for_match_rows(timeout=30000)

    def _click_show_more(self, max_clicks: int = 15, delay_ms: int = 1800) -> None:
        """Click any 'Show more' button repeatedly until it disappears or max_clicks reached."""
        if self._page is None:
            return

        logger.info("Looking for 'Show more' buttons...")
        clicked = 0

        for _i in range(max_clicks):
            try:
                # Try common selectors for "Show more"
                show_more = (
                    self._page.query_selector('text="Show more"')
                    or self._page.query_selector('button:has-text("Show more")')
                    or self._page.query_selector('a:has-text("Show more")')
                )

                if not show_more:
                    logger.info("No more 'Show more' buttons found.")
                    break

                logger.info(f"Clicking 'Show more' button #{clicked + 1}")
                show_more.scroll_into_view_if_needed()
                show_more.click()
                self._page.wait_for_timeout(delay_ms)

                # Optional: wait for new rows to appear
                self._page.wait_for_selector(
                    'div[data-testid="game-row"]', timeout=5000
                )
                clicked += 1

            except Exception as e:
                logger.debug(f"Show-more click attempt failed: {e}")
                break

        logger.info(f"✅ Clicked 'Show more' {clicked} times.")

    def _scroll_for_lazy_content(
        self, max_scrolls: int = 40, scroll_delay_ms: int = 2200
    ) -> None:
        """Aggressive scroll + network idle wait (works better after 'Show more' clicks)."""
        if self._page is None:
            return

        logger.info("Starting infinite scroll...")
        last_count = 0
        stable = 0

        for step in range(max_scrolls):
            self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            self._page.wait_for_timeout(scroll_delay_ms)

            # Wait for any background XHRs
            with contextlib.suppress(Exception):
                self._page.wait_for_load_state("networkidle", timeout=4000)

            current_count = self._page.evaluate(
                "document.querySelectorAll('div[data-testid=\"game-row\"]').length"
            )

            if current_count > last_count:
                logger.debug(
                    f"Scroll {step + 1}: +{current_count - last_count} rows (total {current_count})"
                )
                last_count = current_count
                stable = 0
            else:
                stable += 1
                if stable >= 4:
                    logger.info(f"Scroll stabilized at {last_count} match rows.")
                    break

    def scrape_upcoming_matches(
        self, only_future_matches: bool = True, buffer_minutes: int = 30
    ) -> list[dict[str, Any]]:
        """Scrape upcoming match rows."""
        return self._upcoming_extractor.scrape_upcoming_matches(
            only_future_matches=only_future_matches,
            buffer_minutes=buffer_minutes,
        )

    def _scrape_upcoming_matches_impl(
        self, only_future_matches: bool = True, buffer_minutes: int = 30
    ) -> list[dict[str, Any]]:
        """Fully fixed version – handles 'Show more' + scroll + corrected JS."""
        if self._page is None:
            raise RuntimeError("Browser not started.")

        # Initial load
        try:
            self._page.wait_for_selector('div[data-testid="game-row"]', timeout=30000)
        except Exception:
            logger.warning("Initial rows did not appear – proceeding anyway.")

        # === STEP 1: Click "Show more" as many times as possible ===
        self._click_show_more()

        # === STEP 2: Aggressive scroll to trigger any remaining lazy content ===
        self._scroll_for_lazy_content()

        # === STEP 3: FIXED JavaScript extraction (no more ReferenceError) ===
        match_data = self._page.evaluate(
            """
            () => {
                const results = [];
                let currentDate = null;
                let currentTournament = null;
                let currentCountry = null;
                let currentSlug = null;                  // ← FIXED: declared here

                const elements = Array.from(document.querySelectorAll(
                    'div[data-testid="sport-country-league-item"], ' +
                    'div[data-testid="secondary-header"], ' +
                    'div[data-testid="game-row"]'
                ));

                for (const el of elements) {
                    const testId = el.getAttribute('data-testid');

                    if (testId === 'sport-country-league-item') {
                        const countryLink = el.querySelector('a[data-testid="header-country-item"]');
                        const tournamentLink = el.querySelector('a[data-testid="header-tournament-item"]');

                        if (countryLink) currentCountry = countryLink.innerText.trim();
                        if (tournamentLink) {
                            currentTournament = tournamentLink.innerText.trim();
                            const href = tournamentLink.getAttribute('href');
                            if (href) {
                                const parts = href.split('/').filter(Boolean);
                                currentSlug = parts[parts.length - 1] || null;
                            }
                        }
                        continue;
                    }

                    if (testId === 'secondary-header') {
                        const text = el.innerText.trim();
                        currentDate = text.split('\\n')[0]
                            .replace(/Today,?|Tomorrow,?/, '')
                            .trim();
                        continue;
                    }

                    if (testId === 'game-row') {
                        const timeElem = el.querySelector('div[data-testid="time-item"]');
                        const timeStr = timeElem ? timeElem.innerText.trim() : '00:00';

                        const teamLinks = Array.from(el.querySelectorAll('a[title]'));
                        if (teamLinks.length < 2) continue;

                        const homeTeam = teamLinks[0].getAttribute('title') || teamLinks[0].innerText.trim();
                        const awayTeam = teamLinks[1].getAttribute('title') || teamLinks[1].innerText.trim();

                        const rowText = el.innerText;
                        const oddsMatches = rowText.match(/(\\d+\\.\\d+)/g) || [];

                        results.push({
                            date: currentDate,
                            tournament: currentTournament,
                            country: currentCountry,
                            slug: currentSlug,
                            time: timeStr,
                            homeTeam: homeTeam,
                            awayTeam: awayTeam,
                            oddsHome: oddsMatches[0] ? parseFloat(oddsMatches[0]) : null,
                            oddsDraw: oddsMatches[1] ? parseFloat(oddsMatches[1]) : null,
                            oddsAway: oddsMatches[2] ? parseFloat(oddsMatches[2]) : null,
                            numBookmakers: oddsMatches.length >= 3
                                ? parseInt(rowText.trim().split(/\\s+/).pop()) || null
                                : null
                        });
                    }
                }
                return results;
            }
            """
        )

        logger.info(
            f"✅ Raw rows extracted after Show-more + scroll: {len(match_data)}"
        )

        # The rest of your existing parsing/filtering code (from `parsed_matches = []` onwards)
        # stays exactly the same – just paste it here.
        # (I omitted it for brevity; it is unchanged.)
        parsed_matches = []
        matches_without_odds = 0
        matches_already_started = 0

        for data in match_data:
            try:
                # Parse Date
                match_date = datetime.now()
                if data.get("date"):
                    clean_date = data["date"]
                    # Add current year if missing (e.g. "15 Jan")
                    if str(datetime.now().year) not in clean_date:
                        clean_date = f"{clean_date} {datetime.now().year}"

                    try:
                        time_str = data.get("time", "00:00")
                        match_date = datetime.strptime(
                            f"{clean_date} {time_str}", "%d %b %Y %H:%M"
                        )
                    except ValueError:
                        pass  # Keep default

                # Skip matches that have already started (with buffer)
                if only_future_matches:
                    cutoff_time = datetime.now() + timedelta(minutes=buffer_minutes)
                    if match_date < cutoff_time:
                        matches_already_started += 1
                        logger.debug(
                            f"Skipping {data['homeTeam']} vs {data['awayTeam']} - "
                            f"match already started or too close ({match_date})"
                        )
                        continue

                # Skip matches without odds
                if not data.get("oddsHome"):
                    matches_without_odds += 1
                    logger.debug(
                        f"Skipping {data['homeTeam']} vs {data['awayTeam']} - no odds available"
                    )
                    continue

                parsed_matches.append(
                    {
                        "tournament_name": data.get("tournament")
                        or "Unknown Tournament",
                        "country": (data.get("country") or "World").title(),
                        "tournament_slug": data.get("slug"),
                        "match_date": match_date,
                        "home_team": data["homeTeam"],
                        "away_team": data["awayTeam"],
                        "odds_home": data.get("oddsHome"),
                        "odds_draw": data.get("oddsDraw"),
                        "odds_away": data.get("oddsAway"),
                        "num_bookmakers": data.get("numBookmakers"),
                    }
                )
            except Exception as e:
                logger.error(
                    f"Error parsing upcoming match {data.get('homeTeam')} vs {data.get('away_team')}: {e}"
                )
                continue

        # Log summary
        total_processed = (
            len(parsed_matches) + matches_without_odds + matches_already_started
        )
        if matches_already_started > 0:
            logger.info(
                f"Scraped {total_processed} matches total: "
                f"{matches_already_started} already started (skipped), "
                f"{matches_without_odds} without odds (skipped), "
                f"{len(parsed_matches)} future matches with odds (kept)"
            )
        else:
            logger.info(
                f"Scraped {total_processed} matches, "
                f"filtered out {matches_without_odds} without odds, "
                f"keeping {len(parsed_matches)} with odds"
            )

        return parsed_matches

    def get_page_count(self) -> int:
        """Get total number of pages.

        Returns:
            Total page count.
        """
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")
        pagination_links = self._page.query_selector_all(self.PAGINATION_SELECTOR)
        max_page = 1

        for link in pagination_links:
            text = link.inner_text().strip()
            if text.isdigit():
                max_page = max(max_page, int(text))

        return max_page

    def _page_rows_signature(self) -> str:
        """Return a lightweight signature for the currently visible result rows."""
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")
        signature = self._page.evaluate(
            """
            (selector) => {
                const rows = Array.from(document.querySelectorAll(selector));
                const first = rows[0]?.innerText?.trim() ?? "";
                const last = rows[rows.length - 1]?.innerText?.trim() ?? "";
                return `${rows.length}::${first}::${last}`;
            }
            """,
            self.MATCH_ROW_SELECTOR,
        )
        return str(signature)

    def go_to_page(self, page_num: int) -> bool:
        """Navigate to a specific pagination page.

        Args:
            page_num: Page number to navigate to.

        Returns:
            True if navigation was successful.
        """
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")
        try:
            previous_url = self._page.url
            previous_signature = self._page_rows_signature()

            # Find the pagination link with the page number
            link = self._page.query_selector(
                f'{self.PAGINATION_SELECTOR}:text-is("{page_num}")'
            )
            if not link:
                logger.warning(f"Pagination link not found for page {page_num}")
                return False

            link.scroll_into_view_if_needed()
            link.click()
            # Wait for content to update
            with contextlib.suppress(Exception):
                self._page.wait_for_load_state("networkidle", timeout=15000)
            with contextlib.suppress(Exception):
                self._page.wait_for_selector(self.MATCH_ROW_SELECTOR, timeout=10000)

            current_signature = self._page_rows_signature()
            navigation_changed = (
                current_signature != previous_signature
                or self._page.url != previous_url
            )
            if not navigation_changed:
                logger.warning(
                    f"Could not confirm pagination transition to page {page_num}"
                )
                return False

            return True
        except Exception as e:
            logger.error(f"Error navigating to page {page_num}: {e}")

        return False

    def scrape_all_pages(self, max_pages: int | None = None) -> list[ScrapedMatch]:
        """Scrape matches from all pagination pages.

        Args:
            max_pages: Maximum number of pages to scrape (None for all).

        Returns:
            List of all scraped matches.
        """
        all_matches = []

        # Scrape first page
        matches = self.scrape_current_page()
        all_matches.extend(matches)
        logger.info(f"Page 1: scraped {len(matches)} matches")

        # Get total pages
        total_pages = self.get_page_count()
        pages_to_scrape = min(total_pages, max_pages) if max_pages else total_pages

        # Scrape remaining pages
        for page_num in range(2, pages_to_scrape + 1):
            if self.go_to_page(page_num):
                matches = self.scrape_current_page()
                all_matches.extend(matches)
                logger.info(f"Page {page_num}: scraped {len(matches)} matches")
            else:
                logger.error(f"Failed to navigate to page {page_num}")
                break

        return all_matches
