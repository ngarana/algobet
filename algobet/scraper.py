"""Playwright-based web scraper for OddsPortal football match data."""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from playwright.sync_api import sync_playwright, Page, Browser


@dataclass
class ScrapedMatch:
    """Data class for a scraped match."""

    match_date: datetime
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    odds_home: Optional[float] = None
    odds_draw: Optional[float] = None
    odds_away: Optional[float] = None
    num_bookmakers: Optional[int] = None


@dataclass
class SeasonInfo:
    """Information about a season."""

    name: str  # e.g., "2023/2024"
    url_suffix: Optional[str]  # e.g., "2023-2024" or None for current
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
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def start(self) -> None:
        """Start the browser."""
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self._page = self._browser.new_page()
        # Set generous timeout for slow connections
        self._page.set_default_timeout(120000)

    def close(self) -> None:
        """Close the browser."""
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()

    def navigate_to_results(self, url: str) -> None:
        """Navigate to a results page.

        Args:
            url: The URL of the results page.
        """
        # Use domcontentloaded instead of load for faster initial response
        self._page.goto(url, wait_until="domcontentloaded", timeout=120000)
        # Wait for match rows to load (may take time due to JS rendering)
        try:
            self._page.wait_for_selector(self.MATCH_ROW_SELECTOR, timeout=120000)
        except Exception:
            print(
                "Warning: Timeout waiting for game-row selector in navigate_to_results"
            )

    def get_available_seasons(self) -> list[SeasonInfo]:
        """Get list of available seasons from the current page.

        Returns:
            List of SeasonInfo objects.
        """
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
            match = re.search(r"premier-league-(\d{4}-\d{4})", href)
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
        pattern = r"/([^/]+)/results/"
        replacement = f"/\\1-{season.url_suffix}/results/"
        return re.sub(pattern, replacement, base_url)

    def scrape_current_page(self) -> list[ScrapedMatch]:
        """Scrape all matches from the current page.

        Returns:
            List of ScrapedMatch objects.
        """
        matches = []

        # Ensure content is loaded - wait for at least one game row
        try:
            self._page.wait_for_selector('div[data-testid="game-row"]', timeout=30000)
            # Also wait for odds to populate (they might load slightly later)
            self._page.wait_for_selector(
                'div[data-testid="odd-container-default"]', timeout=10000
            )
        except Exception:
            print(
                "Warning: Timeout waiting for game rows or odds, attempting to scrape anyway..."
            )

        # Use JavaScript to extract all match data with proper date association
        # This iterates through all children of ALL eventRow containers
        match_data = self._page.evaluate("""
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
        """)

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
                        try:
                            match_date = datetime.strptime(date_str, "%d %b %Y")
                        except ValueError:
                            pass

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
                print(f"Error parsing match: {e}")
                continue

        return matches

    def navigate_to_upcoming(
        self, url: str = "https://www.oddsportal.com/matches/football/"
    ) -> None:
        """Navigate to upcoming matches page.

        Args:
            url: The URL of the matches page.
        """
        # Use domcontentloaded instead of load for faster initial response
        self._page.goto(url, wait_until="domcontentloaded", timeout=120000)
        # Wait for match rows to load
        try:
            self._page.wait_for_selector(self.MATCH_ROW_SELECTOR, timeout=120000)
        except Exception:
            print(
                "Warning: Timeout waiting for game-row selector in navigate_to_upcoming"
            )

    def scrape_upcoming_matches(self) -> list[dict]:
        """Scrape upcoming matches from the current page.

        Returns:
            List of dictionaries with match data (including tournament info).
        """
        # Ensure content is loaded
        try:
            self._page.wait_for_selector('div[data-testid="game-row"]', timeout=30000)
            # Also wait for odds to populate (they might load slightly later)
            self._page.wait_for_selector(
                'div[data-testid="odd-container-default"]', timeout=10000
            )
        except Exception:
            print(
                "Warning: Timeout waiting for game rows or odds, attempting to scrape anyway..."
            )

        # Use JavaScript to extract match data with context (Date and Tournament)
        match_data = self._page.evaluate("""
            () => {
                const results = [];
                let currentDate = null;
                let currentTournament = null;
                let currentCountry = null;

                // Find all containers with events
                const containers = Array.from(document.querySelectorAll('.eventRow'));
                
                // Fallback if no eventRow class found
                if (containers.length === 0) {
                    const panel = document.querySelector('div[data-testid="results-panel"]');
                    if (panel) containers.push(panel);
                    // Also check for the main match list container on upcoming page
                    const mainList = document.querySelector('.flex.flex-col.w-full');
                    if (mainList && !containers.includes(mainList)) containers.push(mainList);
                }

                for (const container of containers) {
                    const children = Array.from(container.children);

                    for (const child of children) {
                        // 1. Check for Date Header (e.g. "Today, 15 Jan")
                        // Usually in a div with text-sm or similar class
                        if (child.className && child.className.includes('text-black-main') && child.className.includes('font-bold')) {
                             // Simple heuristic for date header
                             if (/\\d{1,2} [A-Za-z]+/.test(child.innerText)) {
                                 currentDate = child.innerText.trim();
                                 continue;
                             }
                        }
                        // Also check for the specific structure found in results (secondary-header)
                        const dateHeader = child.querySelector('div[data-testid="date-header"]');
                        if (dateHeader) {
                            currentDate = dateHeader.innerText.trim();
                            continue;
                        }

                        // 2. Check for Tournament Header
                        // Structure: Football / Country / Tournament
                        const link = child.querySelector('a[href*="/football/"]');
                        if (link && child.className.includes('bg-gray-light')) { 
                            // This is likely a tournament header row
                            const text = link.innerText.trim();
                            // Try to parse Country / Tournament
                            // Usually breadcrumbs structure isn't directly in text here, 
                            // but often the link title or text helps.
                            // Let's assume the text is the Tournament name
                            currentTournament = text;
                            
                            // Try to find country (often in sibling or part of URL)
                            const href = link.getAttribute('href');
                            if (href) {
                                const parts = href.split('/').filter(p => p);
                                if (parts.length >= 3) {
                                    currentCountry = parts[2].replace(/-/g, ' '); 
                                }
                            }
                            continue;
                        }
                        
                        // 3. Check for Match Row
                        const gameRow = child.querySelector('div[data-testid="game-row"]') || 
                                        (child.getAttribute('data-testid') === 'game-row' ? child : null);
                        
                        if (!gameRow) continue;

                        // Extract time
                        const timeElem = gameRow.querySelector('div[data-testid="time-item"]');
                        const timeStr = timeElem ? timeElem.innerText.trim() : '00:00';

                        // Extract teams
                        const teamLinks = Array.from(gameRow.querySelectorAll('a[title]'));
                        if (teamLinks.length < 2) continue;

                        const homeTeam = teamLinks[0].getAttribute('title') || teamLinks[0].innerText.trim();
                        const awayTeam = teamLinks[1].getAttribute('title') || teamLinks[1].innerText.trim();

                        // Extract odds
                        const rowText = gameRow.innerText;
                        const oddsMatches = rowText.match(/(\\d+\\.\\d+)/g) || [];

                        results.push({
                            date: currentDate,
                            tournament: currentTournament,
                            country: currentCountry,
                            time: timeStr,
                            homeTeam: homeTeam,
                            awayTeam: awayTeam,
                            oddsHome: oddsMatches[0] ? parseFloat(oddsMatches[0]) : null,
                            oddsDraw: oddsMatches[1] ? parseFloat(oddsMatches[1]) : null,
                            oddsAway: oddsMatches[2] ? parseFloat(oddsMatches[2]) : null,
                            numBookmakers: oddsMatches.length >= 3 ? parseInt(rowText.trim().split(/\\s+/).pop()) || null : null
                        });
                        if (oddsMatches.length === 0) {
                             // console.log("No odds found for", homeTeam); // Can't easily pipe console.log to python without setup
                        }
                    }
                }
                return results;
            }
        """)

        parsed_matches = []
        for data in match_data:
            # DEBUG: Print found odds
            if not data.get("oddsHome"):
                print(f"DEBUG: No odds for {data['homeTeam']} vs {data['awayTeam']}")
            else:
                print(f"DEBUG: Found odds for {data['homeTeam']}: {data['oddsHome']}")

            try:
                # Parse Date
                match_date = datetime.now()
                if data.get("date"):
                    clean_date = (
                        data["date"].split(",")[1].strip()
                        if "," in data["date"]
                        else data["date"]
                    )
                    # Add current year if missing (e.g. "15 Jan")
                    if str(datetime.now().year) not in clean_date:
                        clean_date = f"{clean_date} {datetime.now().year}"

                    try:
                        time_str = data.get("time", "00:00")
                        match_date = datetime.strptime(
                            f"{clean_date} {time_str}", "%d %b %Y %H:%M"
                        )
                    except ValueError:
                        pass  # Keep default or try other formats if needed

                parsed_matches.append(
                    {
                        "tournament_name": data.get("tournament")
                        or "Unknown Tournament",
                        "country_name": (data.get("country") or "World").title(),
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
                print(f"Error parsing upcoming match: {e}")
                continue

        return parsed_matches

    def get_page_count(self) -> int:
        """Get total number of pages.

        Returns:
            Total page count.
        """
        pagination_links = self._page.query_selector_all(self.PAGINATION_SELECTOR)
        max_page = 1

        for link in pagination_links:
            text = link.inner_text().strip()
            if text.isdigit():
                max_page = max(max_page, int(text))

        return max_page

    def go_to_page(self, page_num: int) -> bool:
        """Navigate to a specific pagination page.

        Args:
            page_num: Page number to navigate to.

        Returns:
            True if navigation was successful.
        """
        try:
            # Find the pagination link with the page number
            link = self._page.query_selector(
                f'{self.PAGINATION_SELECTOR}:text-is("{page_num}")'
            )
            if link:
                link.click()
                # Wait for content to update
                self._page.wait_for_load_state("networkidle")
                try:
                    self._page.wait_for_selector(self.MATCH_ROW_SELECTOR, timeout=10000)
                except Exception:
                    pass
                return True
        except Exception as e:
            print(f"Error navigating to page {page_num}: {e}")

        return False

    def scrape_all_pages(self, max_pages: Optional[int] = None) -> list[ScrapedMatch]:
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
        print(f"Page 1: scraped {len(matches)} matches")

        # Get total pages
        total_pages = self.get_page_count()
        pages_to_scrape = min(total_pages, max_pages) if max_pages else total_pages

        # Scrape remaining pages
        for page_num in range(2, pages_to_scrape + 1):
            if self.go_to_page(page_num):
                matches = self.scrape_current_page()
                all_matches.extend(matches)
                print(f"Page {page_num}: scraped {len(matches)} matches")
            else:
                print(f"Failed to navigate to page {page_num}")
                break

        return all_matches
