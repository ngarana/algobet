# ruff: noqa: E501
"""Playwright-based web scraper for FBref football player statistics.

Handles Cloudflare protection via cookie persistence and manual
CAPTCHA resolution.  Scrapes per-player-per-match stats from
individual FBref match pages and season-level aggregate stats
from squad/team summary pages.

Usage:
    with FBrefScraper(headless=False) as scraper:
        scraper.navigate_to("https://fbref.com/en/comps/9/2020-2021/...")
        matches = scraper.scrape_season_schedule()
        for url in match_urls:
            stats = scraper.scrape_match_player_stats(url)
"""

from __future__ import annotations

import contextlib
import functools
import io
import json
import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar, cast

import pandas as pd
from lxml import etree, html
from playwright.sync_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

try:
    from playwright_stealth import Stealth

    _STEALTH_AVAILABLE = True
except ImportError:
    _STEALTH_AVAILABLE = False

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

COOKIE_FILE = Path(
    os.environ.get("FBREF_COOKIE_FILE", str(Path.home() / ".fbref_cookies.json"))
)

STORAGE_STATE_FILE = Path(
    os.environ.get(
        "FBREF_STORAGE_STATE",
        str(Path.home() / ".fbref_storage_state.json"),
    )
)

FBREF_BASE = "https://fbref.com"

COMP_IDS: dict[str, dict[str, str]] = {
    "ENG-Premier League": {
        "comp_id": "9",
        "slug": "Premier-League",
    },
    "ESP-La Liga": {
        "comp_id": "12",
        "slug": "La-Liga",
    },
    "FRA-Ligue 1": {
        "comp_id": "13",
        "slug": "Ligue-1",
    },
    "GER-Bundesliga": {
        "comp_id": "20",
        "slug": "Bundesliga",
    },
    "ITA-Serie A": {
        "comp_id": "11",
        "slug": "Serie-A",
    },
}

STAT_PAGE_BY_TYPE = {
    "standard": "stats",
    "keeper": "keepers",
    "keepers": "keepers",
    "shooting": "shooting",
    "playing_time": "playingtime",
    "misc": "misc",
}

PLAYER_MATCH_STAT_TYPES = {"summary", "keepers"}
TEAM_MATCH_STAT_TYPES = {"schedule", "shooting", "keeper", "misc"}

NETWORK_ERROR_MARKERS = (
    "err_name_not_resolved",
    "err_internet_disconnected",
    "err_connection_refused",
    "err_connection_reset",
    "err_connection_timed_out",
    "err_network_changed",
    "net::",
)


def is_retryable_network_error(error: Exception) -> bool:
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
                    if not is_retryable_network_error(e):
                        raise
                    if attempt < max_retries:
                        logger.warning(
                            "Network error in %s: %s. "
                            "Retrying in %.1fs (attempt %d/%d)",
                            func.__name__,
                            e,
                            current_delay,
                            attempt + 1,
                            max_retries,
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            "Max retries (%d) exceeded for %s",
                            max_retries,
                            func.__name__,
                        )
                        raise
            if last_exception:
                raise last_exception
            return None

        return cast(F, wrapper)

    return decorator


def _as_list(value: Any | None) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list | tuple | set):
        return list(value)
    return [value]


def _season_to_url_fragment(season: str | int) -> str:
    season_str = str(season).strip().replace("/", "-")
    if re.fullmatch(r"\d{4}-\d{4}", season_str):
        return season_str
    if re.fullmatch(r"\d{4}-\d{2}", season_str):
        start = int(season_str[:4])
        return f"{start}-{start + 1}"
    if re.fullmatch(r"\d{2}-\d{2}", season_str):
        start_two = int(season_str[:2])
        start = 2000 + start_two if start_two < 70 else 1900 + start_two
        return f"{start}-{start + 1}"
    if re.fullmatch(r"\d{4}", season_str):
        start = int(season_str)
        if 1900 <= start <= 2100:
            return f"{start}-{start + 1}"
        if len(season_str) == 4:
            start = 2000 + int(season_str[:2])
            return f"{start}-{start + 1}"
    raise ValueError(f"Unsupported FBref season format: {season}")


def _default_seasons(seasons: list[Any]) -> list[str]:
    if seasons:
        return [_season_to_url_fragment(s) for s in seasons]
    today = datetime.now()
    start_year = today.year if today.month >= 7 else today.year - 1
    return [f"{start_year}-{start_year + 1}"]


def _to_snake(name: str) -> str:
    normalized = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", str(name))
    normalized = re.sub(r"__([A-Z])", r"_\1", normalized)
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", normalized)
    return (
        normalized.lower()
        .replace("-", "_")
        .replace(" ", "")
        .replace("#", "jersey_number")
    )


def _standardize_colnames(
    df: pd.DataFrame,
    cols: list[str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df
    if df.columns.nlevels > 1 and cols is None:
        new_df = df.copy()
        new_cols = [_to_snake(c) for c in df.columns.levels[0]]
        new_df.columns = new_df.columns.set_levels(new_cols, level=0)
        return new_df
    if cols is None:
        cols = [str(c) for c in df.columns]
    return df.rename(columns={c: _to_snake(c) for c in cols})


def _make_game_id(row: pd.Series) -> str:
    date_val = row.get("date")
    if pd.isna(date_val):
        return f"{row.get('home_team')}-{row.get('away_team')}"
    return (
        f"{pd.to_datetime(date_val).strftime('%Y-%m-%d')} "
        f"{row.get('home_team')}-{row.get('away_team')}"
    )


def _parse_fbref_table(table_el: html.HtmlElement) -> pd.DataFrame:
    for elem in table_el.xpath(".//span[contains(@class, 'f-i')]"):
        parent = elem.getparent()
        if parent is not None:
            etree.strip_elements(parent, "span", with_tail=False)
    for elem in table_el.xpath(".//tbody/tr[contains(@class, 'spacer')]"):
        elem.getparent().remove(elem)
    for elem in table_el.xpath(".//tbody/tr[contains(@class, 'thead')]"):
        elem.getparent().remove(elem)
    (df_table,) = pd.read_html(
        io.StringIO(html.tostring(table_el, encoding="unicode")),
        flavor="lxml",
    )
    return df_table.convert_dtypes()


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or df.columns.nlevels == 1:
        return df.copy()
    flattened = df.copy()
    columns = []
    for col in flattened.columns:
        parts = [str(part) for part in col if not str(part).startswith("Unnamed:")]
        columns.append(_to_snake(parts[-1] if parts else str(col[-1])))
    flattened.columns = columns
    return flattened


@dataclass
class ScrapedScheduleEntry:
    match_date: datetime
    home_team: str
    away_team: str
    home_score: int | None
    away_score: int | None
    match_url: str | None = None


@dataclass
class ScrapedPlayerStat:
    player_name: str
    team_name: str
    is_home: bool
    position: str | None = None
    is_starter: bool | None = None
    minutes_played: int | None = None
    goals: int | None = None
    assists: int | None = None
    shots: int | None = None
    shots_on_target: int | None = None
    fouls_committed: int | None = None
    fouls_suffered: int | None = None
    yellow_cards: int | None = None
    red_cards: int | None = None
    saves: int | None = None
    goals_conceded: int | None = None
    offsides: int | None = None


@dataclass
class ScrapedSeasonPlayerStat:
    player_name: str
    team_name: str
    position: str | None = None
    matches_played: int | None = None
    starts: int | None = None
    minutes: int | None = None
    goals: int | None = None
    assists: int | None = None
    shots: int | None = None
    shots_on_target: int | None = None
    fouls_committed: int | None = None
    fouls_suffered: int | None = None
    yellow_cards: int | None = None
    red_cards: int | None = None
    saves: int | None = None
    goals_conceded: int | None = None
    offsides: int | None = None


@dataclass
class ScrapedMatchStats:
    home_team: str
    away_team: str
    match_date: datetime | None = None
    home_players: list[ScrapedPlayerStat] = field(default_factory=list)
    away_players: list[ScrapedPlayerStat] = field(default_factory=list)


class FBrefScraper:
    """Playwright-based scraper for FBref football data.

    Handles Cloudflare protection through cookie persistence and
    optional manual CAPTCHA resolution.

    Usage:
        with FBrefScraper(headless=False) as scraper:
            scraper.navigate_to(url)
            stats = scraper.scrape_season_player_stats()
    """

    CLOUDFLARE_TITLE = "Just a moment"

    def __init__(
        self,
        leagues: str | list[str] | bool | None = None,
        seasons: str | int | list[str | int] | None = None,
        *,
        headless: bool = False,
        cookie_file: str | Path | None = None,
        storage_state_file: str | Path | None = None,
        timeout: int = 120000,
        no_cache: bool = False,
        no_store: bool = False,
        data_dir: str | Path | None = None,
    ) -> None:
        if isinstance(leagues, bool):
            headless = leagues
            leagues = None
        self.headless = headless
        self.cookie_file = Path(cookie_file) if cookie_file else COOKIE_FILE
        self.storage_state_file = (
            Path(storage_state_file) if storage_state_file else STORAGE_STATE_FILE
        )
        self.timeout = timeout
        self.leagues = _as_list(leagues) or ["ENG-Premier League"]
        self.seasons = _as_list(seasons) or []
        self.no_cache = no_cache
        self.no_store = no_store
        self.data_dir = Path(data_dir) if data_dir else None
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    def __enter__(self) -> FBrefScraper:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def start(self) -> None:
        self._playwright = sync_playwright().start()

        # Use persistent storage state if available (solves CAPTCHA once)
        context_kwargs = {
            "user_agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
            ),
            "viewport": {"width": 1920, "height": 1080},
            "locale": "en-US",
            "java_script_enabled": True,
        }
        if self.storage_state_file.exists():
            logger.info("Loading storage state from %s", self.storage_state_file)
            context_kwargs["storage_state"] = str(self.storage_state_file)

        self._browser = self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
                "--disable-features=IsolateOrigins,site-per-process",
                "--disable-infobars",
                "--window-size=1920,1080",
            ],
        )
        self._context = self._browser.new_context(**context_kwargs)
        self._page = self._context.new_page()

        # Apply playwright-stealth patches
        if _STEALTH_AVAILABLE:
            stealth = Stealth(
                navigator_webdriver=True,
                navigator_languages=True,
                navigator_plugins=True,
                navigator_permissions=True,
                navigator_platform=True,
                chrome_app=True,
                chrome_csi=True,
                chrome_load_times=True,
                chrome_runtime=True,
                webgl_vendor=True,
                hairline=True,
                iframe_content_window=True,
                navigator_hardware_concurrency=True,
                navigator_user_agent=True,
                navigator_user_agent_data=True,
                navigator_vendor=True,
                error_prototype=True,
                sec_ch_ua=True,
                media_codecs=True,
            )
            stealth.use_sync(self._browser)
            logger.info("Applied playwright-stealth patches")
        else:
            logger.warning(
                "playwright-stealth not installed; Cloudflare detection "
                "is likely. Install with: pip install playwright-stealth"
            )

        # Additional anti-detection
        self._page.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US','en']});
            Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
            window.chrome = {runtime: {}};
            """
        )
        self._page.set_default_timeout(self.timeout)

    def close(self) -> None:
        self._save_storage_state()
        self._save_cookies()
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()

    def _save_storage_state(self) -> None:
        if self._context:
            try:
                state = self._context.storage_state()
                with open(self.storage_state_file, "w") as f:
                    json.dump(state, f)
                logger.info("Saved storage state to %s", self.storage_state_file)
            except Exception as e:
                logger.warning("Failed to save storage state: %s", e)

    def _load_cookies(self) -> None:
        if self._context and self.cookie_file.exists():
            try:
                with open(self.cookie_file) as f:
                    cookies = json.load(f)
                self._context.add_cookies(cookies)
                logger.info("Loaded %d cookies from %s", len(cookies), self.cookie_file)
            except Exception as e:
                logger.warning("Failed to load cookies: %s", e)

    def _save_cookies(self) -> None:
        if self._context:
            try:
                cookies = self._context.cookies()
                with open(self.cookie_file, "w") as f:
                    json.dump(cookies, f)
                logger.info("Saved %d cookies to %s", len(cookies), self.cookie_file)
            except Exception as e:
                logger.warning("Failed to save cookies: %s", e)

    @retry_on_network_error(max_retries=3, delay=5.0, backoff=2.0)
    def navigate_to(self, url: str, wait_for_cloudflare: bool = True) -> None:
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")

        logger.info("Navigating to %s", url)
        self._page.goto(url, wait_until="domcontentloaded", timeout=self.timeout)

        if wait_for_cloudflare:
            self._wait_for_cloudflare()

        self._page.wait_for_timeout(2000)
        self._dismiss_overlays()

    def _wait_for_cloudflare(self, max_wait: int = 180) -> None:
        if self._page is None:
            raise RuntimeError("Browser not started.")

        logger.info("Waiting for Cloudflare challenge to resolve...")
        if not self.headless:
            print()
            print("=" * 60)
            print("Cloudflare challenge detected.")
            print("If a CAPTCHA appears in the browser window, solve it manually.")
            print("The scraper will continue once the page loads.")
            print("=" * 60)
            print()
        start = time.time()
        while time.time() - start < max_wait:
            title = self._page.title()
            if self.CLOUDFLARE_TITLE not in title:
                logger.info("Cloudflare challenge resolved. Title: %s", title)
                # Save storage state immediately after passing Cloudflare
                self._save_storage_state()
                return
            self._page.wait_for_timeout(2000)

        raise TimeoutError(
            f"Cloudflare challenge not resolved after {max_wait}s. "
            "If running headless, try running with headless=False "
            "to solve the CAPTCHA manually."
        )

    def _dismiss_overlays(self) -> None:
        if self._page is None:
            return
        dismiss_targets = [
            'button:has-text("Accept")',
            'button:has-text("Reject All")',
            'button:has-text("I Accept")',
            'button:has-text("Close")',
            'button[aria-label="Close"]',
        ]
        for selector in dismiss_targets:
            try:
                locator = self._page.locator(selector).first
                if locator.count() == 0 or not locator.is_visible(timeout=1000):
                    continue
                locator.click(timeout=3000)
                self._page.wait_for_timeout(500)
            except Exception:
                continue

    @staticmethod
    def build_season_url(league_code: str, season: str) -> str:
        comp = COMP_IDS.get(league_code)
        if not comp:
            raise ValueError(f"Unknown league code: {league_code}")
        return (
            f"{FBREF_BASE}/en/comps/{comp['comp_id']}/{season}"
            f"/stats/{season}-{comp['slug']}-Stats"
        )

    @staticmethod
    def build_schedule_url(league_code: str, season: str) -> str:
        comp = COMP_IDS.get(league_code)
        if not comp:
            raise ValueError(f"Unknown league code: {league_code}")
        return (
            f"{FBREF_BASE}/en/comps/{comp['comp_id']}/{season}"
            f"/schedule/{season}-{comp['slug']}-Scores"
        )

    def _iter_league_seasons(self) -> list[tuple[str, str]]:
        seasons = _default_seasons(self.seasons)
        return [(str(league), season) for league in self.leagues for season in seasons]

    def _get_current_tree(self) -> html.HtmlElement:
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")
        content = self._page.content()
        return html.fromstring(content)

    @staticmethod
    def _find_table(
        tree: html.HtmlElement,
        table_id: str | None = None,
        id_contains: str | None = None,
    ) -> html.HtmlElement | None:
        if table_id:
            matches = tree.xpath(f"//table[@id='{table_id}']")
            if matches:
                return matches[0]
            for comment in tree.xpath("//comment()"):
                if table_id not in str(comment.text):
                    continue
                with contextlib.suppress(Exception):
                    parsed = html.fromstring(comment.text)
                    matches = parsed.xpath(f"//table[@id='{table_id}']")
                    if matches:
                        return matches[0]
        if id_contains:
            matches = tree.xpath(f"//table[contains(@id, '{id_contains}')]")
            if matches:
                return matches[0]
            for comment in tree.xpath("//comment()"):
                if id_contains not in str(comment.text):
                    continue
                with contextlib.suppress(Exception):
                    parsed = html.fromstring(comment.text)
                    matches = parsed.xpath(f"//table[contains(@id, '{id_contains}')]")
                    if matches:
                        return matches[0]
        return None

    @staticmethod
    def _parse_teams(tree: html.HtmlElement) -> list[dict[str, str]]:
        teams: list[dict[str, str]] = []
        for team in tree.xpath("//div[contains(@class, 'scorebox')]//strong/a")[:2]:
            href = team.get("href") or ""
            href_parts = href.split("/")
            team_id = href_parts[3] if len(href_parts) > 3 else ""
            teams.append({"id": team_id, "name": team.text_content().strip()})
        return teams

    @staticmethod
    def _extract_match_id_from_url(url: str | None) -> str | None:
        if not url:
            return None
        match = re.search(r"/matches/([^/]+)", url)
        return match.group(1) if match else None

    def _read_table_page(
        self, url: str, table_id: str | None = None
    ) -> html.HtmlElement:
        self.navigate_to(url)
        tree = self._get_current_tree()
        if table_id and self._find_table(tree, table_id=table_id) is None:
            raise RuntimeError(f"Expected FBref table not found: {table_id}")
        return tree

    def read_schedule(self, force_cache: bool = False) -> pd.DataFrame:
        """Return a soccerdata-like FBref schedule dataframe.

        ``force_cache`` is accepted for API compatibility.  This Playwright
        scraper always reads the live page because cookie/state handling happens
        in the browser context rather than the soccerdata cache.
        """
        del force_cache
        frames: list[pd.DataFrame] = []

        for league, season in self._iter_league_seasons():
            schedule_url = self.build_schedule_url(league, season)
            self.navigate_to(schedule_url)
            rows = self._page.evaluate(  # type: ignore[union-attr]
                """
                () => {
                    let table = document.querySelector('table[id^="sched"]');
                    if (!table) {
                        table = Array.from(document.querySelectorAll('table')).find(t => {
                            const headers = Array.from(t.querySelectorAll('th'))
                                .map(h => h.innerText.trim());
                            return headers.includes('Date') && headers.includes('Home');
                        });
                    }
                    if (!table) return [];
                    const body = table.querySelector('tbody');
                    if (!body) return [];
                    const rows = [];
                    for (const row of body.querySelectorAll('tr')) {
                        if (row.classList.contains('thead')) continue;
                        const rowData = {};
                        for (const cell of row.querySelectorAll('th, td')) {
                            const stat = cell.getAttribute('data-stat') || '';
                            if (!stat) continue;
                            rowData[stat] = cell.innerText.trim();
                        }
                        const reportLink = row.querySelector('td[data-stat="match_report"] a');
                        if (reportLink) rowData.match_report = reportLink.getAttribute('href');
                        if (rowData.date && rowData.home_team && rowData.away_team) {
                            rows.push(rowData);
                        }
                    }
                    return rows;
                }
                """
            )
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["league"] = league
            df["season"] = season
            df = df.rename(
                columns={
                    "home_team": "home_team",
                    "away_team": "away_team",
                    "xg_home": "home_xg",
                    "xg_away": "away_xg",
                }
            )
            df = _standardize_colnames(df)
            df["date"] = pd.to_datetime(df["date"], errors="coerce").ffill()
            if "score" in df.columns:
                df["score"] = df["score"].replace("", pd.NA)
            if "match_report" in df.columns:
                df.loc[df["match_report"].notna(), "game_id"] = (
                    df.loc[df["match_report"].notna(), "match_report"]
                    .astype(str)
                    .str.split("/")
                    .str[3]
                )
            else:
                df["match_report"] = pd.NA
                df["game_id"] = pd.NA
            df["game"] = df.apply(_make_game_id, axis=1)
            frames.append(df)

        if not frames:
            return pd.DataFrame().set_index(["league", "season", "game"])

        return (
            _concat_frames(frames).set_index(["league", "season", "game"]).sort_index()
        )

    def read_player_match_stats(
        self,
        stat_type: str = "summary",
        match_id: str | list[str] | None = None,
        force_cache: bool = False,
    ) -> pd.DataFrame:
        """Return FBref player match stats with soccerdata-compatible shape."""
        del force_cache
        if stat_type not in PLAYER_MATCH_STAT_TYPES:
            raise TypeError(
                f"Invalid argument: stat_type should be in {sorted(PLAYER_MATCH_STAT_TYPES)}"
            )

        schedule = self.read_schedule().reset_index()
        if "game_id" not in schedule.columns:
            return pd.DataFrame()
        schedule = schedule[
            schedule["game_id"].notna() & schedule["match_report"].notna()
        ]
        if match_id is not None:
            requested = [match_id] if isinstance(match_id, str) else match_id
            schedule = schedule[schedule["game_id"].isin(requested)]
            if schedule.empty:
                raise ValueError(
                    "No games found with the given IDs in the selected seasons."
                )

        frames: list[pd.DataFrame] = []
        for _, game in schedule.iterrows():
            tree = self._read_table_page(f"{FBREF_BASE}/en/matches/{game['game_id']}")
            teams = self._parse_teams(tree)
            id_format = (
                "keeper_stats_{}"
                if stat_type == "keepers"
                else f"stats_{{}}_{stat_type}"
            )
            for team in teams:
                table = self._find_table(tree, table_id=id_format.format(team["id"]))
                if table is None:
                    logger.warning(
                        "No %s stats found for %s in game_id=%s",
                        stat_type,
                        team["name"],
                        game["game_id"],
                    )
                    continue
                df_table = _parse_fbref_table(table)
                df_table["team"] = team["name"]
                df_table["game"] = game["game"]
                df_table["league"] = game["league"]
                df_table["season"] = game["season"]
                df_table["game_id"] = game["game_id"]
                frames.append(df_table)

        if not frames:
            return pd.DataFrame()

        df = _concat_frames(frames)
        flat = _flatten_columns(df)
        if "player" not in flat.columns:
            flat = _standardize_colnames(flat)
        if "player" in flat.columns:
            flat = flat[~flat["player"].astype(str).str.contains(r"^\d+\sPlayers$")]
        flat = flat.rename(columns={"jersey_number": "jersey_number"})
        return flat.set_index(
            ["league", "season", "game", "team", "player"]
        ).sort_index()

    def read_lineup(
        self,
        match_id: str | list[str] | None = None,
        force_cache: bool = False,
    ) -> pd.DataFrame:
        del force_cache
        schedule = self.read_schedule().reset_index()
        schedule = schedule[
            schedule["game_id"].notna() & schedule["match_report"].notna()
        ]
        if match_id is not None:
            requested = [match_id] if isinstance(match_id, str) else match_id
            schedule = schedule[schedule["game_id"].isin(requested)]
            if schedule.empty:
                raise ValueError(
                    "No games found with the given IDs in the selected seasons."
                )

        lineups: list[pd.DataFrame] = []
        for _, game in schedule.iterrows():
            tree = self._read_table_page(f"{FBREF_BASE}/en/matches/{game['game_id']}")
            teams = self._parse_teams(tree)
            lineup_tables = tree.xpath("//div[contains(@class, 'lineup')]")
            for idx, table in enumerate(lineup_tables[: len(teams)]):
                df_table = _parse_fbref_table(table)
                if len(df_table.columns) < 2:
                    continue
                df_table = df_table.iloc[:, :2]
                df_table.columns = ["jersey_number", "player"]
                bench_rows = df_table.index[
                    df_table["jersey_number"].astype(str) == "Bench"
                ]
                if len(bench_rows) > 0:
                    bench_idx = bench_rows[0]
                    df_table.loc[:bench_idx, "is_starter"] = True
                    df_table.loc[bench_idx:, "is_starter"] = False
                    df_table = df_table.drop(index=bench_idx)
                else:
                    df_table["is_starter"] = pd.NA

                if idx < len(teams):
                    team = teams[idx]
                    stats_table = self._find_table(
                        tree,
                        table_id=f"stats_{team['id']}_summary",
                    )
                    if stats_table is not None:
                        df_stats = _flatten_columns(_parse_fbref_table(stats_table))
                        df_stats = _standardize_colnames(df_stats)
                        wanted = [
                            c
                            for c in ["player", "jersey_number", "pos", "min"]
                            if c in df_stats.columns
                        ]
                        if {"player", "jersey_number"}.issubset(wanted):
                            df_stats = df_stats[wanted].rename(
                                columns={"pos": "position", "min": "minutes_played"}
                            )
                            df_stats["jersey_number"] = df_stats[
                                "jersey_number"
                            ].astype(str)
                            df_table["jersey_number"] = df_table[
                                "jersey_number"
                            ].astype(str)
                            df_table = pd.merge(
                                df_table,
                                df_stats,
                                on=["player", "jersey_number"],
                                how="left",
                            )
                    df_table["team"] = team["name"]
                df_table["game"] = game["game"]
                df_table["league"] = game["league"]
                df_table["season"] = game["season"]
                lineups.append(df_table)

        if not lineups:
            return pd.DataFrame()
        return (
            _concat_frames(lineups).set_index(["league", "season", "game"]).sort_index()
        )

    def read_player_season_stats(self, stat_type: str = "standard") -> pd.DataFrame:
        if stat_type not in STAT_PAGE_BY_TYPE:
            raise TypeError(
                f"Invalid argument: stat_type should be in {sorted(STAT_PAGE_BY_TYPE)}"
            )
        page = STAT_PAGE_BY_TYPE[stat_type]
        stat_id = "keeper" if stat_type == "keepers" else stat_type
        frames: list[pd.DataFrame] = []

        for league, season in self._iter_league_seasons():
            comp = COMP_IDS.get(league)
            if not comp:
                raise ValueError(f"Unknown league code: {league}")
            url = (
                f"{FBREF_BASE}/en/comps/{comp['comp_id']}/{season}/{page}/"
                f"{season}-{comp['slug']}-Stats"
            )
            self.navigate_to(url)
            tree = self._get_current_tree()
            table = self._find_table(tree, id_contains=f"stats_{stat_id}")
            if table is None:
                continue
            df_table = _parse_fbref_table(table)
            df_table["league"] = league
            df_table["season"] = season
            frames.append(df_table)

        if not frames:
            return pd.DataFrame()
        df = _concat_frames(frames)
        flat = _flatten_columns(df)
        flat = flat[
            flat.get("Player", flat.get("player", pd.Series(dtype=str))) != "Player"
        ]
        flat = flat.drop(
            columns=[c for c in ["Rk", "Matches", "rk", "matches"] if c in flat.columns]
        )
        flat = flat.rename(columns={"Squad": "team", "squad": "team"})
        flat = _standardize_colnames(flat)
        return flat.set_index(["league", "season", "team", "player"]).sort_index()

    def read_team_season_stats(
        self,
        stat_type: str = "standard",
        opponent_stats: bool = False,
    ) -> pd.DataFrame:
        if stat_type not in STAT_PAGE_BY_TYPE:
            raise ValueError(
                f"Invalid argument: stat_type should be in {sorted(STAT_PAGE_BY_TYPE)}"
            )
        page = STAT_PAGE_BY_TYPE[stat_type]
        stat_id = "keeper" if stat_type == "keepers" else stat_type
        suffix = "against" if opponent_stats else "for"
        frames: list[pd.DataFrame] = []

        for league, season in self._iter_league_seasons():
            comp = COMP_IDS.get(league)
            if not comp:
                raise ValueError(f"Unknown league code: {league}")
            url = (
                f"{FBREF_BASE}/en/comps/{comp['comp_id']}/{season}/{page}/"
                f"{season}-{comp['slug']}-Stats"
            )
            self.navigate_to(url)
            tree = self._get_current_tree()
            table = self._find_table(
                tree, table_id=f"stats_squads_{stat_id}_{suffix}"
            ) or self._find_table(tree, table_id=f"stats_teams_{stat_id}_{suffix}")
            if table is None:
                continue
            df_table = _parse_fbref_table(table)
            df_table["url"] = table.xpath(".//*[@data-stat='team']/a/@href")
            df_table["league"] = league
            df_table["season"] = season
            frames.append(df_table)

        if not frames:
            return pd.DataFrame()
        flat = _flatten_columns(_concat_frames(frames))
        flat = flat.rename(
            columns={
                "Squad": "team",
                "squad": "team",
                "# Pl": "players_used",
                "number_pl": "players_used",
            }
        )
        flat = _standardize_colnames(flat)
        return flat.set_index(["league", "season", "team"]).sort_index()

    def read_team_match_stats(
        self,
        stat_type: str = "schedule",
        opponent_stats: bool = False,
        team: str | list[str] | None = None,
        force_cache: bool = False,
    ) -> pd.DataFrame:
        del force_cache
        if stat_type not in TEAM_MATCH_STAT_TYPES:
            raise ValueError(
                f"Invalid argument: stat_type should be in {sorted(TEAM_MATCH_STAT_TYPES)}"
            )
        if stat_type == "schedule" and opponent_stats:
            raise ValueError(
                "Opponent stats are not available for the 'schedule' stat type"
            )

        team_stats = self.read_team_season_stats("standard").reset_index()
        if team is not None:
            teams = [team] if isinstance(team, str) else team
            team_stats = team_stats[team_stats["team"].isin(teams)]
            if team_stats.empty:
                raise ValueError(
                    "No data found for the given teams in the selected seasons."
                )

        frames: list[pd.DataFrame] = []
        opp_type = "against" if opponent_stats else "for"
        for _, row in team_stats.iterrows():
            team_url = str(row.get("url", ""))
            if not team_url:
                continue
            season = str(row["season"])
            if len(team_url.split("/")) == 6:
                url = (
                    f"{FBREF_BASE}{team_url.rsplit('/', 1)[0]}"
                    f"/matchlogs/all_comps/{stat_type}"
                )
            else:
                url = (
                    f"{FBREF_BASE}{team_url.rsplit('/', 1)[0]}"
                    f"/{season}/matchlogs/all_comps/{stat_type}"
                )
            self.navigate_to(url)
            tree = self._get_current_tree()
            table = self._find_table(tree, table_id=f"matchlogs_{opp_type}")
            if table is None:
                continue
            for elem in table.xpath(".//th[@data-stat='header_for_against']"):
                elem.text = ""
            for elem in table.xpath(".//tfoot"):
                elem.getparent().remove(elem)
            df_table = _parse_fbref_table(table)
            df_table["season"] = row["season"]
            df_table["team"] = row["team"]
            df_table["league"] = row["league"]
            match_report_links = []
            for link_cell in table.xpath(".//td[@data-stat='match_report']"):
                link = link_cell.xpath("./a/@href")
                match_report_links.append(link[0] if link else None)
            if match_report_links and len(match_report_links) == len(df_table):
                df_table["Match Report"] = match_report_links
            frames.append(df_table)

        if not frames:
            return pd.DataFrame()
        flat = _flatten_columns(_concat_frames(frames))
        flat = flat.rename(columns={"Comp": "league", "Opponent": "opponent"})
        flat = _standardize_colnames(flat)
        flat["date"] = pd.to_datetime(flat["date"], errors="coerce").ffill()
        tmp = flat[["team", "opponent", "venue", "date"]].copy()
        tmp["home_team"] = tmp.apply(
            lambda x: x["team"] if x["venue"] == "Home" else x["opponent"],
            axis=1,
        )
        tmp["away_team"] = tmp.apply(
            lambda x: x["team"] if x["venue"] == "Away" else x["opponent"],
            axis=1,
        )
        flat["game"] = tmp.apply(_make_game_id, axis=1)
        return flat.set_index(["league", "season", "team", "game"]).sort_index()

    def _scrape_match_player_stats_from_tree(
        self, tree: html.HtmlElement
    ) -> ScrapedMatchStats:
        teams = self._parse_teams(tree)
        home_team = teams[0]["name"] if len(teams) > 0 else ""
        away_team = teams[1]["name"] if len(teams) > 1 else ""
        match_date = None
        date_text = ""
        time_node = tree.xpath("//*[@class='venuetime']/@data-venue-date")
        if time_node:
            date_text = str(time_node[0])
        if not date_text:
            text_node = tree.xpath("string(//*[@class='venuetime'])")
            date_text = str(text_node).strip()
        for fmt in ("%Y-%m-%d", "%A %B %d, %Y", "%B %d, %Y"):
            with contextlib.suppress(ValueError):
                match_date = datetime.strptime(date_text, fmt)
                break

        def parse_side(
            team_info: dict[str, str], is_home: bool
        ) -> list[ScrapedPlayerStat]:
            summary = self._find_table(
                tree,
                table_id=f"stats_{team_info['id']}_summary",
            )
            if summary is None:
                return []
            df = _flatten_columns(_parse_fbref_table(summary))
            df = _standardize_colnames(df)
            if "player" in df.columns:
                df = df[~df["player"].astype(str).str.contains(r"^\d+\sPlayers$")]

            keeper = self._find_table(tree, table_id=f"keeper_stats_{team_info['id']}")
            keeper_by_player: dict[str, pd.Series] = {}
            if keeper is not None:
                keeper_df = _standardize_colnames(
                    _flatten_columns(_parse_fbref_table(keeper))
                )
                if "player" in keeper_df.columns:
                    keeper_by_player = {
                        str(row["player"]): row for _, row in keeper_df.iterrows()
                    }

            players: list[ScrapedPlayerStat] = []
            for _, row in df.iterrows():
                player_name = str(row.get("player", "")).strip()
                if not player_name or player_name == "Player":
                    continue
                keeper_row = keeper_by_player.get(player_name)

                def as_int(
                    *columns: str,
                    source_row: pd.Series = row,
                    source_keeper: pd.Series | None = keeper_row,
                ) -> int | None:
                    for col in columns:
                        value = source_row.get(col)
                        if value is None and source_keeper is not None:
                            value = source_keeper.get(col)
                        if value is None or pd.isna(value) or value == "":
                            continue
                        with contextlib.suppress(ValueError, TypeError):
                            return int(float(str(value).replace(",", "")))
                    return None

                players.append(
                    ScrapedPlayerStat(
                        player_name=player_name,
                        team_name=team_info["name"],
                        is_home=is_home,
                        position=str(row.get("pos") or row.get("position") or "")
                        or None,
                        is_starter=as_int("min", "minutes") not in (None, 0),
                        minutes_played=as_int("min", "minutes"),
                        goals=as_int("gls", "goals"),
                        assists=as_int("ast", "assists"),
                        shots=as_int("sh", "shots"),
                        shots_on_target=as_int("sot", "so_t", "shots_on_target"),
                        yellow_cards=as_int("crdy", "crd_y", "cards_yellow"),
                        red_cards=as_int("crdr", "crd_r", "cards_red"),
                        saves=as_int("saves"),
                        goals_conceded=as_int("ga", "goals_against", "goals_conceded"),
                    )
                )
            return players

        return ScrapedMatchStats(
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            home_players=parse_side(teams[0], True) if len(teams) > 0 else [],
            away_players=parse_side(teams[1], False) if len(teams) > 1 else [],
        )

    def scrape_season_schedule(self) -> list[ScrapedScheduleEntry]:
        entries: list[ScrapedScheduleEntry] = []
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")

        table = self._find_table(self._get_current_tree(), id_contains="sched")
        if table is None:
            return entries

        df = _flatten_columns(_parse_fbref_table(table))
        df = _standardize_colnames(df)
        match_report_links = []
        for link_cell in table.xpath(".//td[@data-stat='match_report']"):
            link = link_cell.xpath("./a/@href")
            match_report_links.append(link[0] if link else None)
        if match_report_links and len(match_report_links) == len(df):
            df["match_report"] = match_report_links

        for _, row in df.iterrows():
            try:
                date_val = str(row.get("date", ""))
                match_date: datetime | None = None
                for fmt in ("%Y-%m-%d", "%a %b %d %Y"):
                    with contextlib.suppress(ValueError):
                        match_date = datetime.strptime(date_val, fmt)
                        break
                if not match_date:
                    match_date = datetime.now()
                score_match = re.search(
                    r"(\d+)\s*[–-]\s*(\d+)", str(row.get("score", ""))
                )
                home_score = int(score_match.group(1)) if score_match else None
                away_score = int(score_match.group(2)) if score_match else None

                entries.append(
                    ScrapedScheduleEntry(
                        match_date=match_date,
                        home_team=str(row["home_team"]),
                        away_team=str(row["away_team"]),
                        home_score=home_score,
                        away_score=away_score,
                        match_url=(
                            f"{FBREF_BASE}{row.get('match_report')}"
                            if row.get("match_report")
                            and not str(row.get("match_report")).startswith("http")
                            else row.get("match_report")
                        ),
                    )
                )
            except Exception as e:
                logger.error("Error parsing schedule row: %s", e)
                continue

        logger.info("Scraped %d schedule entries", len(entries))
        return entries

    def scrape_match_player_stats(
        self, match_url: str | None = None
    ) -> ScrapedMatchStats:
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")

        if match_url:
            full_url = (
                match_url
                if match_url.startswith("http")
                else f"{FBREF_BASE}{match_url}"
            )
            self.navigate_to(full_url)

        logger.info("Scraping match player stats from current page...")

        page_title = self._page.title()
        logger.info("Page title: %s", page_title)
        parsed_stats = self._scrape_match_player_stats_from_tree(
            self._get_current_tree()
        )
        if parsed_stats.home_players or parsed_stats.away_players:
            return parsed_stats

        result = self._page.evaluate(
            """
            () => {
                const homePlayers = [];
                const awayPlayers = [];
                let homeTeam = '';
                let awayTeam = '';

                const scoreEl = document.querySelector('[data-slot="score"]');
                if (scoreEl) {
                    const teams = scoreEl.querySelectorAll('[itemprop="name"]');
                    if (teams.length >= 2) {
                        homeTeam = teams[0].innerText.trim();
                        awayTeam = teams[1].innerText.trim();
                    }
                }

                if (!homeTeam) {
                    const h1 = document.querySelector('h1');
                    if (h1) {
                        const m = h1.innerText.match(/(.+?)\\s+v\\.?\\s+(.+)/);
                        if (m) {
                            homeTeam = m[1].trim();
                            awayTeam = m[2].trim();
                        }
                    }
                }

                if (!homeTeam) {
                    const breadcrumbs = document.querySelectorAll('#page ol.breadcrumb li a');
                    if (breadcrumbs.length >= 2) {
                        homeTeam = breadcrumbs[breadcrumbs.length - 2]?.innerText?.trim() || '';
                        awayTeam = breadcrumbs[breadcrumbs.length - 1]?.innerText?.trim() || '';
                    }
                }

                function parseStatsTable(table, isHome) {
                    const players = [];
                    if (!table) return players;

                    const headers = Array.from(table.querySelectorAll('thead th, thead td'))
                        .map(th => th.getAttribute('data-stat') || th.innerText.trim());

                    const body = table.querySelector('tbody');
                    if (!body) return players;

                    for (const row of body.querySelectorAll('tr')) {
                        const cells = Array.from(row.querySelectorAll('td, th'));
                        if (cells.length < 3) continue;

                        const playerLink = row.querySelector('a[href*="/players/"]');
                        if (!playerLink) continue;

                        const playerName = playerLink.innerText.trim();
                        if (!playerName || playerName === 'Player') continue;

                        const rowData = {};
                        cells.forEach(cell => {
                            const stat = cell.getAttribute('data-stat') || '';
                            const text = cell.innerText.trim();
                            if (stat) rowData[stat] = text;
                        });

                        function toInt(val) {
                            if (!val || val === '' || val === '-') return null;
                            const n = parseInt(val);
                            return isNaN(n) ? null : n;
                        }

                        const pos = rowData['position'] || '';
                        const minutes = toInt(rowData['minutes'] || rowData['min']);
                        const isStarter = minutes !== null && minutes > 0;

                        players.push({
                            playerName: playerName,
                            position: pos,
                            isStarter: isStarter,
                            minutesPlayed: minutes,
                            goals: toInt(rowData['goals'] || rowData['gls']),
                            assists: toInt(rowData['assists'] || rowData['ast']),
                            shots: toInt(rowData['shots'] || rowData['sh']),
                            shotsOnTarget: toInt(rowData['shots_on_target'] || rowData['sot']),
                            foulsCommitted: toInt(rowData['fouls_committed'] || rowData['fls'] || rowData['fld']),
                            foulsSuffered: toInt(rowData['fouls_drawn'] || rowData['fld']),
                            yellowCards: toInt(rowData['cards_yellow'] || rowData['crdy']),
                            redCards: toInt(rowData['cards_red'] || rowData['crdr']),
                            saves: toInt(rowData['saves'] || rowData['saves']),
                            goalsConceded: toInt(rowData['goals_conceded'] || rowData['gc']),
                            offsides: toInt(rowData['offsides'] || rowData['off']),
                            isHome: isHome,
                            teamName: isHome ? homeTeam : awayTeam,
                        });
                    }
                    return players;
                }

                const allTables = document.querySelectorAll('table');

                for (const table of allTables) {
                    const id = (table.getAttribute('id') || '').toLowerCase();

                    const hasPlayerLinks = table.querySelector('a[href*="/players/"]');
                    if (!hasPlayerLinks) continue;

                    const thTexts = Array.from(table.querySelectorAll('th, td'))
                        .slice(0, 30)
                        .map(el => el.innerText.trim())
                        .join(' ');

                    const isHomeTable = id.includes('home') ||
                        (!id.includes('away') &&
                         thTexts.includes(homeTeam || ' '));

                    const isAwayTable = id.includes('away') ||
                        (!id.includes('home') &&
                         thTexts.includes(awayTeam || ' '));

                    if (isHomeTable) {
                        const parsed = parseStatsTable(table, true);
                        if (parsed.length > 0) homePlayers.push(...parsed);
                    } else if (isAwayTable) {
                        const parsed = parseStatsTable(table, false);
                        if (parsed.length > 0) awayPlayers.push(...parsed);
                    }
                }

                if (homePlayers.length === 0 && awayPlayers.length === 0) {
                    const tables = Array.from(allTables);
                    for (let i = 0; i < tables.length; i++) {
                        const table = tables[i];
                        if (!table.querySelector('a[href*="/players/"]')) continue;
                        const parsed = parseStatsTable(table, i === 0);
                        if (i === 0) homePlayers.push(...parsed);
                        else awayPlayers.push(...parsed);
                    }
                }

                let matchDate = null;
                const dateEl = document.querySelector('[data-stat="date"]');
                if (dateEl) {
                    matchDate = dateEl.innerText.trim();
                }
                if (!matchDate) {
                    const strongDate = document.querySelector('strong:has(+ .scorebox)');
                    if (strongDate) matchDate = strongDate.innerText.trim();
                }
                if (!matchDate) {
                    const timeInfo = document.querySelector('.venuetime');
                    if (timeInfo) matchDate = timeInfo.innerText.trim();
                }

                return {
                    homeTeam: homeTeam,
                    awayTeam: awayTeam,
                    matchDate: matchDate,
                    homePlayers: homePlayers,
                    awayPlayers: awayPlayers,
                };
            }
            """
        )

        match_date = None
        if result.get("matchDate"):
            date_str = result["matchDate"]
            for fmt in ("%A %B %d, %Y", "%Y-%m-%d", "%B %d, %Y"):
                with contextlib.suppress(ValueError):
                    match_date = datetime.strptime(date_str, fmt)
                    break

        def _parse_player(p: dict[str, Any]) -> ScrapedPlayerStat:
            return ScrapedPlayerStat(
                player_name=p["playerName"],
                team_name=p["teamName"],
                is_home=p["isHome"],
                position=p.get("position") or None,
                is_starter=p.get("isStarter"),
                minutes_played=p.get("minutesPlayed"),
                goals=p.get("goals"),
                assists=p.get("assists"),
                shots=p.get("shots"),
                shots_on_target=p.get("shotsOnTarget"),
                fouls_committed=p.get("foulsCommitted"),
                fouls_suffered=p.get("foulsSuffered"),
                yellow_cards=p.get("yellowCards"),
                red_cards=p.get("redCards"),
                saves=p.get("saves"),
                goals_conceded=p.get("goalsConceded"),
                offsides=p.get("offsides"),
            )

        return ScrapedMatchStats(
            home_team=result.get("homeTeam", ""),
            away_team=result.get("awayTeam", ""),
            match_date=match_date,
            home_players=[_parse_player(p) for p in result.get("homePlayers", [])],
            away_players=[_parse_player(p) for p in result.get("awayPlayers", [])],
        )

    def scrape_season_player_stats(self) -> list[ScrapedSeasonPlayerStat]:
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")

        logger.info("Scraping season player stats from current page...")

        self._dismiss_overlays()

        stats_div = self._page.query_selector("div#div_stats_standard")
        if stats_div is None:
            stats_div = self._page.query_selector("div#all_stats_standard")

        if stats_div:
            click_result = self._page.evaluate(
                """
                (sel) => {
                    const div = document.querySelector(sel);
                    if (div && div.style.display === 'none') {
                        div.style.display = '';
                    }
                    return !!div;
                }
                """,
                "div#div_stats_standard" if stats_div else "div#all_stats_standard",
            )
            logger.info("Stats div found: %s", click_result)

        self._page.wait_for_timeout(1000)

        rows = self._page.evaluate(
            """
            () => {
                const results = [];

                let table = document.querySelector('table#stats_standard');
                if (!table) {
                    table = document.querySelector('table#stats');
                }
                if (!table) {
                    const tables = document.querySelectorAll('table');
                    for (const t of tables) {
                        const id = (t.getAttribute('id') || '').toLowerCase();
                        if (id.includes('standard') || id === 'stats') {
                            table = t;
                            break;
                        }
                    }
                }
                if (!table) {
                    const tables = document.querySelectorAll('table');
                    for (const t of tables) {
                        if (t.querySelector('a[href*="/players/"]')) {
                            table = t;
                            break;
                        }
                    }
                }
                if (!table) return results;

                const headerRows = table.querySelectorAll('thead tr');
                const headers = [];
                for (const hr of headerRows) {
                    for (const th of hr.querySelectorAll('th, td')) {
                        const stat = th.getAttribute('data-stat') || '';
                        const text = th.innerText.trim();
                        if (stat && stat !== 'rank') {
                            headers.push(stat);
                        }
                    }
                }

                const tbody = table.querySelector('tbody');
                if (!tbody) return results;

                for (const row of tbody.querySelectorAll('tr')) {
                    if (row.classList.contains('thead') || row.classList.contains('over_header')) {
                        continue;
                    }

                    const playerLink = row.querySelector('a[href*="/players/"]');
                    if (!playerLink) continue;

                    const playerName = playerLink.innerText.trim();
                    if (!playerName || playerName === 'Player') continue;

                    const rowData = {};
                    for (const cell of row.querySelectorAll('td, th')) {
                        const stat = cell.getAttribute('data-stat') || '';
                        const text = cell.innerText.trim();
                        if (stat) rowData[stat] = text;
                    }

                    function toInt(val) {
                        if (!val || val === '' || val === '-') return null;
                        const n = parseInt(val);
                        return isNaN(n) ? null : n;
                    }

                    results.push({
                        playerName: playerName,
                        teamName: rowData['squad_national'] || rowData['team'] || rowData['squad'] || '',
                        position: rowData['position'] || '',
                        matchesPlayed: toInt(rowData['games'] || rowData['mp']),
                        starts: toInt(rowData['games_starts'] || rowData['starts']),
                        minutes: toInt(rowData['minutes'] || rowData['min']),
                        goals: toInt(rowData['goals'] || rowData['gls']),
                        assists: toInt(rowData['assists'] || rowData['ast']),
                        shots: toInt(rowData['shots'] || rowData['sh']),
                        shotsOnTarget: toInt(rowData['shots_on_target'] || rowData['sot']),
                        foulsCommitted: toInt(rowData['fouls'] || rowData['fld']),
                        foulsSuffered: toInt(rowData['fouled'] || rowData['fls']),
                        yellowCards: toInt(rowData['cards_yellow'] || rowData['crdy'] || rowData['crdy']),
                        redCards: toInt(rowData['cards_red'] || rowData['crdr'] || rowData['crdr']),
                        saves: toInt(rowData['saves'] || rowData['saves']),
                        goalsConceded: toInt(rowData['goals_against'] || rowData['ga']),
                        offsides: toInt(rowData['offsides'] || rowData['off']),
                    });
                }
                return results;
            }
            """
        )

        player_stats: list[ScrapedSeasonPlayerStat] = []
        for row in rows:
            try:
                player_stats.append(
                    ScrapedSeasonPlayerStat(
                        player_name=row["playerName"],
                        team_name=row["teamName"],
                        position=row.get("position") or None,
                        matches_played=row.get("matchesPlayed"),
                        starts=row.get("starts"),
                        minutes=row.get("minutes"),
                        goals=row.get("goals"),
                        assists=row.get("assists"),
                        shots=row.get("shots"),
                        shots_on_target=row.get("shotsOnTarget"),
                        fouls_committed=row.get("foulsCommitted"),
                        fouls_suffered=row.get("foulsSuffered"),
                        yellow_cards=row.get("yellowCards"),
                        red_cards=row.get("redCards"),
                        saves=row.get("saves"),
                        goals_conceded=row.get("goalsConceded"),
                        offsides=row.get("offsides"),
                    )
                )
            except Exception as e:
                logger.error("Error parsing player stat row: %s", e)
                continue

        logger.info("Scraped %d season player stats", len(player_stats))
        return player_stats

    def scrape_all_match_urls_from_schedule(self) -> list[str]:
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")

        self._page.wait_for_selector("table", timeout=30000)

        match_urls: list[str] = []
        seen: set[str] = set()

        links = self._page.evaluate(
            """
            () => {
                const urls = [];
                const links = document.querySelectorAll('a[href*="/matches/"]');
                for (const link of links) {
                    const href = link.getAttribute('href');
                    if (href && !href.includes('/matches/week/')
                        && !href.includes('/matches/today')) {
                        urls.push(href);
                    }
                }
                return urls;
            }
            """
        )

        for href in links:
            full_url = href if href.startswith("http") else f"{FBREF_BASE}{href}"
            if full_url not in seen:
                seen.add(full_url)
                match_urls.append(full_url)

        logger.info("Found %d unique match URLs from schedule page", len(match_urls))
        return match_urls

    def scrape_match_stats_for_season(
        self,
        league_code: str = "ENG-Premier League",
        season: str = "2020-2021",
        max_matches: int | None = None,
    ) -> list[ScrapedMatchStats]:
        schedule_url = self.build_schedule_url(league_code, season)
        logger.info("Navigating to schedule: %s", schedule_url)
        self.navigate_to(schedule_url)

        schedule = self.scrape_season_schedule()
        logger.info("Found %d matches in schedule", len(schedule))

        match_urls = [e.match_url for e in schedule if e.match_url]
        logger.info("Found %d match URLs", len(match_urls))

        if max_matches:
            match_urls = match_urls[:max_matches]

        all_stats: list[ScrapedMatchStats] = []
        for i, url in enumerate(match_urls):
            logger.info("Scraping match %d/%d: %s", i + 1, len(match_urls), url)
            try:
                stats = self.scrape_match_player_stats(url)
                if stats.match_date is None and i < len(schedule):
                    stats.match_date = schedule[i].match_date
                all_stats.append(stats)
                time.sleep(3)
            except Exception as e:
                logger.error("Error scraping match %s: %s", url, e)
                continue

        return all_stats
