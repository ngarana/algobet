"""soccerdata-based importer for historical match data and statistics.

This module provides a service for importing football match data using the
soccerdata library, which scrapes FBref, WhoScored, and other sources with
built-in team name normalization.

Provides a unified API with consistent team names, caching, and rate limiting.

Usage:
    from algobet.importers import SoccerDataImporter
    from algobet.infrastructure.database import session_scope

    with session_scope() as session:
        importer = SoccerDataImporter(session)
        result = importer.import_season(
            league="ENG-Premier League", season="2425"
        )
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from algobet.importers.soccerdata import ScheduleImporter, StatsEnricher
from algobet.matches.models import Match, MatchStatistics, PlayerMatchStats
from algobet.models import Season, Team, TeamAlias, Tournament
from algobet.utils.team_resolver import TeamResolver

logger = logging.getLogger(__name__)

# FBref league code → AlgoBet tournament slug
LEAGUE_MAPPING: dict[str, dict[str, str]] = {
    "ENG-Premier League": {
        "name": "Premier League",
        "country": "England",
        "url_slug": "england-premier-league",
    },
    "ENG-Championship": {
        "name": "Championship",
        "country": "England",
        "url_slug": "championship",
    },
    "ESP-La Liga": {
        "name": "La Liga",
        "country": "Spain",
        "url_slug": "la-liga",
    },
    "FRA-Ligue 1": {
        "name": "Ligue 1",
        "country": "France",
        "url_slug": "ligue-1",
    },
    "GER-Bundesliga": {
        "name": "Bundesliga",
        "country": "Germany",
        "url_slug": "bundesliga",
    },
    "ITA-Serie A": {
        "name": "Serie A",
        "country": "Italy",
        "url_slug": "serie-a",
    },
    "ITA-Serie B": {
        "name": "Serie B",
        "country": "Italy",
        "url_slug": "serie-b",
    },
    "FRA-Ligue 2": {
        "name": "Ligue 2",
        "country": "France",
        "url_slug": "ligue-2",
    },
    "GER-2. Bundesliga": {
        "name": "2. Bundesliga",
        "country": "Germany",
        "url_slug": "2-bundesliga",
    },
    "NED-Eredivisie": {
        "name": "Eredivisie",
        "country": "Netherlands",
        "url_slug": "eredivisie",
    },
    "POR-Primeira Liga": {
        "name": "Primeira Liga",
        "country": "Portugal",
        "url_slug": "primeira-liga",
    },
    "BEL-First Division A": {
        "name": "First Division A",
        "country": "Belgium",
        "url_slug": "first-division-a",
    },
    "TUR-Super Lig": {
        "name": "Super Lig",
        "country": "Turkey",
        "url_slug": "super-lig",
    },
    "GRE-Super League": {
        "name": "Super League",
        "country": "Greece",
        "url_slug": "super-league",
    },
    "SCO-Scottish Premiership": {
        "name": "Scottish Premiership",
        "country": "Scotland",
        "url_slug": "scottish-premiership",
    },
}


@dataclass
class ImportProgress:
    """Progress update for an import operation."""

    total_rows: int = 0
    processed_rows: int = 0
    matches_created: int = 0
    matches_skipped: int = 0
    teams_created: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of the import."""
        if self.processed_rows == 0:
            return 0.0
        return (self.matches_created / self.processed_rows) * 100


@dataclass
class ImportResult:
    """Result of an import operation."""

    success: bool
    progress: ImportProgress
    message: str = ""
    season_id: int | None = None
    tournament_id: int | None = None


class SoccerDataImporter:
    """Import match data using the soccerdata library.

    Uses FBref for schedule/results and optional WhoScored for
    detailed match statistics, with built-in team name normalization
    via teamname_replacements.json.

    Attributes:
        session: SQLAlchemy database session
        progress_callback: Optional callback for progress updates
    """

    def __init__(
        self,
        session: Session,
        progress_callback: Callable[[ImportProgress], None] | None = None,
        resolver: TeamResolver | None = None,
    ) -> None:
        self.session = session
        self.progress_callback = progress_callback
        self._resolver = resolver or TeamResolver()
        self._schedule_importer = ScheduleImporter(self)
        self._stats_enricher = StatsEnricher(self)

    def _resolve_team_name(self, name: str) -> str:
        """Resolve a team name to canonical form via TeamResolver."""
        return self._resolver.resolve(name)

    def _emit_progress(self, progress: ImportProgress) -> None:
        if self.progress_callback:
            self.progress_callback(progress)

    def lookup_team(self, name: str, source: str = "fbref") -> Team | None:
        """Look up an existing team by name, trying all resolution strategies.

        Strategy:
        1. TeamAlias lookup (source-specific)
        2. Exact match on Team.name
        3. Normalized name via teamname_replacements.json
        4. Broad alias lookup (any source)

        Args:
            name: Team name from the data source
            source: Source identifier

        Returns:
            Team instance or None if not found
        """
        # 1. Source-specific alias lookup
        alias = self.session.execute(
            select(TeamAlias).where(TeamAlias.alias == name, TeamAlias.source == source)
        ).scalar_one_or_none()
        if alias:
            return alias.team

        # 2. Exact match
        team = self.session.execute(
            select(Team).where(Team.name == name)
        ).scalar_one_or_none()
        if team:
            return team

        # 3. Normalize via teamname_replacements.json
        normalized = self._resolve_team_name(name)
        if normalized != name:
            team = self.session.execute(
                select(Team).where(Team.name == normalized)
            ).scalar_one_or_none()
            if team:
                return team

        # 4. Broad alias lookup (any source)
        alias = self.session.execute(
            select(TeamAlias).where(TeamAlias.alias == name)
        ).scalar_one_or_none()
        if alias:
            return alias.team

        return None

    def get_or_create_team(self, name: str, source: str = "fbref") -> Team:
        """Get or create a team with normalized name and alias.

        Creates a TeamAlias for source-specific lookups.

        Args:
            name: Team name from the data source
            source: Source identifier

        Returns:
            Team instance
        """
        normalized = self._resolve_team_name(name)

        existing = self.session.execute(
            select(Team).where(Team.name == normalized)
        ).scalar_one_or_none()

        if existing:
            self._ensure_alias(existing, name, source)
            return existing

        team = Team(name=normalized)
        self.session.add(team)
        self.session.flush()

        alias = TeamAlias(team_id=team.id, alias=name, source=source)
        self.session.add(alias)
        self.session.flush()

        return team

    def _ensure_alias(self, team: Team, alias_name: str, source: str) -> None:
        """Ensure a TeamAlias exists for the given team and source.

        Args:
            team: Team instance
            alias_name: Alias name to register
            source: Source identifier
        """
        if alias_name == team.name:
            return

        existing_alias = self.session.execute(
            select(TeamAlias).where(
                TeamAlias.team_id == team.id,
                TeamAlias.alias == alias_name,
                TeamAlias.source == source,
            )
        ).scalar_one_or_none()

        if not existing_alias:
            alias = TeamAlias(team_id=team.id, alias=alias_name, source=source)
            self.session.add(alias)
            self.session.flush()

    def get_or_create_tournament(self, league_code: str) -> Tournament | None:
        """Get or create a tournament from a soccerdata league code.

        Args:
            league_code: soccerdata league ID (e.g., 'ENG-Premier League')

        Returns:
            Tournament instance or None if league is unmapped
        """
        if league_code not in LEAGUE_MAPPING:
            logger.warning("Unknown league code: %s", league_code)
            return None

        info = LEAGUE_MAPPING[league_code]
        tournament = self.session.execute(
            select(Tournament).where(Tournament.url_slug == info["url_slug"])
        ).scalar_one_or_none()

        if not tournament:
            tournament = Tournament(
                name=info["name"],
                country=info["country"],
                url_slug=info["url_slug"],
            )
            self.session.add(tournament)
            self.session.flush()

        return tournament

    def get_or_create_season(self, tournament: Tournament, season_str: str) -> Season:
        """Get or create a season from soccerdata season string.

        Args:
            tournament: Tournament instance
            season_str: Season identifier (e.g., '2021' or '20-21')

        Returns:
            Season instance
        """
        if len(season_str) == 4:
            start_year = int(season_str)
            end_year = start_year + 1
        elif "-" in season_str:
            parts = season_str.split("-")
            if len(parts[0]) == 2:
                start_year = 2000 + int(parts[0])
                end_year = 2000 + int(parts[1])
            else:
                start_year = int(parts[0])
                end_year = int(parts[1])
        else:
            start_year = int(season_str)
            end_year = start_year + 1

        name = f"{start_year}/{end_year}"

        season = self.session.execute(
            select(Season).where(
                Season.tournament_id == tournament.id,
                Season.name == name,
            )
        ).scalar_one_or_none()

        if not season:
            season = Season(
                tournament_id=tournament.id,
                name=name,
                start_year=start_year,
                end_year=end_year,
            )
            self.session.add(season)
            self.session.flush()

        return season

    def _parse_score(self, score_str: str) -> tuple[int | None, int | None]:
        """Parse a score string like '3–1' or '1–0' into home/away scores.

        Args:
            score_str: Score string from FBref (uses en-dash)

        Returns:
            Tuple of (home_score, away_score)
        """
        if not isinstance(score_str, str):
            return None, None

        for sep in ("–", "-", "‑", "‒", "—"):
            if sep in score_str:
                parts = score_str.split(sep)
                try:
                    return int(parts[0].strip()), int(parts[1].strip())
                except (ValueError, IndexError):
                    return None, None
        return None, None

    def _parse_date(self, date_str: str, time_str: str | None = None) -> datetime:
        """Parse a date string to datetime.

        Args:
            date_str: Date string (e.g., '2020-09-12')
            time_str: Optional time string (e.g., '15:00' or '15:00 (16:00)')

        Returns:
            Parsed datetime
        """
        clean_time = time_str if time_str else "15:00"
        if isinstance(clean_time, str) and "(" in clean_time:
            clean_time = clean_time.split("(")[0].strip()

        combined = f"{date_str} {clean_time}"
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(combined, fmt)
            except ValueError:
                continue

        return pd.to_datetime(date_str).to_pydatetime()  # type: ignore[no-any-return]

    def import_schedule(
        self,
        league: str,
        season: str,
        no_cache: bool = False,
        headless: bool = True,
    ) -> ImportResult:
        """Import match schedule from FBref using FBrefScraper.

        Uses the Playwright-based FBrefScraper which handles Cloudflare
        protection via cookie persistence and playwright-stealth, bypassing
        the CAPTCHA issues with soccerdata's FBref implementation.

        Args:
            league: soccerdata league ID (e.g., 'ENG-Premier League')
            season: soccerdata season string (e.g., '2021' or '20-21')
            no_cache: Ignored; FBrefScraper reads live pages
            headless: Whether to run browser in headless mode (default: True)

        Returns:
            ImportResult with import statistics
        """
        return self._schedule_importer.run(
            league,
            season,
            no_cache=no_cache,
            headless=headless,
        )

    def _import_schedule_impl(
        self,
        league: str,
        season: str,
        no_cache: bool = False,
        headless: bool = True,
    ) -> ImportResult:
        """Import match schedule from FBref using FBrefScraper.

        Uses the Playwright-based FBrefScraper which handles Cloudflare
        protection via cookie persistence and playwright-stealth, bypassing
        the CAPTCHA issues with soccerdata's FBref implementation.

        Args:
            league: soccerdata league ID (e.g., 'ENG-Premier League')
            season: soccerdata season string (e.g., '2021' or '20-21')
            no_cache: Ignored; FBrefScraper reads live pages
            headless: Whether to run browser in headless mode (default: True)

        Returns:
            ImportResult with import statistics
        """
        from algobet.fbref_scraper import FBrefScraper

        progress = ImportProgress()

        try:
            with FBrefScraper(
                leagues=league, seasons=season, headless=headless
            ) as scraper:
                schedule: pd.DataFrame = scraper.read_schedule().reset_index()
        except Exception as e:
            error_msg = f"Failed to fetch schedule from FBref: {e}"
            # Check if this is a Cloudflare-related error
            if "Cloudflare challenge not resolved" in str(e) or "CAPTCHA" in str(e):
                error_msg = (
                    "Cloudflare CAPTCHA detected while accessing FBref. "
                    "Run the scraper once with headless=False to solve manually. "
                    "The scraper will save cookies for future headless runs. "
                    "Example: FBrefScraper(headless=False).read_schedule()"
                )
            logger.error(error_msg)
            progress.errors.append(error_msg)
            return ImportResult(success=False, progress=progress, message=error_msg)

        if schedule.empty:
            msg = f"No matches found for {league} season {season}"
            logger.warning(msg)
            return ImportResult(success=True, progress=progress, message=msg)

        progress.total_rows = len(schedule)

        tournament = self.get_or_create_tournament(league)
        if not tournament:
            return ImportResult(
                success=False,
                progress=progress,
                message=f"Unknown league: {league}",
            )

        season_obj = self.get_or_create_season(tournament, season)

        for _idx, row in schedule.iterrows():
            progress.processed_rows += 1

            try:
                home_name = str(row.get("home_team", ""))
                away_name = str(row.get("away_team", ""))
                date_val = row.get("date")
                time_val = row.get("time")

                if not home_name or not away_name:
                    continue

                home_team = self.get_or_create_team(home_name, "fbref")
                away_team = self.get_or_create_team(away_name, "fbref")

                score_str = str(row.get("score", ""))
                home_score, away_score = self._parse_score(score_str)

                match_date = self._parse_date(
                    str(date_val),
                    str(time_val) if pd.notna(time_val) else None,
                )

                status = "FINISHED" if home_score is not None else "SCHEDULED"

                existing = self.session.execute(
                    select(Match).where(
                        Match.tournament_id == tournament.id,
                        Match.season_id == season_obj.id,
                        Match.home_team_id == home_team.id,
                        Match.away_team_id == away_team.id,
                        func.date(Match.match_date) == match_date.date(),
                    )
                ).scalar_one_or_none()

                if existing:
                    if status == "FINISHED" and existing.status != "FINISHED":
                        existing.home_score = home_score
                        existing.away_score = away_score
                        existing.status = "FINISHED"
                        self.session.add(existing)
                    progress.matches_skipped += 1
                else:
                    match = Match(
                        tournament_id=tournament.id,
                        season_id=season_obj.id,
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                        match_date=match_date,
                        home_score=home_score,
                        away_score=away_score,
                        status=status,
                    )
                    self.session.add(match)
                    progress.matches_created += 1

                if progress.processed_rows % 50 == 0:
                    self.session.flush()

            except Exception as e:
                error_msg = f"Error processing row {_idx}: {e}"
                logger.warning(error_msg)
                progress.errors.append(error_msg)

        self.session.flush()
        self._emit_progress(progress)

        return ImportResult(
            success=len(progress.errors) == 0,
            progress=progress,
            message=(
                f"Imported {progress.matches_created} matches, "
                f"skipped {progress.matches_skipped} duplicates "
                f"from {league} {season}"
            ),
            season_id=season_obj.id,
            tournament_id=tournament.id,
        )

    def import_seasons(
        self,
        leagues: list[str],
        seasons: list[str],
        no_cache: bool = False,
        headless: bool = True,
    ) -> list[ImportResult]:
        """Import match data for multiple leagues and seasons.

        Args:
            leagues: List of soccerdata league IDs
            seasons: List of soccerdata season strings
            no_cache: If True, bypass soccerdata cache
            headless: Whether to run browser in headless mode (default: True)

        Returns:
            List of ImportResult objects, one per league+season combination
        """
        results: list[ImportResult] = []
        for league in leagues:
            for season in seasons:
                result = self.import_schedule(
                    league=league, season=season, no_cache=no_cache, headless=headless
                )
                results.append(result)
        return results

    def populate_team_aliases(
        self,
        source: str = "fbref",
        league: str = "ENG-Premier League",
        headless: bool = True,
    ) -> int:
        """Populate TeamAlias table from FBref team names.

        Scrapes team names from FBref for a given league and creates
        TeamAlias records for name variants found in teamname_replacements.json.
        Uses FBrefScraper to bypass Cloudflare CAPTCHA issues.

        Args:
            source: Source identifier (e.g., 'fbref')
            league: soccerdata league ID
            headless: Whether to run browser in headless mode (default: True)

        Returns:
            Number of aliases created
        """
        from algobet.fbref_scraper import FBrefScraper

        aliases_created = 0

        try:
            with FBrefScraper(
                leagues=league, seasons="2425", headless=headless
            ) as scraper:
                schedule: pd.DataFrame = scraper.read_schedule().reset_index()
        except Exception as e:
            # Check if this is a Cloudflare-related error
            if "Cloudflare challenge not resolved" in str(e) or "CAPTCHA" in str(e):
                logger.error(
                    "Cloudflare CAPTCHA detected while accessing FBref. "
                    "Run the scraper once with headless=False to solve manually. "
                    "The scraper will save cookies for future headless runs. "
                    "Example: FBrefScraper(leagues='%s', headless=False).read_schedule()",  # noqa: E501
                    league,
                )
            else:
                logger.error("Failed to populate aliases: %s", e)
            return 0

        all_team_names: set[str] = set()
        if "home_team" in schedule.columns:
            all_team_names.update(schedule["home_team"].dropna().unique())
        if "away_team" in schedule.columns:
            all_team_names.update(schedule["away_team"].dropna().unique())

        for name in all_team_names:
            team = self.lookup_team(name, source)
            if not team:
                team = self.get_or_create_team(name, source)
                aliases_created += 1
            else:
                self._ensure_alias(team, name, source)

        logger.info("Populated %d aliases for source '%s'", aliases_created, source)
        return aliases_created

    def enrich_understat_stats(
        self, league: str = "ENG-Premier League", season: str = "2024"
    ) -> int:
        """Enrich MatchStatistics with Understat xG and advanced metrics."""
        return self._stats_enricher.enrich_understat(league, season)

    def _enrich_understat_stats_impl(
        self, league: str = "ENG-Premier League", season: str = "2024"
    ) -> int:
        """Enrich MatchStatistics with Understat xG and advanced metrics.

        Downloads team-level match stats from Understat (xG, npxG, PPDA,
        deep completions) and saves them to existing MatchStatistics records,
        matching by team name and date.

        Args:
            league: soccerdata league ID
            season: Season string (e.g., '2024' for 2024/25)

        Returns:
            Number of matches enriched
        """
        import soccerdata as sd

        us = sd.Understat(leagues=[league], seasons=[season])
        stats: pd.DataFrame = us.read_team_match_stats()

        # Handle empty DataFrames (seasons with no Understat data coverage)
        if stats.empty or len(stats) == 0:
            logger.warning(
                "No Understat data available for %s season %s - skipping enrichment",
                league,
                season,
            )
            return 0

        # Some old seasons return a malformed DataFrame with index rows but no columns
        if "date" not in stats.columns:
            logger.warning(
                "Understat %s %s missing 'date' column - skipping enrichment",
                league,
                season,
            )
            return 0

        # Get tournament and season objects to filter matches
        tournament = self.get_or_create_tournament(league)
        if not tournament:
            logger.error("Unknown league: %s", league)
            return 0
        season_obj = self.get_or_create_season(tournament, season)

        db_matches = (
            self.session.execute(
                select(Match).where(
                    Match.tournament_id == tournament.id,
                    Match.season_id == season_obj.id,
                )
            )
            .scalars()
            .all()
        )

        match_by_key: dict[tuple[Any, ...], Match] = {}
        for m in db_matches:
            match_by_key[(m.match_date.date(), m.home_team.name, m.away_team.name)] = m

        field_map = {
            "home_np_xg": "home_npxg",
            "away_np_xg": "away_npxg",
            "home_ppda": "home_ppda",
            "away_ppda": "away_ppda",
            "home_deep_completions": "home_deep_completions",
            "away_deep_completions": "away_deep_completions",
            "home_xg": "home_xg",
            "away_xg": "away_xg",
        }

        enriched = 0
        for _idx, row in stats.iterrows():
            try:
                key = (
                    pd.to_datetime(row["date"]).date(),
                    self._resolver.resolve(str(row["home_team"])),
                    self._resolver.resolve(str(row["away_team"])),
                )
            except KeyError as e:
                logger.debug("Skipping row %s: missing column %s", _idx, e)
                continue

            db_match = match_by_key.get(key)
            if not db_match:
                continue

            ms = self.session.execute(
                select(MatchStatistics).where(MatchStatistics.match_id == db_match.id)
            ).scalar_one_or_none()

            if not ms:
                ms = MatchStatistics(match_id=db_match.id)
                self.session.add(ms)

            for src_fld, db_fld in field_map.items():
                val = row.get(src_fld)
                if pd.notna(val):
                    setattr(ms, db_fld, float(val))

            enriched += 1

        self.session.flush()
        logger.info("Enriched %d matches with Understat advanced metrics", enriched)
        return enriched

    def enrich_player_stats(
        self,
        league: str = "ENG-Premier League",
        season: str = "2024",
        skip_existing: bool = True,
    ) -> dict[str, int]:
        """Enrich PlayerMatchStats from ESPN lineup data."""
        return self._stats_enricher.enrich_player_stats(
            league,
            season,
            skip_existing=skip_existing,
        )

    def _enrich_player_stats_impl(
        self,
        league: str = "ENG-Premier League",
        season: str = "2024",
        skip_existing: bool = True,
    ) -> dict[str, int]:
        """Enrich PlayerMatchStats from ESPN lineup data.

        Downloads per-player match statistics (goals, assists, shots, cards,
        saves) from ESPN and saves to the player_match_stats table.

        Args:
            league: soccerdata league ID
            season: Season string (e.g., '2024' for 2024/25)
            skip_existing: If True, skip matches that already have player stats

        Returns:
            Dict with 'players_added' and 'matches_processed' counts
        """
        import soccerdata as sd

        # Get the season object to filter matches
        tournament = self.get_or_create_tournament(league)
        if not tournament:
            logger.error("Unknown league: %s", league)
            return {"players_added": 0, "matches_processed": 0}
        season_obj = self.get_or_create_season(tournament, season)

        espn = sd.ESPN(leagues=[league], seasons=[season])
        schedule = espn.read_schedule()

        # Only fetch matches belonging to the target season
        db_matches = (
            self.session.execute(
                select(Match).where(
                    Match.tournament_id == tournament.id,
                    Match.season_id == season_obj.id,
                )
            )
            .scalars()
            .all()
        )

        match_by_key: dict[tuple[Any, ...], Match] = {}
        for m in db_matches:
            match_by_key[(m.match_date.date(), m.home_team.name, m.away_team.name)] = m

        players_added = 0
        matches_processed = 0

        for _idx, row in schedule.iterrows():
            key = (
                pd.to_datetime(row["date"]).date(),
                self._resolver.resolve(str(row["home_team"])),
                self._resolver.resolve(str(row["away_team"])),
            )
            db_match = match_by_key.get(key)
            if not db_match:
                continue

            game_id = int(row["game_id"])

            if skip_existing:
                existing = self.session.execute(
                    select(PlayerMatchStats)
                    .where(PlayerMatchStats.match_id == db_match.id)
                    .limit(1)
                ).scalar_one_or_none()
                if existing:
                    matches_processed += 1
                    continue

            try:
                lineup = espn.read_lineup(match_id=game_id)
            except Exception:
                continue

            home_id = db_match.home_team_id
            away_id = db_match.away_team_id

            for _p, player in lineup.iterrows():
                name = player.name[-1]

                def _get_int(col: str, p: pd.Series = player) -> int | None:
                    val = p.get(col)
                    return int(val) if val is not None and pd.notna(val) else None

                ps = PlayerMatchStats(
                    match_id=db_match.id,
                    player_name=str(name),
                    team_id=home_id if bool(player["is_home"]) else away_id,
                    is_home=bool(player["is_home"]),
                    position=str(player.get("position", "")),
                    is_starter=True
                    if str(player.get("sub_in", "")) == "start"
                    else None,
                    minutes_played=_get_int("appearances"),
                    goals=_get_int("total_goals"),
                    assists=_get_int("goal_assists"),
                    shots=_get_int("total_shots"),
                    shots_on_target=_get_int("shots_on_target"),
                    fouls_committed=_get_int("fouls_committed"),
                    fouls_suffered=_get_int("fouls_suffered"),
                    yellow_cards=_get_int("yellow_cards"),
                    red_cards=_get_int("red_cards"),
                    saves=_get_int("saves"),
                    goals_conceded=_get_int("goals_conceded"),
                    offsides=_get_int("offsides"),
                    source="espn",
                )
                self.session.add(ps)
                players_added += 1

            matches_processed += 1
            if matches_processed % 20 == 0:
                self.session.flush()

        self.session.flush()
        logger.info(
            "Enriched %d players in %d matches from ESPN",
            players_added,
            matches_processed,
        )
        return {"players_added": players_added, "matches_processed": matches_processed}

    def enrich_all(
        self, league: str = "ENG-Premier League", season: str = "2024"
    ) -> dict[str, int]:
        """Run all enrichment methods in sequence.

        Args:
            league: soccerdata league ID
            season: Season string

        Returns:
            Dict with counts from each enrichment step
        """
        logger.info("Starting full enrichment for %s season %s", league, season)
        understat_count = self.enrich_understat_stats(league=league, season=season)
        player_result = self.enrich_player_stats(league=league, season=season)
        logger.info("Enrichment complete")
        return {
            "understat_enriched": understat_count,
            "players_added": player_result["players_added"],
            "matches_processed": player_result["matches_processed"],
        }
