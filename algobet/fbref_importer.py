"""FBref-based importer for player match statistics.

Uses the FBrefScraper to scrape per-match player stats from FBref
match pages and persist them to the player_match_stats table.

Usage:
    from algobet.fbref_importer import FBrefImporter
    from algobet.fbref_scraper import FBrefScraper
    from algobet.infrastructure.database import session_scope

    with session_scope() as session:
        with FBrefScraper(headless=False) as scraper:
            importer = FBrefImporter(session, scraper)
            result = importer.import_player_stats(
                league="ENG-Premier League",
                season="2020-2021",
                max_matches=10,
            )
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from algobet.fbref_scraper import FBrefScraper, ScrapedMatchStats, ScrapedPlayerStat
from algobet.matches.models import Match, PlayerMatchStats
from algobet.models import Season, Team, Tournament
from algobet.utils.team_resolver import TeamResolver

logger = logging.getLogger(__name__)


class FBrefImporter:
    """Import player match statistics from FBref using Playwright scraper.

    Attributes:
        session: SQLAlchemy database session
        scraper: FBrefScraper instance (must be started externally)
        resolver: TeamResolver for normalizing team names
    """

    def __init__(
        self,
        session: Session,
        scraper: FBrefScraper,
        resolver: TeamResolver | None = None,
    ) -> None:
        self.session = session
        self.scraper = scraper
        self._resolver = resolver or TeamResolver()

    def _resolve_team(self, name: str) -> str:
        return self._resolver.resolve(name)

    def _find_match(
        self,
        home_team_name: str,
        away_team_name: str,
        tournament_id: int,
        season_id: int,
        match_date: Any | None = None,
    ) -> Match | None:
        canonical_home = self._resolve_team(home_team_name)
        canonical_away = self._resolve_team(away_team_name)

        home_team = self.session.execute(
            select(Team).where(Team.name == canonical_home)
        ).scalar_one_or_none()

        away_team = self.session.execute(
            select(Team).where(Team.name == canonical_away)
        ).scalar_one_or_none()

        if not home_team:
            home_team = self.session.execute(
                select(Team).where(Team.name.ilike(f"%{canonical_home}%"))
            ).scalar_one_or_none()

        if not away_team:
            away_team = self.session.execute(
                select(Team).where(Team.name.ilike(f"%{canonical_away}%"))
            ).scalar_one_or_none()

        if not home_team or not away_team:
            logger.warning(
                "Could not find teams: %s (%s) -> %s, %s (%s) -> %s",
                home_team_name,
                canonical_home,
                home_team,
                away_team_name,
                canonical_away,
                away_team,
            )
            return None

        query = select(Match).where(
            Match.tournament_id == tournament_id,
            Match.season_id == season_id,
            Match.home_team_id == home_team.id,
            Match.away_team_id == away_team.id,
        )

        if match_date is not None:
            query = query.where(
                Match.match_date >= match_date.replace(hour=0, minute=0, second=0),
                Match.match_date < match_date.replace(hour=23, minute=59, second=59),
            )

        matches = self.session.execute(query).scalars().all()

        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            if match_date is not None:
                best = min(matches, key=lambda m: abs(m.match_date - match_date))
                return best
            return matches[0]

        broader = select(Match).where(
            Match.tournament_id == tournament_id,
            Match.season_id == season_id,
            Match.home_team_id == home_team.id,
            Match.away_team_id == away_team.id,
        )
        broader_matches = self.session.execute(broader).scalars().all()
        if len(broader_matches) == 1:
            return broader_matches[0]
        if len(broader_matches) > 1:
            return broader_matches[0]

        return None

    def _save_player_stats(
        self,
        match: Match,
        player: ScrapedPlayerStat,
        skip_existing: bool = True,
    ) -> bool:
        if skip_existing:
            existing = self.session.execute(
                select(PlayerMatchStats).where(
                    PlayerMatchStats.match_id == match.id,
                    PlayerMatchStats.player_name == player.player_name,
                    PlayerMatchStats.source == "fbref",
                )
            ).scalar_one_or_none()
            if existing:
                logger.debug(
                    "Skipping existing player stat: %s in match %d",
                    player.player_name,
                    match.id,
                )
                return False

        team_name = self._resolve_team(player.team_name)
        team = self.session.execute(
            select(Team).where(Team.name == team_name)
        ).scalar_one_or_none()

        if not team:
            team = self.session.execute(
                select(Team).where(Team.name.ilike(f"%{team_name}%"))
            ).scalar_one_or_none()

        if not team:
            logger.warning(
                "Could not find team for player %s: %s (resolved: %s)",
                player.player_name,
                player.team_name,
                team_name,
            )
            return False

        stat = PlayerMatchStats(
            match_id=match.id,
            player_name=player.player_name,
            team_id=team.id,
            is_home=player.is_home,
            position=player.position,
            is_starter=player.is_starter,
            minutes_played=player.minutes_played,
            goals=player.goals,
            assists=player.assists,
            shots=player.shots,
            shots_on_target=player.shots_on_target,
            fouls_committed=player.fouls_committed,
            fouls_suffered=player.fouls_suffered,
            yellow_cards=player.yellow_cards,
            red_cards=player.red_cards,
            saves=player.saves,
            goals_conceded=player.goals_conceded,
            offsides=player.offsides,
            source="fbref",
        )
        self.session.add(stat)
        return True

    def import_player_stats(
        self,
        league: str = "ENG-Premier League",
        season: str = "2020-2021",
        max_matches: int | None = None,
        skip_existing: bool = True,
        start_from_match: int = 0,
    ) -> dict[str, int]:
        tournament = self._get_tournament(league)
        if not tournament:
            logger.error("Unknown league: %s", league)
            return {"players_added": 0, "matches_processed": 0, "matches_skipped": 0}

        season_obj = self._get_season(tournament, season)
        if not season_obj:
            logger.error("Season not found: %s %s", league, season)
            return {"players_added": 0, "matches_processed": 0, "matches_skipped": 0}

        logger.info(
            "Importing FBref player stats for %s %s (tournament=%d, season=%d)",
            league,
            season,
            tournament.id,
            season_obj.id,
        )

        match_stats_list = self.scraper.scrape_match_stats_for_season(
            league_code=league,
            season=season,
            max_matches=max_matches,
        )

        if start_from_match > 0:
            match_stats_list = match_stats_list[start_from_match:]

        return self._process_match_stats(
            match_stats_list,
            tournament.id,
            season_obj.id,
            skip_existing=skip_existing,
        )

    def import_player_stats_from_schedule(
        self,
        league: str = "ENG-Premier League",
        season: str = "2020-2021",
        match_urls: list[str] | None = None,
        skip_existing: bool = True,
    ) -> dict[str, int]:
        tournament = self._get_tournament(league)
        if not tournament:
            logger.error("Unknown league: %s", league)
            return {"players_added": 0, "matches_processed": 0, "matches_skipped": 0}

        season_obj = self._get_season(tournament, season)
        if not season_obj:
            logger.error("Season not found: %s %s", league, season)
            return {"players_added": 0, "matches_processed": 0, "matches_skipped": 0}

        all_stats: list[ScrapedMatchStats] = []

        if match_urls:
            urls = match_urls
        else:
            schedule_url = FBrefScraper.build_schedule_url(league, season)
            self.scraper.navigate_to(schedule_url)
            urls = self.scraper.scrape_all_match_urls_from_schedule()

        for i, url in enumerate(urls):
            logger.info("Scraping match %d/%d: %s", i + 1, len(urls), url)
            try:
                stats = self.scraper.scrape_match_player_stats(url)
                all_stats.append(stats)
            except Exception as e:
                logger.error("Error scraping match %s: %s", url, e)
                continue

            if (i + 1) % 10 == 0:
                self.session.flush()

        return self._process_match_stats(
            all_stats,
            tournament.id,
            season_obj.id,
            skip_existing=skip_existing,
        )

    def _process_match_stats(
        self,
        match_stats_list: list[ScrapedMatchStats],
        tournament_id: int,
        season_id: int,
        skip_existing: bool = True,
    ) -> dict[str, int]:
        players_added = 0
        matches_processed = 0
        matches_skipped = 0

        for match_stats in match_stats_list:
            match = self._find_match(
                home_team_name=match_stats.home_team,
                away_team_name=match_stats.away_team,
                tournament_id=tournament_id,
                season_id=season_id,
                match_date=match_stats.match_date,
            )

            if not match:
                logger.warning(
                    "Match not found in DB: %s vs %s (date=%s)",
                    match_stats.home_team,
                    match_stats.away_team,
                    match_stats.match_date,
                )
                matches_skipped += 1
                continue

            if skip_existing:
                existing = self.session.execute(
                    select(PlayerMatchStats)
                    .where(
                        PlayerMatchStats.match_id == match.id,
                        PlayerMatchStats.source == "fbref",
                    )
                    .limit(1)
                ).scalar_one_or_none()
                if existing:
                    logger.debug("Skipping match %d (existing FBref stats)", match.id)
                    matches_skipped += 1
                    continue

            all_players = match_stats.home_players + match_stats.away_players
            match_players_added = 0

            for player in all_players:
                try:
                    saved = self._save_player_stats(
                        match, player, skip_existing=skip_existing
                    )
                    if saved:
                        match_players_added += 1
                except Exception as e:
                    logger.error(
                        "Error saving player stat for %s: %s",
                        player.player_name,
                        e,
                    )
                    continue

            players_added += match_players_added
            matches_processed += 1

            if matches_processed % 10 == 0:
                self.session.flush()
                logger.info(
                    "Progress: %d matches, %d players added",
                    matches_processed,
                    players_added,
                )

        self.session.flush()
        logger.info(
            "FBref import complete: %d matches processed, "
            "%d players added, %d matches skipped",
            matches_processed,
            players_added,
            matches_skipped,
        )

        return {
            "players_added": players_added,
            "matches_processed": matches_processed,
            "matches_skipped": matches_skipped,
        }

    def _get_tournament(self, league: str) -> Tournament | None:
        from algobet.importers.soccerdata_importer import LEAGUE_MAPPING

        info = LEAGUE_MAPPING.get(league)
        if not info:
            return None
        return self.session.execute(
            select(Tournament).where(Tournament.url_slug == info["url_slug"])
        ).scalar_one_or_none()

    def _get_season(self, tournament: Tournament, season: str) -> Season | None:
        if "-" in season:
            parts = season.split("-")
            start_year = int(parts[0])
            end_year = int(parts[1])
        elif len(season) == 4:
            start_year = int(season)
            end_year = start_year + 1
        else:
            start_year = int(season[:4])
            end_year = int(season[-4:]) if len(season) > 4 else start_year + 1

        name = f"{start_year}/{end_year}"
        return self.session.execute(
            select(Season).where(
                Season.tournament_id == tournament.id,
                Season.name == name,
            )
        ).scalar_one_or_none()
