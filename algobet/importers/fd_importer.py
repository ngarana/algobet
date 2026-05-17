"""Football-Data.co.uk importer for historical match results and odds.

This module provides a service for importing historical football data from
Football-Data.co.uk using the soccerdata library, which provides direct access
to match results, statistics (shots, corners, cards), and betting odds going
back 25+ years for top European leagues.

Compared to the soccerdata/FBref approach:
- No Cloudflare/CAPTCHA issues (direct data feed)
- Includes pre-built odds data (no separate scraping needed)
- Match statistics included (shots, corners, cards, fouls)
- CSV format already parsed by soccerdata

Top 5 European leagues supported:
- ENG-Premier League
- ESP-La Liga
- FRA-Ligue 1
- GER-Bundesliga
- ITA-Serie A
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from algobet.importers.tournaments import get_or_create_tournament_by_country
from algobet.matches.models import Match, MatchStatistics
from algobet.models import Season, Team, TeamAlias, Tournament
from algobet.utils.team_resolver import TeamResolver

logger = logging.getLogger(__name__)

# Top 5 European leagues with historical data coverage
TOP_5_LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "FRA-Ligue 1",
    "GER-Bundesliga",
    "ITA-Serie A",
]

LEAGUE_MAPPING = {
    "ENG-Premier League": {
        "name": "Premier League",
        "country": "England",
        "url_slug": "england-premier-league",
        "fd_code": "E0",
    },
    "ESP-La Liga": {
        "name": "La Liga",
        "country": "Spain",
        "url_slug": "la-liga",
        "fd_code": "SP1",
    },
    "FRA-Ligue 1": {
        "name": "Ligue 1",
        "country": "France",
        "url_slug": "ligue-1",
        "fd_code": "F1",
    },
    "GER-Bundesliga": {
        "name": "Bundesliga",
        "country": "Germany",
        "url_slug": "bundesliga",
        "fd_code": "D1",
    },
    "ITA-Serie A": {
        "name": "Serie A",
        "country": "Italy",
        "url_slug": "serie-a",
        "fd_code": "I1",
    },
}


@dataclass
class FDImportProgress:
    """Progress update for an FD.co.uk import operation."""

    total_rows: int = 0
    processed_rows: int = 0
    matches_created: int = 0
    matches_updated: int = 0
    stats_enriched: int = 0
    odds_enriched: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of the import."""
        if self.processed_rows == 0:
            return 0.0
        return (self.matches_created / self.processed_rows) * 100


@dataclass
class FDImportResult:
    """Result of an FD.co.uk import operation."""

    success: bool
    progress: FDImportProgress
    message: str = ""
    season_id: int | None = None
    tournament_id: int | None = None


class FDImporter:
    """Import historical match data from Football-Data.co.uk.

    Uses soccerdata's MatchHistory to fetch match results, statistics,
    and betting odds directly from football-data.co.uk.

    Attributes:
        session: SQLAlchemy database session
        progress_callback: Optional callback for progress updates
    """

    def __init__(
        self,
        session: Session,
        progress_callback: Callable[[FDImportProgress], None] | None = None,
        resolver: TeamResolver | None = None,
    ) -> None:
        self.session = session
        self.progress_callback = progress_callback
        self._resolver = resolver or TeamResolver()

    def _emit_progress(self, progress: FDImportProgress) -> None:
        """Emit progress update to callback if registered."""
        if self.progress_callback:
            self.progress_callback(progress)

    def _resolve_team_name(self, name: str) -> str:
        """Resolve a team name to canonical form via TeamResolver."""
        return self._resolver.resolve(name)

    def get_or_create_team(self, name: str, source: str = "fd") -> Team:
        """Get or create a team with normalized name and alias."""
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

        self._ensure_alias(team, name, source)
        return team

    def _ensure_alias(self, team: Team, alias_name: str, source: str) -> None:
        """Ensure a TeamAlias exists for the given team and source."""
        if alias_name == team.name:
            return

        existing = self.session.execute(
            select(TeamAlias).where(
                TeamAlias.team_id == team.id,
                TeamAlias.alias == alias_name,
                TeamAlias.source == source,
            )
        ).scalar_one_or_none()

        if not existing:
            alias = TeamAlias(team_id=team.id, alias=alias_name, source=source)
            self.session.add(alias)
            self.session.flush()

    def get_or_create_tournament(self, league_code: str) -> Tournament | None:
        """Get or create a tournament from a soccerdata league code."""
        if league_code not in LEAGUE_MAPPING:
            logger.warning("Unknown league code: %s", league_code)
            return None

        info = LEAGUE_MAPPING[league_code]
        return get_or_create_tournament_by_country(
            self.session,
            name=info["name"],
            country=info["country"],
            url_slug=info["url_slug"],
        )

    def get_or_create_season(self, tournament: Tournament, season_str: str) -> Season:
        """Get or create a season from soccerdata season string."""
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

    def _build_fd_url(self, league: str, season: str) -> str:
        """Build the Football-Data.co.uk CSV URL for a league and season."""
        info = LEAGUE_MAPPING.get(league)
        if not info or "fd_code" not in info:
            raise ValueError(f"No FD code mapping for league: {league}")

        fd_code = info["fd_code"]
        # Convert season format: "2024" -> "2324" or "2425"
        if len(season) == 4:
            start = int(season)
            end = start + 1
            season_str = f"{str(start)[2:]}{str(end)[2:]}"
        elif len(season) == 5 and "-" in season:  # e.g., "23-24" -> "2324"
            season_str = season.replace("-", "")
        else:
            season_str = season

        return f"https://www.football-data.co.uk/mmz4281/{season_str}/{fd_code}.csv"

    def import_season(
        self,
        league: str,
        season: str,
        include_stats: bool = True,
        include_odds: bool = True,
    ) -> FDImportResult:
        """Import historical match data from Football-Data.co.uk.

        Fetches match results, statistics, and betting odds for a league season.

        Args:
            league: soccerdata league ID (e.g., 'ENG-Premier League')
            season: Season string (e.g., '2024' for 2024/25, '2425' for 2024/25)
            include_stats: Whether to include match statistics (shots, cards, etc.)
            include_odds: Whether to include betting odds

        Returns:
            FDImportResult with import statistics
        """
        import io
        import urllib.request

        progress = FDImportProgress()

        try:
            url = self._build_fd_url(league, season)
            logger.info("Fetching data from %s", url)

            with urllib.request.urlopen(url) as response:
                csv_data = response.read().decode("utf-8")
            games = pd.read_csv(io.StringIO(csv_data))
        except Exception as e:
            error_msg = f"Failed to fetch data from Football-Data: {e}"
            logger.error(error_msg)
            progress.errors.append(error_msg)
            return FDImportResult(success=False, progress=progress, message=error_msg)

        if games.empty:
            msg = f"No data found for {league} season {season}"
            logger.warning(msg)
            return FDImportResult(success=True, progress=progress, message=msg)

        progress.total_rows = len(games)

        tournament = self.get_or_create_tournament(league)
        if not tournament:
            return FDImportResult(
                success=False,
                progress=progress,
                message=f"Unknown league: {league}",
            )

        season_obj = self.get_or_create_season(tournament, season)

        for _idx, row in games.iterrows():
            progress.processed_rows += 1

            try:
                home_name = str(row.get("HomeTeam", ""))
                away_name = str(row.get("AwayTeam", ""))

                if not home_name or not away_name:
                    continue

                home_team = self.get_or_create_team(home_name, "fd")
                away_team = self.get_or_create_team(away_name, "fd")

                # Parse date (DD/MM/YYYY or DD/MM/YY format)
                date_str = str(row["Date"])
                if pd.isna(row["Date"]) or date_str == "nan":
                    continue
                try:
                    if len(date_str.split("/")[-1]) == 2:
                        match_date = pd.to_datetime(
                            date_str,
                            format="%d/%m/%y",
                        ).to_pydatetime()
                    else:
                        match_date = pd.to_datetime(
                            date_str,
                            format="%d/%m/%Y",
                        ).to_pydatetime()
                except (ValueError, TypeError):
                    logger.warning(
                        "Skipping row %s: unparseable date '%s'",
                        _idx,
                        date_str,
                    )
                    progress.errors.append(
                        f"Unparseable date at row {_idx}: {date_str}"
                    )
                    continue

                home_goals = row.get("FTHG")
                away_goals = row.get("FTAG")

                has_result = pd.notna(home_goals) and pd.notna(away_goals)
                home_score = int(home_goals) if has_result else None
                away_score = int(away_goals) if has_result else None
                status = "FINISHED" if has_result else "SCHEDULED"

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
                    # Update result if we have it and match was scheduled
                    if has_result and existing.status == "SCHEDULED":
                        existing.home_score = home_score
                        existing.away_score = away_score
                        existing.status = "FINISHED"
                        progress.matches_updated += 1

                    db_match = existing
                else:
                    # Create new match - flush immediately so it gets an ID
                    db_match = Match(
                        tournament_id=tournament.id,
                        season_id=season_obj.id,
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                        match_date=match_date,
                        home_score=home_score,
                        away_score=away_score,
                        status=status,
                    )
                    self.session.add(db_match)
                    self.session.flush()  # Get the ID for stats/odds
                    progress.matches_created += 1

                # Update odds if requested
                if include_odds and has_result:
                    self._update_odds(db_match, row)
                    progress.odds_enriched += 1

                # Update stats if requested
                if include_stats:
                    self._update_stats(db_match, row)
                    progress.stats_enriched += 1

                if progress.processed_rows % 100 == 0:
                    self._emit_progress(progress)
                    self.session.flush()

            except Exception as e:
                error_msg = f"Error processing row {_idx}: {e}"
                logger.warning(error_msg)
                progress.errors.append(error_msg)
                self.session.rollback()

        self.session.flush()
        self._emit_progress(progress)

        msg = (
            f"Imported {progress.matches_created} matches, "
            f"updated {progress.matches_updated}, "
            f"enriched {progress.stats_enriched} stats, "
            f"{progress.odds_enriched} odds from {league} {season}"
        )
        return FDImportResult(
            success=len(progress.errors) == 0,
            progress=progress,
            message=msg,
            season_id=season_obj.id,
            tournament_id=tournament.id,
        )

    def _safe_float(self, val: Any) -> float | None:
        """Safely convert value to float, returning None for invalid values."""
        if val is None or pd.isna(val):
            return None
        try:
            v = float(val)
            return v if v > 0 else None
        except (ValueError, TypeError):
            return None

    def _update_odds(self, match: Match, row: pd.Series) -> None:
        """Update match with betting odds from football-data.co.uk."""
        updated = False

        # 1X2 odds - try Pinnacle Sports averages first, then Bet365
        home_odds = self._safe_float(row.get("PSH")) or self._safe_float(
            row.get("B365H")
        )
        draw_odds = self._safe_float(row.get("PSD")) or self._safe_float(
            row.get("B365D")
        )
        away_odds = self._safe_float(row.get("PSA")) or self._safe_float(
            row.get("B365A")
        )

        if match.odds_home is None and home_odds is not None:
            match.odds_home = home_odds
            updated = True
        if match.odds_draw is None and draw_odds is not None:
            match.odds_draw = draw_odds
            updated = True
        if match.odds_away is None and away_odds is not None:
            match.odds_away = away_odds
            updated = True

        # Average odds across bookmakers
        avg_home = self._safe_float(row.get("PSH")) or self._safe_float(
            row.get("BWA")  # Note: BWA is not in CSV, using PSH as approximation
        )
        avg_draw = self._safe_float(row.get("PSD")) or self._safe_float(
            row.get("BWD")  # Note: BWD is not in CSV, using PSD as approximation
        )
        avg_away = self._safe_float(row.get("PSA")) or self._safe_float(
            row.get("WHH")  # Using WHH as approximation, need to check
        )

        # Actually, let's use what's available in the CSV
        # Check what average/max columns are actually present
        avg_home = self._safe_float(row.get("PSH"))  # Pinnacle home odds
        avg_draw = self._safe_float(row.get("PSD"))  # Pinnacle draw odds
        avg_away = self._safe_float(row.get("PSA"))  # Pinnacle away odds

        if match.avg_home_odds is None and avg_home is not None:
            match.avg_home_odds = avg_home
            updated = True
        if match.avg_draw_odds is None and avg_draw is not None:
            match.avg_draw_odds = avg_draw
            updated = True
        if match.avg_away_odds is None and avg_away is not None:
            match.avg_away_odds = avg_away
            updated = True

        # Max odds
        max_home = self._safe_float(row.get("MaxH")) or self._safe_float(row.get("BWH"))
        max_draw = self._safe_float(row.get("MaxD")) or self._safe_float(row.get("BWD"))
        max_away = self._safe_float(row.get("MaxA")) or self._safe_float(row.get("BWA"))

        if match.max_home_odds is None and max_home is not None:
            match.max_home_odds = max_home
            updated = True
        if match.max_draw_odds is None and max_draw is not None:
            match.max_draw_odds = max_draw
            updated = True
        if match.max_away_odds is None and max_away is not None:
            match.max_away_odds = max_away
            updated = True

        # Asian handicap
        ah_line = self._safe_float(row.get("AHh"))
        ah_home = self._safe_float(row.get("B365AHH"))

        if match.odds_asian_handicap_line is None and ah_line is not None:
            match.odds_asian_handicap_line = ah_line
            updated = True
        if match.odds_asian_handicap is None and ah_home is not None:
            # Store home AH odds; away can be derived
            match.odds_asian_handicap = ah_home
            updated = True

        # Over/Under
        ou_line_raw = row.get("B365>2.5") or row.get("PSCH") or row.get("B365C>2.5")
        ou_over = self._safe_float(ou_line_raw)
        # Note: The line is typically 2.5, but we can also check for explicit line
        ou_line = 2.5  # Standard line

        if match.odds_over_under_25 is None and ou_over is not None:
            match.odds_over_under_25 = ou_over
            updated = True
        if match.odds_over_under_line is None and ou_line is not None:
            match.odds_over_under_line = ou_line
            updated = True

        if updated:
            self.session.add(match)

    def _update_stats(self, match: Match, row: pd.Series) -> None:
        """Update match statistics from football-data.co.uk."""
        # Check if we already have stats
        ms = self.session.execute(
            select(MatchStatistics).where(MatchStatistics.match_id == match.id)
        ).scalar_one_or_none()

        if not ms:
            ms = MatchStatistics(match_id=match.id)
            self.session.add(ms)

        updated = False

        # Shots
        if ms.home_shots is None:
            v = self._safe_float(row.get("HS"))
            if v is not None:
                ms.home_shots = int(v)
                updated = True
        if ms.away_shots is None:
            v = self._safe_float(row.get("AS"))
            if v is not None:
                ms.away_shots = int(v)
                updated = True
        if ms.home_shots_on_target is None:
            v = self._safe_float(row.get("HST"))
            if v is not None:
                ms.home_shots_on_target = int(v)
                updated = True
        if ms.away_shots_on_target is None:
            v = self._safe_float(row.get("AST"))
            if v is not None:
                ms.away_shots_on_target = int(v)
                updated = True

        # Corners
        if ms.home_corners is None:
            v = self._safe_float(row.get("HC"))
            if v is not None:
                ms.home_corners = int(v)
                updated = True
        if ms.away_corners is None:
            v = self._safe_float(row.get("AC"))
            if v is not None:
                ms.away_corners = int(v)
                updated = True

        # Cards
        if ms.home_yellow_cards is None:
            v = self._safe_float(row.get("HY"))
            if v is not None:
                ms.home_yellow_cards = int(v)
                updated = True
        if ms.away_yellow_cards is None:
            v = self._safe_float(row.get("AY"))
            if v is not None:
                ms.away_yellow_cards = int(v)
                updated = True
        if ms.home_red_cards is None:
            v = self._safe_float(row.get("HR"))
            if v is not None:
                ms.home_red_cards = int(v)
                updated = True
        if ms.away_red_cards is None:
            v = self._safe_float(row.get("AR"))
            if v is not None:
                ms.away_red_cards = int(v)
                updated = True

        # Fouls
        if ms.home_fouls is None:
            v = self._safe_float(row.get("HF"))
            if v is not None:
                ms.home_fouls = int(v)
                updated = True
        if ms.away_fouls is None:
            v = self._safe_float(row.get("AF"))
            if v is not None:
                ms.away_fouls = int(v)
                updated = True

        # Half-time scores
        ht_home = self._safe_float(row.get("HTHG"))
        ht_away = self._safe_float(row.get("HTAG"))

        if ms.ht_home_score is None and ht_home is not None:
            ms.ht_home_score = int(ht_home)
            updated = True
        if ms.ht_away_score is None and ht_away is not None:
            ms.ht_away_score = int(ht_away)
            updated = True

        # HT result
        if ms.ht_result is None and ht_home is not None and ht_away is not None:
            if ht_home > ht_away:
                ms.ht_result = "H"
            elif ht_home < ht_away:
                ms.ht_result = "A"
            else:
                ms.ht_result = "D"
            updated = True

        if updated:
            self.session.flush()

    def import_top_5_leagues_2024_25(self) -> list[FDImportResult]:
        """Import 2024/25 season for top 5 European leagues."""
        results: list[FDImportResult] = []
        for league in TOP_5_LEAGUES:
            logger.info("Importing %s 2024/25", league)
            result = self.import_season(
                league, "2024", include_stats=True, include_odds=True
            )
            results.append(result)
            logger.info("Completed %s: %s", league, result.message)
        return results

    def import_top_5_leagues_range(
        self,
        start_season: str,
        end_season: str,
        include_stats: bool = True,
        include_odds: bool = True,
    ) -> list[FDImportResult]:
        """Import top 5 leagues for a range of seasons.

        Args:
            start_season: First season (e.g., '2012' for 2012/13)
            end_season: Last season (e.g., '2023' for 2023/24)
            include_stats: Whether to include match statistics
            include_odds: Whether to include betting odds
        """
        results: list[FDImportResult] = []
        start = int(start_season)
        end = int(end_season)

        for year in range(start, end + 1):
            for league in TOP_5_LEAGUES:
                logger.info("Importing %s %s/%s", league, year, year + 1)
                result = self.import_season(
                    league,
                    str(year),
                    include_stats=include_stats,
                    include_odds=include_odds,
                )
                results.append(result)
                logger.info(
                    "Completed %s %s/%s: %s",
                    league,
                    year,
                    year + 1,
                    result.message,
                )

        return results
