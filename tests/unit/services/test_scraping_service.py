"""Focused tests for historical scraping resilience and season persistence."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from algobet.infrastructure.models import Base
from algobet.matches.models import Match
from algobet.scraper import ScrapedMatch
from algobet.services.scraping_service import JobStatus, ScrapingService


def _scraped_match(
    home_team: str,
    away_team: str,
    kickoff: datetime,
) -> ScrapedMatch:
    return ScrapedMatch(
        match_date=kickoff,
        home_team=home_team,
        away_team=away_team,
        home_score=2,
        away_score=1,
        odds_home=1.9,
        odds_draw=3.2,
        odds_away=4.1,
        num_bookmakers=18,
    )


@pytest.fixture
def db_session() -> Session:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_local()

    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_save_result_matches_persists_and_backfills_season_id(
    db_session: Session,
) -> None:
    service = ScrapingService(db_session)
    tournament = service.get_or_create_tournament(
        "England",
        "Premier League",
        "premier-league",
    )
    season = service.get_or_create_season(tournament, "2023-2024")
    assert season is not None

    scraped = _scraped_match(
        "Arsenal",
        "Chelsea",
        datetime(2024, 5, 1, 20, 30),
    )

    saved = service._save_result_matches([scraped], tournament, season=season)
    db_session.commit()

    stored_match = db_session.execute(select(Match)).scalar_one()
    assert saved == 1
    assert stored_match.season_id == season.id

    # Simulate an older row that existed before season-aware persistence.
    stored_match.season_id = None
    db_session.commit()

    saved_again = service._save_result_matches([scraped], tournament, season=season)
    db_session.commit()

    all_matches = db_session.execute(select(Match)).scalars().all()
    assert saved_again == 0
    assert len(all_matches) == 1
    assert all_matches[0].season_id == season.id


def test_scrape_results_commits_each_page_and_accumulates_progress() -> None:
    session = MagicMock()
    service = ScrapingService(session=session)
    mock_scraper = MagicMock()
    mock_scraper.get_page_count.return_value = 2
    first_page = [
        _scraped_match("Arsenal", "Chelsea", datetime(2024, 5, 1, 20, 30)),
        _scraped_match("Liverpool", "Everton", datetime(2024, 5, 2, 20, 30)),
    ]
    second_page = [
        _scraped_match("Brighton", "Brentford", datetime(2024, 5, 3, 20, 30))
    ]
    mock_scraper.scrape_current_page.side_effect = [first_page, second_page]
    mock_scraper.go_to_page.return_value = True
    tournament = MagicMock(id=7)
    season = MagicMock(id=11, name="2023/2024")

    with (
        patch("algobet.services.scraping_service.OddsPortalScraper") as mock_class,
        patch.object(service, "get_or_create_tournament", return_value=tournament),
        patch.object(service, "get_or_create_season", return_value=season),
        patch.object(
            service,
            "_save_result_matches",
            side_effect=[2, 1],
        ) as save_matches,
    ):
        mock_class.return_value.__enter__.return_value = mock_scraper

        progress = service.scrape_results(
            url="https://www.oddsportal.com/football/england/premier-league/results/",
            season="2023-2024",
        )

    assert progress.status == JobStatus.COMPLETED
    assert progress.matches_scraped == 3
    assert progress.matches_saved == 3
    assert session.commit.call_count == 2
    mock_scraper.go_to_page.assert_called_once_with(2)
    save_matches.assert_any_call(
        first_page,
        tournament,
        season=season,
        target_team_id=None,
    )
    save_matches.assert_any_call(
        second_page,
        tournament,
        season=season,
        target_team_id=None,
    )


def test_scrape_results_fails_on_pagination_miss_without_rescraping() -> None:
    session = MagicMock()
    service = ScrapingService(session=session)
    mock_scraper = MagicMock()
    mock_scraper.get_page_count.return_value = 2
    first_page = [_scraped_match("Arsenal", "Chelsea", datetime(2024, 5, 1, 20, 30))]
    mock_scraper.scrape_current_page.return_value = first_page
    mock_scraper.go_to_page.return_value = False
    tournament = MagicMock(id=7)
    season = MagicMock(id=11, name="2023/2024")

    with (
        patch("algobet.services.scraping_service.OddsPortalScraper") as mock_class,
        patch.object(service, "get_or_create_tournament", return_value=tournament),
        patch.object(service, "get_or_create_season", return_value=season),
        patch.object(service, "_save_result_matches", return_value=1),
    ):
        mock_class.return_value.__enter__.return_value = mock_scraper

        progress = service.scrape_results(
            url="https://www.oddsportal.com/football/england/premier-league/results/",
            season="2023-2024",
        )

    assert progress.status == JobStatus.FAILED
    assert progress.matches_scraped == 1
    assert progress.matches_saved == 1
    assert session.commit.call_count == 1
    assert "Failed to navigate to page 2" in progress.message
