"""Helpers for the single-user daily workflow."""

from collections.abc import Iterable
from datetime import datetime, timedelta, timezone

from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

from algobet.api.schemas.match import MatchResponse, MatchStatus
from algobet.models import Match, Prediction, Team, Tournament
from algobet.user_workflow.models import (
    ProfilePreference,
    UserPrediction,
    UserProfile,
    WatchlistEntry,
)

DEFAULT_PROFILE_KEY = "local"


def get_or_create_local_profile(db: Session) -> UserProfile:
    """Return the default local profile, creating preferences on first use."""
    profile = (
        db.query(UserProfile)
        .filter(UserProfile.profile_key == DEFAULT_PROFILE_KEY)
        .first()
    )
    if profile is None:
        profile = UserProfile(
            profile_key=DEFAULT_PROFILE_KEY,
            display_name="Local User",
        )
        db.add(profile)
        try:
            db.flush()
        except IntegrityError:
            db.rollback()
            profile = (
                db.query(UserProfile)
                .filter(UserProfile.profile_key == DEFAULT_PROFILE_KEY)
                .one()
            )

    if profile.preferences is None:
        preferences = ProfilePreference(profile_id=profile.id)
        db.add(preferences)
        try:
            db.flush()
        except IntegrityError:
            db.rollback()
            profile = (
                db.query(UserProfile)
                .filter(UserProfile.profile_key == DEFAULT_PROFILE_KEY)
                .one()
            )
        db.refresh(profile)

    return profile


def get_watchlist_ids(
    db: Session,
    profile_id: int,
    entry_type: str,
) -> list[int]:
    """Return watched IDs for a given entry type."""
    return [
        row.entry_id
        for row in db.query(WatchlistEntry)
        .filter(
            WatchlistEntry.profile_id == profile_id,
            WatchlistEntry.entry_type == entry_type,
        )
        .all()
    ]


def add_watchlist_entry(
    db: Session,
    profile_id: int,
    entry_type: str,
    entry_id: int,
) -> WatchlistEntry:
    """Add a watchlist entry idempotently."""
    existing = (
        db.query(WatchlistEntry)
        .filter(
            WatchlistEntry.profile_id == profile_id,
            WatchlistEntry.entry_type == entry_type,
            WatchlistEntry.entry_id == entry_id,
        )
        .first()
    )
    if existing is not None:
        return existing

    entry = WatchlistEntry(
        profile_id=profile_id,
        entry_type=entry_type,
        entry_id=entry_id,
    )
    db.add(entry)
    db.flush()
    return entry


def result_for_match(match: Match) -> str | None:
    """Return H/D/A when a match has a final score."""
    if (
        match.status != MatchStatus.FINISHED
        or match.home_score is None
        or match.away_score is None
    ):
        return None
    if match.home_score > match.away_score:
        return "H"
    if match.home_score < match.away_score:
        return "A"
    return "D"


def build_match_response(match: Match) -> MatchResponse:
    """Build a match API response with optional display names."""
    tournament = getattr(match, "tournament", None)
    season = getattr(match, "season", None)
    home_team = getattr(match, "home_team", None)
    away_team = getattr(match, "away_team", None)

    return MatchResponse(
        id=match.id,
        tournament_id=match.tournament_id,
        season_id=match.season_id,
        home_team_id=match.home_team_id,
        away_team_id=match.away_team_id,
        match_date=match.match_date,
        home_score=match.home_score,
        away_score=match.away_score,
        status=match.status,
        odds_home=match.odds_home,
        odds_draw=match.odds_draw,
        odds_away=match.odds_away,
        num_bookmakers=match.num_bookmakers,
        created_at=match.created_at,
        updated_at=match.updated_at,
        result=result_for_match(match),
        home_team_name=home_team.name if home_team else None,
        away_team_name=away_team.name if away_team else None,
        tournament_name=tournament.name if tournament else None,
        season_name=season.name if season else None,
    )


def score_user_prediction(
    user_prediction: UserPrediction,
) -> tuple[bool | None, bool | None, int]:
    """Score a user prediction once match results are available."""
    match = user_prediction.match
    actual = result_for_match(match)
    if actual is None:
        return None, None, 0

    outcome_correct = (
        user_prediction.pick_1x2 == actual
        if user_prediction.pick_1x2 is not None
        else None
    )
    exact_score = (
        user_prediction.home_score == match.home_score
        and user_prediction.away_score == match.away_score
        if user_prediction.home_score is not None
        and user_prediction.away_score is not None
        else None
    )

    points = 0
    if outcome_correct:
        points += 3
    if exact_score:
        points += 5

    return outcome_correct, exact_score, points


def day_bounds(date_value: datetime | None = None) -> tuple[datetime, datetime]:
    """Return UTC day boundaries for the provided date or today."""
    base = date_value or datetime.now(timezone.utc)
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)
    start = base.astimezone(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return start, start + timedelta(days=1)


def upcoming_watchlist_filter(
    watched_team_ids: Iterable[int],
    watched_tournament_ids: Iterable[int],
    watched_match_ids: Iterable[int],
) -> list[object]:
    """Build SQLAlchemy filters for watched fixtures."""
    filters: list[object] = []
    team_ids = list(watched_team_ids)
    tournament_ids = list(watched_tournament_ids)
    match_ids = list(watched_match_ids)

    if team_ids:
        filters.append(
            or_(Match.home_team_id.in_(team_ids), Match.away_team_id.in_(team_ids))
        )
    if tournament_ids:
        filters.append(Match.tournament_id.in_(tournament_ids))
    if match_ids:
        filters.append(Match.id.in_(match_ids))

    return filters


def query_predictions_with_context(db: Session):
    """Base prediction query with match/model context loaded."""
    return db.query(Prediction).options(
        joinedload(Prediction.match).joinedload(Match.home_team),
        joinedload(Prediction.match).joinedload(Match.away_team),
        joinedload(Prediction.match).joinedload(Match.tournament),
        joinedload(Prediction.match).joinedload(Match.season),
        joinedload(Prediction.model_version),
    )


def query_matches_with_context(db: Session):
    """Base match query with display context loaded."""
    return db.query(Match).options(
        joinedload(Match.home_team),
        joinedload(Match.away_team),
        joinedload(Match.tournament),
        joinedload(Match.season),
    )


def get_entity_label(
    db: Session,
    entry_type: str,
    entry_id: int,
) -> tuple[str, str | None]:
    """Return a display label and secondary text for a watchlist entry."""
    if entry_type == "team":
        team = db.query(Team).filter(Team.id == entry_id).first()
        return (team.name if team else f"Team #{entry_id}", None)

    if entry_type == "tournament":
        tournament = db.query(Tournament).filter(Tournament.id == entry_id).first()
        if tournament:
            return tournament.name, tournament.country
        return f"Tournament #{entry_id}", None

    match = (
        db.query(Match)
        .options(
            joinedload(Match.home_team),
            joinedload(Match.away_team),
            joinedload(Match.tournament),
        )
        .filter(Match.id == entry_id)
        .first()
    )
    if match:
        return (
            f"{match.home_team.name} vs {match.away_team.name}",
            match.tournament.name if match.tournament else None,
        )
    return f"Match #{entry_id}", None
