"""Routers for the daily user workflow."""

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import or_
from sqlalchemy.orm import Session

from algobet.api.dependencies import get_db
from algobet.api.routers.predictions import _build_prediction_item
from algobet.api.routers.value_bets import build_value_bets
from algobet.api.schemas.match import MatchStatus
from algobet.api.schemas.workflow import (
    DailyWorkflowResponse,
    MatchWorkflowDetailResponse,
    ProfilePreferencesRequest,
    ProfilePreferencesResponse,
    ResultsReviewItemResponse,
    ResultsReviewResponse,
    ResultsSummaryResponse,
    UserPredictionRequest,
    UserPredictionResponse,
    WatchlistEntryRequest,
    WatchlistEntryResponse,
    WatchlistResponse,
)
from algobet.models import Match, Prediction
from algobet.services.match_detail_builder import (
    build_match_detail_response,
    build_match_workflow_detail,
)
from algobet.user_workflow.models import UserPrediction, WatchlistEntry
from algobet.user_workflow.service import (
    add_watchlist_entry,
    build_match_response,
    day_bounds,
    get_entity_label,
    get_or_create_local_profile,
    get_watchlist_ids,
    query_matches_with_context,
    query_predictions_with_context,
    result_for_match,
    score_user_prediction,
    upcoming_watchlist_filter,
)

router = APIRouter()


def _serialize_user_prediction(
    user_prediction: UserPrediction,
    model_prediction: Prediction | None = None,
) -> UserPredictionResponse:
    outcome_correct, exact_score, points = score_user_prediction(user_prediction)
    return UserPredictionResponse(
        id=user_prediction.id,
        match_id=user_prediction.match_id,
        pick_1x2=user_prediction.pick_1x2,
        home_score=user_prediction.home_score,
        away_score=user_prediction.away_score,
        total_goals_line=user_prediction.total_goals_line,
        total_goals_pick=user_prediction.total_goals_pick,
        notes=user_prediction.notes,
        model_prediction=_build_prediction_item(model_prediction)
        if model_prediction is not None
        else None,
        is_correct_1x2=outcome_correct,
        is_exact_score=exact_score,
        points=points,
        created_at=user_prediction.created_at,
        updated_at=user_prediction.updated_at,
    )


def _watchlist_response(db: Session, profile_id: int) -> WatchlistResponse:
    grouped: dict[str, list[WatchlistEntryResponse]] = {
        "team": [],
        "tournament": [],
        "match": [],
    }
    entries = (
        db.query(WatchlistEntry)
        .filter(WatchlistEntry.profile_id == profile_id)
        .order_by(WatchlistEntry.created_at.desc())
        .all()
    )

    for entry in entries:
        label, meta = get_entity_label(db, entry.entry_type, entry.entry_id)
        grouped[entry.entry_type].append(
            WatchlistEntryResponse(
                id=entry.id,
                entry_type=entry.entry_type,
                entry_id=entry.entry_id,
                label=label,
                meta=meta,
                created_at=entry.created_at,
            )
        )

    return WatchlistResponse(
        teams=grouped["team"],
        tournaments=grouped["tournament"],
        matches=grouped["match"],
    )


def _preferences_response(db: Session) -> ProfilePreferencesResponse:
    profile = get_or_create_local_profile(db)
    preferences = profile.preferences
    return ProfilePreferencesResponse(
        profile_key=profile.profile_key,
        display_name=profile.display_name,
        default_days_ahead=preferences.default_days_ahead,
        min_confidence=preferences.min_confidence,
        min_ev=preferences.min_ev,
        favorite_bookie=preferences.favorite_bookie,
        followed_tournament_ids=get_watchlist_ids(db, profile.id, "tournament"),
    )


def _latest_model_prediction(db: Session, match_id: int) -> Prediction | None:
    return (
        db.query(Prediction)
        .filter(Prediction.match_id == match_id)
        .order_by(Prediction.predicted_at.desc())
        .first()
    )


def _summary_for_period(
    db: Session,
    profile_id: int,
    label: str,
    start: datetime,
    end: datetime,
) -> ResultsSummaryResponse:
    predictions = (
        query_predictions_with_context(db)
        .join(Match)
        .filter(
            Match.status == MatchStatus.FINISHED,
            Match.match_date >= start,
            Match.match_date < end,
        )
        .all()
    )
    model_correct = sum(
        1
        for prediction in predictions
        if result_for_match(prediction.match) == prediction.predicted_outcome
    )

    user_predictions = (
        db.query(UserPrediction)
        .join(Match)
        .filter(
            UserPrediction.profile_id == profile_id,
            Match.status == MatchStatus.FINISHED,
            Match.match_date >= start,
            Match.match_date < end,
        )
        .all()
    )
    user_correct = sum(
        1
        for prediction in user_predictions
        if score_user_prediction(prediction)[0] is True
    )

    return ResultsSummaryResponse(
        label=label,
        start_date=start,
        end_date=end,
        model_predictions=len(predictions),
        model_correct=model_correct,
        model_accuracy=model_correct / len(predictions) if predictions else None,
        user_predictions=len(user_predictions),
        user_correct=user_correct,
        user_accuracy=user_correct / len(user_predictions)
        if user_predictions
        else None,
    )


@router.get("/profile/preferences", response_model=ProfilePreferencesResponse)
def get_profile_preferences(
    db: Session = Depends(get_db),
) -> ProfilePreferencesResponse:
    """Return local profile preferences."""
    return _preferences_response(db)


@router.put("/profile/preferences", response_model=ProfilePreferencesResponse)
def update_profile_preferences(
    request: ProfilePreferencesRequest,
    db: Session = Depends(get_db),
) -> ProfilePreferencesResponse:
    """Update local profile preferences and followed tournaments."""
    profile = get_or_create_local_profile(db)
    preferences = profile.preferences

    if request.display_name is not None:
        profile.display_name = request.display_name
    if request.default_days_ahead is not None:
        preferences.default_days_ahead = request.default_days_ahead
    if request.min_confidence is not None:
        preferences.min_confidence = request.min_confidence
    if request.min_ev is not None:
        preferences.min_ev = request.min_ev
    if request.favorite_bookie is not None:
        preferences.favorite_bookie = request.favorite_bookie

    if request.followed_tournament_ids is not None:
        (
            db.query(WatchlistEntry)
            .filter(
                WatchlistEntry.profile_id == profile.id,
                WatchlistEntry.entry_type == "tournament",
            )
            .delete(synchronize_session=False)
        )
        for tournament_id in request.followed_tournament_ids:
            add_watchlist_entry(db, profile.id, "tournament", tournament_id)

    db.flush()
    return _preferences_response(db)


@router.get("/watchlist", response_model=WatchlistResponse)
def get_watchlist(db: Session = Depends(get_db)) -> WatchlistResponse:
    """Return grouped local watchlist entries."""
    profile = get_or_create_local_profile(db)
    return _watchlist_response(db, profile.id)


@router.post("/watchlist", response_model=WatchlistEntryResponse)
def add_to_watchlist(
    request: WatchlistEntryRequest,
    db: Session = Depends(get_db),
) -> WatchlistEntryResponse:
    """Add a team, tournament, or match to the local watchlist."""
    profile = get_or_create_local_profile(db)
    entry = add_watchlist_entry(db, profile.id, request.entry_type, request.entry_id)
    label, meta = get_entity_label(db, entry.entry_type, entry.entry_id)
    return WatchlistEntryResponse(
        id=entry.id,
        entry_type=entry.entry_type,
        entry_id=entry.entry_id,
        label=label,
        meta=meta,
        created_at=entry.created_at,
    )


@router.delete("/watchlist/{entry_type}/{entry_id}")
def remove_from_watchlist(
    entry_type: str,
    entry_id: int,
    db: Session = Depends(get_db),
) -> dict[str, bool]:
    """Remove a watchlist entry."""
    profile = get_or_create_local_profile(db)
    deleted = (
        db.query(WatchlistEntry)
        .filter(
            WatchlistEntry.profile_id == profile.id,
            WatchlistEntry.entry_type == entry_type,
            WatchlistEntry.entry_id == entry_id,
        )
        .delete(synchronize_session=False)
    )
    return {"deleted": bool(deleted)}


@router.get("/user-predictions", response_model=list[UserPredictionResponse])
def list_user_predictions(
    match_id: int | None = Query(None),
    db: Session = Depends(get_db),
) -> list[UserPredictionResponse]:
    """Return local user predictions, optionally for one match."""
    profile = get_or_create_local_profile(db)
    query = db.query(UserPrediction).filter(UserPrediction.profile_id == profile.id)
    if match_id is not None:
        query = query.filter(UserPrediction.match_id == match_id)
    predictions = query.order_by(UserPrediction.created_at.desc()).all()
    return [
        _serialize_user_prediction(
            prediction,
            _latest_model_prediction(db, prediction.match_id),
        )
        for prediction in predictions
    ]


@router.post("/user-predictions", response_model=UserPredictionResponse)
def upsert_user_prediction(
    request: UserPredictionRequest,
    db: Session = Depends(get_db),
) -> UserPredictionResponse:
    """Create or update the local user's prediction for a match."""
    profile = get_or_create_local_profile(db)
    match = db.query(Match).filter(Match.id == request.match_id).first()
    if match is None:
        raise HTTPException(
            status_code=404,
            detail=f"Match {request.match_id} not found",
        )

    prediction = (
        db.query(UserPrediction)
        .filter(
            UserPrediction.profile_id == profile.id,
            UserPrediction.match_id == request.match_id,
        )
        .first()
    )
    if prediction is None:
        prediction = UserPrediction(profile_id=profile.id, match_id=request.match_id)
        db.add(prediction)

    prediction.pick_1x2 = request.pick_1x2
    prediction.home_score = request.home_score
    prediction.away_score = request.away_score
    prediction.total_goals_line = request.total_goals_line
    prediction.total_goals_pick = request.total_goals_pick
    prediction.notes = request.notes
    db.flush()
    db.refresh(prediction)

    return _serialize_user_prediction(
        prediction,
        _latest_model_prediction(db, prediction.match_id),
    )


@router.get("/dashboard/daily", response_model=DailyWorkflowResponse)
def get_daily_dashboard(
    date: datetime | None = Query(None),
    db: Session = Depends(get_db),
) -> DailyWorkflowResponse:
    """Return the daily workflow dashboard payload for the local profile."""
    profile = get_or_create_local_profile(db)
    preferences = profile.preferences
    start, end = day_bounds(date)
    watched_tournament_ids = get_watchlist_ids(db, profile.id, "tournament")
    watched_team_ids = get_watchlist_ids(db, profile.id, "team")
    watched_match_ids = get_watchlist_ids(db, profile.id, "match")

    today_query = (
        query_predictions_with_context(db)
        .join(Match)
        .filter(Match.match_date >= start, Match.match_date < end)
        .order_by(Match.match_date)
    )
    if watched_tournament_ids:
        today_query = today_query.filter(
            Match.tournament_id.in_(watched_tournament_ids)
        )
    today_predictions = today_query.all()

    high_confidence = [
        prediction
        for prediction in today_predictions
        if prediction.confidence >= preferences.min_confidence
    ][:8]

    watch_filters = upcoming_watchlist_filter(
        watched_team_ids,
        watched_tournament_ids,
        watched_match_ids,
    )
    watched_fixtures: list[Match] = []
    if watch_filters:
        watched_fixtures = (
            query_matches_with_context(db)
            .filter(
                Match.match_date >= start,
                Match.match_date
                < start + timedelta(days=preferences.default_days_ahead),
                Match.status.in_([MatchStatus.SCHEDULED, MatchStatus.LIVE]),
                or_(*watch_filters),
            )
            .order_by(Match.match_date)
            .limit(12)
            .all()
        )

    value_bets = build_value_bets(
        db,
        min_ev=preferences.min_ev,
        max_odds=10.0,
        start_date=start,
        end_date=end,
        max_matches=8,
    )

    return DailyWorkflowResponse(
        date=start.date().isoformat(),
        today_matches=[
            _build_prediction_item(prediction) for prediction in today_predictions
        ],
        high_confidence=[
            _build_prediction_item(prediction) for prediction in high_confidence
        ],
        value_bets=value_bets,
        watched_fixtures=[
            build_match_detail_response(db, match) for match in watched_fixtures
        ],
        results_summary=_summary_for_period(db, profile.id, "Today", start, end),
        watchlist=_watchlist_response(db, profile.id),
    )


@router.get("/results/review", response_model=ResultsReviewResponse)
def get_results_review(
    db: Session = Depends(get_db),
) -> ResultsReviewResponse:
    """Return day/week/month model and user-pick review."""
    profile = get_or_create_local_profile(db)
    today_start, today_end = day_bounds()
    week_start = today_start - timedelta(days=6)
    month_start = today_start - timedelta(days=29)

    recent_matches = (
        query_matches_with_context(db)
        .filter(
            Match.status == MatchStatus.FINISHED,
            Match.match_date >= week_start,
            Match.match_date < today_end,
        )
        .order_by(Match.match_date.desc())
        .limit(50)
        .all()
    )

    items = []
    for match in recent_matches:
        model_prediction = _latest_model_prediction(db, match.id)
        user_prediction = (
            db.query(UserPrediction)
            .filter(
                UserPrediction.profile_id == profile.id,
                UserPrediction.match_id == match.id,
            )
            .first()
        )
        actual_result = result_for_match(match)
        items.append(
            ResultsReviewItemResponse(
                match=build_match_response(match),
                model_prediction=_build_prediction_item(model_prediction)
                if model_prediction is not None
                else None,
                user_prediction=_serialize_user_prediction(
                    user_prediction,
                    model_prediction,
                )
                if user_prediction is not None
                else None,
                actual_result=actual_result,
                model_correct=model_prediction.predicted_outcome == actual_result
                if model_prediction is not None and actual_result is not None
                else None,
                user_correct=user_prediction.pick_1x2 == actual_result
                if user_prediction is not None
                and user_prediction.pick_1x2 is not None
                and actual_result is not None
                else None,
            )
        )

    return ResultsReviewResponse(
        summaries=[
            _summary_for_period(db, profile.id, "Today", today_start, today_end),
            _summary_for_period(db, profile.id, "7 Days", week_start, today_end),
            _summary_for_period(db, profile.id, "30 Days", month_start, today_end),
        ],
        items=items,
    )


@router.get("/matches/{match_id}/workflow", response_model=MatchWorkflowDetailResponse)
def get_match_workflow_detail(
    match_id: int,
    db: Session = Depends(get_db),
) -> MatchWorkflowDetailResponse:
    """Return enriched workflow detail for a match."""
    return build_match_workflow_detail(db, match_id)
