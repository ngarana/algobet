"""Response builders for match detail and workflow payloads."""

from datetime import datetime

from fastapi import HTTPException
from sqlalchemy import or_
from sqlalchemy.orm import Session, joinedload

from algobet.api.schemas import (
    MatchDetailResponse,
    MatchResponse,
    MatchStatus,
    PredictionListItemResponse,
    PredictionMatchSummaryResponse,
    PredictionResponse,
)
from algobet.api.schemas.model import ModelVersionResponse
from algobet.api.schemas.team import TeamResponse
from algobet.api.schemas.tournament import SeasonResponse, TournamentResponse
from algobet.api.schemas.workflow import (
    MatchOddsRowResponse,
    MatchWorkflowDetailResponse,
    ModelFeatureExplanationResponse,
    RecentFormResponse,
    RecentTeamMatchResponse,
    SimilarAccuracyResponse,
    StatsComparisonResponse,
    TeamStatsComparisonResponse,
    UserPredictionResponse,
)
from algobet.models import (
    Match,
    MatchStatistics,
    ModelFeature,
    Prediction,
    ScrapedOdds,
)
from algobet.predictions.data.queries import MatchRepository
from algobet.user_workflow.models import UserPrediction
from algobet.user_workflow.service import (
    build_match_response,
    get_or_create_local_profile,
    get_watchlist_ids,
    result_for_match,
    score_user_prediction,
)


def _build_prediction_item(prediction: Prediction) -> PredictionListItemResponse:
    match = prediction.match
    match_summary = (
        PredictionMatchSummaryResponse(
            id=match.id,
            match_date=match.match_date,
            status=match.status,
            home_team_name=match.home_team.name,
            away_team_name=match.away_team.name,
            tournament_name=match.tournament.name if match.tournament else None,
            season_name=match.season.name if match.season else None,
            home_score=match.home_score,
            away_score=match.away_score,
            odds_home=match.odds_home,
            odds_draw=match.odds_draw,
            odds_away=match.odds_away,
        )
        if match is not None
        else None
    )
    model_version = (
        ModelVersionResponse.model_validate(prediction.model_version)
        if prediction.model_version is not None
        else None
    )
    return PredictionListItemResponse(
        id=prediction.id,
        match_id=prediction.match_id,
        model_version_id=prediction.model_version_id,
        prob_home=prediction.prob_home,
        prob_draw=prediction.prob_draw,
        prob_away=prediction.prob_away,
        predicted_outcome=prediction.predicted_outcome,
        confidence=prediction.confidence,
        predicted_at=prediction.predicted_at,
        actual_roi=prediction.actual_roi,
        max_probability=prediction.max_probability,
        match=match_summary,
        model_version=model_version,
    )


def _h2h_responses(db: Session, match: Match, limit: int = 5) -> list[MatchResponse]:
    repo = MatchRepository(db)
    h2h_matches = repo.get_h2h_matches(
        match.home_team_id,
        match.away_team_id,
        limit=limit,
        before_date=match.match_date,
    )
    return [build_match_response(h2h_match) for h2h_match in h2h_matches]


def build_match_detail_response(db: Session, match: Match) -> MatchDetailResponse:
    """Build the standard match detail response."""
    predictions = db.query(Prediction).filter(Prediction.match_id == match.id).all()
    prediction_responses = [
        PredictionResponse(
            id=pred.id,
            match_id=pred.match_id,
            model_version_id=pred.model_version_id,
            prob_home=pred.prob_home,
            prob_draw=pred.prob_draw,
            prob_away=pred.prob_away,
            predicted_outcome=pred.predicted_outcome,
            confidence=pred.confidence,
            predicted_at=pred.predicted_at,
            actual_roi=pred.actual_roi,
            max_probability=pred.max_probability,
        )
        for pred in predictions
    ]

    return MatchDetailResponse(
        **build_match_response(match).model_dump(),
        tournament=TournamentResponse.model_validate(match.tournament)
        if match.tournament
        else None,
        season=SeasonResponse.model_validate(match.season) if match.season else None,
        home_team=TeamResponse.model_validate(match.home_team),
        away_team=TeamResponse.model_validate(match.away_team),
        predictions=prediction_responses,
        h2h_matches=_h2h_responses(db, match),
    )


def _recent_team_form(
    db: Session,
    team_id: int,
    before_date: datetime,
    limit: int = 6,
) -> list[RecentTeamMatchResponse]:
    matches = (
        db.query(Match)
        .options(joinedload(Match.home_team), joinedload(Match.away_team))
        .filter(
            Match.status == MatchStatus.FINISHED,
            Match.match_date < before_date,
            or_(Match.home_team_id == team_id, Match.away_team_id == team_id),
        )
        .order_by(Match.match_date.desc())
        .limit(limit)
        .all()
    )

    rows: list[RecentTeamMatchResponse] = []
    for match in matches:
        if match.home_score is None or match.away_score is None:
            continue
        is_home = match.home_team_id == team_id
        goals_for = match.home_score if is_home else match.away_score
        goals_against = match.away_score if is_home else match.home_score
        if goals_for > goals_against:
            result = "W"
        elif goals_for < goals_against:
            result = "L"
        else:
            result = "D"
        rows.append(
            RecentTeamMatchResponse(
                match_id=match.id,
                match_date=match.match_date,
                opponent_name=match.away_team.name if is_home else match.home_team.name,
                venue="home" if is_home else "away",
                goals_for=goals_for,
                goals_against=goals_against,
                result=result,
            )
        )
    return rows


def _team_stats_comparison(
    db: Session,
    team_id: int,
    team_name: str,
    before_date: datetime,
) -> TeamStatsComparisonResponse:
    matches = (
        db.query(Match)
        .options(joinedload(Match.statistics))
        .filter(
            Match.status == MatchStatus.FINISHED,
            Match.match_date < before_date,
            or_(Match.home_team_id == team_id, Match.away_team_id == team_id),
        )
        .order_by(Match.match_date.desc())
        .limit(6)
        .all()
    )

    goals_for: list[int] = []
    goals_against: list[int] = []
    shots: list[int] = []
    shots_on_target: list[int] = []
    corners: list[int] = []

    for match in matches:
        if match.home_score is None or match.away_score is None:
            continue
        is_home = match.home_team_id == team_id
        goals_for.append(match.home_score if is_home else match.away_score)
        goals_against.append(match.away_score if is_home else match.home_score)
        stats: MatchStatistics | None = getattr(match, "statistics", None)
        if stats is None:
            continue
        shot_value = stats.home_shots if is_home else stats.away_shots
        target_value = (
            stats.home_shots_on_target if is_home else stats.away_shots_on_target
        )
        corner_value = stats.home_corners if is_home else stats.away_corners
        if shot_value is not None:
            shots.append(shot_value)
        if target_value is not None:
            shots_on_target.append(target_value)
        if corner_value is not None:
            corners.append(corner_value)

    def average(values: list[int]) -> float | None:
        return sum(values) / len(values) if values else None

    return TeamStatsComparisonResponse(
        team_id=team_id,
        team_name=team_name,
        matches=len(goals_for),
        avg_goals_for=average(goals_for) or 0.0,
        avg_goals_against=average(goals_against) or 0.0,
        avg_shots=average(shots),
        avg_shots_on_target=average(shots_on_target),
        avg_corners=average(corners),
    )


def _odds_comparison(match: Match, db: Session) -> list[MatchOddsRowResponse]:
    odds_rows = (
        db.query(ScrapedOdds)
        .filter(ScrapedOdds.match_id == match.id)
        .order_by(ScrapedOdds.scraped_at.desc())
        .all()
    )
    rows = [
        MatchOddsRowResponse(
            bookmaker=row.bookmaker,
            odds_home=row.odds_home,
            odds_draw=row.odds_draw,
            odds_away=row.odds_away,
            scraped_at=row.scraped_at,
            source=row.source,
        )
        for row in odds_rows
    ]

    if rows or not (match.odds_home and match.odds_draw and match.odds_away):
        return rows

    return [
        MatchOddsRowResponse(
            bookmaker="Market aggregate",
            odds_home=match.odds_home,
            odds_draw=match.odds_draw,
            odds_away=match.odds_away,
            scraped_at=None,
            source="matches",
        )
    ]


def _model_explanation(
    db: Session,
    match_id: int,
) -> list[ModelFeatureExplanationResponse]:
    feature_row = (
        db.query(ModelFeature)
        .filter(ModelFeature.match_id == match_id)
        .order_by(ModelFeature.computed_at.desc())
        .first()
    )
    if feature_row is None:
        return []

    numeric_features = []
    for feature, raw_value in feature_row.features.items():
        if isinstance(raw_value, int | float):
            value = float(raw_value)
            numeric_features.append((feature, value, abs(value)))

    numeric_features.sort(key=lambda item: item[2], reverse=True)
    return [
        ModelFeatureExplanationResponse(
            feature=feature,
            label=feature.replace("_", " ").title(),
            value=value,
            direction=(
                "positive" if value > 0 else "negative" if value < 0 else "neutral"
            ),
            impact=impact,
        )
        for feature, value, impact in numeric_features[:8]
    ]


def _similar_accuracy(
    db: Session,
    prediction: Prediction | None,
) -> SimilarAccuracyResponse:
    if prediction is None:
        return SimilarAccuracyResponse(
            sample_size=0,
            correct=0,
            accuracy=None,
            description="No model prediction is available for this match.",
        )

    lower = max(0.0, prediction.confidence - 0.1)
    upper = min(1.0, prediction.confidence + 0.1)
    similar_predictions = (
        db.query(Prediction)
        .join(Match)
        .filter(
            Prediction.predicted_outcome == prediction.predicted_outcome,
            Prediction.confidence >= lower,
            Prediction.confidence <= upper,
            Match.status == MatchStatus.FINISHED,
            Match.home_score.isnot(None),
            Match.away_score.isnot(None),
        )
        .limit(200)
        .all()
    )
    correct = sum(
        1
        for similar in similar_predictions
        if result_for_match(similar.match) == similar.predicted_outcome
    )
    return SimilarAccuracyResponse(
        sample_size=len(similar_predictions),
        correct=correct,
        accuracy=correct / len(similar_predictions) if similar_predictions else None,
        description=(
            "Finished matches with the same predicted outcome and similar confidence."
        ),
    )


def _user_prediction_response(
    user_prediction: UserPrediction | None,
    model_prediction: Prediction | None,
) -> UserPredictionResponse | None:
    if user_prediction is None:
        return None
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
        if model_prediction
        else None,
        is_correct_1x2=outcome_correct,
        is_exact_score=exact_score,
        points=points,
        created_at=user_prediction.created_at,
        updated_at=user_prediction.updated_at,
    )


def build_match_workflow_detail(
    db: Session,
    match_id: int,
) -> MatchWorkflowDetailResponse:
    """Build the enriched match workflow detail payload."""
    match = (
        db.query(Match)
        .options(
            joinedload(Match.home_team),
            joinedload(Match.away_team),
            joinedload(Match.tournament),
            joinedload(Match.season),
        )
        .filter(Match.id == match_id)
        .first()
    )
    if match is None:
        raise HTTPException(status_code=404, detail=f"Match {match_id} not found")

    latest_prediction = (
        db.query(Prediction)
        .filter(Prediction.match_id == match.id)
        .order_by(Prediction.predicted_at.desc())
        .first()
    )
    profile = get_or_create_local_profile(db)
    watched_match_ids = get_watchlist_ids(db, profile.id, "match")
    user_prediction = (
        db.query(UserPrediction)
        .filter(
            UserPrediction.profile_id == profile.id,
            UserPrediction.match_id == match.id,
        )
        .first()
    )

    return MatchWorkflowDetailResponse(
        match=build_match_detail_response(db, match),
        odds_comparison=_odds_comparison(match, db),
        recent_form=RecentFormResponse(
            home=_recent_team_form(db, match.home_team_id, match.match_date),
            away=_recent_team_form(db, match.away_team_id, match.match_date),
        ),
        stats_comparison=StatsComparisonResponse(
            home=_team_stats_comparison(
                db,
                match.home_team_id,
                match.home_team.name,
                match.match_date,
            ),
            away=_team_stats_comparison(
                db,
                match.away_team_id,
                match.away_team.name,
                match.match_date,
            ),
        ),
        model_explanation=_model_explanation(db, match.id),
        similar_accuracy=_similar_accuracy(db, latest_prediction),
        user_prediction=_user_prediction_response(user_prediction, latest_prediction),
        watched=match.id in watched_match_ids,
    )
