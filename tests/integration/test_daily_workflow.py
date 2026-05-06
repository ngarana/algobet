"""Integration tests for daily workflow API contracts."""

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from algobet.models import Match, ModelVersion, Prediction, Season, Team, Tournament


def _seed_workflow_data(test_session: Session) -> dict[str, int]:
    tournament = Tournament(
        id=901,
        name="Premier League",
        country="England",
        url_slug="workflow-premier-league",
    )
    season = Season(
        id=901,
        tournament_id=901,
        name="2025/26",
        start_year=2025,
        end_year=2026,
    )
    home_team = Team(id=901, name="Arsenal Workflow")
    away_team = Team(id=902, name="Chelsea Workflow")
    finished_home = Team(id=903, name="Liverpool Workflow")
    finished_away = Team(id=904, name="Everton Workflow")
    model = ModelVersion(
        id=901,
        name="Workflow Model",
        version="workflow-v1",
        algorithm="xgboost",
        accuracy=0.61,
        file_path="data/models/xgboost/workflow-v1/model.pkl",
        is_active=True,
    )
    scheduled_match = Match(
        id=901,
        tournament_id=901,
        season_id=901,
        home_team_id=901,
        away_team_id=902,
        match_date=datetime.now(timezone.utc).replace(hour=14, minute=0, second=0),
        status="SCHEDULED",
        odds_home=2.0,
        odds_draw=3.5,
        odds_away=4.0,
    )
    finished_match = Match(
        id=902,
        tournament_id=901,
        season_id=901,
        home_team_id=903,
        away_team_id=904,
        match_date=datetime.now(timezone.utc) - timedelta(days=1),
        status="FINISHED",
        home_score=2,
        away_score=1,
        odds_home=1.8,
        odds_draw=3.4,
        odds_away=4.5,
    )
    scheduled_prediction = Prediction(
        id=901,
        match_id=901,
        model_version_id=901,
        prob_home=0.58,
        prob_draw=0.25,
        prob_away=0.17,
        predicted_outcome="H",
        confidence=0.58,
        predicted_at=datetime.now(timezone.utc),
    )
    finished_prediction = Prediction(
        id=902,
        match_id=902,
        model_version_id=901,
        prob_home=0.62,
        prob_draw=0.22,
        prob_away=0.16,
        predicted_outcome="H",
        confidence=0.62,
        predicted_at=datetime.now(timezone.utc) - timedelta(days=2),
    )

    test_session.add_all(
        [
            tournament,
            season,
            home_team,
            away_team,
            finished_home,
            finished_away,
            model,
            scheduled_match,
            finished_match,
            scheduled_prediction,
            finished_prediction,
        ]
    )
    test_session.commit()

    return {
        "tournament_id": tournament.id,
        "scheduled_match_id": scheduled_match.id,
        "finished_match_id": finished_match.id,
        "team_id": home_team.id,
    }


def test_daily_workflow_preferences_watchlist_and_picks(
    test_client: TestClient,
    test_session: Session,
) -> None:
    ids = _seed_workflow_data(test_session)

    preferences = test_client.put(
        "/api/v1/workflow/profile/preferences",
        json={
            "display_name": "Local Analyst",
            "followed_tournament_ids": [ids["tournament_id"]],
            "min_confidence": 0.55,
        },
    )
    assert preferences.status_code == 200
    assert preferences.json()["followed_tournament_ids"] == [ids["tournament_id"]]

    watch = test_client.post(
        "/api/v1/workflow/watchlist",
        json={"entry_type": "team", "entry_id": ids["team_id"]},
    )
    assert watch.status_code == 200
    assert watch.json()["label"] == "Arsenal Workflow"

    user_pick = test_client.post(
        "/api/v1/workflow/user-predictions",
        json={
            "match_id": ids["scheduled_match_id"],
            "pick_1x2": "H",
            "home_score": 2,
            "away_score": 0,
        },
    )
    assert user_pick.status_code == 200
    assert user_pick.json()["model_prediction"]["predicted_outcome"] == "H"

    dashboard = test_client.get("/api/v1/workflow/dashboard/daily")
    assert dashboard.status_code == 200
    dashboard_data = dashboard.json()
    assert dashboard_data["today_matches"]
    assert dashboard_data["watchlist"]["teams"][0]["label"] == "Arsenal Workflow"

    detail = test_client.get(
        f"/api/v1/workflow/matches/{ids['scheduled_match_id']}/workflow"
    )
    assert detail.status_code == 200
    detail_data = detail.json()
    assert detail_data["match"]["home_team"]["name"] == "Arsenal Workflow"
    assert detail_data["user_prediction"]["pick_1x2"] == "H"
    assert detail_data["odds_comparison"][0]["bookmaker"] == "Market aggregate"


def test_results_review_scores_user_prediction(
    test_client: TestClient,
    test_session: Session,
) -> None:
    ids = _seed_workflow_data(test_session)

    response = test_client.post(
        "/api/v1/workflow/user-predictions",
        json={
            "match_id": ids["finished_match_id"],
            "pick_1x2": "H",
            "home_score": 2,
            "away_score": 1,
        },
    )
    assert response.status_code == 200
    assert response.json()["points"] == 8

    review = test_client.get("/api/v1/workflow/results/review")
    assert review.status_code == 200
    data = review.json()
    assert data["summaries"][1]["user_correct"] == 1
    assert data["items"][0]["user_correct"] is True
