"""Tests for FastAPI API endpoints."""

from datetime import datetime, timedelta

from algobet.models import Match, Season, Team, Tournament


class TestRootEndpoints:
    """Tests for root and health endpoints."""

    def test_root_endpoint(self, test_client) -> None:
        """Test root endpoint returns API info."""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "AlgoBet API"
        assert data["version"] == "0.1.0"
        assert "docs" in data
        assert "redoc" in data

    def test_health_check(self, test_client) -> None:
        """Test health check endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestTournamentsEndpoints:
    """Tests for tournaments API endpoints."""

    def test_list_tournaments_empty(self, test_client) -> None:
        """Test listing tournaments when database is empty."""
        response = test_client.get("/api/v1/tournaments")
        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_list_tournaments_with_data(self, test_client, test_session) -> None:
        """Test listing tournaments with data."""
        # Create test tournaments
        tournament1 = Tournament(
            name="Premier League", country="England", url_slug="premier-league"
        )
        tournament2 = Tournament(name="La Liga", country="Spain", url_slug="la-liga")
        test_session.add_all([tournament1, tournament2])
        test_session.commit()

        response = test_client.get("/api/v1/tournaments")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["name"] == "La Liga"  # Sorted alphabetically
        assert data[1]["name"] == "Premier League"

    def test_get_tournament(self, test_client, test_session) -> None:
        """Test getting a specific tournament."""
        tournament = Tournament(
            name="Premier League", country="England", url_slug="premier-league"
        )
        test_session.add(tournament)
        test_session.commit()

        response = test_client.get(f"/api/v1/tournaments/{tournament.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == tournament.id
        assert data["name"] == "Premier League"

    def test_get_tournament_not_found(self, test_client) -> None:
        """Test getting a non-existent tournament."""
        response = test_client.get("/api/v1/tournaments/999")
        assert response.status_code == 404

    def test_get_tournament_seasons(self, test_client, test_session) -> None:
        """Test getting seasons for a tournament."""
        tournament = Tournament(
            name="Premier League", country="England", url_slug="premier-league"
        )
        season1 = Season(
            tournament=tournament, name="2023/2024", start_year=2023, end_year=2024
        )
        season2 = Season(
            tournament=tournament, name="2022/2023", start_year=2022, end_year=2023
        )
        test_session.add_all([tournament, season1, season2])
        test_session.commit()

        response = test_client.get(f"/api/v1/tournaments/{tournament.id}/seasons")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        # Should be sorted by year descending
        assert data[0]["name"] == "2023/2024"
        assert data[1]["name"] == "2022/2023"


class TestTeamsEndpoints:
    """Tests for teams API endpoints."""

    def test_list_teams_empty(self, test_client) -> None:
        """Test listing teams when database is empty."""
        response = test_client.get("/api/v1/teams")
        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_list_teams_with_data(self, test_client, test_session) -> None:
        """Test listing teams with data."""
        team1 = Team(name="Arsenal")
        team2 = Team(name="Chelsea")
        test_session.add_all([team1, team2])
        test_session.commit()

        response = test_client.get("/api/v1/teams")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_search_teams(self, test_client, test_session) -> None:
        """Test searching teams by name."""
        team1 = Team(name="Manchester United")
        team2 = Team(name="Manchester City")
        team3 = Team(name="Arsenal")
        test_session.add_all([team1, team2, team3])
        test_session.commit()

        response = test_client.get("/api/v1/teams?search=Manchester")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        for team in data:
            assert "Manchester" in team["name"]

    def test_get_team(self, test_client, test_session) -> None:
        """Test getting a specific team."""
        team = Team(name="Liverpool")
        test_session.add(team)
        test_session.commit()

        response = test_client.get(f"/api/v1/teams/{team.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == team.id
        assert data["name"] == "Liverpool"

    def test_get_team_not_found(self, test_client) -> None:
        """Test getting a non-existent team."""
        response = test_client.get("/api/v1/teams/999")
        assert response.status_code == 404

    def test_list_teams_with_offset(self, test_client, test_session) -> None:
        """Test listing teams with offset pagination."""
        # Create multiple teams
        teams = [Team(name=f"Team {i}") for i in range(5)]
        test_session.add_all(teams)
        test_session.commit()

        # Get first 2 teams
        response = test_client.get("/api/v1/teams?limit=2&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        first_team_id = data[0]["id"]

        # Get next 2 teams with offset
        response = test_client.get("/api/v1/teams?limit=2&offset=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        # Should be different teams
        assert data[0]["id"] != first_team_id

    def test_list_teams_offset_beyond_range(self, test_client, test_session) -> None:
        """Test listing teams with offset beyond available data."""
        team = Team(name="Arsenal")
        test_session.add(team)
        test_session.commit()

        # Offset beyond available data should return empty list
        response = test_client.get("/api/v1/teams?limit=10&offset=100")
        assert response.status_code == 200
        data = response.json()
        assert data == []


class TestMatchesEndpoints:
    """Tests for matches API endpoints."""

    def setup_match_data(self, test_session) -> tuple:
        """Helper to set up match test data."""
        tournament = Tournament(
            name="Premier League", country="England", url_slug="premier-league"
        )
        season = Season(
            tournament=tournament, name="2023/2024", start_year=2023, end_year=2024
        )
        team1 = Team(name="Manchester United")
        team2 = Team(name="Arsenal")

        # Scheduled match
        scheduled_match = Match(
            tournament=tournament,
            season=season,
            home_team=team1,
            away_team=team2,
            match_date=datetime.now() + timedelta(days=1),
            status="SCHEDULED",
            odds_home=2.10,
            odds_draw=3.40,
            odds_away=3.20,
            num_bookmakers=12,
        )

        # Finished match
        finished_match = Match(
            tournament=tournament,
            season=season,
            home_team=team2,
            away_team=team1,
            match_date=datetime.now() - timedelta(days=1),
            home_score=2,
            away_score=1,
            status="FINISHED",
        )

        test_session.add_all(
            [tournament, season, team1, team2, scheduled_match, finished_match]
        )
        test_session.commit()
        return scheduled_match, finished_match

    def test_list_matches_empty(self, test_client) -> None:
        """Test listing matches when database is empty."""
        response = test_client.get("/api/v1/matches")
        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_list_matches_with_data(self, test_client, test_session) -> None:
        """Test listing matches with data."""
        self.setup_match_data(test_session)

        response = test_client.get("/api/v1/matches")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_filter_matches_by_status(self, test_client, test_session) -> None:
        """Test filtering matches by status."""
        scheduled, finished = self.setup_match_data(test_session)

        # Get scheduled matches
        response = test_client.get("/api/v1/matches?status=SCHEDULED")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == scheduled.id
        assert data[0]["status"] == "SCHEDULED"

        # Get finished matches
        response = test_client.get("/api/v1/matches?status=FINISHED")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == finished.id
        assert data[0]["status"] == "FINISHED"

    def test_invalid_status_filter(self, test_client) -> None:
        """Test that invalid status filter returns error."""
        response = test_client.get("/api/v1/matches?status=INVALID")
        assert response.status_code == 400

    def test_get_match(self, test_client, test_session) -> None:
        """Test getting a specific match."""
        scheduled, finished = self.setup_match_data(test_session)

        response = test_client.get(f"/api/v1/matches/{scheduled.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == scheduled.id
        assert data["status"] == "SCHEDULED"
        # Check odds are included
        assert data["odds_home"] == 2.10

    def test_get_match_not_found(self, test_client) -> None:
        """Test getting a non-existent match."""
        response = test_client.get("/api/v1/matches/999")
        assert response.status_code == 404

    def test_get_match_with_result(self, test_client, test_session) -> None:
        """Test that finished matches have result field."""
        scheduled, finished = self.setup_match_data(test_session)

        response = test_client.get(f"/api/v1/matches/{finished.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["result"] == "H"  # Home win (2-1)

    def test_match_preview(self, test_client, test_session) -> None:
        """Test match preview endpoint."""
        scheduled, finished = self.setup_match_data(test_session)

        response = test_client.get(f"/api/v1/matches/{scheduled.id}/preview")
        assert response.status_code == 200
        data = response.json()
        assert "match" in data
        assert "form" in data
        assert data["match"]["id"] == scheduled.id

    def test_get_match_h2h_returns_list(self, test_client, test_session) -> None:
        """Test H2H endpoint returns array of matches directly."""
        scheduled, finished = self.setup_match_data(test_session)

        response = test_client.get(f"/api/v1/matches/{scheduled.id}/h2h")
        assert response.status_code == 200
        data = response.json()
        # Should return array directly (not wrapped in {"h2h": [...]})
        assert isinstance(data, list)
        # The finished match is between the same teams, so it appears in H2H
        assert len(data) == 1
        assert data[0]["result"] == "H"  # Home win from the finished match

    def test_get_match_h2h_with_history(self, test_client, test_session) -> None:
        """Test H2H when teams have previous matches."""
        tournament = Tournament(
            name="Premier League", country="England", url_slug="premier-league"
        )
        season = Season(
            tournament=tournament, name="2023/2024", start_year=2023, end_year=2024
        )
        team1 = Team(name="Manchester United")
        team2 = Team(name="Arsenal")

        # Previous H2H match
        h2h_match = Match(
            tournament=tournament,
            season=season,
            home_team=team1,
            away_team=team2,
            match_date=datetime.now() - timedelta(days=30),
            home_score=1,
            away_score=1,
            status="FINISHED",
        )

        # Current match
        current_match = Match(
            tournament=tournament,
            season=season,
            home_team=team2,
            away_team=team1,
            match_date=datetime.now() + timedelta(days=1),
            status="SCHEDULED",
        )

        test_session.add_all(
            [tournament, season, team1, team2, h2h_match, current_match]
        )
        test_session.commit()

        response = test_client.get(f"/api/v1/matches/{current_match.id}/h2h")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1  # Now returns array directly
        assert data[0]["result"] == "D"  # Draw


class TestPredictionsEndpoints:
    """Tests for predictions API endpoints."""

    def test_list_predictions(self, test_client) -> None:
        """Test listing predictions."""
        response = test_client.get("/api/v1/predictions")
        assert response.status_code == 200
        data = response.json()
        # Returns empty list as no predictions in database
        assert data == []

    def test_generate_predictions_no_model(self, test_client) -> None:
        """Test prediction generation fails when no model is available."""
        response = test_client.post(
            "/api/v1/predictions/generate",
            json={"match_ids": [], "days_ahead": 7},
        )
        # Should return 400 when no active model exists
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "model" in data["detail"].lower() or "No active model" in data["detail"]

    def test_generate_predictions_with_model(self, test_client, test_session) -> None:
        """Test prediction generation with a model version specified.

        Note: This test specifies a model_version that doesn't exist,
        so it will fail with 400. A real implementation would need the
        model file on disk.
        """
        response = test_client.post(
            "/api/v1/predictions/generate",
            json={"match_ids": [], "days_ahead": 7, "model_version": "nonexistent_v1"},
        )
        # Should return 400 because model file doesn't exist
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    def test_upcoming_predictions(self, test_client) -> None:
        """Test upcoming predictions endpoint."""
        response = test_client.get("/api/v1/predictions/upcoming?days=7")
        assert response.status_code == 200
        data = response.json()
        # Returns empty list as no predictions in database
        assert data == []

    def test_prediction_history(self, test_client) -> None:
        """Test prediction history endpoint."""
        response = test_client.get("/api/v1/predictions/history")
        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        assert "overall_accuracy" in data

    def test_value_bets_deprecated(self, test_client) -> None:
        """Test deprecated value bets endpoint (moved to top-level)."""
        response = test_client.get("/api/v1/predictions/value-bets")
        assert response.status_code == 404


class TestValueBetsEndpoints:
    """Tests for value-bets API endpoints."""

    def test_list_value_bets(self, test_client) -> None:
        """Test listing value bets at new top-level endpoint."""
        response = test_client.get("/api/v1/value-bets")
        assert response.status_code == 200
        data = response.json()
        # Returns empty list as predictions not yet implemented
        assert data == []

    def test_value_bets_with_filters(self, test_client) -> None:
        """Test value bets endpoint with query parameters."""
        response = test_client.get(
            "/api/v1/value-bets?min_ev=0.05&max_odds=10.0&days=7&max_matches=20"
        )
        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_value_bets_with_model_version(self, test_client) -> None:
        """Test value bets endpoint with model version filter."""
        response = test_client.get("/api/v1/value-bets?model_version=v1.0.0")
        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_value_bets_with_min_confidence(self, test_client) -> None:
        """Test value bets endpoint with min confidence filter."""
        response = test_client.get("/api/v1/value-bets?min_confidence=0.7")
        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_value_bets_invalid_min_confidence(self, test_client) -> None:
        """Test value bets endpoint with invalid min confidence."""
        response = test_client.get("/api/v1/value-bets?min_confidence=1.5")
        assert response.status_code == 422

    def test_value_bets_invalid_min_ev(self, test_client) -> None:
        """Test value bets endpoint with invalid min_ev (negative)."""
        response = test_client.get("/api/v1/value-bets?min_ev=-0.1")
        assert response.status_code == 422

    def test_value_bets_invalid_max_odds(self, test_client) -> None:
        """Test value bets endpoint with invalid max_odds (less than 1)."""
        response = test_client.get("/api/v1/value-bets?max_odds=0.5")
        assert response.status_code == 422

    def test_value_bets_invalid_days(self, test_client) -> None:
        """Test value bets endpoint with invalid days (too many)."""
        response = test_client.get("/api/v1/value-bets?days=100")
        assert response.status_code == 422


class TestModelsEndpoints:
    """Tests for models API endpoints."""

    def test_list_models(self, test_client) -> None:
        """Test listing models."""
        response = test_client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        # Returns empty list as models not yet implemented
        assert data == []

    def test_get_active_model(self, test_client) -> None:
        """Test getting active model."""
        response = test_client.get("/api/v1/models/active")
        assert response.status_code == 200
        # Returns None as no models yet
        assert response.json() is None

    def test_get_model_not_found(self, test_client) -> None:
        """Test getting a non-existent model."""
        response = test_client.get("/api/v1/models/999")
        assert response.status_code == 404

    def test_activate_model_not_found(self, test_client) -> None:
        """Test activating a non-existent model."""
        response = test_client.post("/api/v1/models/999/activate")
        assert response.status_code == 404

    def test_delete_model_not_found(self, test_client) -> None:
        """Test deleting a non-existent model."""
        response = test_client.delete("/api/v1/models/999")
        assert response.status_code == 404

    def test_get_model_metrics_not_found(self, test_client) -> None:
        """Test getting metrics for non-existent model."""
        response = test_client.get("/api/v1/models/999/metrics")
        assert response.status_code == 404


class TestSeasonsEndpoints:
    """Tests for seasons API endpoints."""

    def setup_season_data(self, test_session) -> tuple:
        """Helper to set up season test data."""
        tournament = Tournament(
            name="Premier League", country="England", url_slug="premier-league"
        )
        season = Season(
            tournament=tournament, name="2023/2024", start_year=2023, end_year=2024
        )
        team1 = Team(name="Manchester United")
        team2 = Team(name="Arsenal")
        team3 = Team(name="Chelsea")

        # Scheduled match
        scheduled_match = Match(
            tournament=tournament,
            season=season,
            home_team=team1,
            away_team=team2,
            match_date=datetime.now() + timedelta(days=1),
            status="SCHEDULED",
            odds_home=2.10,
            odds_draw=3.40,
            odds_away=3.20,
        )

        # Finished match
        finished_match = Match(
            tournament=tournament,
            season=season,
            home_team=team2,
            away_team=team3,
            match_date=datetime.now() - timedelta(days=1),
            home_score=2,
            away_score=1,
            status="FINISHED",
        )

        # Another finished match
        finished_match2 = Match(
            tournament=tournament,
            season=season,
            home_team=team1,
            away_team=team3,
            match_date=datetime.now() - timedelta(days=2),
            home_score=0,
            away_score=0,
            status="FINISHED",
        )

        test_session.add_all(
            [
                tournament,
                season,
                team1,
                team2,
                team3,
                scheduled_match,
                finished_match,
                finished_match2,
            ]
        )
        test_session.commit()
        return season, team1, scheduled_match, finished_match, finished_match2

    def test_get_season_matches(self, test_client, test_session) -> None:
        """Test getting matches for a season."""
        season, team1, scheduled, finished, finished2 = self.setup_season_data(
            test_session
        )

        response = test_client.get(f"/api/v1/seasons/{season.id}/matches")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        # Should be sorted by match_date
        assert data[0]["id"] == finished2.id

    def test_get_season_matches_not_found(self, test_client) -> None:
        """Test getting matches for non-existent season."""
        response = test_client.get("/api/v1/seasons/999/matches")
        assert response.status_code == 404
        assert "Season 999 not found" in response.json()["detail"]

    def test_get_season_matches_filter_by_status(
        self, test_client, test_session
    ) -> None:
        """Test filtering season matches by status."""
        season, team1, scheduled, finished, finished2 = self.setup_season_data(
            test_session
        )

        # Filter by SCHEDULED
        response = test_client.get(
            f"/api/v1/seasons/{season.id}/matches?status=SCHEDULED"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["status"] == "SCHEDULED"

        # Filter by FINISHED
        response = test_client.get(
            f"/api/v1/seasons/{season.id}/matches?status=FINISHED"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        for match in data:
            assert match["status"] == "FINISHED"

    def test_get_season_matches_invalid_status(self, test_client, test_session) -> None:
        """Test that invalid status filter returns error."""
        season, team1, scheduled, finished, finished2 = self.setup_season_data(
            test_session
        )

        response = test_client.get(
            f"/api/v1/seasons/{season.id}/matches?status=INVALID"
        )
        assert response.status_code == 400
        assert "Invalid status" in response.json()["detail"]

    def test_get_season_matches_filter_by_team(self, test_client, test_session) -> None:
        """Test filtering season matches by team ID."""
        season, team1, scheduled, finished, finished2 = self.setup_season_data(
            test_session
        )

        response = test_client.get(
            f"/api/v1/seasons/{season.id}/matches?team_id={team1.id}"
        )
        assert response.status_code == 200
        data = response.json()
        # team1 is in scheduled_match (home) and finished_match2 (home)
        assert len(data) == 2

    def test_get_season_matches_pagination(self, test_client, test_session) -> None:
        """Test pagination with limit and offset."""
        season, team1, scheduled, finished, finished2 = self.setup_season_data(
            test_session
        )

        # Test limit
        response = test_client.get(f"/api/v1/seasons/{season.id}/matches?limit=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

        # Test offset
        response = test_client.get(
            f"/api/v1/seasons/{season.id}/matches?limit=2&offset=2"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers(self, test_client) -> None:
        """Test that CORS headers are set."""
        response = test_client.options(
            "/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # CORS should be configured to allow requests
        assert response.status_code in [200, 204]
