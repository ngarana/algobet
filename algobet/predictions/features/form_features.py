"""Team form calculation features for match predictions."""

from datetime import datetime

from algobet.models import Match
from algobet.predictions.data.queries import MatchRepository


class FormCalculator:
    """Calculator for team form metrics based on recent match history.

    Computes various form indicators including points earned, goals scored/conceded,
    and venue-specific performance metrics.
    """

    def __init__(self, repository: MatchRepository) -> None:
        """Initialize calculator with match repository.

        Args:
            repository: Repository for accessing match data
        """
        self.repository = repository

    def calculate_recent_form(
        self, team_id: int, reference_date: datetime, n_matches: int = 5
    ) -> float:
        """Calculate average points from last N matches.

        Uses 3 points for a win, 1 for a draw, 0 for a loss.

        Args:
            team_id: ID of the team
            reference_date: Date up to which to calculate form
            n_matches: Number of recent matches to consider

        Returns:
            Average points per match (0.0 to 3.0)
        """
        matches = self.repository.get_team_matches(
            team_id=team_id, before_date=reference_date, limit=n_matches
        )

        if not matches:
            return 0.0

        total_points = 0.0
        for match in matches:
            points = self._get_points_for_team(match, team_id)
            total_points += points

        return total_points / len(matches)

    def calculate_goals_scored(
        self, team_id: int, reference_date: datetime, n_matches: int = 5
    ) -> float:
        """Calculate average goals scored in last N matches.

        Args:
            team_id: ID of the team
            reference_date: Date up to which to calculate
            n_matches: Number of recent matches to consider

        Returns:
            Average goals scored per match
        """
        matches = self.repository.get_team_matches(
            team_id=team_id, before_date=reference_date, limit=n_matches
        )

        if not matches:
            return 0.0

        total_goals = sum(self._get_goals_scored(match, team_id) for match in matches)

        return total_goals / len(matches)

    def calculate_goals_conceded(
        self, team_id: int, reference_date: datetime, n_matches: int = 5
    ) -> float:
        """Calculate average goals conceded in last N matches.

        Args:
            team_id: ID of the team
            reference_date: Date up to which to calculate
            n_matches: Number of recent matches to consider

        Returns:
            Average goals conceded per match
        """
        matches = self.repository.get_team_matches(
            team_id=team_id, before_date=reference_date, limit=n_matches
        )

        if not matches:
            return 0.0

        total_goals = sum(self._get_goals_conceded(match, team_id) for match in matches)

        return total_goals / len(matches)

    def calculate_home_away_form(
        self, team_id: int, reference_date: datetime, is_home: bool, n_matches: int = 5
    ) -> float:
        """Calculate venue-specific form (home or away).

        Args:
            team_id: ID of the team
            reference_date: Date up to which to calculate
            is_home: If True, calculate home form; otherwise away form
            n_matches: Number of recent matches to consider

        Returns:
            Average points per match at the specified venue
        """
        matches = self.repository.get_team_matches(
            team_id=team_id,
            before_date=reference_date,
            limit=n_matches,
            home_only=is_home,
            away_only=not is_home,
        )

        if not matches:
            return 0.0

        total_points = 0.0
        for match in matches:
            points = self._get_points_for_team(match, team_id)
            total_points += points

        return total_points / len(matches)

    def get_form_breakdown(
        self, team_id: int, reference_date: datetime, n_matches: int = 5
    ) -> dict[str, float]:
        """Get comprehensive form breakdown for a team.

        Args:
            team_id: ID of the team
            reference_date: Date up to which to calculate
            n_matches: Number of recent matches to consider

        Returns:
            Dictionary with form metrics including:
            - avg_points: Average points per match
            - win_rate: Percentage of matches won
            - draw_rate: Percentage of matches drawn
            - loss_rate: Percentage of matches lost
            - avg_goals_for: Average goals scored
            - avg_goals_against: Average goals conceded
        """
        matches = self.repository.get_team_matches(
            team_id=team_id, before_date=reference_date, limit=n_matches
        )

        if not matches:
            return {
                "avg_points": 0.0,
                "win_rate": 0.0,
                "draw_rate": 0.0,
                "loss_rate": 0.0,
                "avg_goals_for": 0.0,
                "avg_goals_against": 0.0,
            }

        wins = draws = losses = 0
        total_goals_for = total_goals_against = 0.0
        total_points = 0.0

        for match in matches:
            points = self._get_points_for_team(match, team_id)
            total_points += points

            if points == 3:
                wins += 1
            elif points == 1:
                draws += 1
            else:
                losses += 1

            total_goals_for += self._get_goals_scored(match, team_id)
            total_goals_against += self._get_goals_conceded(match, team_id)

        n = len(matches)
        return {
            "avg_points": total_points / n,
            "win_rate": wins / n,
            "draw_rate": draws / n,
            "loss_rate": losses / n,
            "avg_goals_for": total_goals_for / n,
            "avg_goals_against": total_goals_against / n,
        }

    def _get_points_for_team(self, match: Match, team_id: int) -> int:
        """Get points earned by a team in a match.

        Args:
            match: The match object
            team_id: ID of the team

        Returns:
            Points (3 for win, 1 for draw, 0 for loss or undefined)
        """
        home_score = match.home_score or 0
        away_score = match.away_score or 0

        if match.home_team_id == team_id:
            if home_score > away_score:
                return 3
            elif home_score == away_score:
                return 1
            else:
                return 0
        else:  # Away team
            if away_score > home_score:
                return 3
            elif away_score == home_score:
                return 1
            else:
                return 0

    def _get_goals_scored(self, match: Match, team_id: int) -> int:
        """Get goals scored by a team in a match.

        Args:
            match: The match object
            team_id: ID of the team

        Returns:
            Number of goals scored
        """
        if match.home_team_id == team_id:
            return match.home_score or 0
        else:
            return match.away_score or 0

    def _get_goals_conceded(self, match: Match, team_id: int) -> int:
        """Get goals conceded by a team in a match.

        Args:
            match: The match object
            team_id: ID of the team

        Returns:
            Number of goals conceded
        """
        if match.home_team_id == team_id:
            return match.away_score or 0
        else:
            return match.home_score or 0
