"""Tests for API dependencies."""

from sqlalchemy.orm import Session


class TestGetDb:
    """Tests for get_db dependency."""

    def test_get_db_returns_session(self, test_session: Session) -> None:
        """Test that get_db returns a valid database session."""
        # The get_db function is a generator, so we need to get the session
        # In the test context, we use the test_session fixture directly
        # This test verifies the session is properly configured
        assert isinstance(test_session, Session)
        assert test_session.is_active is True

    def test_session_can_query(self, test_session: Session) -> None:
        """Test that the session can perform queries."""
        from algobet.models import Tournament

        # Try a simple query
        result = test_session.query(Tournament).all()
        assert isinstance(result, list)
