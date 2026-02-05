"""FastAPI dependencies for database sessions and other shared resources."""

from collections.abc import Generator

from sqlalchemy.orm import Session

from algobet.database import session_scope


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency for request-scoped database sessions.

    Yields a database session that will be automatically committed or rolled back
    at the end of the request via the session_scope context manager.

    Usage:
        @router.get("/matches")
        def get_matches(db: Session = Depends(get_db)):
            return db.query(Match).all()
    """
    with session_scope() as session:
        yield session
