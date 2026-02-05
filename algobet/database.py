"""Database connection and session management."""

import os
from collections.abc import Generator
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from algobet.models import Base

# Load environment variables
load_dotenv()


def get_db_url() -> str:
    """Get database URL from environment variables."""
    user = os.getenv("POSTGRES_USER", "algobet")
    password = os.getenv("POSTGRES_PASSWORD", "password")
    db = os.getenv("POSTGRES_DB", "football")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def create_db_engine() -> Engine:
    """Create database engine."""
    return create_engine(get_db_url())


def init_db() -> None:
    """Initialize the database tables."""
    engine = create_db_engine()
    Base.metadata.create_all(engine)
    print("Database tables created successfully.")


def get_session() -> Session:
    """Get a new database session."""
    engine = create_db_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
