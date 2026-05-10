"""Shared service-layer primitives."""

from collections.abc import Callable, Iterable
from time import sleep
from typing import Any, TypeVar

from sqlalchemy.orm import Session

T = TypeVar("T")


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    max_retries: int = 3,
    base_delay_seconds: float = 0.25,
) -> T:
    """Run a callable with simple exponential backoff."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep(base_delay_seconds * (2 ** (attempt - 1)))


def batch_persist(
    session: Session,
    items: Iterable[dict[str, Any]],
    model_class: type[T],
) -> list[T]:
    """Create and persist many SQLAlchemy model instances in one transaction."""
    instances = [model_class(**item) for item in items]
    session.add_all(instances)
    session.commit()
    for instance in instances:
        session.refresh(instance)
    return instances
