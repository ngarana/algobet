"""Scheduling-related database models."""

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from algobet.infrastructure.models import Base


class ScheduledTask(Base):
    """Represents a scheduled task for automated scraping or predictions."""

    __tablename__ = "scheduled_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    task_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # "scrape_upcoming", "scrape_results", "predict", etc.
    cron_expression: Mapped[str] = mapped_column(String(100), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Task parameters stored as JSON
    parameters: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict
    )

    # Metadata
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    # Relationships
    executions: Mapped[list["TaskExecution"]] = relationship(
        back_populates="task", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<ScheduledTask(id={self.id}, name='{self.name}', "
            f"type='{self.task_type}', cron='{self.cron_expression}', "
            f"is_active={self.is_active})>"
        )


class TaskExecution(Base):
    """Records the execution history of scheduled tasks."""

    __tablename__ = "task_executions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_id: Mapped[int] = mapped_column(
        ForeignKey("scheduled_tasks.id", ondelete="CASCADE"), nullable=False
    )

    # Execution status
    status: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # "pending", "running", "completed", "failed"
    started_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Execution results
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    # Relationships
    task: Mapped["ScheduledTask"] = relationship(back_populates="executions")

    def __repr__(self) -> str:
        return (
            f"<TaskExecution(id={self.id}, task_id={self.task_id}, "
            f"status='{self.status}', started_at={self.started_at})>"
        )

    @property
    def duration(self) -> float | None:
        """Return execution duration in seconds."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
