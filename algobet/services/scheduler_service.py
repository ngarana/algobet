"""Scheduler service for managing automated tasks."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from algobet.models import ScheduledTask, TaskExecution
from algobet.services.base import BaseService


@dataclass
class TaskDefinition:
    """Definition of a task that can be scheduled."""

    name: str
    task_type: str
    description: str
    default_parameters: dict[str, Any]
    execute: Callable[[Session, dict[str, Any]], dict[str, Any]]


class SchedulerService(BaseService[None]):
    """Service for managing scheduled tasks and their execution."""

    # Registry of available task types
    _task_registry: dict[str, TaskDefinition] = {}

    @classmethod
    def register_task(cls, task_def: TaskDefinition) -> None:
        """Register a task type with the scheduler."""
        cls._task_registry[task_def.task_type] = task_def

    @classmethod
    def get_task_types(cls) -> list[str]:
        """Get all registered task types."""
        return list(cls._task_registry.keys())

    @classmethod
    def get_task_definition(cls, task_type: str) -> TaskDefinition | None:
        """Get the definition for a specific task type."""
        return cls._task_registry.get(task_type)

    def create_schedule(
        self,
        name: str,
        task_type: str,
        cron_expression: str,
        parameters: dict[str, Any] | None = None,
        description: str | None = None,
        is_active: bool = True,
    ) -> ScheduledTask:
        """Create a new scheduled task.

        Args:
            name: Unique name for the scheduled task
            task_type: Type of task to execute
            cron_expression: Cron expression for scheduling
            parameters: Task-specific parameters
            description: Human-readable description
            is_active: Whether the schedule is active

        Returns:
            Created ScheduledTask object
        """
        if task_type not in self._task_registry:
            raise ValueError(f"Unknown task type: {task_type}")

        # Check if name already exists
        existing = self.session.execute(
            select(ScheduledTask).where(ScheduledTask.name == name)
        ).scalar_one_or_none()
        if existing:
            raise ValueError(f"Scheduled task with name '{name}' already exists")

        task = ScheduledTask(
            name=name,
            task_type=task_type,
            cron_expression=cron_expression,
            parameters=parameters or {},
            description=description,
            is_active=is_active,
        )
        self.session.add(task)
        self.session.flush()

        return task

    def get_schedule(self, schedule_id: int) -> ScheduledTask | None:
        """Get a scheduled task by ID."""
        return self.session.execute(
            select(ScheduledTask).where(ScheduledTask.id == schedule_id)
        ).scalar_one_or_none()

    def get_schedule_by_name(self, name: str) -> ScheduledTask | None:
        """Get a scheduled task by name."""
        return self.session.execute(
            select(ScheduledTask).where(ScheduledTask.name == name)
        ).scalar_one_or_none()

    def list_schedules(
        self,
        task_type: str | None = None,
        is_active: bool | None = None,
    ) -> list[ScheduledTask]:
        """List all scheduled tasks, optionally filtered."""
        query = select(ScheduledTask)

        if task_type:
            query = query.where(ScheduledTask.task_type == task_type)
        if is_active is not None:
            query = query.where(ScheduledTask.is_active == is_active)

        query = query.order_by(ScheduledTask.name)
        result = self.session.execute(query)
        return list(result.scalars().all())

    def update_schedule(
        self,
        schedule_id: int,
        cron_expression: str | None = None,
        parameters: dict[str, Any] | None = None,
        description: str | None = None,
        is_active: bool | None = None,
    ) -> ScheduledTask:
        """Update a scheduled task."""
        task = self.get_schedule(schedule_id)
        if not task:
            raise ValueError(f"Scheduled task with ID {schedule_id} not found")

        if cron_expression is not None:
            task.cron_expression = cron_expression
        if parameters is not None:
            task.parameters = parameters
        if description is not None:
            task.description = description
        if is_active is not None:
            task.is_active = is_active

        task.updated_at = datetime.now()
        self.session.flush()

        return task

    def delete_schedule(self, schedule_id: int) -> None:
        """Delete a scheduled task."""
        task = self.get_schedule(schedule_id)
        if not task:
            raise ValueError(f"Scheduled task with ID {schedule_id} not found")

        self.session.delete(task)

    def execute_task(
        self,
        schedule_id: int,
    ) -> TaskExecution:
        """Execute a scheduled task synchronously."""
        task = self.get_schedule(schedule_id)
        if not task:
            raise ValueError(f"Scheduled task with ID {schedule_id} not found")

        if not task.is_active:
            raise ValueError(f"Scheduled task '{task.name}' is not active")

        task_def = self.get_task_definition(task.task_type)
        if not task_def:
            raise ValueError(f"Task definition not found for type: {task.task_type}")

        # Create execution record
        execution = TaskExecution(
            task_id=task.id,
            status="running",
            started_at=datetime.now(),
        )
        self.session.add(execution)
        self.session.flush()

        try:
            # Execute the task
            result = task_def.execute(self.session, task.parameters)

            # Update execution record
            execution.status = "completed"
            execution.completed_at = datetime.now()
            execution.result = result

        except Exception as e:
            # Record error
            execution.status = "failed"
            execution.completed_at = datetime.now()
            execution.error_message = str(e)

        self.session.flush()

        return execution

    def get_execution_history(
        self,
        schedule_id: int,
        limit: int = 100,
    ) -> list[TaskExecution]:
        """Get execution history for a scheduled task."""
        task = self.get_schedule(schedule_id)
        if not task:
            raise ValueError(f"Scheduled task with ID {schedule_id} not found")

        result = self.session.execute(
            select(TaskExecution)
            .where(TaskExecution.task_id == task.id)
            .order_by(TaskExecution.started_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    def get_last_execution(self, schedule_id: int) -> TaskExecution | None:
        """Get the most recent execution for a scheduled task."""
        history = self.get_execution_history(schedule_id, limit=1)
        return history[0] if history else None

    def get_active_schedules(self) -> list[ScheduledTask]:
        """Get all active scheduled tasks."""
        return self.list_schedules(is_active=True)
