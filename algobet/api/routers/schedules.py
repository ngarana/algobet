"""API router for scheduled task management."""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from algobet.api.dependencies import get_db
from algobet.models import ScheduledTask, TaskExecution
from algobet.services.scheduler_service import SchedulerService

router = APIRouter(tags=["schedules"])


# Request/Response Schemas
class ScheduleCreateRequest(BaseModel):
    """Request schema for creating a scheduled task."""

    name: str = Field(
        ...,
        description="Unique name for the scheduled task",
        min_length=1,
        max_length=255,
    )
    task_type: str = Field(..., description="Type of task to execute")
    cron_expression: str = Field(
        ..., description="Cron expression for scheduling (e.g., '0 6 * * *')"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Task-specific parameters"
    )
    description: str | None = Field(None, description="Human-readable description")
    is_active: bool = Field(True, description="Whether the schedule is active")


class ScheduleUpdateRequest(BaseModel):
    """Request schema for updating a scheduled task."""

    cron_expression: str | None = Field(None, description="Updated cron expression")
    parameters: dict[str, Any] | None = Field(
        None, description="Updated task parameters"
    )
    description: str | None = Field(None, description="Updated description")
    is_active: bool | None = Field(None, description="Updated active status")


class ScheduleResponse(BaseModel):
    """Response schema for a scheduled task."""

    id: int = Field(..., description="Unique identifier")
    name: str = Field(..., description="Task name")
    task_type: str = Field(..., description="Task type")
    cron_expression: str = Field(..., description="Cron expression")
    is_active: bool = Field(..., description="Active status")
    parameters: dict[str, Any] = Field(..., description="Task parameters")
    description: str | None = Field(None, description="Task description")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        from_attributes = True


class TaskExecutionResponse(BaseModel):
    """Response schema for a task execution."""

    id: int = Field(..., description="Execution ID")
    task_id: int = Field(..., description="Parent task ID")
    status: str = Field(..., description="Execution status")
    started_at: datetime = Field(..., description="Start timestamp")
    completed_at: datetime | None = Field(None, description="Completion timestamp")
    result: dict[str, Any] | None = Field(None, description="Execution result")
    error_message: str | None = Field(None, description="Error message if failed")

    class Config:
        from_attributes = True


class ScheduleListResponse(BaseModel):
    """Response schema for listing schedules."""

    schedules: list[ScheduleResponse] = Field(..., description="List of schedules")
    total: int = Field(..., description="Total count")


class ExecutionHistoryResponse(BaseModel):
    """Response schema for execution history."""

    executions: list[TaskExecutionResponse] = Field(
        ..., description="List of executions"
    )
    total: int = Field(..., description="Total count")


class ScheduleTaskTypesResponse(BaseModel):
    """Response schema for available task types."""

    task_types: list[str] = Field(..., description="Available task types")


@router.post("/", response_model=ScheduleResponse, status_code=status.HTTP_201_CREATED)
async def create_schedule(
    request: ScheduleCreateRequest,
    db: Session = Depends(get_db),
) -> ScheduledTask:
    """Create a new scheduled task.

    Args:
        request: Schedule creation request
        db: Database session

    Returns:
        Created scheduled task

    Raises:
        HTTPException: If task creation fails
    """
    try:
        scheduler = SchedulerService(db)
        task = scheduler.create_schedule(
            name=request.name,
            task_type=request.task_type,
            cron_expression=request.cron_expression,
            parameters=request.parameters,
            description=request.description,
            is_active=request.is_active,
        )
        return task
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create schedule: {str(e)}",
        ) from e


@router.get("/", response_model=ScheduleListResponse)
async def list_schedules(
    task_type: str | None = None,
    is_active: bool | None = None,
    db: Session = Depends(get_db),
) -> ScheduleListResponse:
    """List all scheduled tasks with optional filtering.

    Args:
        task_type: Optional filter by task type
        is_active: Optional filter by active status
        db: Database session

    Returns:
        List of scheduled tasks
    """
    scheduler = SchedulerService(db)
    schedules = scheduler.list_schedules(task_type=task_type, is_active=is_active)

    # Convert ORM models to Pydantic models
    schedule_responses = [ScheduleResponse.model_validate(s) for s in schedules]

    return ScheduleListResponse(
        schedules=schedule_responses,
        total=len(schedules),
    )


@router.get("/task-types", response_model=ScheduleTaskTypesResponse)
async def get_task_types(
    db: Session = Depends(get_db),
) -> ScheduleTaskTypesResponse:
    """Get all available task types.

    Args:
        db: Database session

    Returns:
        List of available task types
    """
    task_types = SchedulerService.get_task_types()
    return ScheduleTaskTypesResponse(task_types=task_types)


@router.get("/{schedule_id}", response_model=ScheduleResponse)
async def get_schedule(
    schedule_id: int,
    db: Session = Depends(get_db),
) -> ScheduledTask:
    """Get a scheduled task by ID.

    Args:
        schedule_id: Schedule ID
        db: Database session

    Returns:
        Scheduled task details

    Raises:
        HTTPException: If schedule not found
    """
    scheduler = SchedulerService(db)
    task = scheduler.get_schedule(schedule_id)

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Schedule with ID {schedule_id} not found",
        )

    return task


@router.patch("/{schedule_id}", response_model=ScheduleResponse)
async def update_schedule(
    schedule_id: int,
    request: ScheduleUpdateRequest,
    db: Session = Depends(get_db),
) -> ScheduledTask:
    """Update a scheduled task.

    Args:
        schedule_id: Schedule ID
        request: Update request
        db: Database session

    Returns:
        Updated scheduled task

    Raises:
        HTTPException: If schedule not found or update fails
    """
    try:
        scheduler = SchedulerService(db)
        task = scheduler.update_schedule(
            schedule_id=schedule_id,
            cron_expression=request.cron_expression,
            parameters=request.parameters,
            description=request.description,
            is_active=request.is_active,
        )
        return task
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update schedule: {str(e)}",
        ) from e


@router.delete("/{schedule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_schedule(
    schedule_id: int,
    db: Session = Depends(get_db),
) -> None:
    """Delete a scheduled task.

    Args:
        schedule_id: Schedule ID
        db: Database session

    Raises:
        HTTPException: If schedule not found
    """
    try:
        scheduler = SchedulerService(db)
        scheduler.delete_schedule(schedule_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e


@router.post("/{schedule_id}/run", response_model=TaskExecutionResponse)
async def run_schedule_now(
    schedule_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> TaskExecution:
    """Execute a scheduled task immediately.

    Args:
        schedule_id: Schedule ID
        background_tasks: FastAPI background tasks
        db: Database session

    Returns:
        Task execution record

    Raises:
        HTTPException: If schedule not found or execution fails
    """
    try:
        scheduler = SchedulerService(db)
        execution = scheduler.execute_task(schedule_id)
        return execution
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute schedule: {str(e)}",
        ) from e


@router.get("/{schedule_id}/history", response_model=ExecutionHistoryResponse)
async def get_execution_history(
    schedule_id: int,
    limit: int = Query(
        100, ge=1, le=1000, description="Maximum number of executions to return"
    ),
    db: Session = Depends(get_db),
) -> ExecutionHistoryResponse:
    """Get execution history for a scheduled task.





    Args:


        schedule_id: Schedule ID


        limit: Maximum number of executions to return


        db: Database session





    Returns:


        Execution history





    Raises:


        HTTPException: If schedule not found


    """

    try:
        scheduler = SchedulerService(db)

        executions = scheduler.get_execution_history(schedule_id, limit=limit)

        # Convert ORM models to Pydantic models

        execution_responses = [
            TaskExecutionResponse.model_validate(e) for e in executions
        ]

        return ExecutionHistoryResponse(
            executions=execution_responses,
            total=len(executions),
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e


@router.get("/{schedule_id}/last-execution", response_model=TaskExecutionResponse)
async def get_last_execution(
    schedule_id: int,
    db: Session = Depends(get_db),
) -> TaskExecution:
    """Get the most recent execution for a scheduled task.

    Args:
        schedule_id: Schedule ID
        db: Database session

    Returns:
        Last execution record

    Raises:
        HTTPException: If schedule not found or no executions exist
    """
    try:
        scheduler = SchedulerService(db)
        execution = scheduler.get_last_execution(schedule_id)

        if not execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No execution history found for schedule {schedule_id}",
            )

        return execution
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
