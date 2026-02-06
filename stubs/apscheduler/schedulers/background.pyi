from collections.abc import Callable
from typing import Any

class BackgroundScheduler:
    """Background scheduler for APScheduler."""

    def __init__(self) -> None: ...
    def start(self) -> None: ...
    def shutdown(self) -> None: ...
    def add_job(
        self,
        func: Callable[..., Any],
        trigger: Any,
        id: str,
        name: str,
        args: list[Any],
        replace_existing: bool = False,
    ) -> None: ...
    def remove_job(self, job_id: str) -> None: ...
    def remove_all_jobs(self) -> None: ...
