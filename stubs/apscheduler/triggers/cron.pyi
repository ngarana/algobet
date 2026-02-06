from typing import Any

class CronTrigger:
    """Cron trigger for APScheduler."""

    @classmethod
    def from_crontab(cls, cron_expression: str) -> Any: ...
