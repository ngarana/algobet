"""Scraping API orchestration facade."""

from typing import Any


class ScrapingOrchestrator:
    """Own scraping router use-case entry points during router decomposition."""

    def run_scraping_job(self, *args: Any, **kwargs: Any) -> Any:
        from algobet.api.routers.scraping import _run_scraping_job_impl

        return _run_scraping_job_impl(*args, **kwargs)
