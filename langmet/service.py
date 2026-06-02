"""Application service orchestrating repositories and pure analytics."""

from datetime import datetime, timedelta
from typing import Dict, Optional

from .analytics import (
    compute_citation_coverage,
    compute_operational_metrics,
    compute_rag_metrics,
    compute_raga_metrics,
)
from .ports import MetricsRepository


class AnalyticsService:
    """Facade service that exposes metric methods independent of web frameworks."""

    def __init__(self, repo: MetricsRepository):
        self.repo = repo

    def get_operational_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict:
        start, end = _normalize_period(start_date, end_date)
        events = self.repo.fetch_completion_events(start, end)
        return compute_operational_metrics(events, start_date=start, end_date=end)

    def get_rag_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict:
        start, end = _normalize_period(start_date, end_date)
        events = self.repo.fetch_rag_events(start, end)
        return compute_rag_metrics(events, start_date=start, end_date=end)

    def get_citation_coverage(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict:
        start, end = _normalize_period(start_date, end_date)
        events = self.repo.fetch_citation_message_events(start, end)
        return compute_citation_coverage(events)

    def get_raga_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict:
        start, end = _normalize_period(start_date, end_date)
        events = self.repo.fetch_raga_evaluation_events(start, end)
        return compute_raga_metrics(events, start_date=start, end_date=end)


def _normalize_period(
    start_date: Optional[datetime],
    end_date: Optional[datetime],
) -> tuple[datetime, datetime]:
    if end_date is None:
        end_date = datetime.utcnow()
    if start_date is None:
        start_date = end_date - timedelta(days=7)
    return start_date, end_date
