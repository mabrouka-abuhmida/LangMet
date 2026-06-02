"""In-memory repository adapter for tests, demos, and small batch jobs."""

from datetime import datetime
from typing import Sequence

from ..models import (
    CitationMessageEvent,
    CompletionEvent,
    RagaEvaluationEvent,
    RagEvent,
)


def _within(event, start_date: datetime, end_date: datetime) -> bool:
    return start_date <= event.created_at <= end_date


class InMemoryMetricsRepository:
    """Holds event collections in memory and filters them by period.

    Useful for unit tests, the bundled demo, or running analytics over a batch
    of events without a database. Implements the ``MetricsRepository`` protocol.
    """

    def __init__(
        self,
        completion_events: Sequence[CompletionEvent] = (),
        rag_events: Sequence[RagEvent] = (),
        citation_events: Sequence[CitationMessageEvent] = (),
        raga_events: Sequence[RagaEvaluationEvent] = (),
    ):
        self.completion_events = list(completion_events)
        self.rag_events = list(rag_events)
        self.citation_events = list(citation_events)
        self.raga_events = list(raga_events)

    def fetch_completion_events(
        self, start_date: datetime, end_date: datetime
    ) -> Sequence[CompletionEvent]:
        return [e for e in self.completion_events if _within(e, start_date, end_date)]

    def fetch_rag_events(self, start_date: datetime, end_date: datetime) -> Sequence[RagEvent]:
        return [e for e in self.rag_events if _within(e, start_date, end_date)]

    def fetch_citation_message_events(
        self, start_date: datetime, end_date: datetime
    ) -> Sequence[CitationMessageEvent]:
        return [e for e in self.citation_events if _within(e, start_date, end_date)]

    def fetch_raga_evaluation_events(
        self, start_date: datetime, end_date: datetime
    ) -> Sequence[RagaEvaluationEvent]:
        return [e for e in self.raga_events if _within(e, start_date, end_date)]
