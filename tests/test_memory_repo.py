from datetime import datetime, timedelta, timezone

from langmet.adapters.memory import InMemoryMetricsRepository
from langmet.models import CompletionEvent, RagaEvaluationEvent
from langmet.service import AnalyticsService


def _completion(created_at):
    return CompletionEvent(
        provider="openai",
        model="gpt-4o-mini",
        latency_ms=100,
        tokens_total=50,
        error_message=None,
        created_at=created_at,
    )


def test_in_memory_filters_by_period():
    now = datetime.now(timezone.utc)
    inside = _completion(now - timedelta(days=1))
    outside = _completion(now - timedelta(days=30))
    repo = InMemoryMetricsRepository(completion_events=[inside, outside])

    events = repo.fetch_completion_events(now - timedelta(days=7), now)
    assert events == [inside]


def test_in_memory_with_service():
    now = datetime.now(timezone.utc)
    repo = InMemoryMetricsRepository(
        completion_events=[_completion(now)],
        raga_events=[
            RagaEvaluationEvent(
                query_id="q1",
                faithfulness=0.9,
                answer_relevancy=0.8,
                context_precision=None,
                context_recall=None,
                context_relevancy=0.7,
                answer_correctness=None,
                answer_similarity=None,
                created_at=now,
            )
        ],
    )
    service = AnalyticsService(repo)
    op = service.get_operational_metrics(now - timedelta(days=1), now + timedelta(days=1))
    raga = service.get_raga_metrics(now - timedelta(days=1), now + timedelta(days=1))
    cost = service.get_cost_metrics(now - timedelta(days=1), now + timedelta(days=1))

    assert op["overview"]["total_completions"] == 1
    assert raga["overview"]["total_evaluations"] == 1
    assert cost["overview"]["priced_completions"] == 1
