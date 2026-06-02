from datetime import datetime, timedelta, timezone

import pytest

from langmet.analytics import detect_raga_drift
from langmet.models import RagaEvaluationEvent


def _eval(created_at, faithfulness):
    return RagaEvaluationEvent(
        query_id="q",
        faithfulness=faithfulness,
        answer_relevancy=None,
        context_precision=None,
        context_recall=None,
        context_relevancy=None,
        answer_correctness=None,
        answer_similarity=None,
        created_at=created_at,
    )


def test_detect_raga_drift_flags_degradation():
    ref = datetime(2026, 1, 8, 12, 0, tzinfo=timezone.utc)
    # baseline: high faithfulness over the prior week
    baseline = [_eval(ref - timedelta(days=2, hours=i), 0.92 + (i % 3) * 0.01) for i in range(30)]
    # current: degraded faithfulness in the last day
    current = [_eval(ref - timedelta(minutes=59 - i), 0.55 + (i % 2) * 0.01) for i in range(25)]

    result = detect_raga_drift(
        baseline + current,
        metric="faithfulness",
        reference_time=ref,
        min_samples_per_window=20,
    )

    assert result["metric"] == "faithfulness"
    assert result["status"] == "ok"
    assert result["drift_detected"] is True


def test_detect_raga_drift_rejects_unknown_metric():
    with pytest.raises(ValueError):
        detect_raga_drift([], metric="not_a_metric")


def test_detect_raga_drift_skips_none_values():
    ref = datetime(2026, 1, 8, 12, 0, tzinfo=timezone.utc)
    events = [_eval(ref, None) for _ in range(5)]
    result = detect_raga_drift(events, metric="faithfulness", reference_time=ref)
    assert result["status"] == "insufficient_data"
