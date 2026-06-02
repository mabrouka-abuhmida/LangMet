"""Pure analytics functions that work on generic events."""

import re
from datetime import datetime, timedelta
from math import floor
from typing import Dict, Iterable, List, Optional, Sequence

from .models import CitationMessageEvent, CompletionEvent, RagEvent, RagaEvaluationEvent


def compute_operational_metrics(
    events: Iterable[CompletionEvent],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Dict:
    """Compute operational metrics from completion events."""
    events_list = list(events)
    if not events_list:
        return empty_operational_metrics(start_date=start_date, end_date=end_date)

    total_completions = len(events_list)
    error_count = sum(1 for event in events_list if event.error_message)
    success_count = total_completions - error_count

    latencies = [event.latency_ms for event in events_list if event.latency_ms is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    latency_percentiles = _compute_percentiles(latencies)

    total_tokens = sum(event.tokens_total for event in events_list if event.tokens_total)

    by_provider: Dict[str, Dict] = {}
    for event in events_list:
        provider = event.provider or "unknown"
        if provider not in by_provider:
            by_provider[provider] = {
                "count": 0,
                "errors": 0,
                "total_tokens": 0,
                "total_latency": 0.0,
            }

        by_provider[provider]["count"] += 1
        if event.error_message:
            by_provider[provider]["errors"] += 1
        if event.tokens_total:
            by_provider[provider]["total_tokens"] += event.tokens_total
        if event.latency_ms is not None:
            by_provider[provider]["total_latency"] += event.latency_ms

    for provider in by_provider:
        provider_data = by_provider[provider]
        count = provider_data["count"]
        provider_data["avg_latency"] = provider_data["total_latency"] / count if count > 0 else 0
        provider_data["error_rate"] = provider_data["errors"] / count if count > 0 else 0

    return {
        "period": {
            "start": start_date.isoformat() if start_date else None,
            "end": end_date.isoformat() if end_date else None,
        },
        "overview": {
            "total_completions": total_completions,
            "success_count": success_count,
            "error_count": error_count,
            "error_rate": error_count / total_completions if total_completions > 0 else 0,
            "avg_latency_ms": avg_latency,
            "latency_percentiles_ms": latency_percentiles,
            "total_tokens": total_tokens,
            "avg_tokens_per_completion": total_tokens / success_count if success_count > 0 else 0,
        },
        "by_provider": by_provider,
    }


def compute_rag_metrics(
    events: Iterable[RagEvent],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Dict:
    """Compute RAG metrics from RAG events."""
    events_list = list(events)
    if not events_list:
        return empty_rag_metrics(start_date=start_date, end_date=end_date)

    total_queries = len(events_list)

    avg_top_k = sum((event.top_k or 0) for event in events_list) / total_queries if total_queries else 0
    avg_top_n = sum((event.top_n or 0) for event in events_list) / total_queries if total_queries else 0

    retrieval_latencies = [
        event.retrieval_latency_ms for event in events_list if event.retrieval_latency_ms is not None
    ]
    rerank_latencies = [event.rerank_latency_ms for event in events_list if event.rerank_latency_ms is not None]

    avg_retrieval_latency = sum(retrieval_latencies) / len(retrieval_latencies) if retrieval_latencies else 0
    avg_rerank_latency = sum(rerank_latencies) / len(rerank_latencies) if rerank_latencies else 0
    retrieval_latency_percentiles = _compute_percentiles(retrieval_latencies)
    rerank_latency_percentiles = _compute_percentiles(rerank_latencies)

    all_retrieval_scores = []
    all_rerank_scores = []
    for event in events_list:
        all_retrieval_scores.extend(event.retrieval_scores)
        all_rerank_scores.extend(event.rerank_scores)

    avg_retrieval_score = (
        sum(all_retrieval_scores) / len(all_retrieval_scores) if all_retrieval_scores else 0
    )
    avg_rerank_score = sum(all_rerank_scores) / len(all_rerank_scores) if all_rerank_scores else 0

    return {
        "period": {
            "start": start_date.isoformat() if start_date else None,
            "end": end_date.isoformat() if end_date else None,
        },
        "overview": {
            "total_queries": total_queries,
            "avg_top_k": avg_top_k,
            "avg_top_n": avg_top_n,
            "avg_retrieval_latency_ms": avg_retrieval_latency,
            "avg_rerank_latency_ms": avg_rerank_latency,
            "retrieval_latency_percentiles_ms": retrieval_latency_percentiles,
            "rerank_latency_percentiles_ms": rerank_latency_percentiles,
            "total_latency_ms": avg_retrieval_latency + avg_rerank_latency,
        },
        "scores": {
            "avg_retrieval_score": avg_retrieval_score,
            "avg_rerank_score": avg_rerank_score,
            "retrieval_score_count": len(all_retrieval_scores),
            "rerank_score_count": len(all_rerank_scores),
        },
    }


def compute_citation_coverage(events: Iterable[CitationMessageEvent]) -> Dict:
    """Compute citation coverage from assistant message/evidence events."""
    events_list = list(events)
    if not events_list:
        return empty_citation_coverage()

    total_messages = len(events_list)
    messages_with_evidence = sum(1 for event in events_list if event.evidence_count > 0)
    messages_without_evidence = total_messages - messages_with_evidence
    total_evidence = sum(event.evidence_count for event in events_list)

    return {
        "total_messages": total_messages,
        "messages_with_evidence": messages_with_evidence,
        "messages_without_evidence": messages_without_evidence,
        "citation_coverage": messages_with_evidence / total_messages if total_messages > 0 else 0.0,
        "avg_evidence_per_message": total_evidence / total_messages if total_messages > 0 else 0.0,
    }


def empty_operational_metrics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Dict:
    """Return empty operational metrics structure."""
    return {
        "period": {
            "start": start_date.isoformat() if start_date else None,
            "end": end_date.isoformat() if end_date else None,
        },
        "overview": {
            "total_completions": 0,
            "success_count": 0,
            "error_count": 0,
            "error_rate": 0.0,
            "avg_latency_ms": 0,
            "latency_percentiles_ms": _empty_percentiles(),
            "total_tokens": 0,
            "avg_tokens_per_completion": 0,
        },
        "by_provider": {},
    }


def empty_rag_metrics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Dict:
    """Return empty RAG metrics structure."""
    return {
        "period": {
            "start": start_date.isoformat() if start_date else None,
            "end": end_date.isoformat() if end_date else None,
        },
        "overview": {
            "total_queries": 0,
            "avg_top_k": 0,
            "avg_top_n": 0,
            "avg_retrieval_latency_ms": 0,
            "avg_rerank_latency_ms": 0,
            "retrieval_latency_percentiles_ms": _empty_percentiles(),
            "rerank_latency_percentiles_ms": _empty_percentiles(),
            "total_latency_ms": 0,
        },
        "scores": {
            "avg_retrieval_score": 0.0,
            "avg_rerank_score": 0.0,
            "retrieval_score_count": 0,
            "rerank_score_count": 0,
        },
    }


def empty_citation_coverage() -> Dict:
    """Return empty citation metrics structure."""
    return {
        "total_messages": 0,
        "messages_with_evidence": 0,
        "messages_without_evidence": 0,
        "citation_coverage": 0.0,
        "avg_evidence_per_message": 0.0,
    }


def detect_numeric_drift(
    baseline_values: Iterable[float],
    current_values: Iterable[float],
    psi_alert_threshold: float = 0.2,
    mean_shift_alert_threshold: float = 0.2,
) -> Dict:
    """
    Detect drift between two numeric distributions.

    Returns:
        Dict containing PSI, central tendency shifts, and alert flags.
    """
    baseline = sorted(float(value) for value in baseline_values)
    current = sorted(float(value) for value in current_values)

    if not baseline or not current:
        return {
            "status": "insufficient_data",
            "baseline_count": len(baseline),
            "current_count": len(current),
            "psi": 0.0,
            "mean_shift_ratio": 0.0,
            "median_shift_ratio": 0.0,
            "drift_detected": False,
        }

    baseline_mean = sum(baseline) / len(baseline)
    current_mean = sum(current) / len(current)
    baseline_median = _percentile(baseline, 0.5)
    current_median = _percentile(current, 0.5)

    mean_shift_ratio = _safe_relative_change(current_mean, baseline_mean)
    median_shift_ratio = _safe_relative_change(current_median, baseline_median)

    psi = _population_stability_index(baseline, current, bins=10)

    psi_alert = psi >= psi_alert_threshold
    mean_shift_alert = abs(mean_shift_ratio) >= mean_shift_alert_threshold
    drift_detected = psi_alert or mean_shift_alert

    return {
        "status": "ok",
        "baseline_count": len(baseline),
        "current_count": len(current),
        "psi": psi,
        "mean_shift_ratio": mean_shift_ratio,
        "median_shift_ratio": median_shift_ratio,
        "baseline_summary": {
            "mean": baseline_mean,
            "median": baseline_median,
            "percentiles": _compute_percentiles(baseline),
        },
        "current_summary": {
            "mean": current_mean,
            "median": current_median,
            "percentiles": _compute_percentiles(current),
        },
        "alerts": {
            "psi_alert": psi_alert,
            "mean_shift_alert": mean_shift_alert,
        },
        "thresholds": {
            "psi_alert_threshold": psi_alert_threshold,
            "mean_shift_alert_threshold": mean_shift_alert_threshold,
        },
        "drift_detected": drift_detected,
    }


def detect_numeric_drift_windowed(
    observations: Iterable[tuple[datetime, float]],
    current_window: timedelta = timedelta(hours=1),
    baseline_window: timedelta = timedelta(days=7),
    reference_time: Optional[datetime] = None,
    min_samples_per_window: int = 20,
    psi_alert_threshold: float = 0.2,
    mean_shift_alert_threshold: float = 0.2,
) -> Dict:
    """
    Detect numeric drift by auto-splitting one observation stream into two windows.

    Example: compare last 1h (current) against trailing 7d before that (baseline).
    """
    points = [(timestamp, float(value)) for timestamp, value in observations]
    if not points:
        return {
            "status": "insufficient_data",
            "baseline_count": 0,
            "current_count": 0,
            "drift_detected": False,
        }

    if reference_time is None:
        reference_time = max(timestamp for timestamp, _ in points)

    current_start = reference_time - current_window
    baseline_start = current_start - baseline_window

    baseline_values = [
        value
        for timestamp, value in points
        if baseline_start < timestamp <= current_start
    ]
    current_values = [
        value
        for timestamp, value in points
        if current_start < timestamp <= reference_time
    ]

    drift = detect_numeric_drift(
        baseline_values=baseline_values,
        current_values=current_values,
        psi_alert_threshold=psi_alert_threshold,
        mean_shift_alert_threshold=mean_shift_alert_threshold,
    )

    if (
        len(baseline_values) < min_samples_per_window
        or len(current_values) < min_samples_per_window
    ):
        drift["status"] = "insufficient_data"
        drift["drift_detected"] = False

    drift["windows"] = {
        "reference_time": reference_time.isoformat(),
        "baseline_start": baseline_start.isoformat(),
        "baseline_end": current_start.isoformat(),
        "current_start": current_start.isoformat(),
        "current_end": reference_time.isoformat(),
        "baseline_window_seconds": int(baseline_window.total_seconds()),
        "current_window_seconds": int(current_window.total_seconds()),
        "min_samples_per_window": min_samples_per_window,
    }
    return drift


def detect_categorical_drift(
    baseline_labels: Iterable[str],
    current_labels: Iterable[str],
    tvd_alert_threshold: float = 0.15,
) -> Dict:
    """
    Detect drift between two categorical distributions.

    TVD (total variation distance) is in [0, 1], where 0 means identical distributions.
    """
    baseline = list(baseline_labels)
    current = list(current_labels)
    if not baseline or not current:
        return {
            "status": "insufficient_data",
            "baseline_count": len(baseline),
            "current_count": len(current),
            "tvd": 0.0,
            "drift_detected": False,
        }

    baseline_counts = _count_labels(baseline)
    current_counts = _count_labels(current)
    categories = sorted(set(baseline_counts) | set(current_counts))

    baseline_total = len(baseline)
    current_total = len(current)

    baseline_dist = {cat: baseline_counts.get(cat, 0) / baseline_total for cat in categories}
    current_dist = {cat: current_counts.get(cat, 0) / current_total for cat in categories}

    tvd = 0.5 * sum(abs(current_dist[cat] - baseline_dist[cat]) for cat in categories)
    drift_detected = tvd >= tvd_alert_threshold

    return {
        "status": "ok",
        "baseline_count": baseline_total,
        "current_count": current_total,
        "categories": categories,
        "baseline_distribution": baseline_dist,
        "current_distribution": current_dist,
        "tvd": tvd,
        "thresholds": {"tvd_alert_threshold": tvd_alert_threshold},
        "alerts": {"tvd_alert": drift_detected},
        "drift_detected": drift_detected,
    }


def _compute_percentiles(values: Iterable[float]) -> Dict[str, float]:
    points = [50, 90, 95, 99]
    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        return _empty_percentiles()

    return {f"p{point}": _percentile(sorted_values, point / 100) for point in points}


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = quantile * (len(sorted_values) - 1)
    lower_index = floor(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = position - lower_index
    return lower_value + (upper_value - lower_value) * weight


def _empty_percentiles() -> Dict[str, float]:
    return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}


def _safe_relative_change(current: float, baseline: float) -> float:
    epsilon = 1e-12
    denominator = abs(baseline) if abs(baseline) > epsilon else 1.0
    return (current - baseline) / denominator


def _population_stability_index(
    baseline_sorted: list[float],
    current_sorted: list[float],
    bins: int = 10,
) -> float:
    if not baseline_sorted or not current_sorted:
        return 0.0

    edges = [_percentile(baseline_sorted, i / bins) for i in range(1, bins)]

    baseline_bucket_counts = _bucket_counts(baseline_sorted, edges)
    current_bucket_counts = _bucket_counts(current_sorted, edges)

    baseline_total = len(baseline_sorted)
    current_total = len(current_sorted)
    epsilon = 1e-8

    psi = 0.0
    for i in range(len(baseline_bucket_counts)):
        baseline_pct = max(baseline_bucket_counts[i] / baseline_total, epsilon)
        current_pct = max(current_bucket_counts[i] / current_total, epsilon)
        psi += (current_pct - baseline_pct) * _safe_log(current_pct / baseline_pct)
    return psi


def _bucket_counts(values_sorted: list[float], edges: list[float]) -> list[int]:
    counts = [0] * (len(edges) + 1)
    for value in values_sorted:
        bucket_index = 0
        while bucket_index < len(edges) and value > edges[bucket_index]:
            bucket_index += 1
        counts[bucket_index] += 1
    return counts


def _count_labels(values: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        key = value if value else "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _safe_log(value: float) -> float:
    # Local import keeps optional math use minimal in paths that do not need drift.
    from math import log

    return log(value)


# ---------------------------------------------------------------------------
# RAGAS — Retrieval-Augmented Generation Assessment metrics
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an the and or but in on at to for of with is are was were be been being "
    "have has had do does did will would could should may might shall can "
    "it its it's that this these those i me my we our you your he she him her "
    "they them their what which who whom when where why how all any some no not "
    "from by as if about into through during before after above below between "
    "into through under since until while".split()
)


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, remove stopwords, return token list."""
    tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _token_set(text: str) -> set:
    return set(_tokenize(text))


def _f1_token_overlap(text_a: str, text_b: str) -> float:
    """Token-level F1 between two texts."""
    a_tokens = _tokenize(text_a)
    b_tokens = _tokenize(text_b)
    if not a_tokens or not b_tokens:
        return 0.0
    a_set = set(a_tokens)
    b_set = set(b_tokens)
    common = a_set & b_set
    precision = len(common) / len(a_set)
    recall = len(common) / len(b_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_faithfulness(answer: str, contexts: Sequence[str]) -> float:
    """Estimate faithfulness: fraction of answer tokens grounded in the contexts.

    Returns a value in [0, 1]. Higher means the answer is better supported by
    the retrieved passages. Uses token-overlap as a lightweight proxy for NLI.
    """
    answer_tokens = _token_set(answer)
    if not answer_tokens:
        return 0.0
    context_tokens: set = set()
    for ctx in contexts:
        context_tokens |= _token_set(ctx)
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def score_answer_relevancy(question: str, answer: str) -> float:
    """Estimate answer relevancy: how much of the question is addressed by the answer.

    Returns a value in [0, 1].
    """
    question_tokens = _token_set(question)
    if not question_tokens:
        return 0.0
    answer_tokens = _token_set(answer)
    return len(question_tokens & answer_tokens) / len(question_tokens)


def score_context_precision(contexts: Sequence[str], ground_truth: str) -> float:
    """Estimate context precision (signal-to-noise) using position-weighted precision.

    Each context is scored by its token overlap with the ground truth; then
    precision@k is computed as a weighted average that rewards relevant
    contexts appearing earlier in the ranking.

    Returns a value in [0, 1].
    """
    if not contexts or not ground_truth.strip():
        return 0.0

    gt_tokens = _token_set(ground_truth)
    if not gt_tokens:
        return 0.0

    relevance_threshold = 0.1
    relevant_flags = []
    for ctx in contexts:
        ctx_tokens = _token_set(ctx)
        overlap = len(gt_tokens & ctx_tokens) / len(gt_tokens) if gt_tokens else 0.0
        relevant_flags.append(overlap >= relevance_threshold)

    if not any(relevant_flags):
        return 0.0

    cumulative_precision = 0.0
    relevant_seen = 0
    for k, is_relevant in enumerate(relevant_flags, start=1):
        if is_relevant:
            relevant_seen += 1
            cumulative_precision += relevant_seen / k

    total_relevant = sum(relevant_flags)
    return cumulative_precision / total_relevant if total_relevant > 0 else 0.0


def score_context_recall(contexts: Sequence[str], ground_truth: str) -> float:
    """Estimate context recall: how much of the ground truth is covered by the contexts.

    Returns a value in [0, 1].
    """
    gt_tokens = _token_set(ground_truth)
    if not gt_tokens:
        return 0.0
    context_tokens: set = set()
    for ctx in contexts:
        context_tokens |= _token_set(ctx)
    return len(gt_tokens & context_tokens) / len(gt_tokens)


def score_context_relevancy(question: str, contexts: Sequence[str]) -> float:
    """Estimate context relevancy: how relevant the retrieved contexts are to the question.

    Returns a value in [0, 1].
    """
    question_tokens = _token_set(question)
    if not question_tokens or not contexts:
        return 0.0
    context_tokens: set = set()
    for ctx in contexts:
        context_tokens |= _token_set(ctx)
    if not context_tokens:
        return 0.0
    return len(question_tokens & context_tokens) / len(question_tokens)


def score_answer_correctness(answer: str, ground_truth: str) -> float:
    """Estimate answer correctness via token-level F1 against a ground truth answer.

    Returns a value in [0, 1].
    """
    return _f1_token_overlap(answer, ground_truth)


def score_answer_similarity(answer: str, ground_truth: str) -> float:
    """Estimate answer semantic similarity via Jaccard coefficient on tokens.

    Returns a value in [0, 1].
    """
    a_tokens = _token_set(answer)
    b_tokens = _token_set(ground_truth)
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def compute_raga_metrics(
    events: Iterable[RagaEvaluationEvent],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Dict:
    """Aggregate RAGAS evaluation scores from a collection of evaluation events.

    Each field is aggregated separately; None values are excluded from averages so
    that partial evaluations (e.g. no ground truth ⇒ no context_recall) still
    produce valid results for the other metrics.
    """
    events_list = list(events)
    if not events_list:
        return empty_raga_metrics(start_date=start_date, end_date=end_date)

    def _avg(field: str) -> Optional[float]:
        values = [getattr(e, field) for e in events_list if getattr(e, field) is not None]
        return sum(values) / len(values) if values else None

    def _count(field: str) -> int:
        return sum(1 for e in events_list if getattr(e, field) is not None)

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "context_relevancy",
        "answer_correctness",
        "answer_similarity",
    ]

    scores = {name: _avg(name) for name in metric_names}
    counts = {name: _count(name) for name in metric_names}

    # Overall RAGA score: mean of all available metric averages
    available = [v for v in scores.values() if v is not None]
    overall_score = sum(available) / len(available) if available else None

    return {
        "period": {
            "start": start_date.isoformat() if start_date else None,
            "end": end_date.isoformat() if end_date else None,
        },
        "overview": {
            "total_evaluations": len(events_list),
            "overall_score": overall_score,
        },
        "scores": scores,
        "evaluation_counts": counts,
    }


def empty_raga_metrics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Dict:
    """Return empty RAGA metrics structure."""
    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "context_relevancy",
        "answer_correctness",
        "answer_similarity",
    ]
    return {
        "period": {
            "start": start_date.isoformat() if start_date else None,
            "end": end_date.isoformat() if end_date else None,
        },
        "overview": {
            "total_evaluations": 0,
            "overall_score": None,
        },
        "scores": {name: None for name in metric_names},
        "evaluation_counts": {name: 0 for name in metric_names},
    }
