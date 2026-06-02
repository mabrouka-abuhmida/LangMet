"""Threshold-based alerting over computed metrics.

Consumers pass the dictionaries produced by the ``compute_*`` functions plus an
:class:`AlertThresholds` config, and receive a flat list of triggered alerts.
This centralises alerting logic that would otherwise be re-implemented by every
dashboard or monitor.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class AlertThresholds:
    """Tunable thresholds. ``None`` disables the corresponding check."""

    max_error_rate: Optional[float] = 0.05
    max_p95_latency_ms: Optional[float] = None
    max_p99_latency_ms: Optional[float] = None
    max_total_cost_usd: Optional[float] = None
    min_citation_coverage: Optional[float] = None
    min_faithfulness: Optional[float] = 0.7
    min_answer_relevancy: Optional[float] = None
    min_overall_raga: Optional[float] = None
    psi_alert_threshold: float = 0.2


def _alert(metric: str, severity: str, message: str, value, threshold) -> Dict:
    return {
        "metric": metric,
        "severity": severity,
        "message": message,
        "value": value,
        "threshold": threshold,
    }


def evaluate_alerts(
    operational: Optional[Dict] = None,
    rag: Optional[Dict] = None,
    citation: Optional[Dict] = None,
    cost: Optional[Dict] = None,
    raga: Optional[Dict] = None,
    drift: Optional[Dict] = None,
    thresholds: Optional[AlertThresholds] = None,
) -> Dict:
    """Evaluate thresholds against metric payloads and return triggered alerts.

    Returns a dict with ``triggered`` (bool) and ``alerts`` (list). Each metric
    argument is the dict returned by the matching ``compute_*`` function; pass
    only the ones you have.
    """
    thresholds = thresholds or AlertThresholds()
    alerts: List[Dict] = []

    if operational:
        overview = operational.get("overview", {})
        error_rate = overview.get("error_rate")
        if thresholds.max_error_rate is not None and error_rate is not None:
            if error_rate > thresholds.max_error_rate:
                alerts.append(
                    _alert(
                        "error_rate",
                        "critical",
                        f"Error rate {error_rate:.1%} exceeds {thresholds.max_error_rate:.1%}",
                        error_rate,
                        thresholds.max_error_rate,
                    )
                )
        percentiles = overview.get("latency_percentiles_ms", {})
        if thresholds.max_p95_latency_ms is not None:
            p95 = percentiles.get("p95")
            if p95 is not None and p95 > thresholds.max_p95_latency_ms:
                alerts.append(
                    _alert(
                        "p95_latency_ms",
                        "warning",
                        f"p95 latency {p95:.0f}ms exceeds {thresholds.max_p95_latency_ms:.0f}ms",
                        p95,
                        thresholds.max_p95_latency_ms,
                    )
                )
        if thresholds.max_p99_latency_ms is not None:
            p99 = percentiles.get("p99")
            if p99 is not None and p99 > thresholds.max_p99_latency_ms:
                alerts.append(
                    _alert(
                        "p99_latency_ms",
                        "warning",
                        f"p99 latency {p99:.0f}ms exceeds {thresholds.max_p99_latency_ms:.0f}ms",
                        p99,
                        thresholds.max_p99_latency_ms,
                    )
                )

    if cost and thresholds.max_total_cost_usd is not None:
        total = cost.get("overview", {}).get("total_cost_usd")
        if total is not None and total > thresholds.max_total_cost_usd:
            alerts.append(
                _alert(
                    "total_cost_usd",
                    "warning",
                    f"Total cost ${total:.2f} exceeds ${thresholds.max_total_cost_usd:.2f}",
                    total,
                    thresholds.max_total_cost_usd,
                )
            )

    if citation and thresholds.min_citation_coverage is not None:
        coverage = citation.get("citation_coverage")
        if coverage is not None and coverage < thresholds.min_citation_coverage:
            alerts.append(
                _alert(
                    "citation_coverage",
                    "warning",
                    f"Citation coverage {coverage:.1%} below {thresholds.min_citation_coverage:.1%}",
                    coverage,
                    thresholds.min_citation_coverage,
                )
            )

    if raga:
        scores = raga.get("scores", {})
        overview = raga.get("overview", {})
        _check_min(
            alerts, "faithfulness", scores.get("faithfulness"),
            thresholds.min_faithfulness, "critical",
        )
        _check_min(
            alerts, "answer_relevancy", scores.get("answer_relevancy"),
            thresholds.min_answer_relevancy, "warning",
        )
        _check_min(
            alerts, "overall_raga", overview.get("overall_score"),
            thresholds.min_overall_raga, "warning",
        )

    if drift and drift.get("drift_detected"):
        psi = drift.get("psi", 0.0)
        alerts.append(
            _alert(
                "drift",
                "warning",
                f"Distribution drift detected (PSI={psi:.3f})",
                psi,
                thresholds.psi_alert_threshold,
            )
        )

    return {"triggered": bool(alerts), "alerts": alerts}


def _check_min(alerts, name, value, threshold, severity):
    if threshold is not None and value is not None and value < threshold:
        alerts.append(
            _alert(
                name,
                severity,
                f"{name} {value:.2f} below minimum {threshold:.2f}",
                value,
                threshold,
            )
        )
