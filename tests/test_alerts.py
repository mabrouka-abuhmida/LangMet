from langmet.alerts import AlertThresholds, evaluate_alerts


def test_error_rate_alert():
    operational = {"overview": {"error_rate": 0.2, "latency_percentiles_ms": {}}}
    result = evaluate_alerts(operational=operational, thresholds=AlertThresholds(max_error_rate=0.05))
    assert result["triggered"] is True
    assert any(a["metric"] == "error_rate" for a in result["alerts"])


def test_no_alert_when_within_threshold():
    operational = {"overview": {"error_rate": 0.01, "latency_percentiles_ms": {}}}
    result = evaluate_alerts(operational=operational, thresholds=AlertThresholds(max_error_rate=0.05))
    assert result["triggered"] is False
    assert result["alerts"] == []


def test_latency_alerts():
    operational = {
        "overview": {
            "error_rate": 0.0,
            "latency_percentiles_ms": {"p95": 800, "p99": 1500},
        }
    }
    thresholds = AlertThresholds(
        max_error_rate=None, max_p95_latency_ms=500, max_p99_latency_ms=1000
    )
    result = evaluate_alerts(operational=operational, thresholds=thresholds)
    metrics = {a["metric"] for a in result["alerts"]}
    assert "p95_latency_ms" in metrics
    assert "p99_latency_ms" in metrics


def test_cost_alert():
    cost = {"overview": {"total_cost_usd": 150.0}}
    result = evaluate_alerts(cost=cost, thresholds=AlertThresholds(max_total_cost_usd=100.0))
    assert any(a["metric"] == "total_cost_usd" for a in result["alerts"])


def test_faithfulness_alert():
    raga = {"scores": {"faithfulness": 0.5}, "overview": {"overall_score": 0.6}}
    result = evaluate_alerts(raga=raga, thresholds=AlertThresholds(min_faithfulness=0.7))
    assert any(a["metric"] == "faithfulness" and a["severity"] == "critical" for a in result["alerts"])


def test_citation_coverage_alert():
    citation = {"citation_coverage": 0.4}
    result = evaluate_alerts(
        citation=citation, thresholds=AlertThresholds(min_citation_coverage=0.8)
    )
    assert any(a["metric"] == "citation_coverage" for a in result["alerts"])


def test_drift_alert():
    drift = {"drift_detected": True, "psi": 0.35}
    result = evaluate_alerts(drift=drift)
    assert any(a["metric"] == "drift" for a in result["alerts"])
