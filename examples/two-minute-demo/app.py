"""Two-minute LangMet demo: FastAPI backend + static frontend."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from random import Random

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from langmet.alerts import AlertThresholds, evaluate_alerts
from langmet.analytics import (
    compute_citation_coverage,
    compute_operational_metrics,
    compute_rag_metrics,
    compute_raga_metrics,
    detect_numeric_drift_windowed,
)
from langmet.cost import compute_cost_metrics
from langmet.models import CitationMessageEvent, CompletionEvent, RagEvent, RagaEvaluationEvent

APP_DIR = Path(__file__).parent
FRONTEND_DIR = APP_DIR / "frontend"
RNG = Random(42)

app = FastAPI(title="LangMet Two-Minute Demo", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def _build_demo_data() -> dict:
    now = datetime.now(timezone.utc)

    completions = []
    for idx in range(250):
        created_at = now - timedelta(minutes=idx * 3)
        provider = "openai" if idx % 3 else "anthropic"
        model = "gpt-4o-mini" if idx % 3 else "claude-3-5-sonnet"
        latency_base = 180 if idx < 40 else 120
        latency_ms = max(20, int(RNG.gauss(latency_base, 20)))
        prompt_tokens = max(20, int(RNG.gauss(500, 120)))
        completion_tokens = max(10, int(RNG.gauss(200, 60)))
        error_message = "timeout" if idx % 37 == 0 else None
        completions.append(
            CompletionEvent(
                provider=provider,
                model=model,
                latency_ms=latency_ms,
                tokens_total=prompt_tokens + completion_tokens,
                error_message=error_message,
                created_at=created_at,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

    rag_events = []
    for idx in range(160):
        created_at = now - timedelta(minutes=idx * 5)
        rag_events.append(
            RagEvent(
                top_k=10,
                top_n=3,
                retrieval_scores=[round(RNG.uniform(0.55, 0.98), 3) for _ in range(5)],
                rerank_scores=[round(RNG.uniform(0.6, 0.99), 3) for _ in range(3)],
                retrieval_latency_ms=max(8, int(RNG.gauss(55, 12))),
                rerank_latency_ms=max(6, int(RNG.gauss(28, 8))),
                created_at=created_at,
            )
        )

    citation_events = []
    for idx in range(180):
        created_at = now - timedelta(minutes=idx * 4)
        evidence_count = 0 if idx % 5 == 0 else (1 if idx % 3 else 2)
        citation_events.append(
            CitationMessageEvent(
                message_id=f"msg-{idx}",
                evidence_count=evidence_count,
                created_at=created_at,
            )
        )

    raga_events = []
    for idx in range(120):
        created_at = now - timedelta(minutes=idx * 6)
        has_ground_truth = idx % 4 != 0
        raga_events.append(
            RagaEvaluationEvent(
                query_id=f"q-{idx}",
                faithfulness=round(RNG.uniform(0.65, 0.98), 3),
                answer_relevancy=round(RNG.uniform(0.60, 0.97), 3),
                context_precision=round(RNG.uniform(0.55, 0.95), 3) if has_ground_truth else None,
                context_recall=round(RNG.uniform(0.60, 0.96), 3) if has_ground_truth else None,
                context_relevancy=round(RNG.uniform(0.58, 0.94), 3),
                answer_correctness=round(RNG.uniform(0.62, 0.97), 3) if has_ground_truth else None,
                answer_similarity=round(RNG.uniform(0.64, 0.98), 3) if has_ground_truth else None,
                created_at=created_at,
            )
        )

    return {
        "completions": completions,
        "rag_events": rag_events,
        "citation_events": citation_events,
        "raga_events": raga_events,
    }


DEMO_DATA = _build_demo_data()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "demo": "langmet-two-minute"}


@app.get("/api/metrics")
def metrics() -> dict:
    return {
        "operational": compute_operational_metrics(DEMO_DATA["completions"]),
        "rag": compute_rag_metrics(DEMO_DATA["rag_events"]),
        "citation_coverage": compute_citation_coverage(DEMO_DATA["citation_events"]),
        "raga": compute_raga_metrics(DEMO_DATA["raga_events"]),
        "cost": compute_cost_metrics(DEMO_DATA["completions"]),
    }


@app.get("/api/raga")
def raga() -> dict:
    return compute_raga_metrics(DEMO_DATA["raga_events"])


@app.get("/api/cost")
def cost() -> dict:
    return compute_cost_metrics(DEMO_DATA["completions"])


@app.get("/api/alerts")
def alerts() -> dict:
    operational = compute_operational_metrics(DEMO_DATA["completions"])
    citation = compute_citation_coverage(DEMO_DATA["citation_events"])
    raga = compute_raga_metrics(DEMO_DATA["raga_events"])
    cost_metrics = compute_cost_metrics(DEMO_DATA["completions"])
    return evaluate_alerts(
        operational=operational,
        citation=citation,
        raga=raga,
        cost=cost_metrics,
        thresholds=AlertThresholds(
            max_error_rate=0.02,
            max_p95_latency_ms=200,
            min_faithfulness=0.7,
            min_citation_coverage=0.85,
        ),
    )


@app.get("/api/drift")
def drift() -> dict:
    observations = [
        (event.created_at, float(event.latency_ms or 0))
        for event in DEMO_DATA["completions"]
        if event.latency_ms is not None
    ]
    return detect_numeric_drift_windowed(
        observations=observations,
        current_window=timedelta(hours=1),
        baseline_window=timedelta(days=7),
        min_samples_per_window=10,
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")
