# LangMet

![LangMet Logo](https://raw.githubusercontent.com/mabrouka-abuhmida/LangMet/main/LangeMet-Logo.png)

**Observability and drift intelligence for LLM and RAG systems.**

LangMet provides a reusable analytics layer for monitoring operational performance, retrieval quality, and evidence coverage in AI systems.

It separates analytical computation from data access, allowing teams to compute metrics from any telemetry source — SQL databases, log streams, data warehouses, or custom repositories.

**Designed for production AI environments.**

LangMet separates **analytical intelligence** from data access so you can compute metrics from any source: SQL databases, log streams, files, or custom repositories.

# Why LangMet?

Most LLM metrics pipelines are tightly coupled to infrastructure.

# LangMet:

* isolates analytics from storage
* provides percentile-based latency monitoring
* supports windowed drift detection (short-term vs long-term baselines)
* enables evidence coverage analysis for RAG systems
* works with any data source via repository interfaces

**This makes it suitable for:**

* production monitoring
* research evaluation
* safety-critical AI systems
* regulated environments

## Features

- Pure analytics functions for:
  - Operational LLM metrics
  - RAG performance metrics
  - Citation coverage metrics
- Built-in latency percentiles (`p50`, `p90`, `p95`, `p99`) for SLO monitoring
- Drift detection for numeric and categorical signals (PSI + TVD based)
- Windowed drift baselines (compare last 1h vs trailing 7d automatically)
- Repository interface (`MetricsRepository`) for pluggable data access
- SQLAlchemy adapter for existing relational schemas
- Framework-agnostic service layer

## [[Install]]

```bash
pip install langmet
```

With SQLAlchemy adapter support:

```bash
pip install "langmet[sqlalchemy]"
```

## 2-Minute Demo

Most engineers want proof it works before reading internals. A runnable backend + frontend demo is included:

- `examples/two-minute-demo/README.md`

Quick run:

```bash
python -m pip install -e ".[fastapi]"
python -m pip install uvicorn
uvicorn app:app --app-dir examples/two-minute-demo --reload
```

Open `http://127.0.0.1:8000/`.


### Example UI Demo

![Example UI Demo - Overview](https://raw.githubusercontent.com/mabrouka-abuhmida/LangMet/main/examples/two-minute-demo/image/README/1770853195159.png)

![Example UI Demo - Drift](https://raw.githubusercontent.com/mabrouka-abuhmida/LangMet/main/examples/two-minute-demo/image/README/1770853181629.png)

## [[Quickstart]] (Pure Functions)

```python
from datetime import datetime
from langmet.models import CompletionEvent
from langmet.analytics import compute_operational_metrics

events = [
    CompletionEvent(
        provider="openai",
        model="gpt-4o-mini",
        latency_ms=320,
        tokens_total=850,
        error_message=None,
        created_at=datetime.utcnow(),
    )
]

metrics = compute_operational_metrics(events)
print(metrics["overview"]["avg_latency_ms"])
```

Drift detection:

```python
from datetime import datetime, timedelta
from langmet.analytics import (
    detect_numeric_drift,
    detect_categorical_drift,
    detect_numeric_drift_windowed,
)

latency_drift = detect_numeric_drift(
    baseline_values=[120, 130, 115, 125],
    current_values=[210, 220, 205, 215],
)

provider_drift = detect_categorical_drift(
    baseline_labels=["openai", "openai", "anthropic"],
    current_labels=["anthropic", "anthropic", "openai"],
)

# Automatic window split: last 1h vs trailing 7d.
ref = datetime.utcnow()
observations = [
    (ref - timedelta(hours=2), 120.0),
    (ref - timedelta(minutes=40), 220.0),
]
windowed_drift = detect_numeric_drift_windowed(
    observations=observations,
    reference_time=ref,
)
```

## Quickstart (Repository + Service)

```python
from datetime import datetime, timedelta
from langmet.service import AnalyticsService
from langmet.adapters.sqlalchemy_repo import SQLAlchemyMetricsRepository

repo = SQLAlchemyMetricsRepository(db_session)
service = AnalyticsService(repo)

start = datetime.utcnow() - timedelta(days=7)
end = datetime.utcnow()

all_operational = service.get_operational_metrics(start, end)
all_rag = service.get_rag_metrics(start, end)
citation = service.get_citation_coverage(start, end)
```

## Production Integration Guide

Use this path when wiring LangMet to a real service.

- ### 1) Capture telemetry events in your app

For each request or pipeline run, emit these fields:
- Completion events: `provider`, `model`, `latency_ms`, `tokens_total`, `error_message`, `created_at`
- RAG events: `top_k`, `top_n`, `retrieval_scores`, `rerank_scores`, `retrieval_latency_ms`, `rerank_latency_ms`, `created_at`
- Citation events: `message_id`, `evidence_count`, `created_at`

- ### 2) Example SQL schema (PostgreSQL)

```sql
CREATE TABLE completion_logs (
  id BIGSERIAL PRIMARY KEY,
  provider TEXT NOT NULL,
  model TEXT,
  latency_ms DOUBLE PRECISION,
  tokens_total INTEGER,
  error_message TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE rag_logs (
  id BIGSERIAL PRIMARY KEY,
  top_k INTEGER,
  top_n INTEGER,
  retrieval_scores JSONB,
  rerank_scores JSONB,
  retrieval_latency_ms DOUBLE PRECISION,
  rerank_latency_ms DOUBLE PRECISION,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE citation_events (
  id BIGSERIAL PRIMARY KEY,
  message_id TEXT NOT NULL,
  evidence_count INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_completion_logs_created_at ON completion_logs (created_at);
CREATE INDEX idx_rag_logs_created_at ON rag_logs (created_at);
CREATE INDEX idx_citation_events_created_at ON citation_events (created_at);
```

### 3) Wire repository and service

```python
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from langmet.adapters.sqlalchemy_repo import SQLAlchemyMetricsRepository
from langmet.service import AnalyticsService

def get_metrics_payload(db: Session) -> dict:
    repo = SQLAlchemyMetricsRepository(db)
    svc = AnalyticsService(repo)
    start = datetime.utcnow() - timedelta(days=7)
    end = datetime.utcnow()
    return {
        "operational": svc.get_operational_metrics(start, end),
        "rag": svc.get_rag_metrics(start, end),
        "citation_coverage": svc.get_citation_coverage(start, end),
    }
```

### 4) Expose in API

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/api/metrics")
def metrics():
    # replace with your Session management
    payload = get_metrics_payload(db_session)
    return payload
```

### 5) Add drift monitoring

```python
from datetime import timedelta
from langmet.analytics import detect_numeric_drift_windowed

drift = detect_numeric_drift_windowed(
    observations=latency_observations,  # list[(timestamp, latency_ms)]
    current_window=timedelta(hours=1),
    baseline_window=timedelta(days=7),
    min_samples_per_window=20,
)
```

### 6) Frontend contract

Your UI only needs:

- `GET /api/metrics` for overview cards and tables
- `GET /api/drift` (or drift in same payload) for alerts

Keep response keys stable:

- `operational.overview`
- `rag.overview`
- `citation_coverage`

### 7) Production checklist

- Store timestamps in UTC (`TIMESTAMPTZ`)
- Index `created_at` on telemetry tables
- Add cache TTL for dashboard polling endpoints
- Define alert thresholds for:
  - latency percentiles (`p95`, `p99`)
  - error rate
  - drift (`psi`, `tvd`)
- Add data retention policy (for example 30–90 days hot storage)

## Core Concepts

- `langmet.models`: event contracts used by analytics
- `langmet.analytics`: pure computation functions
- `langmet.ports`: repository protocol your project can implement
- `langmet.service`: orchestration facade
- `langmet.adapters`: optional infrastructure adapters

## Development

```bash
pip install -e ".[dev,sqlalchemy]"
ruff check .
pytest
python -m build
twine check dist/*
```

## License

MIT
