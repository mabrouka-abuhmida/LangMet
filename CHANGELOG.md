# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0]

### Added
- **Cost & token-budget metrics** (`langmet.cost`): `compute_cost_metrics` with a
  per-model `DEFAULT_PRICE_TABLE`, input/output token accounting, and a blended
  fallback for `tokens_total`. `CompletionEvent` gained optional `prompt_tokens`
  and `completion_tokens` fields. New `AnalyticsService.get_cost_metrics`.
- **Pluggable RAGAS scorers** (`langmet.scoring`): `RagaScorer` protocol,
  `TokenOverlapScorer` (default, dependency-free), `EmbeddingScorer` (optional
  `[embeddings]` extra, sentence-transformers), and `score_query` helper.
- **RAGA drift detection** (`detect_raga_drift`): track a RAGAS metric over time
  using the existing windowed drift engine.
- **Alerting layer** (`langmet.alerts`): `evaluate_alerts` + `AlertThresholds`
  for centralised threshold checks across operational, cost, citation, RAGA, and
  drift metrics.
- **In-memory adapter** (`InMemoryMetricsRepository`) for tests, demos, and batch
  jobs; SQLAlchemy adapter import is now optional/guarded.
- Packaging: `py.typed` marker (PEP 561), `CONTRIBUTING.md`, `[embeddings]` extra.
- Two-minute demo: `/api/cost` and `/api/alerts` endpoints, cost + alerts panels
  in the frontend, and realistic priced model names.

## [0.2.0]

### Added
- **RAGAS evaluation metrics** for RAG quality assessment:
  - `RagaEvaluationEvent` domain model holding per-query scores.
  - Pure-Python, dependency-free scoring functions: `score_faithfulness`,
    `score_answer_relevancy`, `score_context_precision`, `score_context_recall`,
    `score_context_relevancy`, `score_answer_correctness`, `score_answer_similarity`.
  - `compute_raga_metrics` / `empty_raga_metrics` aggregation producing per-metric
    averages, evaluation counts, and an overall RAGA score.
  - `MetricsRepository.fetch_raga_evaluation_events` port method.
  - `SQLAlchemyMetricsRepository` support for a `raga_evaluations` table.
  - `AnalyticsService.get_raga_metrics` service method.
- Two-minute demo now generates synthetic RAGAS data, exposes a `/api/raga`
  endpoint, includes RAGA scores in `/api/metrics`, and renders a "RAGAS Quality"
  section in the frontend.
- Documentation: RAGAS usage examples, SQL schema, and integration guidance in the README.

## [0.1.2]

- Initial public release with operational, RAG, and citation coverage metrics,
  drift detection (PSI / TVD / windowed), repository ports, and the SQLAlchemy adapter.
