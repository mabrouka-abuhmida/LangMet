# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
