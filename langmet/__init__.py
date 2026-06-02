"""LangMet - reusable analytics for LLM and RAG systems."""

from .analytics import (
    compute_citation_coverage,
    compute_operational_metrics,
    compute_rag_metrics,
    compute_raga_metrics,
    detect_categorical_drift,
    detect_numeric_drift,
    detect_numeric_drift_windowed,
    score_answer_correctness,
    score_answer_relevancy,
    score_answer_similarity,
    score_context_precision,
    score_context_recall,
    score_context_relevancy,
    score_faithfulness,
)
from .service import AnalyticsService

__all__ = [
    "AnalyticsService",
    "compute_operational_metrics",
    "compute_rag_metrics",
    "compute_citation_coverage",
    "compute_raga_metrics",
    "detect_numeric_drift",
    "detect_numeric_drift_windowed",
    "detect_categorical_drift",
    "score_faithfulness",
    "score_answer_relevancy",
    "score_context_precision",
    "score_context_recall",
    "score_context_relevancy",
    "score_answer_correctness",
    "score_answer_similarity",
]

__version__ = "0.2.0"
