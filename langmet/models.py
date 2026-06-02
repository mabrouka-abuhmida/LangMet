"""Core domain models used by the analytics engine."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence


@dataclass(frozen=True)
class CompletionEvent:
    """A single completion invocation.

    ``prompt_tokens`` and ``completion_tokens`` are optional. When present they
    enable accurate input/output cost accounting; otherwise cost analytics fall
    back to a blended rate applied to ``tokens_total``.
    """

    provider: str
    model: Optional[str]
    latency_ms: Optional[float]
    tokens_total: Optional[int]
    error_message: Optional[str]
    created_at: datetime
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


@dataclass(frozen=True)
class RagEvent:
    """A single RAG pipeline execution event."""

    top_k: Optional[int]
    top_n: Optional[int]
    retrieval_scores: Sequence[float]
    rerank_scores: Sequence[float]
    retrieval_latency_ms: Optional[float]
    rerank_latency_ms: Optional[float]
    created_at: datetime


@dataclass(frozen=True)
class CitationMessageEvent:
    """Assistant message with a pre-aggregated evidence count."""

    message_id: str
    evidence_count: int
    created_at: datetime


@dataclass(frozen=True)
class RagaEvaluationEvent:
    """Pre-computed RAGAS scores for a single RAG query."""

    query_id: str
    faithfulness: Optional[float]
    answer_relevancy: Optional[float]
    context_precision: Optional[float]
    context_recall: Optional[float]
    context_relevancy: Optional[float]
    answer_correctness: Optional[float]
    answer_similarity: Optional[float]
    created_at: datetime
