"""Pluggable RAGAS scorers.

The default :class:`TokenOverlapScorer` is pure-Python and dependency-free. The
optional :class:`EmbeddingScorer` uses sentence-transformers (install via the
``[embeddings]`` extra) for semantically aware similarity-style metrics.

All scorers conform to the :class:`RagaScorer` protocol, so callers can swap
implementations without changing their pipeline.
"""

from datetime import datetime, timezone
from typing import Optional, Protocol, Sequence, runtime_checkable

from . import analytics
from .models import RagaEvaluationEvent


@runtime_checkable
class RagaScorer(Protocol):
    """Interface for computing the seven RAGAS dimensions for one query."""

    def faithfulness(self, answer: str, contexts: Sequence[str]) -> float: ...

    def answer_relevancy(self, question: str, answer: str) -> float: ...

    def context_precision(self, contexts: Sequence[str], ground_truth: str) -> float: ...

    def context_recall(self, contexts: Sequence[str], ground_truth: str) -> float: ...

    def context_relevancy(self, question: str, contexts: Sequence[str]) -> float: ...

    def answer_correctness(self, answer: str, ground_truth: str) -> float: ...

    def answer_similarity(self, answer: str, ground_truth: str) -> float: ...


class TokenOverlapScorer:
    """Default scorer: thin wrapper over the pure-Python token-overlap functions."""

    def faithfulness(self, answer: str, contexts: Sequence[str]) -> float:
        return analytics.score_faithfulness(answer, contexts)

    def answer_relevancy(self, question: str, answer: str) -> float:
        return analytics.score_answer_relevancy(question, answer)

    def context_precision(self, contexts: Sequence[str], ground_truth: str) -> float:
        return analytics.score_context_precision(contexts, ground_truth)

    def context_recall(self, contexts: Sequence[str], ground_truth: str) -> float:
        return analytics.score_context_recall(contexts, ground_truth)

    def context_relevancy(self, question: str, contexts: Sequence[str]) -> float:
        return analytics.score_context_relevancy(question, contexts)

    def answer_correctness(self, answer: str, ground_truth: str) -> float:
        return analytics.score_answer_correctness(answer, ground_truth)

    def answer_similarity(self, answer: str, ground_truth: str) -> float:
        return analytics.score_answer_similarity(answer, ground_truth)


class EmbeddingScorer:
    """Embedding-backed scorer using cosine similarity of sentence embeddings.

    Requires ``sentence-transformers`` (``pip install "langmet[embeddings]"``).
    Similarity-style metrics (relevancy, recall, correctness, similarity) use
    cosine similarity; the rest delegate to token overlap, which is robust
    without a judge model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", model=None):
        if model is not None:
            self._model = model
        else:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - exercised only without extra
                raise ImportError(
                    "EmbeddingScorer requires sentence-transformers. "
                    'Install it with: pip install "langmet[embeddings]"'
                ) from exc
            self._model = SentenceTransformer(model_name)
        self._fallback = TokenOverlapScorer()

    def _cosine(self, text_a: str, text_b: str) -> float:
        if not text_a.strip() or not text_b.strip():
            return 0.0
        embeddings = self._model.encode([text_a, text_b])
        vec_a, vec_b = embeddings[0], embeddings[1]
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        # Clamp to [0, 1]; cosine can be slightly negative for unrelated text.
        return max(0.0, min(1.0, dot / (norm_a * norm_b)))

    def faithfulness(self, answer: str, contexts: Sequence[str]) -> float:
        if not contexts:
            return 0.0
        return max(self._cosine(answer, ctx) for ctx in contexts)

    def answer_relevancy(self, question: str, answer: str) -> float:
        return self._cosine(question, answer)

    def context_precision(self, contexts: Sequence[str], ground_truth: str) -> float:
        return self._fallback.context_precision(contexts, ground_truth)

    def context_recall(self, contexts: Sequence[str], ground_truth: str) -> float:
        if not contexts:
            return 0.0
        return max(self._cosine(ground_truth, ctx) for ctx in contexts)

    def context_relevancy(self, question: str, contexts: Sequence[str]) -> float:
        if not contexts:
            return 0.0
        return max(self._cosine(question, ctx) for ctx in contexts)

    def answer_correctness(self, answer: str, ground_truth: str) -> float:
        return self._cosine(answer, ground_truth)

    def answer_similarity(self, answer: str, ground_truth: str) -> float:
        return self._cosine(answer, ground_truth)


def score_query(
    question: str,
    answer: str,
    contexts: Sequence[str],
    ground_truth: Optional[str] = None,
    query_id: str = "",
    scorer: Optional[RagaScorer] = None,
    created_at: Optional[datetime] = None,
) -> RagaEvaluationEvent:
    """Score a single query into a :class:`RagaEvaluationEvent`.

    Metrics that require a reference answer (context precision/recall, answer
    correctness/similarity) are left as ``None`` when ``ground_truth`` is absent.
    """
    scorer = scorer or TokenOverlapScorer()
    has_truth = bool(ground_truth and ground_truth.strip())

    return RagaEvaluationEvent(
        query_id=query_id,
        faithfulness=scorer.faithfulness(answer, contexts),
        answer_relevancy=scorer.answer_relevancy(question, answer),
        context_precision=scorer.context_precision(contexts, ground_truth) if has_truth else None,
        context_recall=scorer.context_recall(contexts, ground_truth) if has_truth else None,
        context_relevancy=scorer.context_relevancy(question, contexts),
        answer_correctness=scorer.answer_correctness(answer, ground_truth) if has_truth else None,
        answer_similarity=scorer.answer_similarity(answer, ground_truth) if has_truth else None,
        created_at=created_at or datetime.now(timezone.utc),
    )
