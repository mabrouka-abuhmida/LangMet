"""SQLAlchemy repository adapter for LangMet."""

import json
from datetime import datetime
from typing import Sequence

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..models import CitationMessageEvent, CompletionEvent, RagEvent, RagaEvaluationEvent


class SQLAlchemyMetricsRepository:
    """Fetches raw events from relational tables and maps them to domain models."""

    def __init__(self, db: Session):
        self.db = db

    def fetch_completion_events(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Sequence[CompletionEvent]:
        rows = self.db.execute(
            text(
                """
                SELECT provider, model, latency_ms, tokens_total, error_message, created_at
                FROM completion_logs
                WHERE created_at >= :start_date AND created_at <= :end_date
                """
            ),
            {"start_date": start_date, "end_date": end_date},
        ).fetchall()

        return [
            CompletionEvent(
                provider=row.provider or "unknown",
                model=row.model,
                latency_ms=row.latency_ms,
                tokens_total=row.tokens_total,
                error_message=row.error_message,
                created_at=row.created_at,
            )
            for row in rows
        ]

    def fetch_rag_events(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Sequence[RagEvent]:
        rows = self.db.execute(
            text(
                """
                SELECT top_k, top_n, retrieval_scores, rerank_scores,
                       retrieval_latency_ms, rerank_latency_ms, created_at
                FROM rag_logs
                WHERE created_at >= :start_date AND created_at <= :end_date
                """
            ),
            {"start_date": start_date, "end_date": end_date},
        ).fetchall()

        result: list[RagEvent] = []
        for row in rows:
            retrieval_scores = _parse_score_list(row.retrieval_scores)
            rerank_scores = _parse_score_list(row.rerank_scores)
            result.append(
                RagEvent(
                    top_k=row.top_k,
                    top_n=row.top_n,
                    retrieval_scores=retrieval_scores,
                    rerank_scores=rerank_scores,
                    retrieval_latency_ms=row.retrieval_latency_ms,
                    rerank_latency_ms=row.rerank_latency_ms,
                    created_at=row.created_at,
                )
            )
        return result

    def fetch_citation_message_events(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Sequence[CitationMessageEvent]:
        rows = self.db.execute(
            text(
                """
                SELECT m.id, COUNT(me.id) as evidence_count, m.created_at
                FROM messages m
                LEFT JOIN message_evidence me ON m.id = me.message_id
                WHERE m.role = 'assistant'
                  AND m.created_at >= :start_date
                  AND m.created_at <= :end_date
                GROUP BY m.id, m.created_at
                """
            ),
            {"start_date": start_date, "end_date": end_date},
        ).fetchall()

        return [
            CitationMessageEvent(
                message_id=str(row.id),
                evidence_count=int(row.evidence_count or 0),
                created_at=row.created_at,
            )
            for row in rows
        ]


    def fetch_raga_evaluation_events(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Sequence[RagaEvaluationEvent]:
        rows = self.db.execute(
            text(
                """
                SELECT query_id,
                       faithfulness, answer_relevancy,
                       context_precision, context_recall, context_relevancy,
                       answer_correctness, answer_similarity,
                       created_at
                FROM raga_evaluations
                WHERE created_at >= :start_date AND created_at <= :end_date
                """
            ),
            {"start_date": start_date, "end_date": end_date},
        ).fetchall()

        return [
            RagaEvaluationEvent(
                query_id=str(row.query_id),
                faithfulness=_parse_optional_float(row.faithfulness),
                answer_relevancy=_parse_optional_float(row.answer_relevancy),
                context_precision=_parse_optional_float(row.context_precision),
                context_recall=_parse_optional_float(row.context_recall),
                context_relevancy=_parse_optional_float(row.context_relevancy),
                answer_correctness=_parse_optional_float(row.answer_correctness),
                answer_similarity=_parse_optional_float(row.answer_similarity),
                created_at=row.created_at,
            )
            for row in rows
        ]


def _parse_optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_score_list(raw_scores) -> list[float]:
    if raw_scores is None:
        return []
    if isinstance(raw_scores, str):
        try:
            raw_scores = json.loads(raw_scores)
        except json.JSONDecodeError:
            return []
    if isinstance(raw_scores, list):
        values = []
        for value in raw_scores:
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        return values
    return []
