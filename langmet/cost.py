"""Cost and token-budget analytics for completion events.

Prices are expressed in USD per 1,000 tokens, split into ``input`` (prompt) and
``output`` (completion) rates. Provide your own table to match your contracts;
the bundled :data:`DEFAULT_PRICE_TABLE` is a convenience starting point and is
not guaranteed to be current.
"""

from datetime import datetime
from typing import Dict, Iterable, Mapping, Optional

from .models import CompletionEvent

# USD per 1,000 tokens. Keys are matched case-insensitively against event.model.
DEFAULT_PRICE_TABLE: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
}


def _lookup_price(model: Optional[str], price_table: Mapping[str, Mapping[str, float]]):
    if not model:
        return None
    model_lower = model.lower()
    if model_lower in price_table:
        return price_table[model_lower]
    # Allow prefix matches like "gpt-4o-2024-05-13" -> "gpt-4o".
    for key, price in price_table.items():
        if model_lower.startswith(key):
            return price
    return None


def _event_cost(event: CompletionEvent, price: Mapping[str, float]) -> Optional[float]:
    input_rate = price.get("input", 0.0)
    output_rate = price.get("output", 0.0)

    if event.prompt_tokens is not None or event.completion_tokens is not None:
        prompt = event.prompt_tokens or 0
        completion = event.completion_tokens or 0
        return (prompt * input_rate + completion * output_rate) / 1000.0

    if event.tokens_total:
        blended_rate = (input_rate + output_rate) / 2.0
        return event.tokens_total * blended_rate / 1000.0

    return None


def compute_cost_metrics(
    events: Iterable[CompletionEvent],
    price_table: Optional[Mapping[str, Mapping[str, float]]] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Dict:
    """Compute cost metrics from completion events using a per-model price table.

    Events whose model is absent from ``price_table`` are counted under
    ``unpriced_completions`` and excluded from cost totals.
    """
    table = {k.lower(): v for k, v in (price_table or DEFAULT_PRICE_TABLE).items()}
    events_list = list(events)
    if not events_list:
        return empty_cost_metrics(start_date=start_date, end_date=end_date)

    total_cost = 0.0
    priced_count = 0
    unpriced_count = 0
    by_model: Dict[str, Dict] = {}
    by_provider: Dict[str, Dict] = {}

    for event in events_list:
        price = _lookup_price(event.model, table)
        if price is None:
            unpriced_count += 1
            continue

        cost = _event_cost(event, price)
        if cost is None:
            unpriced_count += 1
            continue

        priced_count += 1
        total_cost += cost

        model_key = event.model or "unknown"
        model_bucket = by_model.setdefault(model_key, {"count": 0, "cost_usd": 0.0, "tokens": 0})
        model_bucket["count"] += 1
        model_bucket["cost_usd"] += cost
        model_bucket["tokens"] += event.tokens_total or 0

        provider_key = event.provider or "unknown"
        provider_bucket = by_provider.setdefault(
            provider_key, {"count": 0, "cost_usd": 0.0, "tokens": 0}
        )
        provider_bucket["count"] += 1
        provider_bucket["cost_usd"] += cost
        provider_bucket["tokens"] += event.tokens_total or 0

    return {
        "period": {
            "start": start_date.isoformat() if start_date else None,
            "end": end_date.isoformat() if end_date else None,
        },
        "overview": {
            "total_cost_usd": total_cost,
            "priced_completions": priced_count,
            "unpriced_completions": unpriced_count,
            "avg_cost_per_completion_usd": total_cost / priced_count if priced_count else 0.0,
        },
        "by_model": by_model,
        "by_provider": by_provider,
    }


def empty_cost_metrics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Dict:
    """Return empty cost metrics structure."""
    return {
        "period": {
            "start": start_date.isoformat() if start_date else None,
            "end": end_date.isoformat() if end_date else None,
        },
        "overview": {
            "total_cost_usd": 0.0,
            "priced_completions": 0,
            "unpriced_completions": 0,
            "avg_cost_per_completion_usd": 0.0,
        },
        "by_model": {},
        "by_provider": {},
    }
