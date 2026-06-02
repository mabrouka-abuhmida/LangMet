from datetime import datetime, timezone

from langmet.cost import compute_cost_metrics
from langmet.models import CompletionEvent


def _event(model, tokens_total=None, prompt_tokens=None, completion_tokens=None, provider="openai"):
    return CompletionEvent(
        provider=provider,
        model=model,
        latency_ms=100,
        tokens_total=tokens_total,
        error_message=None,
        created_at=datetime.now(timezone.utc),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def test_cost_with_token_split():
    # gpt-4o: input 0.0025, output 0.01 per 1k
    events = [_event("gpt-4o", prompt_tokens=1000, completion_tokens=1000)]
    result = compute_cost_metrics(events)
    # 1000/1000*0.0025 + 1000/1000*0.01 = 0.0125
    assert abs(result["overview"]["total_cost_usd"] - 0.0125) < 1e-9
    assert result["overview"]["priced_completions"] == 1
    assert result["by_model"]["gpt-4o"]["count"] == 1


def test_cost_blended_when_only_total():
    # gpt-4o blended rate = (0.0025 + 0.01)/2 = 0.00625 per 1k
    events = [_event("gpt-4o", tokens_total=1000)]
    result = compute_cost_metrics(events)
    assert abs(result["overview"]["total_cost_usd"] - 0.00625) < 1e-9


def test_cost_prefix_match():
    events = [_event("gpt-4o-2024-05-13", prompt_tokens=1000, completion_tokens=0)]
    result = compute_cost_metrics(events)
    assert abs(result["overview"]["total_cost_usd"] - 0.0025) < 1e-9


def test_cost_unpriced_model():
    events = [_event("mystery-model", tokens_total=1000)]
    result = compute_cost_metrics(events)
    assert result["overview"]["total_cost_usd"] == 0.0
    assert result["overview"]["unpriced_completions"] == 1
    assert result["overview"]["priced_completions"] == 0


def test_cost_custom_price_table():
    table = {"my-llm": {"input": 1.0, "output": 2.0}}
    events = [_event("my-llm", prompt_tokens=1000, completion_tokens=1000)]
    result = compute_cost_metrics(events, price_table=table)
    assert abs(result["overview"]["total_cost_usd"] - 3.0) < 1e-9


def test_cost_empty():
    result = compute_cost_metrics([])
    assert result["overview"]["total_cost_usd"] == 0.0
    assert result["by_model"] == {}
