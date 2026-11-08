"""llm.py: retry with backoff, circuit breaker, fallback, cost tracking.  No network."""
from __future__ import annotations

import dataclasses

import pytest

from backend.agents.llm import (
    CircuitBreaker, CircuitOpen, FakeLLM, LLMClient, LLMConfig, LLMError, LLMResponse,
    ProviderUnavailable, UsageLedger, build_client,
)


def make_config(**over) -> LLMConfig:
    base = LLMConfig(
        provider="fake", base_url=None, model=None, api_key=None,
        fallback_base_url=None, fallback_model=None, fallback_api_key=None,
        timeout_s=5, max_attempts=3, backoff_base_s=0.5, backoff_max_s=8,
        breaker_threshold=3, breaker_reset_s=30, cost_per_1k_prompt=1.0, cost_per_1k_completion=2.0,
        fake_scenario="default",
    )
    return dataclasses.replace(base, **over)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_success_records_tokens_cost_and_step():
    ledger = UsageLedger()
    llm = LLMClient(FakeLLM([LLMResponse(content="ok", prompt_tokens=1000, completion_tokens=500)]), make_config())
    resp = llm.call("analysis", [{"role": "user", "content": "hi"}], ledger=ledger)
    assert resp.content == "ok"
    (rec,) = ledger.records
    assert rec.step == "analysis" and rec.prompt_tokens == 1000 and rec.completion_tokens == 500
    assert rec.cost_usd == pytest.approx(1.0 + 1.0)           # 1k * $1 + 0.5k * $2
    assert ledger.by_step()["analysis"]["calls"] == 1


def test_transient_failures_are_retried_with_exponential_backoff():
    sleeps: list[float] = []
    fake = FakeLLM([TimeoutError(), ConnectionError(), "ok"])
    llm = LLMClient(fake, make_config(breaker_threshold=10), sleep=sleeps.append)
    assert llm.call("analysis", [{"role": "user", "content": "x"}]).content == "ok"
    assert len(fake.calls) == 3
    # base 0.5 then 1.0, plus jitter in [0, 0.5]
    assert 0.5 <= sleeps[0] <= 1.0 and 1.0 <= sleeps[1] <= 1.5


def test_retries_are_bounded_then_provider_unavailable():
    fake = FakeLLM([TimeoutError()] * 10)
    llm = LLMClient(fake, make_config(breaker_threshold=100), sleep=lambda s: None)
    with pytest.raises(ProviderUnavailable):
        llm.call("analysis", [{"role": "user", "content": "x"}])
    assert len(fake.calls) == 3                                  # max_attempts


def test_non_transient_error_is_not_retried():
    fake = FakeLLM([LLMError("400 bad request"), "never reached"])
    llm = LLMClient(fake, make_config(), sleep=lambda s: None)
    with pytest.raises(LLMError):
        llm.call("analysis", [{"role": "user", "content": "x"}])
    assert len(fake.calls) == 1


def test_circuit_breaker_opens_after_repeated_failures_and_stops_calling():
    clock = Clock()
    fake = FakeLLM([TimeoutError()] * 100)
    breaker = CircuitBreaker(threshold=3, reset_s=30, clock=clock)
    llm = LLMClient(fake, make_config(), breaker=breaker, sleep=lambda s: None)
    with pytest.raises(ProviderUnavailable):
        llm.call("analysis", [{"role": "user", "content": "x"}])
    assert breaker.state == CircuitBreaker.OPEN
    n = len(fake.calls)
    with pytest.raises(CircuitOpen):                              # fails fast, provider untouched
        llm.call("analysis", [{"role": "user", "content": "x"}])
    assert len(fake.calls) == n


def test_breaker_half_open_trial_closes_or_reopens():
    clock = Clock()
    breaker = CircuitBreaker(threshold=1, reset_s=30, clock=clock)
    breaker.record_failure()
    assert breaker.state == "open" and not breaker.allow()
    clock.now = 31
    assert breaker.allow() and breaker.state == "half_open"
    assert not breaker.allow()                                    # only one trial at a time
    breaker.record_failure()
    assert breaker.state == "open"
    clock.now = 62
    assert breaker.allow()
    breaker.record_success()
    assert breaker.state == "closed" and breaker.allow()


def test_fallback_provider_answers_when_primary_is_down():
    ledger = UsageLedger()
    primary = FakeLLM([TimeoutError()] * 10, name="primary")
    fallback = FakeLLM(["from fallback"], name="fallback")
    llm = LLMClient(primary, make_config(), fallback=fallback, sleep=lambda s: None)
    assert llm.call("analysis", [{"role": "user", "content": "x"}], ledger=ledger).content == "from fallback"
    assert ledger.records[0].provider == "fallback"


def test_config_comes_only_from_environment(monkeypatch):
    for name in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(LLMError, match="missing base URL, model, API key"):
        build_client()
    monkeypatch.setenv("LLM_COST_PER_1K_PROMPT", "0.25")
    assert LLMConfig.from_env().cost_per_1k_prompt == 0.25
