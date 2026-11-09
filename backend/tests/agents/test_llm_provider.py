"""Provider behaviour for a free tier (Gemini, OpenAI-compatible): 429 + Retry-After, 400 bodies,
pacing between calls, env-configurable run budget, the llm_check script.  No network: FakeLLM, or the
real OpenAI SDK over an in-memory httpx transport."""
from __future__ import annotations

import json
import logging

import httpx
import pytest

from backend.agents.llm import (
    FakeLLM, LLMClient, LLMError, OpenAICompatProvider, ProviderUnavailable, TransientError,
)
from backend.agents.llm_check import run_checks
from backend.agents.state import MAX_STEPS, Budget
from backend.agents.tools import TOOL_SPECS
from backend.tests.agents.test_llm import make_config

OK_BODY = {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
           "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "pong"}}],
           "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}}
MSG = [{"role": "user", "content": "ping"}]


class FakeClock:
    """A clock that only moves when we 'sleep'."""
    def __init__(self):
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self):
        return self.now

    def sleep(self, s: float):
        self.sleeps.append(s)
        self.now += s


def sdk_provider(responses: list[httpx.Response]) -> tuple[OpenAICompatProvider, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responses.pop(0)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatProvider("primary", "http://llm.test/v1", "m", "k", http_client=client), seen


# ------------------------------------------------------------------ 429 / Retry-After
def test_429_waits_at_least_retry_after():
    clock = FakeClock()
    llm = LLMClient(FakeLLM([TransientError("429", retry_after=7.0), "ok"]), make_config(),
                    sleep=clock.sleep, clock=clock)
    assert llm.call("analysis", MSG).content == "ok"
    assert clock.sleeps[0] >= 7.0


def test_absurd_retry_after_is_capped():
    clock = FakeClock()
    llm = LLMClient(FakeLLM([TransientError("429", retry_after=3600), "ok"]), make_config(),
                    sleep=clock.sleep, clock=clock)
    llm.call("analysis", MSG)
    assert clock.sleeps[0] == 60.0


def test_no_retry_when_the_wait_would_pass_the_run_deadline():
    clock = FakeClock()
    fake = FakeLLM([TransientError("429", retry_after=30.0), "never reached"])
    llm = LLMClient(fake, make_config(breaker_threshold=10), sleep=clock.sleep, clock=clock)
    with pytest.raises(ProviderUnavailable, match="would pass the deadline"):
        llm.call("analysis", MSG, timeout_s=5)
    assert len(fake.calls) == 1 and clock.sleeps == []


def test_real_sdk_429_retry_after_header_then_success():
    clock = FakeClock()
    provider, seen = sdk_provider([
        httpx.Response(429, headers={"retry-after": "3"}, json={"error": {"message": "quota"}}),
        httpx.Response(200, json=OK_BODY),
    ])
    llm = LLMClient(provider, make_config(), sleep=clock.sleep, clock=clock)
    resp = llm.call("analysis", MSG)
    assert resp.content == "pong" and resp.prompt_tokens == 5 and len(seen) == 2
    assert clock.sleeps[0] >= 3.0


def test_real_sdk_429_google_retry_delay_in_body():
    clock = FakeClock()
    body = [{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12s"}]}}]
    provider, _ = sdk_provider([httpx.Response(429, json=body), httpx.Response(200, json=OK_BODY)])
    LLMClient(provider, make_config(), sleep=clock.sleep, clock=clock).call("analysis", MSG)
    assert clock.sleeps[0] >= 12.0


# ------------------------------------------------------------------ 400 logged with body, not retried
def test_real_sdk_400_is_not_retried_and_body_is_logged(caplog):
    caplog.set_level(logging.INFO, logger="dispute.agents")
    body = {"error": {"code": 400, "message": "Invalid JSON payload received. Unknown name \"default\" at "
                                              "'tools[0].function_declarations[0].parameters.properties[1]'"}}
    provider, seen = sdk_provider([httpx.Response(400, json=body)])
    llm = LLMClient(provider, make_config(), sleep=lambda s: None)
    with pytest.raises(LLMError, match="Unknown name") as err:
        llm.call("analysis", MSG, tools=TOOL_SPECS)
    assert not isinstance(err.value, ProviderUnavailable)
    assert len(seen) == 1                                                  # a 400 is never retried
    logged = [json.loads(r.message) for r in caplog.records]
    (entry,) = [x for x in logged if x["event"] == "llm_provider_error"]
    assert entry["status"] == 400 and "Unknown name" in entry["body"]
    assert entry["sent_tools"] == [t["function"]["name"] for t in TOOL_SPECS]
    assert llm.breaker.state == "closed"                                   # our bug, not an outage


def test_tool_specs_use_the_conservative_schema_subset():
    def walk(node):
        if isinstance(node, dict):
            assert not {"default", "minimum", "maximum"} & set(node), node
            if node.get("type") == "object":
                assert node.get("properties"), "object schema with empty properties"
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(TOOL_SPECS)


# ------------------------------------------------------------------ LLM_MIN_INTERVAL_S pacing
def test_min_interval_spaces_out_calls():
    clock = FakeClock()
    llm = LLMClient(FakeLLM(["a", "b", "c"]), make_config(min_interval_s=4.0), sleep=clock.sleep, clock=clock)
    for _ in range(3):
        llm.call("analysis", MSG)
    assert clock.sleeps == [4.0, 4.0]                                      # first call immediate


def test_min_interval_counts_time_already_elapsed():
    clock = FakeClock()
    llm = LLMClient(FakeLLM(["a", "b"]), make_config(min_interval_s=4.0), sleep=clock.sleep, clock=clock)
    llm.call("analysis", MSG)
    clock.now += 3.0                                                       # 3 s of other work
    llm.call("analysis", MSG)
    assert clock.sleeps == [pytest.approx(1.0)]


def test_min_interval_default_is_off(monkeypatch):
    monkeypatch.delenv("LLM_MIN_INTERVAL_S", raising=False)
    from backend.agents.llm import LLMConfig
    assert LLMConfig.from_env().min_interval_s == 0.0
    monkeypatch.setenv("LLM_MIN_INTERVAL_S", "6.5")
    assert LLMConfig.from_env().min_interval_s == 6.5


# ------------------------------------------------------------------ budget from env
def test_budget_from_env(monkeypatch):
    for n in ("AGENT_MAX_MODEL_CALLS", "AGENT_MAX_TOOL_CALLS", "AGENT_WALL_CLOCK_S"):
        monkeypatch.delenv(n, raising=False)
    b = Budget.from_env()
    assert (b.max_steps, b.max_tool_calls, b.wall_clock_s) == (MAX_STEPS, 16, 60.0)
    monkeypatch.setenv("AGENT_MAX_MODEL_CALLS", "8")
    monkeypatch.setenv("AGENT_MAX_TOOL_CALLS", "5")
    monkeypatch.setenv("AGENT_WALL_CLOCK_S", "120")
    b = Budget.from_env()
    assert (b.max_steps, b.max_tool_calls, b.wall_clock_s) == (8, 5, 120.0)
    monkeypatch.setenv("AGENT_MAX_TOOL_CALLS", "0")
    with pytest.raises(ValueError):
        Budget.from_env()


def test_graph_uses_env_budget(monkeypatch, retriever, request_ok):
    from backend.agents.fake_scenarios import scenario_script
    from backend.agents.graph import run_dispute_graph
    monkeypatch.setenv("AGENT_MAX_MODEL_CALLS", "1")
    llm = LLMClient(FakeLLM(scenario_script("default")), make_config(), sleep=lambda s: None)
    r = run_dispute_graph(request_ok, "r", "c", llm=llm, retriever=retriever)
    assert r.status == "escalated" and r.escalation.reason_code.value == "budget_exhausted"


# ------------------------------------------------------------------ llm_check script
def test_llm_check_passes_with_a_working_provider(capsys):
    from backend.agents.llm import LLMResponse, ToolCall
    call = lambda: LLMResponse(tool_calls=[ToolCall("1", "get_time", {"timezone": "Asia/Kolkata"})])  # noqa: E731
    fake = FakeLLM(["pong", call(), call(), "It is 14:05 IST."])
    assert run_checks(LLMClient(fake, make_config(), sleep=lambda s: None)) is True
    out = capsys.readouterr().out
    assert "content: 'pong'" in out and "get_time" in out and "prompt_tokens=" in out and "3/3 checks passed" in out
    assert "dummy fallback" in out and "turn 2 content: 'It is 14:05 IST.'" in out


def test_llm_check_reports_errors_clearly(capsys):
    fake = FakeLLM([LLMError("BadRequestError 400: Unknown name \"default\""), "no tool call here"])
    assert run_checks(LLMClient(fake, make_config(), sleep=lambda s: None)) is False
    out = capsys.readouterr().out
    assert "FAIL" in out and "Unknown name" in out and "without calling the tool" in out and "0/3" in out


def test_library_config_never_reads_dotenv(monkeypatch):
    """Only CLI entry points load .env; tests and the worker see os.environ alone."""
    import dotenv

    def boom(*a, **k):
        raise AssertionError("LLMConfig.from_env must not load .env")
    monkeypatch.setattr(dotenv, "load_dotenv", boom)
    from backend.agents.llm import LLMConfig
    LLMConfig.from_env()
