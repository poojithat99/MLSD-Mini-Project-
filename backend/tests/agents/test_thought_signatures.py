"""Gemini thought signatures: tool calls must be replayed with their extra_content unchanged.

Uses the real OpenAI SDK over an in-memory transport that behaves like Gemini's
OpenAI-compatible endpoint: it attaches extra_content.google.thought_signature to
every tool call it returns, and rejects (400) any later request whose replayed tool
calls do not carry the right signature.  No network.
"""
from __future__ import annotations

import json

import httpx

from backend.agents.fake_scenarios import ScriptedAgents
from backend.agents.graph import run_dispute_graph
from backend.agents.llm import (
    THOUGHT_SIGNATURE_FALLBACK, FakeLLM, LLMClient, LLMResponse, OpenAICompatProvider, ToolCall,
    assistant_tool_call_message,
)
from backend.agents.llm_check import run_checks
from backend.tests.agents.test_llm import make_config

MISSING = "Function call is missing a thought_signature"


def completion(content=None, tool_calls=None) -> dict:
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if tool_calls else "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


def gemini_call(call_id: str, name: str, args: dict, signature: str | None = None) -> dict:
    tc = {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
    if signature:
        tc["extra_content"] = {"google": {"thought_signature": signature}}
    return tc


class GeminiLike:
    """Fake endpoint.  ``brain(messages, tools)`` decides the reply (an LLMResponse);
    every tool call gets signature ``sig-<id>``.  Replays without it are rejected."""

    def __init__(self, brain):
        self.brain = brain
        self.requests: list[dict] = []
        self.rejections: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        for m in body["messages"]:
            for tc in m.get("tool_calls") or []:
                sig = ((tc.get("extra_content") or {}).get("google") or {}).get("thought_signature")
                if sig != f"sig-{tc['id']}":
                    self.rejections.append(f"{tc['id']}: {sig!r}")
                    return httpx.Response(400, json={"error": {"code": 400, "message": MISSING}})
        resp = self.brain(body["messages"], body.get("tools"))
        if isinstance(resp, Exception):
            return httpx.Response(500, json={"error": {"message": str(resp)}})
        calls = [gemini_call(tc.id, tc.name, tc.arguments, f"sig-{tc.id}") for tc in resp.tool_calls]
        return httpx.Response(200, json=completion(resp.content, calls))


def client_for(handler) -> LLMClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatProvider("primary", "http://gemini.test/v1beta/openai", "m", "k", http_client=http)
    return LLMClient(provider, make_config(), sleep=lambda s: None)


# ------------------------------------------------------------------ capture + replay
def test_extra_content_is_captured_and_replayed_unchanged_in_order():
    sent: list[dict] = []
    replies = [
        completion(tool_calls=[gemini_call("c1", "get_time", {"timezone": "Asia/Kolkata"}, "SIG-A"),
                               gemini_call("c2", "get_time", {"timezone": "UTC"}, "SIG-B")]),
        completion("It is 14:05 IST."),
    ]

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=replies.pop(0))

    llm = client_for(handler)
    first = llm.call("t", [{"role": "user", "content": "time?"}], tools=[{"type": "function", "function": {"name": "get_time"}}])
    assert [tc.extra_content for tc in first.tool_calls] == [{"google": {"thought_signature": "SIG-A"}},
                                                           {"google": {"thought_signature": "SIG-B"}}]
    messages = [{"role": "user", "content": "time?"}, assistant_tool_call_message(first)]
    messages += [{"role": "tool", "tool_call_id": tc.id, "content": "{}"} for tc in first.tool_calls]
    llm.call("t", messages)

    replayed = sent[1]["messages"][1]["tool_calls"]                     # what actually went over the wire
    assert [tc["id"] for tc in replayed] == ["c1", "c2"]
    assert [tc["extra_content"] for tc in replayed] == [{"google": {"thought_signature": "SIG-A"}},
                                                      {"google": {"thought_signature": "SIG-B"}}]
    assert replayed[0]["function"]["arguments"] == json.dumps({"timezone": "Asia/Kolkata"})   # verbatim


def test_other_extra_content_fields_are_kept():
    resp = LLMResponse(tool_calls=[ToolCall("c1", "f", {}, extra_content={"google": {"thought_signature": "S",
                                                                                    "other": 1}, "x": 2})])
    (tc,) = assistant_tool_call_message(resp)["tool_calls"]
    assert tc["extra_content"] == {"google": {"thought_signature": "S", "other": 1}, "x": 2}


# ------------------------------------------------------------------ fallback
def test_missing_signature_falls_back_to_documented_dummy():
    resp = LLMResponse(tool_calls=[ToolCall("c1", "f", {"a": 1}), ToolCall("c2", "g", {})])   # FakeLLM-style
    msg = assistant_tool_call_message(resp)
    assert [tc["extra_content"] for tc in msg["tool_calls"]] == [
        {"google": {"thought_signature": THOUGHT_SIGNATURE_FALLBACK}}] * 2
    assert msg["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'


def test_fallback_goes_over_the_wire():
    sent: list[dict] = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=completion("ok"))

    history = [{"role": "user", "content": "q"},
               assistant_tool_call_message(LLMResponse(tool_calls=[ToolCall("c9", "f", {})])),
               {"role": "tool", "tool_call_id": "c9", "content": "{}"}]
    client_for(handler).call("t", history)
    assert sent[0]["messages"][1]["tool_calls"][0]["extra_content"] == {
        "google": {"thought_signature": THOUGHT_SIGNATURE_FALLBACK}}


# ------------------------------------------------------------------ whole graph, including a verifier retry
def test_full_run_with_retry_never_loses_a_signature(retriever, request_ok):
    endpoint = GeminiLike(ScriptedAgents(fabricate="first_attempt"))   # forces a verifier retry
    r = run_dispute_graph(request_ok, "run-sig", "corr-sig", llm=client_for(endpoint), retriever=retriever)
    assert endpoint.rejections == []
    assert r.status == "recommended" and r.verification.attempts == 2   # re-entered the analyst and finished
    replayed = [tc for body in endpoint.requests for m in body["messages"] for tc in m.get("tool_calls") or []]
    assert replayed and all(tc["extra_content"]["google"]["thought_signature"] == f"sig-{tc['id']}"
                            for tc in replayed)


def test_without_the_fix_gemini_would_reject(retriever, request_ok):
    """Guard against regressions: dropping extra_content reproduces the production 400."""
    endpoint = GeminiLike(ScriptedAgents())
    llm = client_for(endpoint)
    first = llm.call("t", [{"role": "system", "content": "analysis"},
                           {"role": "user", "content": '"working_days": 1 card used online'}])
    stripped = assistant_tool_call_message(first)
    for tc in stripped["tool_calls"]:
        del tc["extra_content"]
    http = endpoint(httpx.Request("POST", "http://x", json={"messages": [stripped]}))
    assert http.status_code == 400 and MISSING in http.text


# ------------------------------------------------------------------ llm_check (d)
def test_llm_check_two_turn_against_gemini_like_endpoint(capsys):
    def brain(messages, tools):
        if any(m["role"] == "tool" for m in messages):
            return LLMResponse(content="It is 14:05 IST.")
        if tools:
            return LLMResponse(tool_calls=[ToolCall("t1", "get_time", {"timezone": "Asia/Kolkata"})])
        return LLMResponse(content="pong")

    endpoint = GeminiLike(brain)
    assert run_checks(client_for(endpoint)) is True
    out = capsys.readouterr().out
    assert "present (" in out and "turn 2 content: 'It is 14:05 IST.'" in out and endpoint.rejections == []


def test_llm_check_two_turn_reports_missing_signature_400(capsys):
    """If signatures were dropped, (d) shows the provider's 400 message clearly."""
    calls = [LLMResponse(content="pong"),
             LLMResponse(tool_calls=[ToolCall("t1", "get_time", {})]),
             LLMResponse(tool_calls=[ToolCall("t2", "get_time", {})]),
             ValueError(f"BadRequestError 400: {MISSING}")]
    assert run_checks(LLMClient(FakeLLM(calls), make_config(), sleep=lambda s: None)) is False
    assert MISSING in capsys.readouterr().out
