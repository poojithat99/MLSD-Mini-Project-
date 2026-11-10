"""AGENT_SEED_PROVISIONS (code pre-fetches the Table 2 row for the delay + RBI 7(i)) and the
skipped modifier check for out-of-scope drafts."""
from __future__ import annotations

import json

import pytest

from backend.agents.fake_scenarios import ScriptedAgents
from backend.agents.graph import run_dispute_graph
from backend.agents.llm import FakeLLM, LLMClient, LLMResponse, ToolCall
from backend.agents.state import env_flag
from backend.tests.agents.test_llm import make_config
from backend.tests.agents.test_real_failures import OOC, GeminiLike


def run(request, retriever, script, **kw):
    fake = FakeLLM(script)
    llm = LLMClient(fake, make_config(), sleep=lambda s: None)
    return run_dispute_graph(request, "run-s", "corr-s", llm=llm, retriever=retriever, **kw), fake


def config_entry(r) -> dict:
    (entry,) = [t for t in r.trace if t.name == "run.config"]
    return entry.detail


def tools_before_first_model_call(r) -> list[str]:
    out = []
    for t in r.trace:
        if t.type == "llm":
            break
        if t.type == "tool":
            out.append(f"{t.name} {t.detail['arguments']}")
    return out


# ------------------------------------------------------------------ flag on (default)
def test_seed_on_by_default_and_recorded(monkeypatch, retriever, request_ok):
    monkeypatch.delenv("AGENT_SEED_PROVISIONS", raising=False)
    r, _ = run(request_ok, retriever, ScriptedAgents())
    assert config_entry(r)["seed_provisions"] is True
    assert r.trace[0].name == "run.config"                       # first entry: which mode produced this


def test_seed_on_gives_41_escalation_with_both_provisions(monkeypatch, retriever, request_41):
    monkeypatch.setenv("AGENT_SEED_PROVISIONS", "on")
    r, _ = run(request_41, retriever, GeminiLike("41"))           # model cites only SBI 4.6.2(b)(i), says decline
    assert r.status == "escalated" and r.escalation.reason_code.value == "conflicting_provisions"
    cited = {(c.doc_id, c.clause_id) for c in r.citations}
    assert {("rbi_circular_2017", "7(i)"), ("rbi_circular_2017", "Table 2")} <= cited
    seeded = tools_before_first_model_call(r)
    assert any("'Table 2'" in x for x in seeded) and any("'7(i)'" in x for x in seeded)


# ------------------------------------------------------------------ flag off
def test_seed_off_prefetches_nothing_and_applies_no_conflict_rule(monkeypatch, retriever, request_41):
    monkeypatch.setenv("AGENT_SEED_PROVISIONS", "off")
    r, fake = run(request_41, retriever, GeminiLike("41"))
    assert config_entry(r)["seed_provisions"] is False
    assert tools_before_first_model_call(r) == []                # no seeded tool calls
    assert not any(t.name == "analysis.conflict_rule" for t in r.trace)
    case_message = fake.calls[0][1]["content"]
    assert "not pre-fetched" in case_message and '"clause_id":"7(i)"' not in case_message
    # the agent alone: this model's own (wrong) answer goes through; that is what "off" measures
    assert r.status == "recommended" and r.route == "decline"


def test_argument_overrides_env(monkeypatch, retriever, request_ok):
    monkeypatch.setenv("AGENT_SEED_PROVISIONS", "on")
    r, _ = run(request_ok, retriever, ScriptedAgents(), seed_provisions=False)
    assert config_entry(r)["seed_provisions"] is False and tools_before_first_model_call(r) == []


@pytest.mark.parametrize("raw, expected", [("1", True), ("true", True), ("ON", True), ("0", False),
                                           ("false", False), ("off", False), ("", True)])
def test_env_flag_values(monkeypatch, raw, expected):
    monkeypatch.setenv("AGENT_SEED_PROVISIONS", raw)
    assert env_flag("AGENT_SEED_PROVISIONS", True) is expected


def test_env_flag_rejects_garbage(monkeypatch):
    monkeypatch.setenv("AGENT_SEED_PROVISIONS", "maybe")
    with pytest.raises(ValueError):
        env_flag("AGENT_SEED_PROVISIONS", True)


# ------------------------------------------------------------------ out-of-scope skips the modifier check
def one_search_then_answer(in_scope: bool):
    """Searches once, then answers. In scope it gets nudged (search again), out of scope it should not."""
    def script(messages, tools):
        if any(m["role"] == "tool" for m in messages):
            if any(m["role"] == "user" and m["content"].startswith("MODIFIER CHECK") for m in messages):
                if sum(1 for m in messages if m.get("tool_calls")) < 2:
                    return LLMResponse(tool_calls=[ToolCall("s2", "search_corpus", {"query": "second"})])
            draft = {"in_scope": in_scope, "classification": None if not in_scope else "cnp_fraud",
                     "escalate": not in_scope, "route": None, "citations": []}
            return LLMResponse(content=json.dumps(draft))
        return LLMResponse(tool_calls=[ToolCall("s1", "search_corpus", {"query": "cryptocurrency investment"})])
    return script


def test_out_of_scope_draft_skips_modifier_check(retriever):
    r, fake = run(OOC, retriever, one_search_then_answer(in_scope=False))
    assert r.escalation.reason_code.value == "out_of_corpus"
    sent = [m["content"] for call in fake.calls for m in call if m["role"] == "user"]
    assert not any(c.startswith("MODIFIER CHECK") for c in sent)
    assert len(fake.calls) == 2                                   # search + answer


def test_in_scope_draft_still_gets_modifier_check_one_more_call(retriever):
    r, fake = run(OOC, retriever, one_search_then_answer(in_scope=True))
    sent = [m["content"] for call in fake.calls for m in call if m["role"] == "user"]
    assert any(c.startswith("MODIFIER CHECK") for c in sent)
    assert len([c for c in fake.calls if "verification agent" not in c[0]["content"]]) >= 3
