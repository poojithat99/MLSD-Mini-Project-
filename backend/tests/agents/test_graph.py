"""The whole graph, end to end, with FakeLLM + the keyword retriever over the real corpus.

Required by the brief / plan:
  * section 4.1 ends in escalation citing BOTH provisions
  * an out-of-corpus question escalates
  * a fabricated citation is caught by the verifier
  * budget exhaustion yields an escalation, not an answer
  * the circuit breaker opens after repeated provider failures
"""
from __future__ import annotations

import json
import logging
from datetime import date

from backend.agents.fake_scenarios import FABRICATED_QUOTE, ScriptedAgents, scenario_script
from backend.agents.graph import run_dispute_graph
from backend.agents.llm import CircuitBreaker, FakeLLM, LLMClient, LLMResponse, ToolCall
from backend.agents.state import Budget, DisputeInput
from backend.tests.agents.test_llm import make_config


def client(script, breaker=None, **cfg) -> LLMClient:
    return LLMClient(FakeLLM(script), make_config(**cfg), breaker=breaker, sleep=lambda s: None)


def run(request, retriever, script="default", **kw):
    llm = kw.pop("llm", None) or client(scenario_script(script) if isinstance(script, str) else script)
    return run_dispute_graph(request, "run-1", "corr-1", llm=llm, retriever=retriever, **kw)


def node_entries(result, name):
    return [t for t in result.trace if t.type == "node" and t.name == name]


# ------------------------------------------------------------------ section 4.1
def test_section_41_escalates_citing_both_provisions(retriever, request_41):
    r = run(request_41, retriever)
    assert r.status == "escalated" and r.route is None
    assert r.escalation.reason_code.value == "conflicting_provisions"
    cited = {(c.doc_id, c.clause_id) for c in r.citations}
    assert ("rbi_circular_2017", "7(i)") in cited                       # customer negligence
    assert ("rbi_circular_2017", "Table 2") in cited                    # beyond 7 working days -> board policy
    assert set(r.escalation.competing_provisions) == {"rbi_circular_2017 7(i)", "rbi_circular_2017 Table 2"}
    assert r.verification.passed and r.reporting_delay_working_days == 17


def test_modifier_check_forces_a_second_search(retriever, request_41):
    r = run(request_41, retriever)
    searches = [t.detail["arguments"]["query"] for t in r.trace if t.type == "tool" and t.name == "search_corpus"]
    assert len(set(searches)) >= 2
    assert "beyond seven working days" in searches[1]                   # derived from what was read + the facts
    assert node_entries(r, "analysis.modifier_check")[0].detail["done"] is True


def test_modifier_check_nudges_a_model_that_answers_after_one_search(retriever, request_ok):
    agents = ScriptedAgents()
    llm = client(agents)
    run(request_ok, retriever, llm=llm)
    nudges = [m for call in llm.primary.calls for m in call
              if m["role"] == "user" and m["content"].startswith("MODIFIER CHECK")]
    assert nudges, "the agent must be asked to look for modifying provisions"


# ------------------------------------------------------------------ ordinary + out of corpus
def test_ordinary_case_is_recommended_with_resolving_citation(retriever, request_ok):
    r = run(request_ok, retriever)
    assert r.status == "recommended" and r.route == "provisional_credit"
    assert r.citations[0].clause_id == "6(ii)" and r.verification.passed
    assert r.prompt_version == "analysis_v1+verification_v1"


def test_out_of_corpus_question_escalates(retriever):
    req = DisputeInput(narrative="I lost money on a cryptocurrency exchange investment and want it back.",
                       channel="card_not_present", amount=90000, transaction_date=date(2026, 10, 1),
                       communication_date=date(2026, 10, 1), reported_date=date(2026, 10, 2))
    r = run(req, retriever)
    assert r.status == "escalated" and r.escalation.reason_code.value == "out_of_corpus"
    assert r.route is None and r.citations == []


# ------------------------------------------------------------------ verification
def test_fabricated_citation_is_caught_then_retry_succeeds(retriever, request_ok):
    r = run(request_ok, retriever, script="fabricate_citation")
    verdicts = [t.detail for t in node_entries(r, "verifier.verdict")]
    assert verdicts[0]["decision"] == "retry" and verdicts[0]["strategy_hint"] == "neighbour_clause"
    assert "does not resolve" in verdicts[0]["unsupported_claims"][0]
    assert verdicts[-1]["decision"] == "pass"
    assert r.status == "recommended" and r.verification.attempts == 2
    assert all(c.quote != FABRICATED_QUOTE for c in r.citations)
    retry_tools = [t for t in r.trace if t.type == "tool" and t.name == "get_clause"]
    assert retry_tools, "the retry fetched the cited clause by id (a different strategy)"


def test_persistent_fabrication_escalates_and_never_leaks(retriever, request_ok):
    r = run(request_ok, retriever, script="fabricate_always")
    assert r.status == "escalated" and r.escalation.reason_code.value == "verification_failed"
    assert r.verification.attempts == 3 and not r.verification.passed
    assert all(c.quote != FABRICATED_QUOTE for c in r.citations)
    strategies = [t.detail["strategy"] for t in node_entries(r, "analysis.modifier_check")]
    assert strategies[0] is None and len(set(strategies[1:])) == 2      # two retries, two different strategies


def test_llm_verifier_rejects_unsupported_claim(retriever, request_ok):
    agents = ScriptedAgents()
    first = {"done": False}

    def script(messages, tools):
        if "verification agent" in messages[0]["content"] and not first["done"]:
            first["done"] = True
            return LLMResponse(content=json.dumps({"claims": [{"index": 0, "supported": False}],
                                                   "missed_provision": None}))
        return agents(messages, tools)

    r = run(request_ok, retriever, script=script)
    v = node_entries(r, "verifier.verdict")[0].detail
    assert v["decision"] == "retry" and v["strategy_hint"] == "rewrite_query"
    assert r.verification.attempts == 2


def test_verifier_sees_only_claims_and_passages(retriever, request_41):
    llm = client(scenario_script("default"))
    run(request_41, retriever, llm=llm)
    (verifier_call,) = [c for c in llm.primary.calls if "verification agent" in c[0]["content"]]
    assert [m["role"] for m in verifier_call] == ["system", "user"]            # no analyst messages / tool calls
    payload = json.loads(verifier_call[1]["content"])
    assert set(payload) == {"claims", "outcome", "passages", "facts_computed_by_code"}
    assert "OTP" not in verifier_call[1]["content"]                            # not even the customer narrative


def test_verifier_model_down_forces_escalation(retriever, request_ok):
    r = run(request_ok, retriever, script="verifier_down")
    assert r.status == "escalated" and r.escalation.reason_code.value == "provider_unavailable"
    assert "deterministic citation check passed" in r.escalation.note
    assert r.citations and not r.verification.passed                    # evidence kept, clearly not verified


# ------------------------------------------------------------------ budget + provider failures
def test_budget_exhaustion_yields_escalation_not_answer(retriever, request_ok):
    def looping(messages, tools):                                        # a model that never stops searching
        n = sum(1 for m in messages if m["role"] == "tool")
        return LLMResponse(tool_calls=[ToolCall(id=f"c{n}", name="search_corpus", arguments={"query": f"q{n}"})])

    r = run(request_ok, retriever, script=looping, budget=Budget(max_tool_calls=3))
    assert r.status == "escalated" and r.escalation.reason_code.value == "budget_exhausted"
    assert r.route is None


def test_wall_clock_deadline_yields_timeout_escalation(retriever, request_ok):
    r = run(request_ok, retriever, budget=Budget(wall_clock_s=0))
    assert r.status == "escalated" and r.escalation.reason_code.value == "timeout"


def test_circuit_breaker_opens_after_repeated_provider_failures(retriever, request_ok):
    breaker = CircuitBreaker(threshold=3, reset_s=60)
    llm = client(scenario_script("provider_down"), breaker=breaker)
    r1 = run(request_ok, retriever, llm=llm)
    assert r1.status == "escalated" and r1.escalation.reason_code.value == "provider_unavailable"
    assert breaker.state == "open"
    calls = len(llm.primary.calls)
    r2 = run(request_ok, retriever, llm=llm)                               # next run fails fast
    assert r2.escalation.reason_code.value == "provider_unavailable"
    assert "circuit breaker open" in r2.escalation.note
    assert len(llm.primary.calls) == calls                                 # provider not called at all


# ------------------------------------------------------------------ plumbing
def test_logs_and_trace_carry_ids_and_cost(retriever, request_ok, caplog):
    caplog.set_level(logging.INFO, logger="dispute.agents")
    r = run(request_ok, retriever)
    lines = [json.loads(rec.message) for rec in caplog.records]
    assert lines and all(x["run_id"] == "run-1" and x["correlation_id"] == "corr-1" for x in lines)
    llm_steps = {t.name for t in r.trace if t.type == "llm"}
    assert llm_steps == {"analysis", "verification"}
    assert all(t.prompt_tokens > 0 for t in r.trace if t.type == "llm")


def test_request_may_be_a_plain_dict(retriever, request_ok):
    assert run(request_ok.model_dump(), retriever).run_id == "run-1"
