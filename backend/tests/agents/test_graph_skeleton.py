"""graph.py end to end with FakeLLM: every non-clean ending is a coded escalation."""
from __future__ import annotations

import json
import logging

from backend.agents.graph import run_dispute_graph
from backend.agents.llm import FakeLLM, LLMClient
from backend.agents.state import Budget
from backend.tests.agents.test_llm import make_config

FABRICATED = json.dumps({
    "classification": "cnp_fraud", "route": "provisional_credit", "basis": "zero liability",
    "liability_position": "customer has zero liability",
    "citations": [{"doc_id": "rbi_circular_2017", "clause_id": "6(ii)",
                   "quote": "the customer shall always be refunded in full within one day"}],
})
OUT_OF_CORPUS = json.dumps({"escalate": True, "reason_code": "out_of_corpus", "note": "no provision covers this"})


def client(script, **cfg) -> LLMClient:
    return LLMClient(FakeLLM(script), make_config(**cfg), sleep=lambda s: None)


def test_unverifiable_answer_escalates_after_two_retries(retriever, request_ok):
    llm = client([FABRICATED] * 3)
    r = run_dispute_graph(request_ok, "run-1", "corr-1", llm=llm, retriever=retriever)
    assert r.status == "escalated" and r.route is None
    assert r.escalation.reason_code.value == "verification_failed"
    assert r.verification.attempts == 3 and not r.verification.passed
    assert r.citations == []                                     # the fabricated citation never leaks out
    assert [t.name for t in r.trace if t.type == "node"].count("analysis_agent") == 3


def test_verified_escalation_goes_through_finalize(retriever, request_ok):
    r = run_dispute_graph(request_ok, "run-2", "corr-2", llm=client([OUT_OF_CORPUS]), retriever=retriever)
    assert r.status == "escalated" and r.escalation.reason_code.value == "out_of_corpus"
    assert r.prompt_version == "analysis_v1+verification_v1"
    assert r.reporting_delay_working_days == 1


def test_provider_down_escalates_provider_unavailable(retriever, request_ok):
    r = run_dispute_graph(request_ok, "run-3", "corr-3", llm=client([TimeoutError()] * 10), retriever=retriever)
    assert r.status == "escalated" and r.escalation.reason_code.value == "provider_unavailable"


def test_budget_exhaustion_escalates_not_answers(retriever, request_ok):
    r = run_dispute_graph(request_ok, "run-4", "corr-4", llm=client([FABRICATED] * 3),
                          retriever=retriever, budget=Budget(max_steps=1))
    assert r.status == "escalated" and r.escalation.reason_code.value == "budget_exhausted"


def test_trace_and_logs_carry_run_and_correlation_ids(retriever, request_ok, caplog):
    caplog.set_level(logging.INFO, logger="dispute.agents")
    r = run_dispute_graph(request_ok, "run-5", "corr-5", llm=client([OUT_OF_CORPUS]), retriever=retriever)
    lines = [json.loads(rec.message) for rec in caplog.records]
    assert lines and all(x["run_id"] == "run-5" and x["correlation_id"] == "corr-5" for x in lines)
    llm_entries = [t for t in r.trace if t.type == "llm"]
    assert llm_entries and llm_entries[0].name == "analysis" and llm_entries[0].prompt_tokens > 0


def test_request_may_be_a_plain_dict(retriever, request_ok):
    r = run_dispute_graph(request_ok.model_dump(), "run-6", "corr-6", llm=client([OUT_OF_CORPUS]),
                          retriever=retriever)
    assert r.run_id == "run-6"
