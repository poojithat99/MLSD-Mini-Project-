"""Round-3 rules: the code-written conflict basis quotes the provisions verbatim; an unknown alert date
means no recommendation (contested_fault); seeded provisions are announced as already in context."""
from __future__ import annotations

import json
from datetime import date

from backend.agents.fake_scenarios import _sentence_with
from backend.agents.graph import run_dispute_graph
from backend.agents.llm import FakeLLM, LLMClient, LLMResponse, ToolCall
from backend.agents.state import AnalysisDraft, DisputeInput
from backend.agents.verification_agent import route_problems
from backend.tests.agents.test_llm import make_config
from backend.tests.agents.test_real_failures import GeminiLike

# S07-like: skimming, no alert date (delay counted from the transaction date by our calculator)
NO_ALERT = DisputeInput(
    narrative="My card was copied at a shop terminal and used for purchases I did not make. I did not get or did "
              "not see any SMS alert. I reported it about two weeks after the transactions.",
    channel="card_present", amount=18000, transaction_date=date(2026, 10, 1), communication_date=None,
    reported_date=date(2026, 10, 16), merchant="LOCAL STORE")


def client(script):
    return LLMClient(FakeLLM(script), make_config(), sleep=lambda s: None)


# ------------------------------------------------------------------ conflict basis
def test_conflict_basis_quotes_both_provisions_verbatim_and_is_marked_code_written(retriever, request_41):
    r = run_dispute_graph(request_41, "c", "c", llm=client(GeminiLike("41")), retriever=retriever)
    assert r.escalation.reason_code.value == "conflicting_provisions"
    quotes = {(c.doc_id, c.clause_id): c.quote for c in r.citations}
    assert quotes[("rbi_circular_2017", "7(i)")] in r.basis
    assert quotes[("rbi_circular_2017", "Table 2")] in r.basis
    assert "different outcomes" not in r.basis                    # no inference in the basis any more
    (rule,) = [t for t in r.trace if t.name == "analysis.conflict_rule"]
    assert rule.detail["code_written_basis"] == r.basis


def test_seeded_provisions_are_announced_as_already_in_context(retriever, request_41):
    llm = client(GeminiLike("41"))
    run_dispute_graph(request_41, "c", "c", llm=llm, retriever=retriever)
    case_message = llm.primary.calls[0][1]["content"]
    assert case_message.count("already in your context, do not fetch it again") == 3


# ------------------------------------------------------------------ unknown alert date
def test_route_problems_unknown_alert_requires_escalation():
    d = AnalysisDraft(classification="skimming", route="decline")
    assert route_problems(d, 12, alert_date_known=True) == []
    assert "contested_fault" in route_problems(d, 12, alert_date_known=False)[0]
    # bank fault and card disputes are decided without the delay, so they still go through
    assert route_problems(AnalysisDraft(classification="duplicate", route="chargeback"), 1, alert_date_known=False) == []
    assert route_problems(d.model_copy(update={"escalate": True, "route": None}), 12, alert_date_known=False) == []


class NoAlertModel:
    """Two searches, then: escalates contested_fault citing RBI 5 + Table 2 if the prompt has the
    alert-receipt rule, or (always_decline=True) insists on declining."""

    def __init__(self, always_decline: bool = False):
        self.always_decline = always_decline

    def __call__(self, messages, tools):
        if "verification agent" in messages[0]["content"]:
            p = json.loads(messages[1]["content"])
            return LLMResponse(content=json.dumps({
                "claims": [{"index": c["index"], "supported": True} for c in p["claims"]],
                "case_facts": [{"index": f["index"], "status": "consistent"} for f in p["case_facts"]],
                "missed_provision": None}))
        n = sum(1 for m in messages if m.get("tool_calls"))
        if n < 2:
            q = ["SMS alerts shall mandatorily be sent to the customers", "skimming card copied liability"][n]
            return LLMResponse(tool_calls=[ToolCall(f"n{n}", "search_corpus", {"query": q})])
        passages = [p for m in messages if m["role"] == "tool" for p in (json.loads(m["content"]) or [])
                    if isinstance(p, dict)]
        five = next(({"doc_id": "rbi_circular_2017", "clause_id": "5", "quote": _sentence_with(p["text"], "mandatorily be sent")}
                     for p in passages if p["doc_id"] == "rbi_circular_2017" and "5" in p["clause_ids"]
                     and _sentence_with(p["text"], "mandatorily be sent")), None)
        row = json.loads(messages[1]["content"].split("LIABILITY_TABLE_ROW: ", 1)[1].split("\n", 1)[0])
        if "Alert receipt unknown" in messages[0]["content"] and not self.always_decline:
            d = {"classification": "skimming", "escalate": True, "reason_code": "contested_fault",
                 "citations": [five, row], "competing_provisions": ["rbi_circular_2017 5", "rbi_circular_2017 Table 2"],
                 "note": "Whether the bank sent the mandatory alert or the customer missed it is not settled."}
        else:
            d = {"classification": "skimming", "route": "decline", "citations": [row],
                 "basis": "Beyond 7 working days.", "liability_position": "Per the bank's policy."}
        return LLMResponse(content=json.dumps(d))


def test_unknown_alert_date_escalates_contested_fault(retriever):
    r = run_dispute_graph(NO_ALERT, "a", "a", llm=client(NoAlertModel()), retriever=retriever)
    assert r.status == "escalated" and r.escalation.reason_code.value == "contested_fault"
    assert {("rbi_circular_2017", "5"), ("rbi_circular_2017", "Table 2")} <= {(c.doc_id, c.clause_id) for c in r.citations}


def test_unknown_alert_date_never_yields_a_confident_decline(retriever):
    r = run_dispute_graph(NO_ALERT, "a", "a", llm=client(NoAlertModel(always_decline=True)), retriever=retriever)
    assert r.status == "escalated" and r.route is None
    verdicts = [t.detail for t in r.trace if t.name == "verifier.verdict"]
    assert any("contested_fault" in u for v in verdicts for u in v["unsupported_claims"])
