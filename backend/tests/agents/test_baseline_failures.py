"""Reproductions of baseline evaluation failures (evaluation/results/analysis_v1_seed-on.json, 2026-11-10).

``BaselineLike`` mimics what the real model did in the saved traces, on the real scenario inputs:
  S03  bank-side breach, 16 days: anchors on the seeded Table 2 "Beyond 7" row and declines
  S13  alert missed, nothing shared, 17 days: cites the seeded 7(i) while proposing provisional_credit
  S19/S20  duplicate charge / goods not delivered: takes the out-of-scope exit
Like a real model it fills an output field only if the prompt asks for it, follows an explicit
scope instruction only if the prompt contains it, and on a retry corrects what the verifier names.
"""
from __future__ import annotations

import json
import re

from backend.agents.analysis_agent import RETRY_MARKER
from backend.agents.fake_scenarios import _sentence_with
from backend.agents.graph import run_dispute_graph
from backend.agents.llm import FakeLLM, LLMClient, LLMResponse, ToolCall
from backend.agents.state import AnalysisDraft
from backend.tests.agents.test_llm import make_config
from evaluation.evaluate import load_scenarios

SC = {s["id"]: s for s in load_scenarios()}
QUERIES = {
    "S03": ("Table 2 third party breach zero liability deficiency on the part of the bank",
            "Board approved policy beyond 7 working days"),
    "S13": ("Beyond 7 working days As per bank Board approved policy",
            "beyond seven working days the customer liability shall be full"),
    "S11": ("negligence by a customer shared the payment credentials entire loss",
            "beyond seven working days Board approved policy"),
    "S19": ("complaints relating to Point of Sales Card Not Present transactions Master Card VISA guidelines",
            "duplicate charge card network"),
    "S20": ("complaints relating to Point of Sales Card Not Present transactions Master Card VISA guidelines",
            "goods not delivered merchant dispute"),
    "S14": ("cryptocurrency exchange investment", "unauthorised electronic banking transaction"),
}


class BaselineLike:
    def __init__(self, sid: str, fact_status: str = "consistent"):
        self.sid = sid
        self.fact_status = fact_status        # what the verifier says about every case fact
        self._n = 0

    def __call__(self, messages, tools):
        if "verification agent" in messages[0]["content"]:
            payload = json.loads(messages[1]["content"])
            self.verifier_payloads = getattr(self, "verifier_payloads", []) + [payload]
            return LLMResponse(content=json.dumps({
                "claims": [{"index": c["index"], "supported": True} for c in payload["claims"]],
                "case_facts": [{"index": f["index"], "status": self.fact_status} for f in payload["case_facts"]],
                "missed_provision": None}))
        searches = [tc for m in messages if m["role"] == "assistant"
                    for tc in m.get("tool_calls") or [] if tc["function"]["name"] == "search_corpus"]
        if len(searches) < 2:
            self._n += 1
            return LLMResponse(tool_calls=[ToolCall(f"b{self._n}", "search_corpus",
                                                    {"query": QUERIES[self.sid][len(searches)], "k": 5})])
        return LLMResponse(content=json.dumps(self._draft(messages)))

    # ------------------------------------------------------------------ the model's answer
    def _draft(self, messages) -> dict:
        system, case = messages[0]["content"], messages[1]["content"]
        said = "\n".join(m["content"] or "" for m in messages if m["role"] == "user")
        retry_problems = said.split(RETRY_MARKER, 1)[1] if RETRY_MARKER in said else ""
        passages = []
        for m in messages:
            if m["role"] == "tool":
                data = json.loads(m["content"])
                passages += data if isinstance(data, list) else []
        seeded = {name: json.loads(js) for name, js in re.findall(r"([A-Z_]+): (\{[^\n]*\})", case)}

        def cite(doc, clause, kw):
            for p in passages:
                if p["doc_id"] == doc and clause in p["clause_ids"]:
                    q = _sentence_with(p["text"], kw)
                    if q:
                        return {"doc_id": doc, "clause_id": clause, "quote": q}
            return None

        asks = lambda field: f'"{field}"' in system           # noqa: E731  (the JSON template lists the field)
        d = {"in_scope": True, "classification": "cnp_fraud", "route": None, "basis": "", "liability_position": "",
             "case_facts": [], "claims": [], "citations": [], "escalate": False, "reason_code": None,
             "competing_provisions": [], "note": ""}
        row = seeded.get("LIABILITY_TABLE_ROW") or cite("rbi_circular_2017", "Table 2", "Beyond 7 working days")
        sbi_full = cite("sbi_policy_2021", "Table 2", "customer liability shall be full")
        if self.sid == "S03":
            if asks("bank_fault"):
                d["bank_fault"] = True
            if "6(i)" in retry_problems:                       # told that bank fault takes precedence
                six = seeded.get("BANK_DEFICIENCY_PROVISION") or cite("rbi_circular_2017", "6(i)", "deficiency on the part of the bank")
                d.update(route="provisional_credit", citations=[six], basis="RBI para 6(i): deficiency on the part of the bank.",
                         liability_position="Zero liability.",
                         claims=["Zero liability arises for deficiency on the part of the bank irrespective of reporting."])
            else:                                              # baseline behaviour: anchored on the delay row
                d.update(route="decline", citations=[c for c in (row, sbi_full) if c],
                         basis="Reported beyond 7 working days; SBI policy makes liability full.",
                         liability_position="Full liability.", claims=["Beyond 7 working days liability is full."])
        elif self.sid == "S13":
            if asks("credentials_shared"):
                d["credentials_shared"] = False
            neg = seeded.get("NEGLIGENCE_PROVISION")
            route = "decline" if "contradicts" in retry_problems else "provisional_credit"
            d.update(route=route, citations=[c for c in (row, neg, sbi_full) if c],
                     basis="Third-party breach reported beyond 7 working days.",
                     liability_position="Per the bank's policy.", claims=["Beyond 7 working days liability is full."])
        elif self.sid == "S11":
            if asks("credentials_shared"):
                d["credentials_shared"] = True
            d.update(route="decline", citations=[c for c in (cite("rbi_circular_2017", "7(i)", "entire loss"),) if c],
                     basis="Negligence.", liability_position="Entire loss.", claims=["Negligence means entire loss."])
        elif self.sid in ("S19", "S20"):
            in_scope_disputes = "charged more than once" in system   # the prompt says card disputes are in scope
            if not in_scope_disputes:
                d.update(in_scope=False, classification=None, escalate=True, reason_code="out_of_corpus")
            else:
                five = cite("sbi_policy_2021", "5(i)", "Card Not Present")
                d.update(classification="duplicate" if self.sid == "S19" else "merchant_dispute", route="chargeback",
                         citations=[five] if five else [], basis="SBI customer-rights para 5(i).",
                         liability_position="Raised with the card network.",
                         claims=["POS / CNP complaints are dealt with as per the card network guidelines."])
        else:  # S14: a real non-dispute stays out of scope
            d.update(in_scope=False, classification=None, escalate=True, reason_code="out_of_corpus")
        return d


def run(sid, retriever, **kw):
    fake = BaselineLike(sid, **kw)
    llm = LLMClient(FakeLLM(fake), make_config(), sleep=lambda s: None)
    return run_dispute_graph(SC[sid]["request"], f"b-{sid}", "b", llm=llm, retriever=retriever), fake


def cited(r) -> set:
    return {(c.doc_id, c.clause_id) for c in r.citations}


# ------------------------------------------------------------------ 1. S03: para 6 before the delay table
def test_s03_bank_side_deficiency_gives_zero_liability_via_6i(retriever):
    r, fake = run("S03", retriever)
    assert r.status == "recommended" and r.route == "provisional_credit"
    assert ("rbi_circular_2017", "6(i)") in cited(r)
    facts = [f["fact"] for p in fake.verifier_payloads for f in p["case_facts"]]
    assert any("bank's own" in f for f in facts)                # the bank-fault premise is checked against the request


def test_s03_bank_fault_not_confirmed_by_request_is_not_recommended(retriever):
    r, _ = run("S03", retriever, fact_status="not_stated")
    assert not (r.status == "recommended" and r.route == "provisional_credit")


# ------------------------------------------------------------------ 2. S13: negligence escalation only on shared credentials
def test_s13_citing_7i_without_shared_credentials_does_not_escalate(retriever):
    r, _ = run("S13", retriever)
    assert not any(t.name == "analysis.conflict_rule" for t in r.trace)
    assert r.status == "recommended" and r.route == "decline"
    assert {("rbi_circular_2017", "Table 2"), ("sbi_policy_2021", "Table 2")} <= cited(r)


def test_s11_shared_credentials_still_escalates_with_both(retriever):
    r, _ = run("S11", retriever)
    assert r.status == "escalated" and r.escalation.reason_code.value == "conflicting_provisions"
    assert {("rbi_circular_2017", "7(i)"), ("rbi_circular_2017", "Table 2")} <= cited(r)


def test_shared_credentials_must_be_confirmed_by_the_request(retriever):
    r, _ = run("S11", retriever, fact_status="not_stated")
    assert not (r.status == "escalated" and r.escalation.reason_code.value == "conflicting_provisions"
                and r.verification.passed)


# ------------------------------------------------------------------ 3. S19/S20: card disputes route to chargeback
def test_s19_duplicate_charge_routes_to_chargeback(retriever):
    r, _ = run("S19", retriever)
    assert (r.status, r.route, r.classification) == ("recommended", "chargeback", "duplicate")
    assert ("sbi_policy_2021", "5(i)") in cited(r)


def test_s20_goods_not_delivered_routes_to_chargeback(retriever):
    r, _ = run("S20", retriever)
    assert (r.status, r.route, r.classification) == ("recommended", "chargeback", "merchant_dispute")
    assert ("sbi_policy_2021", "5(i)") in cited(r)


def test_s14_non_dispute_still_out_of_corpus(retriever):
    r, _ = run("S14", retriever)
    assert r.escalation.reason_code.value == "out_of_corpus" and r.route is None and r.citations == []


# ------------------------------------------------------------------ the route-consistency check itself
def test_route_problems_rules():
    from backend.agents.verification_agent import route_problems
    def d(**kw):
        return AnalysisDraft(**{"classification": "cnp_fraud", "route": "provisional_credit", **kw})
    assert route_problems(d(), 2) == [] and route_problems(d(), 7) == []
    assert route_problems(d(), 8)                                             # beyond 7 -> decline
    assert route_problems(d(route="decline"), 8) == []
    assert route_problems(d(route="decline", bank_fault=True), 16)            # bank fault -> zero liability
    assert route_problems(d(bank_fault=True), 16)                             # ... and must cite 6(i)
    assert route_problems(d(classification="duplicate", route="decline"), 1)  # card dispute -> chargeback
    assert route_problems(d(credentials_shared=True, route="decline"), 1) == []   # handled by escalation
    assert route_problems(d(escalate=True, route=None), 20) == []
