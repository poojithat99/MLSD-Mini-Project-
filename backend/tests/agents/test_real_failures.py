"""Reproductions of the failures seen against real Gemini (smoke runs, 2026-11-09).

``GeminiLike`` mimics what the real model did on each case, while following
whatever output fields the system prompt asks for (as a real model would):
  ok   correct answer, but states case facts ("did not share payment details") as claims; sends "note": null
  41   cites only the SBI negligence clause and recommends decline
  ooc  forces merchant_dispute / chargeback on a crypto investment loss
  41_otp (second real run) searches with "OTP" wording, never cites the negligence clause,
       and settles on the Table 2 row alone
Its verifier is strict and literal: a claim is supported only by passages, a
case fact only by the request text.
"""
from __future__ import annotations

import json
import re
from datetime import date

from backend.agents.analysis_agent import CREDENTIALS_FACT
from backend.agents.fake_scenarios import ScriptedAgents, _sentence_with
from backend.agents.graph import run_dispute_graph
from backend.agents.llm import FakeLLM, LLMClient, LLMResponse, ToolCall
from backend.agents.state import DisputeInput
from backend.tests.agents.test_llm import make_config

OOC = DisputeInput(narrative="I lost money on a cryptocurrency exchange investment and want it back.",
                   channel="card_not_present", amount=90000, transaction_date=date(2026, 10, 1),
                   communication_date=date(2026, 10, 1), reported_date=date(2026, 10, 2))

# case fact -> text that must appear in the request for the fact to be "consistent"
FACT_ANCHORS = {
    "The transaction was an unauthorised card-not-present transaction.": "card-not-present charge i never made",
    "The customer did not share any payment details.": "never shared any details",
    "The customer shared an OTP with a caller claiming to be from the bank.": "shared an otp",
    "The customer authorised a payment to a cryptocurrency exchange as an investment.": "cryptocurrency exchange",
    "The customer shared their card PIN with a friend.": "pin with a friend",
    "The customer shared payment credentials (an OTP) with a caller.": "shared an otp",
    CREDENTIALS_FACT: "shared an otp",
}
UNSUPPORTED_REGULATORY = {
    "The customer bears liability for authorised merchant transactions but may raise a chargeback.",
}
QUERIES = {
    "ok": ("third party breach customer notifies the bank within three working days zero liability",
           "negligence by a customer shared the payment credentials"),
    "41": ("negligence by a customer shared the payment credentials entire loss",
           "beyond seven working days customer liability"),
    "41_otp": ("customer liability sharing OTP", "OTP shared Board approved policy"),
    "ooc": ("cryptocurrency exchange investment unauthorised transaction",
            "merchant dispute chargeback card not present"),
}


class GeminiLike:
    def __init__(self, kind: str, contradicting_fact: bool = False):
        self.kind = kind
        self.contradicting_fact = contradicting_fact
        self._n = 0

    def __call__(self, messages, tools):
        if "verification agent" in messages[0]["content"]:
            return self._verify(json.loads(messages[1]["content"]))
        searches = [tc for m in messages if m["role"] == "assistant"
                    for tc in m.get("tool_calls") or [] if tc["function"]["name"] == "search_corpus"]
        if len(searches) < 2:
            self._n += 1
            return LLMResponse(tool_calls=[ToolCall(f"g{self._n}", "search_corpus",
                                                    {"query": QUERIES[self.kind][len(searches)]})])
        return LLMResponse(content=json.dumps(self._draft(messages)))

    # -------------------------------------------------------------- analyst
    def _draft(self, messages) -> dict:
        system, case = messages[0]["content"], messages[1]["content"]
        passages = []
        for m in messages:
            if m["role"] == "tool":
                data = json.loads(m["content"])
                passages += data if isinstance(data, list) else []
        row = re.search(r"LIABILITY_TABLE_ROW: (\{.*\})", case)
        row = json.loads(row.group(1)) if row else None

        def cite(doc, clause, kw):
            for p in passages:
                if p["doc_id"] == doc and clause in p["clause_ids"]:
                    q = _sentence_with(p["text"], kw)
                    if q:
                        return {"doc_id": doc, "clause_id": clause, "quote": q}
            return None

        d = {"classification": "cnp_fraud", "route": None, "basis": "", "liability_position": "",
             "citations": [], "escalate": False, "reason_code": None, "competing_provisions": [],
             "note": None, "in_scope": True}                                     # Gemini sends null
        if self.kind == "ok":
            facts = ["The transaction was an unauthorised card-not-present transaction.",
                     "The customer did not share any payment details."]
            claims = ["Under para 6(ii), a customer has zero liability for a third party breach notified "
                      "within three working days."]
            cites = [cite("rbi_circular_2017", "6(ii)", "within three working days"),
                     row or cite("rbi_circular_2017", "Table 2", "Within 3 working days")]
            d.update(route="provisional_credit", basis="RBI para 6(ii) and Table 2.",
                     liability_position="Zero liability.")
        elif self.kind == "41":
            facts = ["The customer shared an OTP with a caller claiming to be from the bank."]
            claims = ["Under SBI 4.6.2(b)(i), where the loss is due to customer negligence such as sharing "
                      "payment credentials, the customer bears the entire loss until reporting."]
            cites = [cite("sbi_policy_2021", "4.6.2(b)(i)", "entire loss")]
            d.update(route="decline", basis="SBI 4.6.2(b)(i): customer negligence.",
                     liability_position="Customer bears the entire loss.")
        elif self.kind == "41_otp":
            facts = ["The customer shared payment credentials (an OTP) with a caller."]
            claims = ["For a report beyond 7 working days, liability is as per the bank's Board approved policy."]
            cites = [row or cite("rbi_circular_2017", "Table 2", "Beyond 7 working days")]
            d.update(route="decline", basis="RBI Table 2: beyond 7 working days.",
                     liability_position="As per the bank's Board-approved policy.")
            if "credentials_shared" in system:          # asked explicitly, the model answers from the facts
                d["credentials_shared"] = True
        else:
            facts = ["The customer authorised a payment to a cryptocurrency exchange as an investment."]
            claims = ["The customer bears liability for authorised merchant transactions but may raise a chargeback."]
            first = passages[0]
            cites = [{"doc_id": first["doc_id"], "clause_id": first["clause_ids"][0],
                      "quote": re.split(r"(?<=\.)\s+", first["text"])[0].rstrip(".")}]
            d.update(classification="merchant_dispute", route="chargeback", basis="Card network dispute rules.")
            if "in_scope" in system:                    # a model told how to say "not covered" uses it
                d["in_scope"] = False
        if self.contradicting_fact:
            facts = ["The customer shared their card PIN with a friend."]
        d["citations"] = [c for c in cites if c]
        if "case_facts" in system:                      # the prompt asks for facts separately
            d["case_facts"], d["claims"] = facts, claims
        else:
            d["claims"] = facts + claims
        return d

    # -------------------------------------------------------------- verifier
    def _verify(self, payload: dict) -> LLMResponse:
        request_text = json.dumps(payload.get("request", {})).lower()
        claims = [{"index": c["index"],
                   "supported": c["claim"] not in FACT_ANCHORS and c["claim"] not in UNSUPPORTED_REGULATORY}
                  for c in payload["claims"]]
        facts = [{"index": f["index"],
                  "status": "consistent" if FACT_ANCHORS.get(f["fact"], "\0") in request_text else "contradicts"}
                 for f in payload.get("case_facts", [])]
        return LLMResponse(content=json.dumps({"claims": claims, "case_facts": facts, "missed_provision": None}))


def run(request, retriever, kind, **kw):
    llm = LLMClient(FakeLLM(GeminiLike(kind, **kw)), make_config(), sleep=lambda s: None)
    return run_dispute_graph(request, "run-g", "corr-g", llm=llm, retriever=retriever)


def model_calls(r) -> int:
    return sum(1 for t in r.trace if t.type == "llm")


# ------------------------------------------------------------------ 1. ok
def test_ok_case_facts_do_not_fail_verification(retriever, request_ok):
    r = run(request_ok, retriever, "ok")
    assert r.status == "recommended" and r.route == "provisional_credit"
    assert r.verification.passed and r.verification.attempts == 1
    cited = {(c.doc_id, c.clause_id) for c in r.citations}
    assert ("rbi_circular_2017", "6(ii)") in cited and ("rbi_circular_2017", "Table 2") in cited
    assert model_calls(r) <= 8


def test_case_fact_contradicting_the_request_is_a_failure(retriever, request_ok):
    r = run(request_ok, retriever, "ok", contradicting_fact=True)
    assert r.status == "escalated" and r.escalation.reason_code.value == "verification_failed"
    verdicts = [t.detail for t in r.trace if t.name == "verifier.verdict"]
    assert any("contradicts the request" in u for v in verdicts for u in v["unsupported_claims"])


def test_verifier_gets_the_request_as_delimited_data(retriever, request_ok):
    llm = LLMClient(FakeLLM(GeminiLike("ok")), make_config(), sleep=lambda s: None)
    run_dispute_graph(request_ok, "r", "c", llm=llm, retriever=retriever)
    (call,) = [c for c in llm.primary.calls if "verification agent" in c[0]["content"]]
    payload = json.loads(call[1]["content"])
    assert payload["request"]["narrative"].startswith("<<<UNTRUSTED narrative")
    assert set(payload) == {"case_facts", "request", "claims", "outcome", "passages", "facts_computed_by_code"}


# ------------------------------------------------------------------ 2. section 4.1
def test_41_negligence_vs_table_row_escalates_with_both(retriever, request_41):
    r = run(request_41, retriever, "41")
    assert r.status == "escalated" and r.route is None
    assert r.escalation.reason_code.value == "conflicting_provisions"
    cited = {(c.doc_id, c.clause_id): c.quote for c in r.citations}
    assert ("rbi_circular_2017", "7(i)") in cited
    assert "Beyond 7 working days" in cited[("rbi_circular_2017", "Table 2")]
    assert r.escalation.competing_provisions == ["rbi_circular_2017 7(i)", "rbi_circular_2017 Table 2"]
    assert r.verification.passed


def test_41_model_that_never_cites_negligence_still_escalates_with_both(retriever, request_41):
    r = run(request_41, retriever, "41_otp")
    assert r.status == "escalated" and r.escalation.reason_code.value == "conflicting_provisions"
    cited = {(c.doc_id, c.clause_id) for c in r.citations}
    assert {("rbi_circular_2017", "7(i)"), ("rbi_circular_2017", "Table 2")} <= cited
    assert r.verification.passed and model_calls(r) <= 8


# ------------------------------------------------------------------ 3. out of corpus
def test_ooc_customer_authorised_investment_is_out_of_corpus(retriever):
    r = run(OOC, retriever, "ooc")
    assert r.status == "escalated" and r.escalation.reason_code.value == "out_of_corpus"
    assert r.route is None and r.citations == [] and r.classification is None
    assert model_calls(r) <= 8


# ------------------------------------------------------------------ D. unverified text on escalation
def test_unverified_escalation_shows_no_basis_or_citations(retriever, request_ok):
    llm = LLMClient(FakeLLM(ScriptedAgents(fabricate="always")), make_config(), sleep=lambda s: None)
    r = run_dispute_graph(request_ok, "r", "c", llm=llm, retriever=retriever)
    assert r.escalation.reason_code.value == "verification_failed"
    assert r.basis == "" and r.liability_position == "" and r.citations == []
    assert r.escalation.competing_provisions == []
    (audit,) = [t for t in r.trace if t.name == "escalate.unverified_draft"]
    assert audit.detail["basis"] and audit.detail["citations"]            # kept for the analyst, labelled
