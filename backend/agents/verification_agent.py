"""Verification agent (requirements A2, A3; brief section 5 step 4).

A separate agent with its own prompt.  It sees only the CLAIMS and CASE FACTS,
the CITED PASSAGES, the request's own fields (as delimited untrusted data) and
facts computed by code, never the analysis agent's messages, tool calls or
reasoning, so it cannot be talked into agreeing.

(a) Deterministic citation check (no model): every citation must point at a
    passage actually retrieved in this run (same doc_id, clause_id inside that
    passage) and its quote must be a normalised substring of that passage's text
    (Member 1's ``normalize_for_match``).
(b) Model check: are the regulatory claims supported by the cited passages, and are the
    case facts consistent with the request?

Returns a Verdict: pass | retry (with a strategy hint) | escalate.
If the verifier's model is unavailable, only (a) runs and the run is forced to
escalate: an answer whose claims were not checked is never released.
"""
from __future__ import annotations

import json

from backend.retrieval.ingest import normalize_for_match
from backend.retrieval.search import _norm_clause_id

from .llm import LLMError, parse_json_object
from .observability import log_event
from .prompts import load_prompt
from .state import LOAD_BEARING_FACTS, AnalysisDraft, Citation, ReasonCode, RetrievedPassage, Verdict

MIN_QUOTE_CHARS = 20          # a quote must be long enough to mean something ("the bank" is not evidence)


def _clause_key(clause_id: str) -> str:
    return _norm_clause_id(clause_id).replace(" ", "")


def citation_resolves(c: Citation, evidence: dict[str, RetrievedPassage]) -> bool:
    if len(c.quote.strip()) < MIN_QUOTE_CHARS:
        return False
    quote = normalize_for_match(c.quote)
    for p in evidence.values():
        if p.doc_id != c.doc_id:
            continue
        if _clause_key(c.clause_id) not in {_clause_key(x) for x in p.clause_ids}:
            continue
        if quote in normalize_for_match(p.text + " " + p.lead_in):
            return True
    return False


def deterministic_check(draft: AnalysisDraft, evidence: dict[str, RetrievedPassage]) -> list[str]:
    """Problems found without any model.  Empty list = passed."""
    problems = []
    for c in draft.citations:
        if not citation_resolves(c, evidence):
            problems.append(f"citation {c.doc_id} {c.clause_id} does not resolve to retrieved text: {c.quote[:300]!r}")
    if draft.escalate:
        if draft.reason_code in (ReasonCode.CONFLICTING_PROVISIONS, ReasonCode.CONTESTED_FAULT):
            cited = {(c.doc_id, _clause_key(c.clause_id)) for c in draft.citations}
            if len(cited) < 2:
                problems.append("an escalation for conflicting provisions / contested fault must cite both provisions")
    else:
        if draft.route is None:
            problems.append("a recommendation must name a route")
        if not draft.citations:
            problems.append("a recommendation must cite at least one provision")
    return problems


def _cited_passages(draft: AnalysisDraft, evidence: dict[str, RetrievedPassage]) -> list[dict]:
    """The retrieved passages the citations point to: the only text the verifier reads."""
    keys = {(c.doc_id, _clause_key(c.clause_id)) for c in draft.citations}
    out = []
    for p in evidence.values():
        if any((p.doc_id, _clause_key(x)) in keys for x in p.clause_ids):
            out.append({"doc_id": p.doc_id, "clause_ids": p.clause_ids, "context": p.lead_in, "text": p.text})
    return out


def _claims(draft: AnalysisDraft) -> list[str]:
    claims = list(draft.claims)
    for extra in (draft.liability_position, draft.basis):
        if extra and extra not in claims:
            claims.append(extra)
    return claims


def llm_check(run, draft: AnalysisDraft) -> Verdict:
    """(b) Model check, in two parts.  The verifier never sees the analyst's messages,
    tool calls or reasoning.

      case_facts  statements about THIS dispute -> checked against the request's own
                  fields (passed as delimited, untrusted data).  "consistent" or
                  "not_stated" is fine; "contradicts" (or a regulatory statement filed
                  as a fact: "not_a_case_fact") is a failure.
      claims      regulatory statements -> must be supported by the cited passages.

    A statement the model did not give a verdict on counts as unverified (failure)."""
    system, _ = load_prompt(run.verification_prompt)
    claims, facts = _claims(draft), list(draft.case_facts)
    payload = {
        "case_facts": [{"index": i, "fact": f} for i, f in enumerate(facts)],
        "request": run.toolbox.get_request_fields(),        # free text already wrapped as untrusted data
        "claims": [{"index": i, "claim": c} for i, c in enumerate(claims)],
        "outcome": "escalate: " + (draft.reason_code.value if draft.reason_code else "") if draft.escalate
                   else f"route: {draft.route}",
        "passages": _cited_passages(draft, run.evidence),
        "facts_computed_by_code": {"reporting_delay": run.delay},
    }
    resp = run.call_llm("verification", [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ])
    out = parse_json_object(resp.content)

    claim_ok = {r.get("index"): bool(r.get("supported")) for r in out.get("claims") or []}
    fact_status = {r.get("index"): r.get("status") for r in out.get("case_facts") or []}
    problems = [f"unsupported by the cited passages: {c}" for i, c in enumerate(claims) if not claim_ok.get(i)]
    for i, f in enumerate(facts):
        status = fact_status.get(i)
        if f in LOAD_BEARING_FACTS and status != "consistent":
            problems.append(f"load-bearing case fact not confirmed by the request ({status}): {f}")
        elif status == "contradicts":
            problems.append(f"case fact contradicts the request: {f}")
        elif status == "not_a_case_fact":
            problems.append(f"regulatory statement filed as a case fact (not checked against passages): {f}")
        elif status not in ("consistent", "not_stated"):
            problems.append(f"case fact not verified: {f}")
    if problems:
        return Verdict(decision="retry", strategy_hint="rewrite_query", unsupported_claims=problems)
    missed = out.get("missed_provision")
    if missed and not draft.escalate:
        return Verdict(decision="retry", strategy_hint="wider_k",
                       unsupported_claims=[f"answer ignores a modifying provision: {missed}"])
    return Verdict(decision="pass")


UNAUTHORISED_CLASSES = {"cnp_fraud", "skimming", "upi_social_eng"}
CARD_DISPUTE_CLASSES = {"duplicate", "merchant_dispute"}


def route_problems(draft: AnalysisDraft, working_days: int | None, alert_date_known: bool = True) -> list[str]:
    """Deterministic route check: is the proposed route consistent with the liability rules,
    given the model's own findings and the delay computed by code?  Applies in this order:

      card dispute (duplicate / merchant_dispute)        -> chargeback
      bank at fault (bank_fault, RBI 6(i))               -> provisional_credit citing RBI 6(i),
                                                            whatever the reporting delay
      credentials shared                                  -> not checked here (escalation decides)
      communication (alert) date unknown                  -> no recommendation: Table 2 counts from receipt
                                                            of the communication, so escalate contested_fault
      otherwise a third-party breach, by RBI Table 2:
        <= 7 working days (zero / limited liability)      -> provisional_credit
        >  7 working days (bank's Board-approved policy;
            SBI Table 2: customer liability is full)      -> decline

    Escalations and out-of-scope drafts are not checked.  A mismatch is a verification problem:
    the analyst is asked to retry; the code never changes the route itself."""
    if draft.escalate or not draft.in_scope or draft.route is None:
        return []
    cls, route = draft.classification, draft.route
    if cls in CARD_DISPUTE_CLASSES:
        return [] if route == "chargeback" else [f"a {cls} is routed to chargeback, not {route}"]
    if cls not in UNAUTHORISED_CLASSES or draft.credentials_shared is True:
        return []
    if draft.bank_fault is True:
        problems = []
        if route != "provisional_credit":
            problems.append(f"bank_fault is true: RBI 6(i) gives zero liability irrespective of when the "
                            f"transaction is reported, so the route must be provisional_credit (got {route})")
        if not any(c.doc_id == "rbi_circular_2017" and _clause_key(c.clause_id) == "6(i)" for c in draft.citations):
            problems.append("bank_fault is true: the recommendation must cite rbi_circular_2017 6(i)")
        return problems
    if not alert_date_known:
        return [f"the date the customer received the bank's communication is unknown, so the RBI Table 2 row "
                f"cannot be applied (it counts from receipt) and fault is unsettled (RBI para 5: mandatory SMS "
                f"alerts): escalate with reason_code contested_fault instead of route {route}"]
    if working_days is None:
        return []
    expected = "provisional_credit" if working_days <= 7 else "decline"
    if route != expected:
        return [f"route {route} contradicts the liability table for a third-party breach reported after "
                f"{working_days} working days (RBI Table 2; SBI Table 2 beyond 7): expected {expected}"]
    return []


def verify(run, draft: AnalysisDraft) -> Verdict:
    # (a) deterministic, free, always runs first: citations (unchanged), then the route check
    problems = deterministic_check(draft, run.evidence)
    if not problems:
        # The table rule needs the delay counted from receipt of the bank's communication; when that date
        # is unknown our calculator counts from the transaction date, an assumption no route may rest on.
        delay = run.delay or {}
        known = delay.get("counted_from_field", "communication_date") == "communication_date"
        problems = route_problems(draft, delay.get("working_days"), alert_date_known=known)
    if problems:
        hint = "neighbour_clause" if any("does not resolve" in p for p in problems) else "wider_k"
        log_event("verification", stage="deterministic", decision="retry", problems=problems)
        return Verdict(decision="retry", strategy_hint=hint, unsupported_claims=problems)
    if not draft.citations:                  # e.g. out_of_corpus escalation: nothing for a model to check
        return Verdict(decision="pass")
    # (b) model check; if the verifier's model is down we must not release the answer
    try:
        verdict = llm_check(run, draft)
    except LLMError as exc:                  # includes ProviderUnavailable / CircuitOpen
        log_event("verification", stage="llm", decision="escalate", error=str(exc))
        return Verdict(decision="escalate", reason_code=ReasonCode.PROVIDER_UNAVAILABLE,
                       note=f"verifier model unavailable ({exc}); deterministic citation check passed, "
                            "claims not verified")
    log_event("verification", stage="llm", decision=verdict.decision, unsupported=verdict.unsupported_claims)
    return verdict
