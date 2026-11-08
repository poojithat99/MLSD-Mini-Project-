"""Verification agent (requirements A2, A3; brief section 5 step 4).

A separate agent with its own prompt.  It sees only the CLAIMS, the CITED
PASSAGES and facts computed by code, never the analysis agent's messages, tool
calls or reasoning, so it cannot be talked into agreeing.

(a) Deterministic citation check (no model): every citation must point at a
    passage actually retrieved in this run (same doc_id, clause_id inside that
    passage) and its quote must be a normalised substring of that passage's text
    (Member 1's ``normalize_for_match``).
(b) Model check: is each claim supported by the cited passages?

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
from .state import AnalysisDraft, Citation, ReasonCode, RetrievedPassage, Verdict

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
            problems.append(f"citation {c.doc_id} {c.clause_id} does not resolve to retrieved text: {c.quote[:80]!r}")
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
    """(b) Does each claim follow from the cited passages?  The verifier never sees
    the analyst's messages, tool calls or reasoning: only claims + passages + facts."""
    system, _ = load_prompt(run.verification_prompt)
    claims = _claims(draft)
    payload = {
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
    unsupported = [claims[r["index"]] for r in out.get("claims", [])
                   if not r.get("supported") and isinstance(r.get("index"), int) and r["index"] < len(claims)]
    missed = out.get("missed_provision")
    if unsupported:
        return Verdict(decision="retry", strategy_hint="rewrite_query", unsupported_claims=unsupported)
    if missed and not draft.escalate:
        return Verdict(decision="retry", strategy_hint="wider_k",
                       unsupported_claims=[f"answer ignores a modifying provision: {missed}"])
    return Verdict(decision="pass")


def verify(run, draft: AnalysisDraft) -> Verdict:
    # (a) deterministic, free, always runs first
    problems = deterministic_check(draft, run.evidence)
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
