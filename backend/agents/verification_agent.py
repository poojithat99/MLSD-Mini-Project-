"""Verification agent (requirement A3).

(a) Deterministic citation check: every citation must point at a passage that was
    actually retrieved in this run (same doc_id, clause_id inside that passage)
    and its quote must be a normalised substring of that passage's text.
"""
from __future__ import annotations

from backend.retrieval.ingest import normalize_for_match
from backend.retrieval.search import _norm_clause_id

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


def verify(run, draft: AnalysisDraft) -> Verdict:
    problems = deterministic_check(draft, run.evidence)
    if problems:
        hint = "neighbour_clause" if any("does not resolve" in p for p in problems) else "wider_k"
        return Verdict(decision="retry", strategy_hint=hint, unsupported_claims=problems)
    return Verdict(decision="pass")
