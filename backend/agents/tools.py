"""Read-only tools the analysis agent may choose to call (requirement A1).

    search_corpus(query, k, expand_refs)   semantic search over the corpus      (Member 1's Retriever.search)
    get_clause(doc_id, clause_id)          exact clause / table lookup by id    (Member 1's Retriever.get_clause)
    get_request_fields()                   the dispute's own structured fields
    compute_reporting_delay()              working days from bank alert to report, computed in CODE

Every call:
  * is charged to the run's Budget (tool-call limit + wall-clock deadline),
  * runs under a timeout (TOOL_TIMEOUT_S),
  * adds every passage it returns to the run's evidence set (the only text the
    verifier will accept citations from),
  * appends a trace entry (tool name, arguments, chunk ids returned, latency).

Nothing here writes anywhere: all tools are read-only.
"""
from __future__ import annotations

import contextvars
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import date, timedelta
from typing import Any, Callable, Protocol

from .observability import log_event, span
from .state import TOOL_TIMEOUT_S, Budget, DisputeInput, RetrievedPassage, TraceEntry


# --------------------------------------------------------------------------- #
# working days (deterministic, never the model)
# --------------------------------------------------------------------------- #
WORKING_DAY_ASSUMPTION = (
    "Counted from the day after the bank's communication up to and including the reporting date "
    "(RBI circular Table 2 note: 'excluding the date of receiving the communication'). The customer's "
    "home-branch calendar is not available, so Sundays and the 2nd and 4th Saturdays of the month "
    "(RBI's standard bank holidays) are excluded, plus any listed public holidays."
)


def is_working_day(d: date, holidays: frozenset[date] = frozenset()) -> bool:
    if d in holidays or d.weekday() == 6:                   # Sunday
        return False
    if d.weekday() == 5:                                    # Saturday: 2nd and 4th are holidays
        nth = (d.day - 1) // 7 + 1
        return nth not in (2, 4)
    return True


def working_days_between(communication: date, reported: date, holidays: frozenset[date] = frozenset()) -> int:
    """Working days after ``communication`` up to and including ``reported``."""
    if reported < communication:
        raise ValueError("reported date is before the communication date")
    days, d = 0, communication + timedelta(days=1)
    while d <= reported:
        days += is_working_day(d, holidays)
        d += timedelta(days=1)
    return days


def compute_reporting_delay(request: DisputeInput, holidays: frozenset[date] = frozenset()) -> dict:
    """Reporting delay as the liability table counts it.  Falls back to the transaction
    date when the communication date is unknown, and says so."""
    start = request.communication_date or request.transaction_date
    source = "communication_date" if request.communication_date else "transaction_date (communication date unknown)"
    return {
        "working_days": working_days_between(start, request.reported_date, holidays),
        "calendar_days": (request.reported_date - start).days,
        "counted_from": start.isoformat(),
        "counted_from_field": source,
        "reported_date": request.reported_date.isoformat(),
        "assumption": WORKING_DAY_ASSUMPTION,
    }


# --------------------------------------------------------------------------- #
# the retrieval interface we depend on
# --------------------------------------------------------------------------- #
class RetrieverPort(Protocol):
    """What the agents need from retrieval.  ``backend.retrieval.search.Retriever``
    satisfies it today; an HTTP client to a separate retrieval service can later."""

    def search(self, query: str, k: int | None = None, *, expand_refs: bool = False) -> list: ...
    def get_clause(self, doc_id: str, clause_id: str, part: str | None = None) -> list: ...
    def get_chunk(self, chunk_id: str) -> Any: ...


def to_record(p: Any, retriever: Any) -> RetrievedPassage:
    """Convert Member 1's ``Passage`` into our evidence record, adding every clause id
    the chunk contains (a chunk can hold several, e.g. 7(i) and 7(ii))."""
    chunk = getattr(retriever, "by_id", {}).get(p.chunk_id, {})
    clause_ids = list(chunk.get("clause_ids") or [x for x in (p.supporting_clause,) if x])
    table_no = chunk.get("table_no") or (re.match(r"Table\s+(\d+)", p.clause or "") or [None, None])[1]
    if table_no:
        clause_ids.append(f"Table {table_no}")
    return RetrievedPassage(
        chunk_id=p.chunk_id, doc_id=p.doc_id, clause_ids=clause_ids, label=p.clause, citation=p.citation,
        text=p.text, lead_in=p.lead_in or "", supporting_text=p.supporting_text or "",
        score=float(p.score), via=p.via or "",
        refs=list(p.refs or []), referenced_by=list(p.referenced_by or []),
    )


# --------------------------------------------------------------------------- #
# timeouts
# --------------------------------------------------------------------------- #
class ToolTimeout(RuntimeError):
    pass


_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent-tool")


def run_with_timeout(fn: Callable[[], Any], timeout_s: float) -> Any:
    """Run ``fn`` in a worker thread; give up waiting after ``timeout_s``.
    (Python cannot kill the thread, but the agent stops waiting and moves on.)"""
    ctx = contextvars.copy_context()                     # keep run_id / correlation_id in logs
    future = _pool.submit(ctx.run, fn)
    try:
        return future.result(timeout=timeout_s)
    except FutureTimeout as exc:
        raise ToolTimeout(f"tool call exceeded {timeout_s:.1f} s") from exc


# --------------------------------------------------------------------------- #
# tool specs shown to the model (OpenAI function-calling format)
# Kept to the JSON-schema subset Gemini's OpenAI-compatible endpoint accepts: no "default",
# "minimum" or "maximum" (k is clamped in code), and no "parameters" at all for tools without
# arguments (an empty "properties" object can be rejected).  Check: python -m backend.agents.llm_check --agent-tools
# --------------------------------------------------------------------------- #
TOOL_SPECS: list[dict] = [
    {"type": "function", "function": {
        "name": "search_corpus",
        "description": "Semantic search over the RBI circular, the SBI policy and the RBI Ombudsman scheme. "
                       "Returns passages with chunk_id, doc_id, clause_ids and full text. The corpus uses formal "
                       "wording (e.g. 'payment credentials', not 'OTP').",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "k": {"type": "integer", "description": "number of passages, 1 to 10 (default 5)"},
            "expand_refs": {"type": "boolean",
                            "description": "also return passages that cite, or are cited by, the hits "
                                           "(default false)"},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_clause",
        "description": "Exact lookup of a clause or table by identifier, e.g. doc_id='rbi_circular_2017', "
                       "clause_id='7(ii)' or 'Table 2'.",
        "parameters": {"type": "object", "properties": {
            "doc_id": {"type": "string", "enum": ["rbi_circular_2017", "sbi_policy_2021", "rbi_ios_2026"]},
            "clause_id": {"type": "string"},
        }, "required": ["doc_id", "clause_id"]}}},
    {"type": "function", "function": {
        "name": "get_request_fields",
        "description": "The dispute's structured fields (channel, amount, dates, account type)."}},
    {"type": "function", "function": {
        "name": "compute_reporting_delay",
        "description": "Working days between the bank's communication and the customer's report, as the "
                       "liability table counts them. Always use this; never count days yourself."}},
]
TOOL_NAMES = {t["function"]["name"] for t in TOOL_SPECS}
CODE_ONLY_TOOLS = {"fetch_chunk"}     # used by the agent's own retry code, never offered to the model


# --------------------------------------------------------------------------- #
# the toolbox bound to one run
# --------------------------------------------------------------------------- #
class Toolbox:
    def __init__(
        self, retriever: RetrieverPort, request: DisputeInput, budget: Budget,
        evidence: dict[str, RetrievedPassage], trace: list[TraceEntry],
        timeout_s: float = TOOL_TIMEOUT_S, safe_fields: dict | None = None,
    ):
        self.retriever = retriever
        self.request = request
        self.budget = budget
        self.evidence = evidence          # chunk_id -> passage, shared with the verifier
        self.trace = trace
        self.timeout_s = timeout_s
        self.safe_fields = safe_fields    # request fields after guardrails (free text wrapped as data)
        self.queries: list[str] = []      # every search query issued in this run

    # -- the four tools ---------------------------------------------------- #
    def search_corpus(self, query: str, k: int = 5, expand_refs: bool = False) -> list[RetrievedPassage]:
        k = max(1, min(int(k), 10))
        self.queries.append(query)
        hits = self.retriever.search(query, k, expand_refs=bool(expand_refs))
        return self._remember(hits)

    def get_clause(self, doc_id: str, clause_id: str) -> list[RetrievedPassage]:
        return self._remember(self.retriever.get_clause(doc_id, clause_id))

    def fetch_chunk(self, chunk_id: str) -> list[RetrievedPassage]:
        """Re-read a passage by chunk id (follows a cross-reference)."""
        p = self.retriever.get_chunk(chunk_id)
        return self._remember([p] if p else [])

    def get_request_fields(self) -> dict:
        if self.safe_fields is not None:
            return self.safe_fields
        return self.request.model_dump(mode="json")

    def compute_reporting_delay(self) -> dict:
        return compute_reporting_delay(self.request)

    # -- dispatcher used by the agent loop --------------------------------- #
    def run(self, name: str, arguments: dict) -> str:
        """Execute one tool call; returns JSON text for the model.  Tool errors are
        returned to the model as data, budget/deadline errors propagate."""
        self.budget.charge_tool()
        start = time.perf_counter()
        detail: dict[str, Any] = {"arguments": arguments}
        with span("tool." + name, tool=name):
            try:
                if name not in TOOL_NAMES | CODE_ONLY_TOOLS:
                    raise ValueError(f"unknown tool {name!r}")
                fn = getattr(self, name)
                timeout = min(self.timeout_s, max(0.1, self.budget.remaining_s()))
                out = run_with_timeout(lambda: fn(**arguments), timeout)
                if isinstance(out, list):
                    detail["chunk_ids"] = [p.chunk_id for p in out]
                    payload = [self._for_model(p) for p in out] or {"result": "no passages found"}
                else:
                    payload = out
            except (ToolTimeout, ValueError, TypeError, KeyError) as exc:
                detail["error"] = str(exc)
                payload = {"error": str(exc)}
        latency = (time.perf_counter() - start) * 1000
        self.trace.append(TraceEntry(type="tool", name=name, latency_ms=latency, detail=detail))
        log_event("tool_call", tool=name, latency_ms=round(latency, 1), **detail)
        return json.dumps(payload, ensure_ascii=False, default=str)

    # -- helpers ----------------------------------------------------------- #
    def _remember(self, passages: list) -> list[RetrievedPassage]:
        records = [to_record(p, self.retriever) for p in passages]
        for r in records:
            self.evidence.setdefault(r.chunk_id, r)
        return records

    @staticmethod
    def _for_model(p: RetrievedPassage) -> dict:
        return {"chunk_id": p.chunk_id, "doc_id": p.doc_id, "clause_ids": p.clause_ids, "citation": p.citation,
                "context": p.lead_in, "text": p.text, "refs": p.refs, "via": p.via}
