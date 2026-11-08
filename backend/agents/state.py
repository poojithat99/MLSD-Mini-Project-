"""Data types shared by the graph, the two agents and the tools.

* ``DisputeInput``   the request fields the agents read (see note below)
* ``DisputeResult``  the one output of a run (what the API returns)
* ``ReasonCode``     why a run escalated
* ``Budget``         the loop bounds of requirement A1: steps, tool calls, wall-clock
* ``RetrievedPassage``  a passage the agents have actually read; the verifier's evidence set
* ``GraphState``     what flows between LangGraph nodes

Note on ``DisputeInput``: the published request contract belongs to Member 3
(``backend/app/schemas.py``), which does not exist yet.  These are the fields the
agents need; validation (types, ranges, cross-field) happens at the API boundary
before a run starts, so here we only declare shapes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------- #
# vocabularies
# --------------------------------------------------------------------------- #
Classification = Literal["cnp_fraud", "skimming", "upi_social_eng", "merchant_dispute", "duplicate"]
Route = Literal["chargeback", "provisional_credit", "decline"]
CLASSIFICATIONS: tuple[str, ...] = Classification.__args__
ROUTES: tuple[str, ...] = Route.__args__


class ReasonCode(str, Enum):
    CONTESTED_FAULT = "contested_fault"                # who was at fault cannot be settled from the facts
    CONFLICTING_PROVISIONS = "conflicting_provisions"  # two provisions apply and point different ways
    OUT_OF_CORPUS = "out_of_corpus"                    # the corpus does not cover the case
    VERIFICATION_FAILED = "verification_failed"        # answer not supported by retrieved text after retries
    BUDGET_EXHAUSTED = "budget_exhausted"              # step or tool-call limit hit
    TIMEOUT = "timeout"                                # wall-clock deadline hit
    PROVIDER_UNAVAILABLE = "provider_unavailable"      # model provider down / breaker open


# --------------------------------------------------------------------------- #
# input
# --------------------------------------------------------------------------- #
class DisputeInput(BaseModel):
    narrative: str                         # customer's own words (untrusted free text)
    channel: str                           # e.g. "card_not_present", "card_present", "atm", "upi"
    amount: float
    transaction_date: date
    communication_date: date | None = None  # when the bank's SMS/e-mail alert reached the customer
    reported_date: date
    merchant: str | None = None            # untrusted free text
    account_type: str | None = None        # e.g. "savings", "credit_card"


FREE_TEXT_FIELDS = ("narrative", "merchant")   # scanned for prompt injection before any model sees them


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
class Citation(BaseModel):
    clause_id: str        # clause label as the corpus numbers it: "7(i)", "Table 2", "4.6.2(b)(i)"
    doc_id: str           # "rbi_circular_2017" | "sbi_policy_2021" | "rbi_ios_2026"
    quote: str            # verbatim text from that clause


class Escalation(BaseModel):
    reason_code: ReasonCode
    competing_provisions: list[str] = Field(default_factory=list)   # "doc_id clause_id", e.g. "rbi_circular_2017 7(i)"
    note: str = ""


class Verification(BaseModel):
    passed: bool = False
    attempts: int = 0
    unsupported_claims: list[str] = Field(default_factory=list)


class TraceEntry(BaseModel):
    type: Literal["node", "tool", "llm"]
    name: str
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    detail: dict[str, Any] = Field(default_factory=dict)   # query, chunk ids returned, error ... (D5)


class DisputeResult(BaseModel):
    run_id: str
    status: Literal["recommended", "escalated"]
    classification: Classification | None = None
    reporting_delay_working_days: int | None = None
    route: Route | None = None
    basis: str = ""
    liability_position: str = ""
    citations: list[Citation] = Field(default_factory=list)
    escalation: Escalation | None = None
    verification: Verification = Field(default_factory=Verification)
    prompt_version: str = ""
    trace: list[TraceEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _confident_only_if_verified(self) -> "DisputeResult":
        """The type itself refuses a confident answer that was not verified."""
        if self.status == "recommended":
            if self.route is None or self.escalation is not None or not self.verification.passed or not self.citations:
                raise ValueError("a recommendation needs a route, citations and passed verification, and no escalation")
        else:
            if self.escalation is None or self.route is not None:
                raise ValueError("an escalation needs a reason code and must not carry a route")
        return self


# --------------------------------------------------------------------------- #
# loop bounds (A1)
# --------------------------------------------------------------------------- #
class BudgetExhausted(RuntimeError):
    """Step or tool-call limit reached."""


class DeadlineExceeded(BudgetExhausted):
    """Wall-clock limit reached."""


# Defaults sized for the declared SLO (plan section 9: 60 s hard cap per run).  A normal run makes
# ~5 model calls (analyst 4 incl. the modifier check, verifier 1); 14 leaves room for both retries.
MAX_STEPS = 14          # model calls per run, both agents, all retries
MAX_TOOL_CALLS = 16     # tool calls per run
WALL_CLOCK_S = 60.0     # hard cap per run
TOOL_TIMEOUT_S = 10.0   # per tool call


@dataclass
class Budget:
    """Shared by every agent and every retry of one run, so retries cannot extend it."""
    max_steps: int = MAX_STEPS
    max_tool_calls: int = MAX_TOOL_CALLS
    wall_clock_s: float = WALL_CLOCK_S
    clock: Any = time.monotonic
    steps: int = 0
    tool_calls: int = 0
    started: float = field(default=-1.0)

    def __post_init__(self) -> None:
        if self.started < 0:
            self.started = self.clock()

    def remaining_s(self) -> float:
        return self.wall_clock_s - (self.clock() - self.started)

    def check_deadline(self) -> None:
        if self.remaining_s() <= 0:
            raise DeadlineExceeded(f"wall-clock limit of {self.wall_clock_s:.0f} s reached")

    def charge_step(self) -> None:
        """Call before every model call."""
        self.check_deadline()
        if self.steps >= self.max_steps:
            raise BudgetExhausted(f"step limit of {self.max_steps} model calls reached")
        self.steps += 1

    def charge_tool(self) -> None:
        """Call before every tool call."""
        self.check_deadline()
        if self.tool_calls >= self.max_tool_calls:
            raise BudgetExhausted(f"tool-call limit of {self.max_tool_calls} reached")
        self.tool_calls += 1


# --------------------------------------------------------------------------- #
# evidence + graph state
# --------------------------------------------------------------------------- #
class RetrievedPassage(BaseModel):
    """A passage returned by a tool during this run.  Citations may only point here."""
    chunk_id: str
    doc_id: str
    clause_ids: list[str]        # every clause label inside the passage (+ "Table N" for tables)
    label: str                   # the chunk's own label, e.g. "7(i)-(ii)" or "Table 2 (para 8)"
    citation: str                # human-readable, e.g. "RBI Circular ..., para 7(i), p. 3"
    text: str
    lead_in: str = ""
    score: float = 0.0
    via: str = ""                # why it was retrieved, if not a direct hit
    refs: list[str] = Field(default_factory=list)
    referenced_by: list[str] = Field(default_factory=list)


class AnalysisDraft(BaseModel):
    """What the analysis agent hands to the verifier."""
    classification: Classification | None = None
    route: Route | None = None
    basis: str = ""
    liability_position: str = ""
    claims: list[str] = Field(default_factory=list)       # each factual statement the answer rests on
    citations: list[Citation] = Field(default_factory=list)
    escalate: bool = False
    reason_code: ReasonCode | None = None
    competing_provisions: list[str] = Field(default_factory=list)
    note: str = ""


class Verdict(BaseModel):
    """What the verifier hands back to the graph."""
    decision: Literal["pass", "retry", "escalate"]
    strategy_hint: str | None = None                      # "rewrite_query" | "neighbour_clause" | "wider_k"
    unsupported_claims: list[str] = Field(default_factory=list)
    reason_code: ReasonCode | None = None
    note: str = ""


class GraphState(TypedDict, total=False):
    request: DisputeInput
    run: Any                            # RunContext (graph.py): services + budget + ledger + trace + evidence
    flags: list[str]                    # guardrail findings on the input
    draft: AnalysisDraft
    verdict: Verdict
    attempts: int                       # analysis attempts so far (1 + retries)
    strategies_used: list[str]
    escalation: Escalation              # set when the run must escalate
    result: DisputeResult
