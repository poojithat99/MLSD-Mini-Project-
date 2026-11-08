"""The dispute triage graph (requirements A1, A2, A3, A5, F7).

    guard_input -> analysis_agent -> verifier --pass-----> finalize -> END
                        ^               |
                        +----retry------+   (at most 2 retries, a different strategy each time)
                                        |
                                        +--escalate--> escalate -> END

Any node may also jump straight to ``escalate`` if something goes wrong
(budget exhausted, deadline passed, provider down, unexpected error).

THE RULE: a run that does not complete cleanly returns an escalation with a
reason code, never a confident answer.  ``DisputeResult`` itself enforces that a
"recommended" result carries passed verification and citations.

Entry point:  run_dispute_graph(request, run_id, correlation_id) -> DisputeResult
"""
from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from . import analysis_agent, guardrails, verification_agent
from .llm import LLMClient, LLMError, LLMResponse, ProviderUnavailable, UsageLedger, get_llm
from .observability import log_event, run_context, span
from .prompts import load_prompt
from .state import (
    Budget, BudgetExhausted, DeadlineExceeded, DisputeInput, DisputeResult, Escalation, GraphState,
    ReasonCode, RetrievedPassage, TraceEntry, Verification,
)
from .tools import RetrieverPort, Toolbox, compute_reporting_delay

MAX_RETRIES = 2
STRATEGIES = ("rewrite_query", "neighbour_clause", "wider_k")


# --------------------------------------------------------------------------- #
# per-run context: services, limits, cost ledger, trace and evidence
# --------------------------------------------------------------------------- #
@dataclass
class RunContext:
    run_id: str
    correlation_id: str
    llm: LLMClient
    retriever: RetrieverPort
    request: DisputeInput
    budget: Budget
    analysis_prompt: str = "analysis_v1"
    verification_prompt: str = "verification_v1"
    ledger: UsageLedger = field(default_factory=UsageLedger)
    trace: list[TraceEntry] = field(default_factory=list)
    evidence: dict[str, RetrievedPassage] = field(default_factory=dict)
    delay: dict | None = None
    toolbox: Toolbox | None = None

    def __post_init__(self) -> None:
        self.toolbox = Toolbox(self.retriever, self.request, self.budget, self.evidence, self.trace)

    def call_llm(self, step: str, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        """Every model call of the run: charged to the budget, costed, traced."""
        self.budget.charge_step()
        n = len(self.ledger.records)
        resp = self.llm.call(step, messages, tools, ledger=self.ledger, timeout_s=self.budget.remaining_s())
        for r in self.ledger.records[n:]:
            self.trace.append(TraceEntry(
                type="llm", name=step, latency_ms=r.latency_ms, prompt_tokens=r.prompt_tokens,
                completion_tokens=r.completion_tokens, cost_usd=r.cost_usd, detail={"provider": r.provider},
            ))
        self.budget.check_deadline()          # a slow answer that arrives after the deadline is not used
        return resp

    def prompt_version(self) -> str:
        return f"{load_prompt(self.analysis_prompt)[1]}+{load_prompt(self.verification_prompt)[1]}"


# --------------------------------------------------------------------------- #
# node wrapper: span + log + latency, and failures become escalations
# --------------------------------------------------------------------------- #
def _reason_for(exc: Exception) -> ReasonCode:
    if isinstance(exc, DeadlineExceeded):
        return ReasonCode.TIMEOUT
    if isinstance(exc, BudgetExhausted):
        return ReasonCode.BUDGET_EXHAUSTED
    if isinstance(exc, (ProviderUnavailable, LLMError)):
        return ReasonCode.PROVIDER_UNAVAILABLE
    return ReasonCode.VERIFICATION_FAILED      # unexpected error: nothing was verified, so no answer


def traced_node(name: str) -> Callable:
    def wrap(fn: Callable[[GraphState], dict]) -> Callable[[GraphState], dict]:
        @functools.wraps(fn)
        def node(state: GraphState) -> dict:
            run: RunContext = state["run"]
            start = time.perf_counter()
            with span("node." + name, node=name):
                try:
                    update = fn(state)
                    outcome = "ok"
                except Exception as exc:          # noqa: BLE001 - every failure must end in an escalation
                    reason = _reason_for(exc)
                    outcome = f"failed: {type(exc).__name__}"
                    update = {"escalation": Escalation(reason_code=reason, note=f"{name}: {exc}")}
            latency = (time.perf_counter() - start) * 1000
            run.trace.append(TraceEntry(type="node", name=name, latency_ms=latency, detail={"outcome": outcome}))
            log_event("node", node=name, outcome=outcome, latency_ms=round(latency, 1))
            return update
        return node
    return wrap


# --------------------------------------------------------------------------- #
# nodes
# --------------------------------------------------------------------------- #
@traced_node("guard_input")
def guard_input(state: GraphState) -> dict:
    run: RunContext = state["run"]
    safe_fields, flags = guardrails.guard_request(state["request"])
    run.toolbox.safe_fields = safe_fields                           # models only ever see these
    run.delay = compute_reporting_delay(state["request"])          # code, not the model
    run.trace.append(TraceEntry(type="node", name="guardrails.input", detail={"flags": flags}))
    if flags:
        log_event("guardrail_input", flags=flags)
    injected = guardrails.injected_free_text_fields(flags)
    if injected:               # stop before ANY model call; name the field, never echo the text
        return {"flags": flags, "escalation": Escalation(
            reason_code=ReasonCode.INJECTION_DETECTED,
            note=f"possible prompt injection in field(s): {', '.join(injected)}; no model was called. "
                 "An analyst should read the original request.")}
    return {"flags": flags}


@traced_node("analysis_agent")
def analysis_node(state: GraphState) -> dict:
    run: RunContext = state["run"]
    used = list(state.get("strategies_used", []))
    strategy = None
    if state.get("attempts", 0) > 0:                                # this is a retry: pick a NEW strategy
        hint = state["verdict"].strategy_hint
        strategy = hint if hint in STRATEGIES and hint not in used else next(s for s in STRATEGIES if s not in used)
        used.append(strategy)
    draft = analysis_agent.run_analysis(run, state["request"], strategy=strategy,
                                        verdict=state.get("verdict"), previous=state.get("draft"))
    return {"draft": draft, "attempts": state.get("attempts", 0) + 1, "strategies_used": used}


@traced_node("verifier")
def verifier_node(state: GraphState) -> dict:
    run: RunContext = state["run"]
    verdict = verification_agent.verify(run, state["draft"])
    run.trace.append(TraceEntry(type="node", name="verifier.verdict", detail=verdict.model_dump(mode="json")))
    return {"verdict": verdict}


@traced_node("finalize")
def finalize_node(state: GraphState) -> dict:
    run: RunContext = state["run"]
    d = state["draft"]
    common = dict(
        run_id=run.run_id, classification=d.classification, basis=d.basis,
        reporting_delay_working_days=run.delay["working_days"] if run.delay else None,
        liability_position=d.liability_position, citations=d.citations,
        verification=Verification(passed=True, attempts=state["attempts"]),
        prompt_version=run.prompt_version(),
    )
    if d.escalate:            # a VERIFIED escalation: e.g. section 4.1, both provisions cited and checked
        result = DisputeResult(status="escalated", route=None, escalation=Escalation(
            reason_code=d.reason_code or ReasonCode.CONTESTED_FAULT,
            competing_provisions=d.competing_provisions, note=d.note), **common)
    else:
        result = DisputeResult(status="recommended", route=d.route, **common)
    blocked = guardrails.output_violations(result)
    run.trace.append(TraceEntry(type="node", name="guardrails.output", detail={"blocked": blocked}))
    if blocked:                # never return it as it is: escalate instead
        log_event("guardrail_output_blocked", problems=blocked)
        kinds = sorted({p.split(":")[0] for p in blocked})          # the blocked text itself stays in the trace only
        return {"escalation": Escalation(reason_code=ReasonCode.VERIFICATION_FAILED,
                                         note="output filter blocked the answer (" + ", ".join(kinds) + ")")}
    return {"result": result}


def escalate_node(state: GraphState) -> dict:
    """Builds the escalation.  Never raises: it is the last line of defence."""
    run: RunContext = state["run"]
    esc = state.get("escalation")
    verdict = state.get("verdict")
    if esc is None and verdict is not None:
        if verdict.decision == "retry":                              # retries used up
            esc = Escalation(reason_code=ReasonCode.VERIFICATION_FAILED,
                             note=f"answer not supported after {state.get('attempts', 0)} attempts")
        else:
            esc = Escalation(reason_code=verdict.reason_code or ReasonCode.VERIFICATION_FAILED, note=verdict.note)
    esc = esc or Escalation(reason_code=ReasonCode.VERIFICATION_FAILED, note="run did not complete")
    draft = state.get("draft")
    # keep only citations that resolve to retrieved text: an analyst can use them, nothing unverified leaks out
    cites = [c for c in (draft.citations if draft else []) if verification_agent.citation_resolves(c, run.evidence)]
    if draft and not esc.competing_provisions:
        esc.competing_provisions = draft.competing_provisions
    esc.note = guardrails.scrub(esc.note)                           # model-written text: filter it too
    try:
        version = run.prompt_version()
    except (OSError, ValueError):
        version = "unknown"
    result = DisputeResult(
        run_id=run.run_id, status="escalated", route=None, escalation=esc,
        classification=draft.classification if draft else None,
        reporting_delay_working_days=run.delay["working_days"] if run.delay else None,
        basis=guardrails.scrub(draft.basis) if draft else "", liability_position="", citations=cites,
        verification=Verification(passed=False, attempts=state.get("attempts", 0),
                                  unsupported_claims=verdict.unsupported_claims if verdict else []),
        prompt_version=version,
    )
    run.trace.append(TraceEntry(type="node", name="escalate", detail={"reason_code": esc.reason_code.value}))
    log_event("node", node="escalate", reason_code=esc.reason_code.value, note=esc.note)
    return {"result": result, "escalation": esc}


# --------------------------------------------------------------------------- #
# edges
# --------------------------------------------------------------------------- #
def _ok_or_escalate(next_node: str) -> Callable[[GraphState], str]:
    return lambda state: "escalate" if state.get("escalation") else next_node


def after_verifier(state: GraphState) -> str:
    if state.get("escalation"):
        return "escalate"
    decision = state["verdict"].decision
    if decision == "pass":
        return "finalize"
    if decision == "retry" and state.get("attempts", 0) <= MAX_RETRIES:
        return "analysis_agent"
    return "escalate"


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("guard_input", guard_input)
    g.add_node("analysis_agent", analysis_node)
    g.add_node("verifier", verifier_node)
    g.add_node("finalize", finalize_node)
    g.add_node("escalate", escalate_node)
    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", _ok_or_escalate("analysis_agent"), ["analysis_agent", "escalate"])
    g.add_conditional_edges("analysis_agent", _ok_or_escalate("verifier"), ["verifier", "escalate"])
    g.add_conditional_edges("verifier", after_verifier, ["finalize", "analysis_agent", "escalate"])
    g.add_conditional_edges("finalize", _ok_or_escalate(END), [END, "escalate"])
    g.add_edge("escalate", END)
    return g.compile()


_GRAPH = build_graph()


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def _default_retriever() -> RetrieverPort:
    from backend.retrieval.search import Retriever   # loads FAISS + embedding model

    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever()
    return _RETRIEVER


_RETRIEVER: RetrieverPort | None = None


def run_dispute_graph(
    request: Any, run_id: str, correlation_id: str, *,
    llm: LLMClient | None = None, retriever: RetrieverPort | None = None, budget: Budget | None = None,
    analysis_prompt: str = "analysis_v1",
) -> DisputeResult:
    """Run one dispute through the graph.  Always returns a DisputeResult, never raises."""
    if not isinstance(request, DisputeInput):
        data = request.model_dump() if hasattr(request, "model_dump") else dict(request)
        request = DisputeInput.model_validate(data)
    with run_context(run_id, correlation_id), span("dispute_run"):
        start = time.perf_counter()
        run = RunContext(
            run_id=run_id, correlation_id=correlation_id, llm=llm or get_llm(),
            retriever=retriever or _default_retriever(), request=request, budget=budget or Budget(),
            analysis_prompt=analysis_prompt,
        )
        state: GraphState = {"request": request, "run": run, "attempts": 0, "strategies_used": []}
        try:
            final = _GRAPH.invoke(state)
            result: DisputeResult = final["result"]
        except Exception as exc:                  # noqa: BLE001 - last resort, still an escalation
            final = escalate_node({**state, "escalation": Escalation(
                reason_code=_reason_for(exc), note=f"graph error: {exc}")})
            result = final["result"]
        result.trace = run.trace
        log_event("run_finished", status=result.status, route=result.route,
                  reason_code=result.escalation.reason_code.value if result.escalation else None,
                  steps=run.budget.steps, tool_calls=run.budget.tool_calls,
                  cost_usd=round(run.ledger.total_cost(), 6),
                  latency_ms=round((time.perf_counter() - start) * 1000, 1))
        return result
