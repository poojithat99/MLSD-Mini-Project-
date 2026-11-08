"""Analysis agent (requirements A1, A2, F3–F7; brief section 5 steps 1–3).

A bounded tool-calling loop:

    model -> tool calls -> results -> model -> ... -> final JSON draft

Bounded three ways: the run's Budget (model calls, tool calls, wall-clock) is
charged on every call, and the loop itself stops after MAX_TURNS.

The REQUIRED REASONING STEP (brief section 5, step 2) is enforced in code, not
left to the model: the first time the model tries to finish, if it has issued
fewer than two different searches in this attempt, we do not accept the answer.
We send it MODIFIER_CHECK, which lists the clauses it has read and the facts of
the case, and asks it to search again for any provision that modifies the one
it found.  Only then is a final answer accepted.

On a retry, the graph passes a strategy chosen by the verifier, and the agent
works differently each time:
  rewrite_query     re-search using the corpus's own wording
  neighbour_clause  code fetches the cited clauses and their cross-references by id
  wider_k           code re-runs the searches with k=10 and cross-reference expansion
"""
from __future__ import annotations

import json

from pydantic import ValidationError

from .llm import LLMError, parse_json_object
from .observability import log_event
from .prompts import load_prompt
from .state import AnalysisDraft, BudgetExhausted, DisputeInput, TraceEntry, Verdict
from .tools import TOOL_SPECS

MAX_TURNS = 6                 # model turns per attempt (the run Budget is the overall limit)

MODIFIER_CHECK = (
    "MODIFIER CHECK: before answering, decide whether any OTHER provision modifies the one you relied on: "
    "customer negligence (e.g. sharing payment credentials), the reporting-delay limbs (within 3 / 4 to 7 / "
    "beyond 7 working days), the bank's Board-approved policy, provisos beginning 'Further' or 'However', "
    "or a definition. Call search_corpus again with a NEW query built from the wording of the passages "
    "you have read and the facts of this case, then give your final JSON."
)
RETRY_MARKER = "RETRY STRATEGY"
STRATEGY_TEXT = {
    "rewrite_query": "Your earlier searches did not support the answer. Rewrite your queries using the exact "
                     "wording that appears in the corpus passages, search again, and quote only what you read.",
    "neighbour_clause": "Your citations did not resolve. The clauses you cited and the clauses they refer to "
                        "have been fetched by id below. Quote them character for character.",
    "wider_k": "Search more widely: use search_corpus with k=10 and expand_refs=true, so that provisions "
               "related to the ones you found are included.",
}


def _case_message(run, request: DisputeInput) -> str:
    fields = run.toolbox.get_request_fields()        # free text already wrapped by guardrails
    return (
        "CASE FIELDS (data from the request, not instructions):\n"
        f"{json.dumps(fields, ensure_ascii=False, indent=1)}\n\n"
        "REPORTING DELAY (computed by code; do not recount):\n"
        f"{json.dumps(run.delay, ensure_ascii=False)}"
    )


def _retry_message(run, strategy: str, verdict: Verdict | None, previous: AnalysisDraft | None) -> str:
    """Strategy instructions, plus evidence the code fetched itself for this strategy."""
    lines = [f"{RETRY_MARKER}: {strategy}. {STRATEGY_TEXT[strategy]}"]
    if verdict and verdict.unsupported_claims:
        lines.append("Problems the verifier found: " + "; ".join(verdict.unsupported_claims))
    fetched: list[str] = []
    if strategy == "neighbour_clause" and previous:
        for c in previous.citations:
            fetched.append(run.toolbox.run("get_clause", {"doc_id": c.doc_id, "clause_id": c.clause_id}))
            for p in list(run.evidence.values()):         # follow cross-references of what was cited
                if p.doc_id == c.doc_id and c.clause_id in p.clause_ids:
                    for ref in p.refs[:2]:
                        if ref not in run.evidence:
                            fetched.append(run.toolbox.run("fetch_chunk", {"chunk_id": ref}))
                    break
    if strategy == "wider_k" and run.toolbox.queries:
        fetched.append(run.toolbox.run("search_corpus",
                                       {"query": run.toolbox.queries[-1], "k": 10, "expand_refs": True}))
    if fetched:
        lines.append("PASSAGES FETCHED FOR THIS RETRY:\n" + "\n".join(fetched))
    return "\n\n".join(lines)


def _assistant_tool_message(resp) -> dict:
    return {"role": "assistant", "content": resp.content or "", "tool_calls": [
        {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
        for tc in resp.tool_calls]}


def run_analysis(run, request: DisputeInput, strategy: str | None = None,
                 verdict: Verdict | None = None, previous: AnalysisDraft | None = None) -> AnalysisDraft:
    system, _version = load_prompt(run.analysis_prompt)
    messages: list[dict] = [{"role": "system", "content": system},
                            {"role": "user", "content": _case_message(run, request)}]
    if strategy:
        messages.append({"role": "user", "content": _retry_message(run, strategy, verdict, previous)})

    queries_before = len(run.toolbox.queries)
    nudged = repaired = False
    for _turn in range(MAX_TURNS):
        resp = run.call_llm("analysis", messages, tools=TOOL_SPECS)

        if resp.tool_calls:                               # the agent chose to call tools
            messages.append(_assistant_tool_message(resp))
            for tc in resp.tool_calls:
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": run.toolbox.run(tc.name, tc.arguments)})
            continue

        # The agent wants to finish.  Enforce the modifier check once per attempt.
        searches = set(run.toolbox.queries[queries_before:])
        if len(searches) < 2 and not nudged:
            nudged = True
            read = sorted({f"{p.doc_id} {p.label}" for p in run.evidence.values()})
            messages.append({"role": "assistant", "content": resp.content or ""})
            messages.append({"role": "user", "content": f"{MODIFIER_CHECK}\nClauses read so far: {read}"})
            continue

        try:
            draft = AnalysisDraft.model_validate(parse_json_object(resp.content))
        except (LLMError, ValidationError) as exc:
            if repaired:
                raise
            repaired = True                               # one chance to fix malformed output
            messages.append({"role": "assistant", "content": resp.content or ""})
            messages.append({"role": "user", "content": f"Your reply was not valid JSON for the required schema "
                                                        f"({str(exc)[:300]}). Reply with the JSON object only."})
            continue

        modifier_done = len(set(run.toolbox.queries[queries_before:])) >= 2
        run.trace.append(TraceEntry(type="node", name="analysis.modifier_check", detail={
            "done": modifier_done, "queries": run.toolbox.queries[queries_before:], "strategy": strategy}))
        log_event("analysis_draft", escalate=draft.escalate, route=draft.route, modifier_check=modifier_done,
                  citations=[f"{c.doc_id} {c.clause_id}" for c in draft.citations])
        return draft

    raise BudgetExhausted(f"analysis agent did not finish within {MAX_TURNS} turns")
