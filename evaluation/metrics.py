"""Evaluation metrics (requirements L3, L5).

Everything is computed from a finished run's ``DisputeResult`` (as a dict) and its scenario, so stored
results can be re-scored without re-running the agents.

What a run is judged on
-----------------------
* RETRIEVED passages: every chunk id returned by a tool call in the trace (search_corpus,
  get_clause, fetch_chunk, including code-seeded lookups), with its text read from
  ``data/index/chunks.jsonl``.
* OUTPUT CLAIMS: the sentences of the result's ``basis`` and ``liability_position``. That is
  what the model asserts about the regulation to the analyst.  An unverified escalation
  shows no basis, so it has no claims; a basis written by the conflict rule (code quoting two
  provisions verbatim) is not a model claim and is skipped.
* CITED passages: retrieved passages that hold a cited (doc_id, clause_id).

One judge call per run
----------------------
LLM-judged parts use ONE call per run, through ``backend/agents/llm.py`` (step "judge") with
the versioned prompt ``evaluation/prompts/judge_v1.txt``.  The judge returns, for each claim,
the ids of the retrieved passages that support it, and, for each search query, a relevance
score (0 / 0.5 / 1) for each passage it returned.  Judgements are cached on disk by a SHA-256 of
(judge prompt version + exact payload), so re-scoring identical output costs no model call.
A run with no claims and no searches needs no judge call.
"""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.agents.llm import LLMClient, UsageLedger, parse_json_object
from backend.agents.state import Citation, RetrievedPassage
from backend.agents.verification_agent import citation_resolves

REPO = Path(__file__).resolve().parents[1]
CHUNKS_FILE = REPO / "data" / "index" / "chunks.jsonl"
JUDGE_PROMPT = Path(__file__).resolve().parent / "prompts" / "judge_v1.txt"
DEFAULT_CACHE = Path(__file__).resolve().parent / ".cache" / "judge"
RETRIEVAL_TOOLS = {"search_corpus", "get_clause", "fetch_chunk"}


# --------------------------------------------------------------------------- #
# reading a run
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _chunks() -> dict[str, dict]:
    with open(CHUNKS_FILE, encoding="utf-8") as fh:
        return {c["chunk_id"]: c for c in (json.loads(line) for line in fh if line.strip())}


def _record(chunk: dict) -> RetrievedPassage:
    ids = list(chunk["clause_ids"]) + ([f"Table {chunk['table_no']}"] if chunk.get("table_no") else [])
    return RetrievedPassage(chunk_id=chunk["chunk_id"], doc_id=chunk["doc_id"], clause_ids=ids,
                            label=chunk.get("clause", ""), citation="", text=chunk["text"],
                            lead_in=chunk.get("lead_in", ""))


def retrieved_passages(result: dict) -> dict[str, RetrievedPassage]:
    """Every passage a tool returned during the run, in first-seen order."""
    out: dict[str, RetrievedPassage] = {}
    for t in result.get("trace", []):
        if t["type"] == "tool" and t["name"] in RETRIEVAL_TOOLS:
            for cid in t["detail"].get("chunk_ids", []):
                if cid not in out and cid in _chunks():
                    out[cid] = _record(_chunks()[cid])
    return out


def search_pairs(result: dict) -> list[tuple[str, list[str]]]:
    """(query, chunk ids returned) for every search_corpus call."""
    return [(t["detail"]["arguments"].get("query", ""), t["detail"].get("chunk_ids", []))
            for t in result.get("trace", [])
            if t["type"] == "tool" and t["name"] == "search_corpus" and t["detail"].get("chunk_ids")]


# Sentence boundary: ., ! or ? followed by whitespace, but not after common abbreviations ("Rs. 10,000",
# "No. 5", "e.g.", "i.e.", "viz.", "para.") and not when the next token is a number.
_SENTENCE_END = re.compile(r"(?<!\bRs\.)(?<!\bNo\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bviz\.)(?<!\bpara\.)"
                           r"(?<=[.!?])\s+(?![0-9])")


def code_written_basis(result: dict) -> str | None:
    """The basis written by the conflict rule (code, not the model), if this run used it."""
    for t in result.get("trace", []):
        if t["name"] == "analysis.conflict_rule" and t["detail"].get("code_written_basis"):
            return t["detail"]["code_written_basis"]
    return None


def output_claims(result: dict) -> list[str]:
    """The MODEL's claims: sentences of basis + liability_position.  A basis written by the
    conflict rule (code that quotes two provisions verbatim) is not a model claim and is skipped."""
    basis = result.get("basis", "")
    if basis and basis == code_written_basis(result):
        basis = ""
    text = " ".join(x for x in (basis, result.get("liability_position", "")) if x)
    return [s.strip() for s in _SENTENCE_END.split(text) if len(s.strip()) > 3]


def cited_chunk_ids(result: dict, passages: dict[str, RetrievedPassage]) -> set[str]:
    out = set()
    for c in result.get("citations", []):
        for p in passages.values():
            if p.doc_id == c["doc_id"] and c["clause_id"] in p.clause_ids:
                out.add(p.chunk_id)
    return out


def unresolved_citations(result: dict, passages: dict[str, RetrievedPassage]) -> list[dict]:
    """Citations whose clause/quote is not in the retrieved text (the verifier's own rule)."""
    return [c for c in result.get("citations", []) if not citation_resolves(Citation(**c), passages)]


def model_calls(result: dict) -> int:
    return sum(1 for t in result.get("trace", []) if t["type"] == "llm")


# --------------------------------------------------------------------------- #
# the judge: one call per run, cached
# --------------------------------------------------------------------------- #
def judge_version() -> str:
    return JUDGE_PROMPT.read_text(encoding="utf-8").partition("\n")[0].partition(":")[2].strip()


def judge_payload(result: dict) -> tuple[dict, dict[str, str]]:
    """Payload for the judge, using short passage ids (P1, P2, ...) to save tokens.
    Returns (payload, short id -> chunk id)."""
    passages = retrieved_passages(result)
    short = {cid: f"P{i + 1}" for i, cid in enumerate(passages)}
    payload = {
        "claims": [{"id": i, "claim": c} for i, c in enumerate(output_claims(result))],
        "passages": [{"id": short[cid], "text": p.text} for cid, p in passages.items()],
        "searches": [{"id": i, "query": q, "passages": [short[c] for c in ids if c in short]}
                     for i, (q, ids) in enumerate(search_pairs(result))],
    }
    return payload, {v: k for k, v in short.items()}


def judge_run(result: dict, llm: LLMClient | None, cache_dir: Path = DEFAULT_CACHE,
              ledger: UsageLedger | None = None) -> dict | None:
    """Support + relevance judgement for one run, keyed by chunk id.

    Returns {"support": {claim_index: [chunk ids]}, "relevance": [(search_index, chunk_id, score)],
    "cached": bool}, or None if there is nothing to judge or no judge is available."""
    payload, back = judge_payload(result)
    if not payload["claims"] and not payload["searches"]:
        return None
    system = JUDGE_PROMPT.read_text(encoding="utf-8").partition("\n")[2].strip()
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    key = hashlib.sha256(f"{judge_version()}\n{body}".encode()).hexdigest()
    path = cache_dir / f"{key}.json"
    if path.exists():
        raw, cached = json.loads(path.read_text(encoding="utf-8")), True
    else:
        if llm is None:
            return None
        resp = llm.call("judge", [{"role": "system", "content": system}, {"role": "user", "content": body}],
                        ledger=ledger)
        raw, cached = parse_json_object(resp.content), False
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw), encoding="utf-8")
    support = {int(s["claim"]): [back[p] for p in s.get("passages", []) if p in back]
               for s in raw.get("support", []) if isinstance(s.get("claim"), int)}
    relevance = [(int(r["search"]), back[r["passage"]], float(r["score"]))
                 for r in raw.get("relevance", []) if r.get("passage") in back]
    return {"support": support, "relevance": relevance, "cached": cached}


# --------------------------------------------------------------------------- #
# the four L3 metrics (per run; aggregate() averages them)
# --------------------------------------------------------------------------- #
def groundedness(claims: list[str], judgement: dict | None) -> float | None:
    """Share of output claims supported by ANY passage retrieved in the run.

    = (claims with >=1 supporting retrieved passage, per the judge) / (number of claims).
    None when the run has no claims (e.g. an out-of-corpus or unverified escalation) or no judgement."""
    if not claims or judgement is None:
        return None
    return sum(1 for i in range(len(claims)) if judgement["support"].get(i)) / len(claims)


def faithfulness(claims: list[str], judgement: dict | None, cited: set[str]) -> float | None:
    """Share of output claims supported by the CITED passages only.

    = (claims with >=1 supporting passage that is also a cited passage) / (number of claims).
    A claim backed only by a retrieved-but-uncited passage counts as unfaithful: the answer
    does not show its evidence.  None when there are no claims or no judgement."""
    if not claims or judgement is None:
        return None
    return sum(1 for i in range(len(claims)) if set(judgement["support"].get(i, [])) & cited) / len(claims)


def context_relevance(judgement: dict | None) -> float | None:
    """Mean judged relevance (0 / 0.5 / 1) of each passage to the search query that retrieved it.

    Averaged over all (search_corpus query, returned passage) pairs of the run.  Exact
    by-id lookups (get_clause, seeded provisions) have no query and are not included.
    None when the run made no searches or there is no judgement."""
    if judgement is None or not judgement["relevance"]:
        return None
    return sum(s for _, _, s in judgement["relevance"]) / len(judgement["relevance"])


def hallucinated(unresolved: list[dict], claims: list[str], judgement: dict | None) -> bool | None:
    """True if the run has at least one citation that does not resolve to retrieved text
    (deterministic, the verifier's own rule) OR at least one claim that no retrieved passage supports
    (judge).  hallucination_rate = share of runs where this is True.  If there is no
    judgement, only the citation part is evaluated (None if it is clean, so the run is not
    counted as clean on half the evidence)."""
    if unresolved:
        return True
    if not claims:
        return False
    if judgement is None:
        return None
    return any(not judgement["support"].get(i) for i in range(len(claims)))


# --------------------------------------------------------------------------- #
# outcome accuracy (deterministic)
# --------------------------------------------------------------------------- #
def outcome_checks(scenario: dict, result: dict) -> dict[str, Any]:
    """Expected vs actual, no model involved.

    status/route/classification/reason_code: exact match (None == None counts as a match).
    citation_recall = |expected doc-qualified clause ids that were cited| / |expected|; None when
    nothing is expected.  delay: reporting_delay_working_days == expected.  For prompt-injection
    scenarios, no_model_call: the trace has no model call.
    outcome_correct = status and reason_code and route all match (the headline accuracy)."""
    esc = result.get("escalation") or {}
    cited = {f"{c['doc_id']} {c['clause_id']}" for c in result.get("citations", [])}
    expected = set(scenario["expected_clause_ids"])
    checks = {
        "status": result["status"] == scenario["expected_status"],
        "route": result.get("route") == scenario["expected_route"],
        "classification": result.get("classification") == scenario["expected_classification"],
        "reason_code": esc.get("reason_code") == scenario["expected_reason_code"],
        "citation_recall": (len(expected & cited) / len(expected)) if expected else None,
        "delay": result.get("reporting_delay_working_days") == scenario["expected_delay_working_days"],
    }
    if "prompt_injection" in scenario["tags"]:
        checks["no_model_call"] = model_calls(result) == 0
    checks["outcome_correct"] = checks["status"] and checks["reason_code"] and checks["route"]
    return checks


# --------------------------------------------------------------------------- #
# per-run scoring and aggregation
# --------------------------------------------------------------------------- #
def score_run(scenario: dict, result: dict, judge_llm: LLMClient | None, cache_dir: Path = DEFAULT_CACHE,
              judge_ledger: UsageLedger | None = None) -> dict:
    passages = retrieved_passages(result)
    claims = output_claims(result)
    cited = cited_chunk_ids(result, passages)
    unresolved = unresolved_citations(result, passages)
    judgement = judge_run(result, judge_llm, cache_dir, judge_ledger)
    return {
        "outcome": outcome_checks(scenario, result),
        "groundedness": groundedness(claims, judgement),
        "faithfulness": faithfulness(claims, judgement, cited),
        "context_relevance": context_relevance(judgement),
        "hallucinated": hallucinated(unresolved, claims, judgement),
        "n_claims": len(claims), "n_retrieved": len(passages), "unresolved_citations": unresolved,
        "judged": judgement is not None, "judge_cached": bool(judgement and judgement["cached"]),
    }


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def aggregate(scored: list[tuple[dict, dict]]) -> dict:
    """Aggregate (scenario, score) pairs.  The four metrics are means over runs where they are
    defined (n_* says how many).  hallucination_rate = runs flagged / runs evaluated.
    Outcome accuracy is reported for confirmed scenarios and separately for needs_review ones."""
    def block(pairs):
        outs = [s["outcome"] for _, s in pairs]
        flags = [s["hallucinated"] for _, s in pairs if s["hallucinated"] is not None]
        return {
            "n": len(pairs),
            "outcome_accuracy": _mean([float(o["outcome_correct"]) for o in outs]),
            "status_accuracy": _mean([float(o["status"]) for o in outs]),
            "route_accuracy": _mean([float(o["route"]) for o in outs]),
            "classification_accuracy": _mean([float(o["classification"]) for o in outs]),
            "reason_code_accuracy": _mean([float(o["reason_code"]) for o in outs]),
            "citation_recall": _mean([o["citation_recall"] for o in outs]),
            "groundedness": _mean([s["groundedness"] for _, s in pairs]),
            "n_groundedness": sum(1 for _, s in pairs if s["groundedness"] is not None),
            "faithfulness": _mean([s["faithfulness"] for _, s in pairs]),
            "context_relevance": _mean([s["context_relevance"] for _, s in pairs]),
            "n_context_relevance": sum(1 for _, s in pairs if s["context_relevance"] is not None),
            "hallucination_rate": (sum(flags) / len(flags)) if flags else None,
            "n_hallucination": len(flags),
        }
    confirmed = [(sc, s) for sc, s in scored if not sc.get("needs_review")]
    review = [(sc, s) for sc, s in scored if sc.get("needs_review")]
    tags = sorted({t for sc, _ in scored for t in sc["tags"]})
    return {"confirmed": block(confirmed), "needs_review": block(review), "all": block(scored),
            "by_tag": {t: block([(sc, s) for sc, s in scored if t in sc["tags"]]) for t in tags}}


# --------------------------------------------------------------------------- #
# cost (L5)
# --------------------------------------------------------------------------- #
def run_usage(result: dict) -> dict[str, dict[str, float]]:
    """Tokens and cost per step (analysis / verification) for one run, from its trace."""
    steps: dict[str, dict[str, float]] = {}
    for t in result.get("trace", []):
        if t["type"] == "llm":
            s = steps.setdefault(t["name"], {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})
            s["calls"] += 1
            s["prompt_tokens"] += t["prompt_tokens"]
            s["completion_tokens"] += t["completion_tokens"]
            s["cost_usd"] += t["cost_usd"]
    return steps


def cost_summary(results: list[dict], judge_usage: list[dict] | None = None) -> dict:
    """Per-step and per-run tokens/cost over runs, cost per 100 runs, largest cost driver.

    cost_per_100_runs = mean system cost per run x 100 (the agents only; the judge is an
    evaluation cost, reported separately).  largest_cost_driver = the step with the highest
    total cost; when every cost is 0 (free tier, LLM_COST_PER_1K_* = 0) it is the step with the
    most tokens, and ``driver_basis`` says which was used."""
    per_step: dict[str, dict[str, float]] = {}
    per_run_cost, per_run_tokens = [], []
    for r in results:
        usage = run_usage(r)
        per_run_cost.append(sum(s["cost_usd"] for s in usage.values()))
        per_run_tokens.append(sum(s["prompt_tokens"] + s["completion_tokens"] for s in usage.values()))
        for step, s in usage.items():
            agg = per_step.setdefault(step, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})
            for k in agg:
                agg[k] += s[k]
    n = len(results) or 1
    total_cost = sum(per_run_cost)
    if per_step:
        basis = "cost_usd" if total_cost > 0 else "tokens"
        key = (lambda kv: kv[1]["cost_usd"]) if basis == "cost_usd" else \
              (lambda kv: kv[1]["prompt_tokens"] + kv[1]["completion_tokens"])
        driver = max(per_step.items(), key=key)[0]
    else:
        basis, driver = None, None
    judge = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
    for u in judge_usage or []:
        for k in judge:
            judge[k] += u.get(k, 0)
    return {
        "runs": len(results), "per_step": per_step,
        "mean_cost_per_run_usd": total_cost / n, "mean_tokens_per_run": sum(per_run_tokens) / n,
        "mean_model_calls_per_run": sum(model_calls(r) for r in results) / n,
        "cost_per_100_runs_usd": total_cost / n * 100, "tokens_per_100_runs": sum(per_run_tokens) / n * 100,
        "largest_cost_driver": driver, "driver_basis": basis, "judge": judge,
    }
