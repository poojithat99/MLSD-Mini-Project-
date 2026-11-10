"""Batch evaluation of the agent graph on the fixed scenario set (requirements L2, L3, L5).

    python -m evaluation.evaluate --prompt analysis_v1 --seed on                 # baseline
    python -m evaluation.evaluate --prompt analysis_v1 --seed off                # L3: before/after the change
    python -m evaluation.evaluate --prompt analysis_v2 --seed on                 # L2: prompt variant
    python -m evaluation.evaluate --compare evaluation/results/A.json evaluation/results/B.json

Options: --limit N (first N scenarios), --only S01,S11, --out PATH, --no-judge (deterministic
metrics only, no judge calls), --retriever real|keyword (keyword = offline, for smoke tests).

Behaviour
* Scenarios run one at a time; LLM_MIN_INTERVAL_S pacing applies inside llm.py.
* The results file is rewritten after every scenario.  Re-running the same command resumes:
  scenarios already done (same prompt, seed and scenario content) are skipped.
* Quota safety: if a scenario ends with provider_unavailable (quota exhausted, provider down,
  breaker open), it is stored as "deferred", not scored, and the WHOLE batch stops with exit
  code 3.  It is never retried in a loop; run the same command later to resume.  The same
  applies if the judge call fails ("judge_pending": only the judge is redone on resume).
* Watchdog: each scenario runs under a hard per-scenario timeout (EVAL_SCENARIO_TIMEOUT_S, default
  300 s).  A scenario that does not finish is recorded as "timeout" (not scored) and the batch
  moves on instead of hanging; resuming re-runs it.
* Versions: the results file records the prompt versions and a fingerprint of the code that produced
  it (agents, prompts, metrics, judge prompt).  Resuming a file produced by different prompt or
  code versions is refused: start a new --out (old files belong in results/archive/).
* No model id, endpoint or key is written to the results (requirement D4).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable

from backend.agents.graph import run_dispute_graph
from backend.agents.llm import LLMClient, LLMError, UsageLedger, get_llm, load_dotenv_if_present
from backend.agents.prompts import load_prompt

from . import metrics

HERE = Path(__file__).resolve().parent
SCENARIOS = HERE / "scenarios.jsonl"
RESULTS_DIR = HERE / "results"
EXIT_DEFERRED = 3
REPO = HERE.parent
FINGERPRINTED = ["backend/agents/*.py", "backend/prompts/*.txt", "evaluation/metrics.py", "evaluation/prompts/*.txt"]


def scenario_timeout_s() -> float:
    raw = os.environ.get("EVAL_SCENARIO_TIMEOUT_S", "").strip()
    return float(raw) if raw else 300.0


def code_fingerprint() -> str:
    """SHA-256 over the code and prompts that determine a result, so results from different
    versions can never be mixed in one file."""
    h = hashlib.sha256()
    for pattern in FINGERPRINTED:
        for f in sorted(REPO.glob(pattern)):
            h.update(f.relative_to(REPO).as_posix().encode() + b"\0" + f.read_bytes())
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- #
# scenarios and results file
# --------------------------------------------------------------------------- #
def load_scenarios(path: Path = SCENARIOS) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def scenario_hash(s: dict) -> str:
    return hashlib.sha256(json.dumps(s, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _load_results(path: Path, meta: dict) -> dict:
    if not path.exists():
        return {"meta": {**meta, "created_at": _now()}, "runs": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    for k in ("analysis_prompt", "seed_provisions", "prompt_versions", "code_fingerprint"):
        if data["meta"].get(k) != meta[k]:
            raise SystemExit(f"refusing to resume {path}: it was produced with {k}={data['meta'].get(k)!r}, "
                             f"but the current run has {k}={meta[k]!r}. Use a new --out (move the old file to "
                             f"evaluation/results/archive/) instead of mixing versions.")
    return data


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data["meta"]["updated_at"] = _now()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)                                          # atomic: a crash never leaves half a file


# --------------------------------------------------------------------------- #
# running
# --------------------------------------------------------------------------- #
class ScenarioTimeout(RuntimeError):
    pass


def _with_watchdog(fn: Callable[[], Any], timeout_s: float) -> Any:
    """Run ``fn`` in a daemon thread and stop waiting after ``timeout_s``.  A daemon thread, so
    a hung scenario cannot keep the process alive at exit either."""
    box: dict[str, Any] = {}

    def target():
        try:
            box["result"] = fn()
        except BaseException as exc:          # noqa: BLE001 - re-raised in the caller's thread
            box["error"] = exc

    t = threading.Thread(target=target, name="eval-scenario", daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise ScenarioTimeout(f"no result within {timeout_s:.0f} s")
    if "error" in box:
        raise box["error"]
    return box["result"]


def _ledger_totals(ledger: UsageLedger) -> dict:
    return {"calls": len(ledger.records),
            "prompt_tokens": sum(r.prompt_tokens for r in ledger.records),
            "completion_tokens": sum(r.completion_tokens for r in ledger.records),
            "cost_usd": ledger.total_cost()}


def run_batch(
    scenarios: list[dict], *, prompt: str, seed: bool, out_path: Path, llm: LLMClient, retriever: Any,
    judge_llm: LLMClient | None, cache_dir: Path = metrics.DEFAULT_CACHE, log: Callable[[str], None] = print,
    timeout_s: float | None = None,
) -> int:
    """Run (or resume) the batch.  Returns 0 when every scenario is done or timed out, EXIT_DEFERRED
    if it stopped on a provider failure."""
    timeout_s = timeout_s or scenario_timeout_s()
    meta = {"analysis_prompt": prompt, "seed_provisions": seed,
            "prompt_versions": f"{load_prompt(prompt)[1]}+{load_prompt('verification_v1')[1]}",
            "code_fingerprint": code_fingerprint(),
            "judge_prompt": metrics.judge_version() if judge_llm is not None else None}
    data = _load_results(out_path, meta)
    data["meta"].update({k: v for k, v in meta.items() if v is not None})
    for s in scenarios:
        h = scenario_hash(s)
        rec = data["runs"].get(s["id"])
        if rec and rec["scenario_hash"] == h and rec["status"] == "done":
            continue                                           # resume: already scored
        if rec and rec["scenario_hash"] == h and rec["status"] == "judge_pending":
            result = rec["result"]                             # agent run is fine; only the judge is redone
        else:
            log(f"{s['id']}: running")
            try:
                result = _with_watchdog(lambda: run_dispute_graph(
                    s["request"], f"eval-{s['id']}", f"eval-{prompt}-{'on' if seed else 'off'}", llm=llm,
                    retriever=retriever, analysis_prompt=prompt, seed_provisions=seed).model_dump(mode="json"),
                    timeout_s)
            except ScenarioTimeout as exc:
                data["runs"][s["id"]] = {"status": "timeout", "scenario_hash": h, "error": str(exc)}
                _save(out_path, data)
                log(f"{s['id']}: TIMEOUT ({exc}); recorded as failed, moving on")
                continue
            if (result.get("escalation") or {}).get("reason_code") == "provider_unavailable":
                data["runs"][s["id"]] = {"status": "deferred", "scenario_hash": h, "result": result}
                _save(out_path, data)
                log(f"{s['id']}: provider unavailable ({result['escalation']['note'][:120]}). "
                    f"Stopping the batch; resume later with the same command.")
                return EXIT_DEFERRED
        ledger = UsageLedger()
        try:
            score = metrics.score_run(s, result, judge_llm, cache_dir, ledger)
        except LLMError as exc:
            data["runs"][s["id"]] = {"status": "judge_pending", "scenario_hash": h, "result": result}
            _save(out_path, data)
            log(f"{s['id']}: judge call failed ({exc}). Stopping the batch; resume later.")
            return EXIT_DEFERRED
        data["runs"][s["id"]] = {"status": "done", "scenario_hash": h, "result": result, "metrics": score,
                                 "judge_usage": _ledger_totals(ledger)}
        _save(out_path, data)
    data["summary"] = summarize(data, scenarios)
    _save(out_path, data)
    return 0


def summarize(data: dict, scenarios: list[dict]) -> dict:
    by_id = {s["id"]: s for s in scenarios}
    done = [(by_id[i], r) for i, r in data["runs"].items() if r["status"] == "done" and i in by_id]
    results = [r["result"] for _, r in done]
    return {
        "metrics": metrics.aggregate([(s, r["metrics"]) for s, r in done]),
        "cost": metrics.cost_summary(results, [r.get("judge_usage", {}) for _, r in done]),
        "prompt_versions_seen": sorted({r["prompt_version"] for r in results}),
        "seed_provisions_seen": sorted({str(_seed_of(r)) for r in results}),
        "done": len(done), "pending": sorted(i for i, r in data["runs"].items() if r["status"] != "done"),
        "timeouts": sorted(i for i, r in data["runs"].items() if r["status"] == "timeout"),
    }


def _seed_of(result: dict) -> Any:
    for t in result.get("trace", []):
        if t["name"] == "run.config":
            return t["detail"].get("seed_provisions")
    return None


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _fmt(x: Any, pct: bool = False) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "NO"
    if isinstance(x, float):
        return f"{x:.0%}" if pct else f"{x:.3f}"
    return str(x)


def _outcome_text(status: str, route: Any, reason: Any) -> str:
    return f"{status[:3]} {route or reason or ''}".strip()


def print_report(data: dict, scenarios: list[dict], out: Callable[[str], None] = print) -> None:
    m = data["meta"]
    out(f"\nprompt={m['analysis_prompt']} ({m.get('prompt_versions')})  seed_provisions={m['seed_provisions']}  "
        f"judge={m.get('judge_prompt') or 'off'}")
    out(f"{'id':<4} {'expected':<34} {'actual':<34} {'class':<5} {'recall':<6} {'calls':<5} ok")
    for s in scenarios:
        rec = data["runs"].get(s["id"])
        exp = _outcome_text(s["expected_status"], s["expected_route"], s["expected_reason_code"])
        if not rec or rec["status"] != "done":
            out(f"{s['id']:<4} {exp:<34} {'(' + (rec['status'] if rec else 'not run') + ')':<34}")
            continue
        r, o = rec["result"], rec["metrics"]["outcome"]
        act = _outcome_text(r["status"], r.get("route"), (r.get("escalation") or {}).get("reason_code"))
        flag = ("yes" if o["outcome_correct"] else "NO") + ("  (needs_review)" if s.get("needs_review") else "")
        out(f"{s['id']:<4} {exp:<34} {act:<34} {_fmt(o['classification']):<5} "
            f"{_fmt(o['citation_recall'], pct=True):<6} {metrics.model_calls(r):<5} {flag}")
    summ = summarize(data, scenarios)                          # always fresh (a stored summary may be stale)
    c = summ["metrics"]["confirmed"]
    out(f"\nconfirmed scenarios: {c['n']}  (needs_review reported separately: {summ['metrics']['needs_review']['n']})")
    for k in ("outcome_accuracy", "status_accuracy", "route_accuracy", "classification_accuracy",
              "reason_code_accuracy", "citation_recall"):
        out(f"  {k:<24} {_fmt(c[k], pct=True)}")
    a = summ["metrics"]["all"]
    out("L3 metrics (all scenarios):")
    out(f"  groundedness             {_fmt(a['groundedness'])}  (n={a['n_groundedness']})")
    out(f"  faithfulness             {_fmt(a['faithfulness'])}  (n={a['n_groundedness']})")
    out(f"  context_relevance        {_fmt(a['context_relevance'])}  (n={a['n_context_relevance']})")
    out(f"  hallucination_rate       {_fmt(a['hallucination_rate'], pct=True)}  (n={a['n_hallucination']})")
    cost = summ["cost"]
    out(f"L5 cost: {cost['mean_model_calls_per_run']:.1f} model calls/run, {cost['mean_tokens_per_run']:.0f} tokens/run, "
        f"${cost['mean_cost_per_run_usd']:.5f}/run, ${cost['cost_per_100_runs_usd']:.3f} per 100 runs, "
        f"largest driver: {cost['largest_cost_driver']} (by {cost['driver_basis']})")
    for step, s in sorted(cost["per_step"].items()):
        out(f"  {step:<13} calls={s['calls']:<4} prompt_tokens={s['prompt_tokens']:<7} "
            f"completion_tokens={s['completion_tokens']:<6} cost=${s['cost_usd']:.5f}")
    out(f"  judge (evaluation only, not system cost): {cost['judge']['calls']} calls, "
        f"${cost['judge']['cost_usd']:.5f}")
    if summ["timeouts"]:
        out(f"timed out (watchdog, not scored): {', '.join(summ['timeouts'])}")
    if summ["pending"]:
        out(f"pending (resume with the same command): {', '.join(summ['pending'])}")


COMPARE_KEYS = ["outcome_accuracy", "citation_recall", "groundedness", "faithfulness",
                "context_relevance", "hallucination_rate"]


def compare(path_a: Path, path_b: Path, scenarios: list[dict], out: Callable[[str], None] = print) -> dict:
    """Side-by-side metrics with deltas (B - A), plus scenarios whose outcome changed."""
    a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (path_a, path_b))
    sa, sb = summarize(a, scenarios), summarize(b, scenarios)
    out(f"A: {path_a}  prompt={a['meta']['analysis_prompt']} ({a['meta'].get('prompt_versions')}) "
        f"seed={a['meta']['seed_provisions']}  code={a['meta'].get('code_fingerprint')}  done={sa['done']}")
    out(f"B: {path_b}  prompt={b['meta']['analysis_prompt']} ({b['meta'].get('prompt_versions')}) "
        f"seed={b['meta']['seed_provisions']}  code={b['meta'].get('code_fingerprint')}  done={sb['done']}")
    if set(i for i, r in a["runs"].items() if r["status"] == "done") != \
       set(i for i, r in b["runs"].items() if r["status"] == "done"):
        out("WARNING: the two runs did not complete the same scenarios; deltas are not like for like.")
    rows = {}
    out(f"\n{'metric':<24} {'A':>8} {'B':>8} {'B - A':>8}")
    for block, keys in (("confirmed", COMPARE_KEYS[:2]), ("all", COMPARE_KEYS[2:])):
        for k in keys:
            va, vb = sa["metrics"][block][k], sb["metrics"][block][k]
            delta = None if va is None or vb is None else vb - va
            rows[k] = {"A": va, "B": vb, "delta": delta}
            out(f"{k:<24} {_fmt(va):>8} {_fmt(vb):>8} {('+' if (delta or 0) >= 0 else '') + _fmt(delta):>8}")
    for k in ("mean_model_calls_per_run", "mean_tokens_per_run", "cost_per_100_runs_usd"):
        va, vb = sa["cost"][k], sb["cost"][k]
        rows[k] = {"A": va, "B": vb, "delta": vb - va}
        out(f"{k:<24} {va:>8.1f} {vb:>8.1f} {vb - va:>+8.1f}")
    out(f"largest cost driver      A={sa['cost']['largest_cost_driver']}  B={sb['cost']['largest_cost_driver']}")
    changed = []
    for s in scenarios:
        ra, rb = a["runs"].get(s["id"]), b["runs"].get(s["id"])
        if not (ra and rb and ra["status"] == rb["status"] == "done"):
            continue
        oa, ob = ra["metrics"]["outcome"]["outcome_correct"], rb["metrics"]["outcome"]["outcome_correct"]
        ta = _outcome_text(ra["result"]["status"], ra["result"].get("route"), (ra["result"].get("escalation") or {}).get("reason_code"))
        tb = _outcome_text(rb["result"]["status"], rb["result"].get("route"), (rb["result"].get("escalation") or {}).get("reason_code"))
        if oa != ob or ta != tb:
            changed.append(s["id"])
            out(f"  {s['id']}: A {ta} ({'ok' if oa else 'wrong'})  ->  B {tb} ({'ok' if ob else 'wrong'})")
    if not changed:
        out("  no scenario changed outcome")
    return {"rows": rows, "changed": changed}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default="analysis_v1", help="analysis prompt file name in backend/prompts/")
    ap.add_argument("--seed", choices=["on", "off"], default="on", help="AGENT_SEED_PROVISIONS for these runs")
    ap.add_argument("--limit", type=int, help="only the first N scenarios")
    ap.add_argument("--only", help="comma-separated scenario ids")
    ap.add_argument("--out", help="results file (default evaluation/results/<prompt>_seed-<on|off>.json)")
    ap.add_argument("--no-judge", action="store_true", help="skip the judge call (deterministic metrics only)")
    ap.add_argument("--retriever", choices=["real", "keyword"], default="real")
    ap.add_argument("--scenarios", default=str(SCENARIOS))
    ap.add_argument("--compare", nargs=2, metavar=("A_JSON", "B_JSON"))
    args = ap.parse_args(argv)
    scenarios = load_scenarios(Path(args.scenarios))
    if args.compare:
        compare(Path(args.compare[0]), Path(args.compare[1]), scenarios)
        return 0
    if args.only:
        wanted = set(args.only.split(","))
        scenarios = [s for s in scenarios if s["id"] in wanted]
    if args.limit:
        scenarios = scenarios[: args.limit]
    load_dotenv_if_present()
    if args.retriever == "keyword":
        from backend.agents.fake_retriever import KeywordRetriever
        retriever = KeywordRetriever()
    else:
        from backend.retrieval.search import Retriever
        retriever = Retriever()
    llm = get_llm()
    out = Path(args.out) if args.out else RESULTS_DIR / f"{args.prompt}_seed-{args.seed}.json"
    code = run_batch(scenarios, prompt=args.prompt, seed=args.seed == "on", out_path=out, llm=llm,
                     retriever=retriever, judge_llm=None if args.no_judge else llm)
    print_report(json.loads(out.read_text(encoding="utf-8")), scenarios)
    print(f"\nresults: {out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
