"""Retrieval probe: does the index surface the provisions we expect?

    python -m backend.retrieval.probe            # table of ranks + summary
    python -m backend.retrieval.probe --json     # machine-readable (for before/after comparison, L3)

Each probe is a query plus the provision(s) that should answer it, written as
(doc_id, clause-or-table).  A probe passes at rank r if the r-th passage is a chunk
that *contains* that clause (so a chunk holding 7(i)+7(ii) satisfies "7(ii)").

This is a small smoke set, not the L3 evaluation set.  Its jobs are:
  * compare index variants with identical queries (e.g. EMBED_HEADER_MODE=full vs light,
    CHUNK_MAX_WORDS, embedding model);
  * show the vocabulary gap the agent must bridge (probe 2 vs 1: the corpus never says "OTP");
  * give a first look at the score range of in-corpus vs out-of-corpus queries, which is
    what an escalation threshold for F7 has to separate.
"""
from __future__ import annotations

import argparse
import json

from .config import Settings
from .search import Retriever

CIRC, SBI, OMB = "rbi_circular_2017", "sbi_policy_2021", "rbi_ios_2026"

PROBES: list[dict] = [
    {"id": "negligence-credentials", "q": "customer shared payment credentials with a caller, negligence, who bears the loss",
     "expect": [(CIRC, "7(i)"), (SBI, "4.6.2(b)(i)")]},
    {"id": "negligence-otp", "q": "customer shared an OTP with someone who called saying he was from the bank",
     "expect": [(CIRC, "7(i)"), (SBI, "4.6.2(b)(i)")]},
    {"id": "zero-liability-bank-fault", "q": "zero liability when the bank is at fault or contributory negligence by the bank",
     "expect": [(CIRC, "6(i)"), (SBI, "4.6.2(a)(i)")]},
    {"id": "report-within-3-days", "q": "customer reports unauthorised transaction within three working days, what is the liability",
     "expect": [(CIRC, "table:2"), (SBI, "table:2")]},
    {"id": "report-after-7-days", "q": "reported more than seven working days after the bank's communication, liability as per board approved policy",
     "expect": [(CIRC, "table:2"), (CIRC, "table:1"), (SBI, "table:2"), (SBI, "4.6.3")]},
    {"id": "working-days-counting", "q": "how are working days counted when computing the number of days taken to report",
     "expect": [(CIRC, "table:2"), (SBI, "table:2")]},
    {"id": "max-liability-amounts", "q": "maximum liability of a customer for credit card with limit up to 5 lakh",
     "expect": [(CIRC, "table:1"), (SBI, "table:1")]},
    {"id": "burden-of-proof", "q": "who has to prove that the customer is liable for the unauthorised transaction",
     "expect": [(CIRC, "12")]},
    {"id": "reversal-timeline", "q": "how many days does the bank have to credit the disputed amount back to the customer's account",
     "expect": [(CIRC, "9"), (CIRC, "10")]},
    {"id": "ombudsman-complaint", "q": "how can a customer complain to the RBI Ombudsman if the bank does not resolve the dispute",
     "expect": [(OMB, "*")]},
]
#: far-domain: easy to reject.  near-domain: banking vocabulary but NOT answerable from the corpus; these are
#: the hard cases for an escalation threshold (F7), because they share terms with real provisions.
OUT_OF_CORPUS = [
    "how do I renew my passport", "what is the capital gains tax rate on mutual funds", "recipe for chocolate cake",
    "what is the current RBI repo rate", "what is the interest rate on a home loan",
    "how do I apply for a new debit card", "what is the daily ATM cash withdrawal limit",
    "how do I close my savings account", "what are the NEFT transfer charges",
]


def _matches(passage: dict, retriever: Retriever, doc: str, clause: str) -> bool:
    c = retriever.by_id[passage["chunk_id"]]
    if c["doc_id"] != doc:
        return False
    if clause == "*":
        return True
    if clause.startswith("table:"):
        return c["table_no"] == clause[6:]
    return clause in c["clause_ids"]


def run(retriever: Retriever, k: int = 10) -> dict:
    rows = []
    for pr in PROBES:
        plain = retriever.search(pr["q"], k)
        expanded = retriever.search(pr["q"], 5, expand_refs=True)
        ranks = []
        for r, p in enumerate(plain, 1):
            d = p.to_dict()
            if any(_matches(d, retriever, doc, cl) for doc, cl in pr["expect"]):
                ranks.append(r)
        found = {(doc, cl): next((r for r, p in enumerate(plain, 1) if _matches(p.to_dict(), retriever, doc, cl)), None)
                 for doc, cl in pr["expect"]}
        in_exp = any(_matches(p.to_dict(), retriever, doc, cl) for p in expanded for doc, cl in pr["expect"])
        rows.append({
            "id": pr["id"], "first_rank": ranks[0] if ranks else None,
            "top_score": round(plain[0].score, 3) if plain else None,
            "found_with_expand_k5": in_exp, "per_expected": {f"{d}:{c}": r for (d, c), r in found.items()},
        })
    ranked = [r["first_rank"] for r in rows]
    summary = {
        "hit@1": sum(1 for r in ranked if r == 1) / len(rows),
        "hit@5": sum(1 for r in ranked if r and r <= 5) / len(rows),
        "mrr": round(sum(1 / r for r in ranked if r) / len(rows), 3),
        "hit@5_with_expand": sum(1 for r in rows if r["found_with_expand_k5"]) / len(rows),
    }
    ood = [{"q": q, "top_score": round(retriever.search(q, 1)[0].score, 3)} for q in OUT_OF_CORPUS]
    in_scores = [r["top_score"] for r in rows if r["top_score"] is not None]
    summary["in_corpus_top_score_min"] = min(in_scores)
    summary["out_of_corpus_top_score_max"] = max(o["top_score"] for o in ood)
    summary["threshold_separable"] = summary["in_corpus_top_score_min"] > summary["out_of_corpus_top_score_max"]
    summary["margin"] = round(summary["in_corpus_top_score_min"] - summary["out_of_corpus_top_score_max"], 3)
    return {"embedder": retriever.manifest["embedder"]["name"], "chunker": retriever.manifest["chunker"],
            "probes": rows, "out_of_corpus": ood, "summary": summary}


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe retrieval quality on a fixed query set.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    res = run(Retriever(Settings.from_env()))
    if args.json:
        print(json.dumps(res, indent=2))
        return
    print(f"embedder: {res['embedder']}\nchunker : {res['chunker']}\n")
    print(f"{'probe':26} {'1st hit':>7} {'top score':>9} {'hit w/ expand':>14}   rank of each expected provision")
    for r in res["probes"]:
        per = ", ".join(f"{k.split(':', 1)[1]}={v or '-'}" for k, v in r["per_expected"].items())
        print(f"{r['id']:26} {str(r['first_rank'] or '-'):>7} {r['top_score']:>9} {str(r['found_with_expand_k5']):>14}   {per}")
    print("\nout-of-corpus queries (top cosine score):")
    for o in res["out_of_corpus"]:
        print(f"  {o['top_score']:>6}  {o['q']}")
    s = res["summary"]
    print(f"\nSUMMARY  hit@1={s['hit@1']:.2f}  hit@5={s['hit@5']:.2f}  MRR={s['mrr']}  hit@5+expand={s['hit@5_with_expand']:.2f}")
    print(f"         lowest in-corpus top score = {s['in_corpus_top_score_min']}, "
          f"highest out-of-corpus top score = {s['out_of_corpus_top_score_max']}  (margin {s['margin']})")
    if not s["threshold_separable"]:
        print("         -> NO single score threshold separates answerable from unanswerable queries here:\n"
              "            escalation (F7) must also use the agent's verification step (A3), not score alone.")


if __name__ == "__main__":
    main()
