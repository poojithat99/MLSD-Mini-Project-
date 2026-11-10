"""metrics.py with FakeLLM: the four L3 metrics, the one-call cached judge, outcome checks, L5 cost."""
from __future__ import annotations

import copy

import pytest

from backend.agents.llm import FakeLLM, LLMClient
from backend.tests.agents.test_llm import make_config
from backend.tests.evaluation.conftest import agent_llm, judge_llm, scripted_judge
from evaluation import metrics


# ------------------------------------------------------------------ metric arithmetic
def test_groundedness_faithfulness_relevance_arithmetic():
    claims = ["a.", "b.", "c."]
    judgement = {"support": {0: ["A"], 1: ["B"], 2: []}, "relevance": [(0, "A", 1.0), (0, "B", 0.5), (1, "C", 0.0)],
                 "cached": False}
    assert metrics.groundedness(claims, judgement) == pytest.approx(2 / 3)            # any retrieved passage
    assert metrics.faithfulness(claims, judgement, cited={"A"}) == pytest.approx(1 / 3)  # cited passages only
    assert metrics.context_relevance(judgement) == pytest.approx(0.5)
    assert metrics.groundedness([], judgement) is None and metrics.faithfulness(claims, None, set()) is None


def test_hallucination_rules():
    j_ok = {"support": {0: ["A"]}, "relevance": [], "cached": False}
    j_bad = {"support": {0: []}, "relevance": [], "cached": False}
    assert metrics.hallucinated([{"doc_id": "x"}], ["a."], j_ok) is True               # unresolved citation
    assert metrics.hallucinated([], ["a."], j_bad) is True                             # unsupported claim
    assert metrics.hallucinated([], ["a."], j_ok) is False
    assert metrics.hallucinated([], [], None) is False                                 # nothing asserted
    assert metrics.hallucinated([], ["a."], None) is None                              # half the evidence only


# ------------------------------------------------------------------ reading a real run
def test_run_reading_uses_trace_chunks_and_output_text(scenarios, run_scenario):
    r = run_scenario(scenarios["S04"])                                   # scripted: provisional_credit citing 7(ii)
    assert r["status"] == "recommended"
    passages = metrics.retrieved_passages(r)
    assert "rbi_circular_2017:table-2" in passages                       # seeded by code, still "retrieved"
    assert metrics.output_claims(r)                                      # basis/liability sentences
    assert metrics.cited_chunk_ids(r, passages) >= {"rbi_circular_2017:7(i)"}   # chunk that holds 7(ii)
    assert metrics.unresolved_citations(r, passages) == []
    assert metrics.search_pairs(r) and all(q for q, _ in metrics.search_pairs(r))


def test_fabricated_citation_counts_as_hallucination(scenarios, run_scenario):
    r = copy.deepcopy(run_scenario(scenarios["S04"]))
    r["citations"][0]["quote"] = "The bank shall always refund the customer in full within one day"
    passages = metrics.retrieved_passages(r)
    assert metrics.unresolved_citations(r, passages)
    score = metrics.score_run(scenarios["S04"], r, None)
    assert score["hallucinated"] is True


# ------------------------------------------------------------------ judge: one call per run, cached
def test_one_judge_call_per_run_and_cache(tmp_path, scenarios, run_scenario):
    r = run_scenario(scenarios["S04"])
    judge = judge_llm()
    first = metrics.score_run(scenarios["S04"], r, judge, cache_dir=tmp_path)
    assert len(judge.primary.calls) == 1 and first["judged"] and not first["judge_cached"]
    again = metrics.score_run(scenarios["S04"], r, judge, cache_dir=tmp_path)
    assert len(judge.primary.calls) == 1 and again["judge_cached"]                     # no second model call
    assert again["groundedness"] == first["groundedness"] == 1.0
    assert 0 < first["context_relevance"] < 1


def test_judge_payload_is_data_only_with_short_ids(scenarios, run_scenario):
    r = run_scenario(scenarios["S01"])
    payload, back = metrics.judge_payload(r)
    assert set(payload) == {"claims", "passages", "searches"}
    assert all(p["id"].startswith("P") for p in payload["passages"]) and set(back.values()) == set(
        metrics.retrieved_passages(r))


def test_no_judge_call_when_nothing_to_judge(tmp_path, scenarios, run_scenario):
    r = run_scenario(scenarios["S17"])                                                  # injection: no model at all
    judge = judge_llm()
    score = metrics.score_run(scenarios["S17"], r, judge, cache_dir=tmp_path)
    assert judge.primary.calls == [] and not score["judged"] and score["hallucinated"] is False


def test_judge_cost_goes_to_its_own_ledger(tmp_path, scenarios, run_scenario):
    from backend.agents.llm import UsageLedger
    ledger = UsageLedger()
    metrics.score_run(scenarios["S01"], run_scenario(scenarios["S01"]), judge_llm(), tmp_path, ledger)
    assert [r.step for r in ledger.records] == ["judge"]


# ------------------------------------------------------------------ outcome accuracy (deterministic)
def test_outcome_checks_section_41(scenarios, run_scenario):
    o = metrics.outcome_checks(scenarios["S11"], run_scenario(scenarios["S11"]))
    assert o["outcome_correct"] and o["reason_code"] and o["citation_recall"] == 1.0 and o["delay"]


def test_outcome_checks_injection_requires_no_model_call(scenarios, run_scenario):
    o = metrics.outcome_checks(scenarios["S17"], run_scenario(scenarios["S17"]))
    assert o["outcome_correct"] and o["no_model_call"] and o["citation_recall"] is None


def test_outcome_checks_detect_a_wrong_answer(scenarios, run_scenario):
    r = copy.deepcopy(run_scenario(scenarios["S11"]))
    r.update(status="recommended", route="decline", escalation=None)
    o = metrics.outcome_checks(scenarios["S11"], r)
    assert not o["outcome_correct"] and not o["status"] and not o["reason_code"]


def test_aggregate_separates_needs_review(scenarios, run_scenario):
    pairs = [(scenarios[i], metrics.score_run(scenarios[i], run_scenario(scenarios[i]), None))
             for i in ("S01", "S11", "S17", "S07")]
    agg = metrics.aggregate(pairs)
    assert agg["confirmed"]["n"] == 3 and agg["needs_review"]["n"] == 1 and agg["all"]["n"] == 4
    assert "section_4_1" in agg["by_tag"]


# ------------------------------------------------------------------ L5 cost
def test_cost_summary_tokens_basis_on_free_tier(scenarios, run_scenario):
    runs = [run_scenario(scenarios[i]) for i in ("S04", "S11")]
    c = metrics.cost_summary(runs)
    assert set(c["per_step"]) == {"analysis", "verification"}
    assert c["driver_basis"] == "tokens" and c["largest_cost_driver"] == "analysis"
    assert c["tokens_per_100_runs"] == pytest.approx(c["mean_tokens_per_run"] * 100)


def test_cost_summary_cost_basis_when_rates_set(scenarios, retriever):
    from backend.agents.graph import run_dispute_graph
    llm = agent_llm(cost_per_1k_prompt=1.0, cost_per_1k_completion=2.0)
    r = run_dispute_graph(scenarios["S01"]["request"], "c", "c", llm=llm, retriever=retriever).model_dump(mode="json")
    c = metrics.cost_summary([r], [{"calls": 1, "cost_usd": 0.5}])
    assert c["driver_basis"] == "cost_usd" and c["mean_cost_per_run_usd"] > 0
    assert c["cost_per_100_runs_usd"] == pytest.approx(c["mean_cost_per_run_usd"] * 100)
    assert c["judge"]["cost_usd"] == 0.5                                               # reported separately


def test_scripted_judge_is_usable_directly():
    llm = LLMClient(FakeLLM(scripted_judge), make_config(), sleep=lambda s: None)
    assert llm is not None


def test_code_written_conflict_basis_is_not_a_model_claim(scenarios, run_scenario):
    r = run_scenario(scenarios["S11"])
    assert metrics.code_written_basis(r) == r["basis"] and r["basis"]
    assert metrics.output_claims(r) == []                      # nothing the model wrote is asserted


def test_sentence_split_keeps_rupee_amounts_whole():
    r = {"basis": "Liability is limited to Rs. 10,000, whichever is lower. See e.g. Table 1.",
         "liability_position": "Capped at Rs. 10,000."}
    assert metrics.output_claims(r) == ["Liability is limited to Rs. 10,000, whichever is lower.",
                                        "See e.g. Table 1.", "Capped at Rs. 10,000."]
