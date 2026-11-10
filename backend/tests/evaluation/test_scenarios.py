"""The scenario set is checked against the REAL corpus: every expected clause exists, every
supporting passage is verbatim corpus text, every delay matches the calculator."""
from __future__ import annotations

import json
from collections import Counter

from backend.agents.guardrails import guard_request, injected_free_text_fields
from backend.agents.state import CLASSIFICATIONS, ROUTES, DisputeInput, ReasonCode
from backend.agents.tools import compute_reporting_delay
from backend.retrieval.ingest import normalize_for_match

CHUNKS = [json.loads(line) for line in open("data/index/chunks.jsonl", encoding="utf-8") if line.strip()]
FIELDS = {"id", "tags", "request", "expected_status", "expected_route", "expected_classification",
          "expected_reason_code", "expected_clause_ids", "expected_delay_working_days", "supporting_passage",
          "rationale", "needs_review"}


def labels(chunk: dict) -> set[str]:
    return set(chunk["clause_ids"]) | ({f"Table {chunk['table_no']}"} if chunk["table_no"] else set())


def test_twenty_unique_scenarios_with_all_fields(scenarios):
    assert len(scenarios) == 20 and sorted(scenarios) == [f"S{i:02d}" for i in range(1, 21)]
    for s in scenarios.values():
        assert set(s) == FIELDS, s["id"]
        assert s["rationale"].strip()


def test_requests_are_valid_dispute_inputs(scenarios):
    for s in scenarios.values():
        DisputeInput.model_validate(s["request"])


def test_expected_values_use_the_real_vocabularies(scenarios):
    for s in scenarios.values():
        assert s["expected_status"] in ("recommended", "escalated")
        assert s["expected_route"] in (*ROUTES, None)
        assert s["expected_classification"] in (*CLASSIFICATIONS, None)
        assert s["expected_reason_code"] in ({r.value for r in ReasonCode} | {None})
        if s["expected_status"] == "recommended":
            assert s["expected_route"] and s["expected_reason_code"] is None, s["id"]
        else:
            assert s["expected_route"] is None and s["expected_reason_code"], s["id"]


def test_every_expected_clause_and_passage_exists_verbatim_in_the_corpus(scenarios):
    for s in scenarios.values():
        cited = set()
        for p in s["supporting_passage"]:
            hits = [c for c in CHUNKS if c["doc_id"] == p["doc_id"] and p["clause_id"] in labels(c)
                    and normalize_for_match(p["text"]) in normalize_for_match(c["text"] + " " + c["lead_in"])]
            assert hits, f"{s['id']}: {p['doc_id']} {p['clause_id']} text not found verbatim"
            cited.add(f"{p['doc_id']} {p['clause_id']}")
        assert set(s["expected_clause_ids"]) == cited, s["id"]


def test_delays_match_the_calculator(scenarios):
    for s in scenarios.values():
        got = compute_reporting_delay(DisputeInput.model_validate(s["request"]))["working_days"]
        assert got == s["expected_delay_working_days"], s["id"]


def test_boundaries_cover_the_table_rows(scenarios):
    delays = {s["id"]: s["expected_delay_working_days"] for s in scenarios.values()}
    assert delays["S02"] == 3 and delays["S04"] == 4 and delays["S05"] == 7 and delays["S06"] == 8


def test_required_mix(scenarios):
    tags = Counter(t for s in scenarios.values() for t in s["tags"])
    assert tags["zero_liability"] >= 3 and tags["limited_liability"] >= 2 and tags["late_report"] >= 2
    assert tags["negligence"] >= 3 and tags["section_4_1"] == 3 and tags["out_of_corpus"] >= 2
    assert tags["prompt_injection"] == 2 and tags["duplicate"] >= 1 and tags["merchant_dispute"] >= 1


def test_needs_review_is_exactly_the_agreed_set(scenarios):
    assert {i for i, s in scenarios.items() if s["needs_review"]} == {"S07", "S19", "S20"}
    assert "assumption" in scenarios["S07"]["rationale"] and "thin" in scenarios["S19"]["rationale"]


def test_negligence_and_41_expectations(scenarios):
    for i in ("S08", "S09", "S10", "S11", "S12"):
        assert scenarios[i]["expected_reason_code"] == "conflicting_provisions"
        assert "rbi_circular_2017 7(i)" in scenarios[i]["expected_clause_ids"]
    assert scenarios["S11"]["request"]["narrative"].startswith("The bank's SMS alert for the transaction came the same day")


def test_injection_scenarios_actually_trigger_the_guardrail(scenarios):
    for i in ("S17", "S18"):
        _, flags = guard_request(DisputeInput.model_validate(scenarios[i]["request"]))
        assert injected_free_text_fields(flags), i
    for i, s in scenarios.items():
        if "prompt_injection" not in s["tags"]:
            _, flags = guard_request(DisputeInput.model_validate(s["request"]))
            assert not injected_free_text_fields(flags), f"{i} would be escalated as an injection"
