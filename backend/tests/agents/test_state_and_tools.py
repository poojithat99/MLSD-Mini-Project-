"""state.py (result invariants, Budget) and tools.py (tools, working days, timeouts)."""
from __future__ import annotations

import json
import time
from datetime import date

import pytest

from backend.agents.state import (
    Budget, BudgetExhausted, Citation, DeadlineExceeded, DisputeResult, Escalation, ReasonCode, Verification,
)
from backend.agents.tools import Toolbox, ToolTimeout, run_with_timeout, working_days_between


# ------------------------------------------------------------------ DisputeResult invariants
def test_recommendation_requires_passed_verification_and_citations():
    cite = Citation(clause_id="6(ii)", doc_id="rbi_circular_2017", quote="within three working days")
    with pytest.raises(ValueError):
        DisputeResult(run_id="r", status="recommended", route="provisional_credit", citations=[cite],
                      verification=Verification(passed=False))
    ok = DisputeResult(run_id="r", status="recommended", route="provisional_credit", citations=[cite],
                       verification=Verification(passed=True, attempts=1))
    assert ok.escalation is None


def test_escalation_requires_reason_and_no_route():
    with pytest.raises(ValueError):
        DisputeResult(run_id="r", status="escalated", route="decline",
                      escalation=Escalation(reason_code=ReasonCode.TIMEOUT))
    with pytest.raises(ValueError):
        DisputeResult(run_id="r", status="escalated")
    assert len(ReasonCode) == 8 and ReasonCode("injection_detected")


# ------------------------------------------------------------------ Budget
class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_budget_limits_steps_tools_and_wall_clock():
    clock = Clock()
    b = Budget(max_steps=2, max_tool_calls=1, wall_clock_s=10, clock=clock)
    b.charge_step(); b.charge_step()
    with pytest.raises(BudgetExhausted):
        b.charge_step()
    b.charge_tool()
    with pytest.raises(BudgetExhausted):
        b.charge_tool()
    clock.now += 11
    with pytest.raises(DeadlineExceeded):
        b.check_deadline()


# ------------------------------------------------------------------ working days
def test_working_days_section_41_dates():
    # Alert Thu 1 Oct 2026, reported Thu 22 Oct: excludes the alert day, 3 Sundays, 2nd Saturday (10 Oct)
    assert working_days_between(date(2026, 10, 1), date(2026, 10, 22)) == 17
    # with Gandhi Jayanti (2 Oct) as a listed holiday
    assert working_days_between(date(2026, 10, 1), date(2026, 10, 22), frozenset({date(2026, 10, 2)})) == 16


def test_working_days_boundaries():
    assert working_days_between(date(2026, 10, 5), date(2026, 10, 5)) == 0      # same day
    assert working_days_between(date(2026, 10, 5), date(2026, 10, 8)) == 3      # Mon -> Thu
    assert working_days_between(date(2026, 10, 2), date(2026, 10, 3)) == 1      # 1st Saturday is working
    assert working_days_between(date(2026, 10, 9), date(2026, 10, 11)) == 0     # 2nd Sat + Sunday
    with pytest.raises(ValueError):
        working_days_between(date(2026, 10, 5), date(2026, 10, 1))


# ------------------------------------------------------------------ tools
def make_toolbox(retriever, request, **budget):
    evidence, trace = {}, []
    return Toolbox(retriever, request, Budget(**budget), evidence, trace), evidence, trace


def test_search_records_evidence_and_trace(retriever, request_41):
    tb, evidence, trace = make_toolbox(retriever, request_41)
    out = json.loads(tb.run("search_corpus", {"query": "customer shared payment credentials negligence", "k": 3}))
    assert out[0]["chunk_id"] in evidence
    assert "rbi_circular_2017:7(i)" in evidence
    assert evidence["rbi_circular_2017:7(i)"].clause_ids == ["7(i)", "7(ii)"]
    assert trace[0].type == "tool" and trace[0].name == "search_corpus"
    assert trace[0].detail["arguments"]["query"].startswith("customer shared")


def test_get_clause_adds_table_label(retriever, request_41):
    tb, evidence, _ = make_toolbox(retriever, request_41)
    tb.run("get_clause", {"doc_id": "rbi_circular_2017", "clause_id": "Table 2"})
    assert "Table 2" in evidence["rbi_circular_2017:table-2"].clause_ids


def test_reporting_delay_tool_is_deterministic(retriever, request_41):
    tb, _, _ = make_toolbox(retriever, request_41)
    out = json.loads(tb.run("compute_reporting_delay", {}))
    assert out["working_days"] == 17 and out["counted_from_field"] == "communication_date"
    assert "2nd and 4th Saturdays" in out["assumption"]


def test_tool_errors_go_back_to_the_model_not_crash(retriever, request_41):
    tb, _, trace = make_toolbox(retriever, request_41)
    assert "unknown tool" in json.loads(tb.run("delete_everything", {}))["error"]
    assert "error" in json.loads(tb.run("get_clause", {"doc_id": "rbi_circular_2017"}))   # missing arg
    assert trace[-1].detail["error"]


def test_tool_call_budget_is_enforced(retriever, request_41):
    tb, _, _ = make_toolbox(retriever, request_41, max_tool_calls=1)
    tb.run("get_request_fields", {})
    with pytest.raises(BudgetExhausted):
        tb.run("get_request_fields", {})


def test_tool_timeout():
    with pytest.raises(ToolTimeout):
        run_with_timeout(lambda: time.sleep(1), timeout_s=0.05)


def test_slow_retriever_times_out_inside_toolbox(request_41):
    class Slow:
        def search(self, *a, **k):
            time.sleep(1)
            return []
    evidence, trace = {}, []
    tb = Toolbox(Slow(), request_41, Budget(), evidence, trace, timeout_s=0.05)
    assert "exceeded" in json.loads(tb.run("search_corpus", {"query": "x"}))["error"]
