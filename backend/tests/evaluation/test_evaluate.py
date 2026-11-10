"""evaluate.py with FakeLLM: batch run, resume, quota stop, judge-pending, compare, CLI."""
from __future__ import annotations

import json

import pytest

from backend.agents.fake_scenarios import ScriptedAgents
from backend.agents.llm import LLMError, reset_llm
from backend.tests.evaluation.conftest import agent_llm, judge_llm
from evaluation import evaluate

THREE = ("S04", "S11", "S17")          # recommendation, section 4.1 escalation, injection (no model call)


def batch(scenarios, ids, tmp_path, out="r.json", llm=None, judge=None, seed=True, prompt="analysis_v1", **kw):
    return evaluate.run_batch([scenarios[i] for i in ids], prompt=prompt, seed=seed, out_path=tmp_path / out,
                              llm=llm or agent_llm(), retriever=kw.pop("retriever"),
                              judge_llm=judge, cache_dir=tmp_path / "cache", log=lambda m: None, **kw)


def test_batch_runs_scores_and_records_versions(tmp_path, scenarios, retriever):
    assert batch(scenarios, THREE, tmp_path, judge=judge_llm(), retriever=retriever) == 0
    data = json.loads((tmp_path / "r.json").read_text())
    assert data["meta"]["seed_provisions"] is True and data["meta"]["judge_prompt"] == "judge_v1"
    from backend.agents.prompts import load_prompt
    expected = f"{load_prompt('analysis_v1')[1]}+{load_prompt('verification_v1')[1]}"
    assert data["meta"]["prompt_versions"] == expected
    assert all(data["runs"][i]["status"] == "done" for i in THREE)
    summary = data["summary"]
    assert summary["prompt_versions_seen"] == [expected]
    assert summary["seed_provisions_seen"] == ["True"]
    assert summary["metrics"]["confirmed"]["outcome_accuracy"] == 1.0
    assert summary["cost"]["largest_cost_driver"] == "analysis"
    assert data["runs"]["S04"]["judge_usage"]["calls"] == 1 and data["runs"]["S17"]["judge_usage"]["calls"] == 0


def test_results_contain_no_model_ids_or_endpoints(tmp_path, scenarios, retriever):
    batch(scenarios, ("S04",), tmp_path, retriever=retriever)
    text = (tmp_path / "r.json").read_text().lower()
    assert "base_url" not in text and "api_key" not in text and "http" not in text


def test_resume_skips_done_scenarios(tmp_path, scenarios, retriever):
    batch(scenarios, THREE, tmp_path, retriever=retriever)
    def must_not_be_called(messages, tools):
        raise AssertionError("resume re-ran a finished scenario")
    assert batch(scenarios, THREE, tmp_path, llm=agent_llm(must_not_be_called), retriever=retriever) == 0


def test_provider_failure_defers_and_stops_then_resumes(tmp_path, scenarios, retriever):
    down = agent_llm(lambda m, t: TimeoutError("quota exhausted"))
    code = batch(scenarios, ("S17", "S04", "S11"), tmp_path, llm=down, retriever=retriever)
    assert code == evaluate.EXIT_DEFERRED
    runs = json.loads((tmp_path / "r.json").read_text())["runs"]
    assert runs["S17"]["status"] == "done"                     # no model needed: finished before the outage
    assert runs["S04"]["status"] == "deferred" and "S11" not in runs   # stopped; nothing after it was attempted
    n_attempts = len(down.primary.calls)
    assert n_attempts <= 3                                     # one bounded call (max_attempts), no tight loop
    assert batch(scenarios, ("S17", "S04", "S11"), tmp_path, retriever=retriever) == 0
    runs = json.loads((tmp_path / "r.json").read_text())["runs"]
    assert all(runs[i]["status"] == "done" for i in ("S17", "S04", "S11"))


def test_judge_failure_is_pending_and_only_the_judge_is_redone(tmp_path, scenarios, retriever):
    broken = judge_llm(lambda m, t: LLMError("400 bad judge request"))
    assert batch(scenarios, ("S04",), tmp_path, judge=broken, retriever=retriever) == evaluate.EXIT_DEFERRED
    assert json.loads((tmp_path / "r.json").read_text())["runs"]["S04"]["status"] == "judge_pending"
    def no_agent_calls(messages, tools):
        raise AssertionError("agent re-run although only the judge was pending")
    assert batch(scenarios, ("S04",), tmp_path, llm=agent_llm(no_agent_calls), judge=judge_llm(),
                 retriever=retriever) == 0


def test_refuses_to_mix_seed_modes_in_one_file(tmp_path, scenarios, retriever):
    batch(scenarios, ("S17",), tmp_path, seed=True, retriever=retriever)
    with pytest.raises(SystemExit, match="seed_provisions"):
        batch(scenarios, ("S17",), tmp_path, seed=False, retriever=retriever)


def test_seed_off_is_recorded_per_result(tmp_path, scenarios, retriever):
    batch(scenarios, ("S04",), tmp_path, seed=False, retriever=retriever)
    data = json.loads((tmp_path / "r.json").read_text())
    assert data["meta"]["seed_provisions"] is False and data["summary"]["seed_provisions_seen"] == ["False"]


def test_compare_prints_side_by_side_deltas_and_changed_outcomes(tmp_path, scenarios, retriever):
    batch(scenarios, THREE, tmp_path, out="a.json", retriever=retriever)
    # B: a run where the analyst always fabricates -> every recommendation fails verification
    batch(scenarios, THREE, tmp_path, out="b.json", llm=agent_llm(ScriptedAgents(fabricate="always")),
          retriever=retriever)
    lines: list[str] = []
    res = evaluate.compare(tmp_path / "a.json", tmp_path / "b.json", list(scenarios.values()), out=lines.append)
    text = "\n".join(lines)
    assert "B - A" in text and "outcome_accuracy" in text and "cost_per_100_runs_usd" in text
    assert res["rows"]["outcome_accuracy"]["delta"] < 0 and "S04" in res["changed"]


def test_report_prints_expected_vs_actual(tmp_path, scenarios, retriever):
    batch(scenarios, THREE, tmp_path, retriever=retriever)
    lines: list[str] = []
    evaluate.print_report(json.loads((tmp_path / "r.json").read_text()), [scenarios[i] for i in THREE],
                          out=lines.append)
    text = "\n".join(lines)
    assert "S11" in text and "esc conflicting_provisions" in text and "groundedness" in text
    assert "largest driver: analysis" in text


def test_cli_end_to_end_with_fake_provider(tmp_path, monkeypatch, capsys):
    # never read the developer's .env in tests (it would also leak into os.environ for later tests)
    monkeypatch.setattr(evaluate, "load_dotenv_if_present", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("LLM_FAKE_SCENARIO", "default")
    monkeypatch.setenv("LLM_MIN_INTERVAL_S", "0")
    reset_llm()
    try:
        code = evaluate.main(["--prompt", "analysis_v2", "--only", "S04,S17", "--retriever", "keyword",
                              "--no-judge", "--out", str(tmp_path / "cli.json")])
    finally:
        reset_llm()
    assert code == 0
    out = capsys.readouterr().out
    from backend.agents.prompts import load_prompt
    assert f"prompt=analysis_v2 ({load_prompt('analysis_v2')[1]}+" in out and "results:" in out
    data = json.loads((tmp_path / "cli.json").read_text())
    assert set(data["runs"]) == {"S04", "S17"} and data["meta"]["judge_prompt"] is None


def test_watchdog_records_a_hung_scenario_and_moves_on(tmp_path, scenarios, retriever, monkeypatch):
    import time as _time
    real = evaluate.run_dispute_graph

    def hanging_for_s04(request, run_id, *a, **k):
        if run_id == "eval-S04":
            _time.sleep(3)
        return real(request, run_id, *a, **k)
    monkeypatch.setattr(evaluate, "run_dispute_graph", hanging_for_s04)
    start = _time.perf_counter()
    assert batch(scenarios, ("S04", "S17"), tmp_path, retriever=retriever, timeout_s=0.3) == 0
    assert _time.perf_counter() - start < 2.5
    data = json.loads((tmp_path / "r.json").read_text())
    assert data["runs"]["S04"]["status"] == "timeout" and data["runs"]["S17"]["status"] == "done"
    assert data["summary"]["timeouts"] == ["S04"]


def test_scenario_timeout_is_env_configurable(monkeypatch):
    monkeypatch.delenv("EVAL_SCENARIO_TIMEOUT_S", raising=False)
    assert evaluate.scenario_timeout_s() == 300.0
    monkeypatch.setenv("EVAL_SCENARIO_TIMEOUT_S", "45")
    assert evaluate.scenario_timeout_s() == 45.0


def test_resume_refuses_different_prompt_or_code_versions(tmp_path, scenarios, retriever, monkeypatch):
    batch(scenarios, ("S17",), tmp_path, retriever=retriever)
    monkeypatch.setattr(evaluate, "code_fingerprint", lambda: "0" * 12)
    with pytest.raises(SystemExit, match="code_fingerprint"):
        batch(scenarios, ("S17",), tmp_path, retriever=retriever)
    monkeypatch.undo()
    data = json.loads((tmp_path / "r.json").read_text())
    data["meta"]["prompt_versions"] = "analysis_v1.2+verification_v1.1"          # an old file
    (tmp_path / "r.json").write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="prompt_versions"):
        batch(scenarios, ("S17",), tmp_path, retriever=retriever)


def test_fingerprint_changes_when_a_prompt_changes(tmp_path, monkeypatch):
    (tmp_path / "backend" / "prompts").mkdir(parents=True)
    f = tmp_path / "backend" / "prompts" / "x.txt"
    f.write_text("version: a\nhello")
    monkeypatch.setattr(evaluate, "REPO", tmp_path)
    before = evaluate.code_fingerprint()
    f.write_text("version: a\nhello!")
    assert evaluate.code_fingerprint() != before
