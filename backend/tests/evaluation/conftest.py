"""Fixtures for evaluation tests: FakeLLM only, keyword retriever over the real chunks, no network."""
from __future__ import annotations

import json

import pytest

from backend.agents.fake_retriever import KeywordRetriever
from backend.agents.fake_scenarios import ScriptedAgents
from backend.agents.graph import run_dispute_graph
from backend.agents.llm import FakeLLM, LLMClient, LLMResponse
from backend.tests.agents.test_llm import make_config
from evaluation.evaluate import load_scenarios


@pytest.fixture(scope="session")
def retriever():
    return KeywordRetriever()


@pytest.fixture(scope="session")
def scenarios() -> dict[str, dict]:
    return {s["id"]: s for s in load_scenarios()}


def agent_llm(script=None, **cfg) -> LLMClient:
    """Scripted agents.  Free-tier cost rates (0) unless a test sets them.
    Note: ScriptedAgents detects shared credentials by regex and ignores negation, so a narrative
    like S01's "I never shared any card details, OTP or PIN" looks like sharing to it; tests that
    need a scripted recommendation use S04 instead."""
    cfg = {"cost_per_1k_prompt": 0.0, "cost_per_1k_completion": 0.0, **cfg}
    return LLMClient(FakeLLM(script or ScriptedAgents()), make_config(**cfg), sleep=lambda s: None)


def scripted_judge(messages, tools):
    """Judge stand-in: every claim is supported by the first passage; every searched passage is
    relevant (1) except the last one returned by each search (0)."""
    payload = json.loads(messages[1]["content"])
    first = payload["passages"][0]["id"] if payload["passages"] else None
    support = [{"claim": c["id"], "passages": [first] if first else []} for c in payload["claims"]]
    relevance = [{"search": s["id"], "passage": p, "score": 0 if i == len(s["passages"]) - 1 else 1}
                 for s in payload["searches"] for i, p in enumerate(s["passages"])]
    return LLMResponse(content=json.dumps({"support": support, "relevance": relevance}))


def judge_llm(script=scripted_judge) -> LLMClient:
    return LLMClient(FakeLLM(script), make_config(), sleep=lambda s: None)


@pytest.fixture
def run_scenario(retriever):
    def _run(scenario: dict, **kw) -> dict:
        return run_dispute_graph(scenario["request"], f"t-{scenario['id']}", "t", llm=agent_llm(),
                                 retriever=retriever, **kw).model_dump(mode="json")
    return _run
