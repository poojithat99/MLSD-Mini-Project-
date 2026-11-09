"""Check the configured model provider through llm.py (the same path the agents use).

    python -m backend.agents.llm_check                 # (a) plain completion, (b) one trivial tool,
                                                       # (d) two-turn tool loop (thought signatures)
    python -m backend.agents.llm_check --agent-tools   # (c) also send the agents' real tool definitions

Reads LLM_* from the environment (and .env if present).  Prints each result,
token counts, latency and cost, or the error.  For a provider 400 the error
includes the response body (e.g. which tool-schema feature Gemini rejected).
Exit code 0 = every check passed.  The API key is never printed.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from urllib.parse import urlparse

from .llm import (
    THOUGHT_SIGNATURE_FALLBACK, LLMClient, LLMConfig, UsageLedger, assistant_tool_call_message, build_client,
    load_dotenv_if_present,
)
from .tools import TOOL_SPECS

TRIVIAL_TOOL = {"type": "function", "function": {
    "name": "get_time",
    "description": "Returns the current time in a time zone.",
    "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]},
}}


def _check(name: str, fn) -> bool:
    print(f"\n[{name}]")
    start = time.perf_counter()
    try:
        lines = fn()
    except Exception as exc:  # noqa: BLE001 - report every failure clearly, keep going
        print(f"  FAIL after {(time.perf_counter() - start):.1f}s: {type(exc).__name__}: {exc}")
        return False
    for line in lines:
        print("  " + line)
    print(f"  OK ({(time.perf_counter() - start):.1f}s)")
    return True


def _usage(ledger: UsageLedger) -> str:
    r = ledger.records[-1]
    return (f"prompt_tokens={r.prompt_tokens} completion_tokens={r.completion_tokens} "
            f"cost_usd={r.cost_usd:.6f} latency_ms={r.latency_ms:.0f} provider={r.provider}")


def run_checks(client: LLMClient, agent_tools: bool = False) -> bool:
    ledger = UsageLedger()

    def plain():
        resp = client.call("llm_check.plain", [{"role": "user", "content": "Reply with exactly one word: pong"}],
                           ledger=ledger)
        return [f"content: {resp.content!r}", _usage(ledger)]

    def tool_call():
        resp = client.call("llm_check.tool", [
            {"role": "user", "content": "What time is it in Asia/Kolkata? Use the get_time tool."}],
            tools=[TRIVIAL_TOOL], ledger=ledger)
        out = [f"tool_calls: {[(tc.name, tc.arguments) for tc in resp.tool_calls]}", _usage(ledger)]
        if not resp.tool_calls:
            raise RuntimeError(f"model answered without calling the tool: {resp.content!r}")
        return out

    def real_tools():
        resp = client.call("llm_check.agent_tools", [
            {"role": "user", "content": "Find the RBI rule on customers who shared payment credentials. "
                                        "Use the search_corpus tool."}], tools=TOOL_SPECS, ledger=ledger)
        return [f"accepted {len(TOOL_SPECS)} tool definitions",
                f"tool_calls: {[(tc.name, tc.arguments) for tc in resp.tool_calls]}", _usage(ledger)]

    def two_turn():
        """Model calls a tool, we send back a fake result, model answers: the history replays the
        assistant tool call, which is exactly where Gemini demands the thought signature."""
        messages = [{"role": "user", "content": "What time is it in Asia/Kolkata? Use the get_time tool, "
                                                "then answer in one sentence."}]
        first = client.call("llm_check.two_turn.1", messages, tools=[TRIVIAL_TOOL], ledger=ledger)
        if not first.tool_calls:
            raise RuntimeError(f"turn 1: model answered without calling the tool: {first.content!r}")
        replay = assistant_tool_call_message(first)
        sigs = [(c.get("extra_content") or {}).get("google", {}).get("thought_signature") for c in replay["tool_calls"]]
        lines = ["turn 1 " + _usage(ledger), "thought_signature per call: " + ", ".join(
            "dummy fallback" if x == THOUGHT_SIGNATURE_FALLBACK else f"present ({len(x)} chars)" for x in sigs)]
        messages.append(replay)
        for tc in first.tool_calls:
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": '{"time": "14:05 IST"}'})
        second = client.call("llm_check.two_turn.2", messages, tools=[TRIVIAL_TOOL], ledger=ledger)
        return lines + [f"turn 2 content: {second.content!r}", "turn 2 " + _usage(ledger)]

    results = [_check("a) plain completion", plain), _check("b) trivial tool call", tool_call)]
    if agent_tools:
        results.append(_check("c) agents' tool definitions", real_tools))
    results.append(_check("d) two-turn tool loop (tool call -> result -> answer)", two_turn))
    print(f"\n{sum(results)}/{len(results)} checks passed; total cost_usd={ledger.total_cost():.6f}")
    return all(results)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent-tools", action="store_true", help="also send the agents' real tool definitions")
    ap.add_argument("--logs", action="store_true", help="print structured log lines (attempts, 4xx bodies)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.logs else logging.WARNING, format="%(message)s")

    load_dotenv_if_present()
    config = LLMConfig.from_env()
    print(f"provider={config.provider} endpoint_host={urlparse(config.base_url or '').netloc or '-'} "
          f"model={config.model or '-'} api_key={'set' if config.api_key else 'MISSING'}")
    print(f"timeout_s={config.timeout_s} max_attempts={config.max_attempts} "
          f"min_interval_s={config.min_interval_s} breaker_threshold={config.breaker_threshold}")
    if config.provider == "fake":
        print("LLM_PROVIDER=fake: nothing to check (the fake never touches the network).")
        sys.exit(0)
    try:
        client = build_client(config)
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAIL: cannot build the client: {type(exc).__name__}: {exc}")
        sys.exit(2)
    sys.exit(0 if run_checks(client, agent_tools=args.agent_tools) else 1)


if __name__ == "__main__":
    main()
