"""Scripted model behaviour for FakeLLM: tests and demos with no network.

``ScriptedAgents`` plays both agents.  It is a small rule-based stand-in for the
model, NOT a model: it reacts to the conversation the way a well-behaved model
should (search, search again after the modifier check, answer in JSON), and
it quotes only text that the tools actually returned, so its citations resolve
against the real corpus.

Scenarios (LLM_PROVIDER=fake, LLM_FAKE_SCENARIO=...):
  default              normal behaviour
  fabricate_citation   first attempt quotes text that is not in the corpus;
                       the verifier catches it and the retry quotes correctly
  fabricate_always     every attempt fabricates -> escalation "verification_failed"
  provider_down        every call times out -> breaker opens -> "provider_unavailable"
  verifier_down        analyst works, verifier's model times out -> forced escalation
"""
from __future__ import annotations

import itertools
import json
import re

from .analysis_agent import MODIFIER_CHECK, RETRY_MARKER
from .llm import LLMResponse, ToolCall

FABRICATED_QUOTE = "The bank shall refund the full amount to the customer in every case of unauthorised transaction"

OUT_OF_CORPUS = ("crypto", "bitcoin", "insurance", "mutual fund", "stock market", "lottery", "home loan")
SHARED_CREDENTIALS = re.compile(
    r"\b(shared|gave|told|disclosed|read out)\b.{0,60}\b(otp|pin|cvv|password|credentials?)\b"
    r"|\b(otp|pin|cvv)\b.{0,40}\b(shared|given|told)\b", re.IGNORECASE | re.DOTALL)

# kind -> (first query, modifier query), worded like the corpus
QUERIES = {
    "out_of_corpus": ("investment loss on a cryptocurrency exchange",
                      "unauthorised electronic banking transaction customer liability"),
    "conflict": ("negligence by a customer shared the payment credentials entire loss",
                 "delay in reporting beyond seven working days Board approved policy"),
    "negligence": ("negligence by a customer shared the payment credentials entire loss",
                   "third party breach customer notifies the bank within three working days"),
    "zero": ("third party breach customer notifies the bank within three working days zero liability",
             "negligence by a customer shared the payment credentials entire loss"),
    "limited": ("delay of four to seven working days customer liability Table 1",
                "negligence by a customer shared the payment credentials entire loss"),
    "late": ("beyond seven working days customer liability shall be full",
             "negligence by a customer shared the payment credentials entire loss"),
}


def _sentence_with(text: str, keyword: str) -> str:
    """The line or sentence of ``text`` containing ``keyword`` (verbatim)."""
    for line in text.split("\n"):
        if keyword.lower() in line.lower():
            for sentence in re.split(r"(?<=\.)\s+", line):
                if keyword.lower() in sentence.lower():
                    return sentence.strip().rstrip(".")
    return ""


class ScriptedAgents:
    def __init__(self, fabricate: str = "never", verifier_down: bool = False):
        self.fabricate = fabricate          # "never" | "first_attempt" | "always"
        self.verifier_down = verifier_down
        self._ids = itertools.count(1)

    def __call__(self, messages: list[dict], tools: list[dict] | None):
        if "verification agent" in messages[0]["content"]:
            return TimeoutError("verifier model timed out") if self.verifier_down else self._verify(messages)
        return self._analyse(messages)

    # ------------------------------------------------------------------ verifier
    def _verify(self, messages: list[dict]) -> LLMResponse:
        payload = json.loads(messages[1]["content"])
        claims = [{"index": c["index"], "supported": True, "reason": "stated in the cited passage"}
                  for c in payload["claims"]]
        return LLMResponse(content=json.dumps({"claims": claims, "missed_provision": None}))

    # ------------------------------------------------------------------ analyst
    def _analyse(self, messages: list[dict]) -> LLMResponse:
        case = messages[1]["content"]
        said = "\n".join(m["content"] or "" for m in messages if m["role"] == "user")
        delay = int(re.search(r'"working_days": (\d+)', case).group(1))
        kind = self._kind(case, delay)
        searches = [tc for m in messages if m["role"] == "assistant"
                    for tc in m.get("tool_calls", []) if tc["function"]["name"] == "search_corpus"]
        first, modifier = QUERIES[kind]
        if not searches:
            return self._tool("search_corpus", {"query": first, "k": 5, "expand_refs": True})
        if MODIFIER_CHECK[:15] in said and len(searches) < 2:
            return self._tool("search_corpus", {"query": modifier, "k": 5, "expand_refs": True})
        draft = self._draft(kind, delay, case, self._passages(messages, said))
        retry = RETRY_MARKER in said
        if draft["citations"] and (self.fabricate == "always" or (self.fabricate == "first_attempt" and not retry)):
            draft["citations"][0]["quote"] = FABRICATED_QUOTE
        return LLMResponse(content=json.dumps(draft))

    def _tool(self, name: str, args: dict) -> LLMResponse:
        return LLMResponse(tool_calls=[ToolCall(id=f"call_{next(self._ids)}", name=name, arguments=args)])

    @staticmethod
    def _kind(case: str, delay: int) -> str:
        low = case.lower()
        if any(k in low for k in OUT_OF_CORPUS):
            return "out_of_corpus"
        shared = bool(SHARED_CREDENTIALS.search(case))
        if shared:
            return "conflict" if delay > 7 else "negligence"
        return "zero" if delay <= 3 else "limited" if delay <= 7 else "late"

    @staticmethod
    def _passages(messages: list[dict], said: str) -> list[dict]:
        """Every passage the tools returned in this conversation (tool results + retry prefetch)."""
        out: list[dict] = []
        chunks = [m["content"] for m in messages if m["role"] == "tool"]
        chunks += re.findall(r"^\[.*\]$", said, flags=re.MULTILINE)       # prefetched lists in a retry message
        for content in chunks:
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(data, list):
                out.extend(data)
        return out

    @staticmethod
    def _cite(passages: list[dict], doc_id: str, clause_id: str, keyword: str) -> dict | None:
        for p in passages:
            if p["doc_id"] == doc_id and clause_id in p["clause_ids"]:
                quote = _sentence_with(p["text"], keyword)
                if quote:
                    return {"doc_id": doc_id, "clause_id": clause_id, "quote": quote}
        return None

    def _draft(self, kind: str, delay: int, case: str, passages: list[dict]) -> dict:
        low = case.lower()
        classification = ("upi_social_eng" if "upi" in low else
                          "skimming" if re.search(r"card_present|\batm\b|\bpos\b", low) else "cnp_fraud")
        d = {"classification": classification, "route": None, "basis": "", "liability_position": "",
             "claims": [f"The reporting delay is {delay} working days."], "citations": [],
             "escalate": False, "reason_code": None, "competing_provisions": [], "note": ""}
        cite = lambda *a: self._cite(passages, *a)   # noqa: E731
        if kind == "out_of_corpus":
            d.update(classification=None, escalate=True, reason_code="out_of_corpus",
                     note="No provision in the corpus covers this kind of loss.")
        elif kind == "conflict":
            a = cite("rbi_circular_2017", "7(i)", "entire loss")
            b = cite("rbi_circular_2017", "Table 2", "Beyond 7 working days")
            d.update(escalate=True, reason_code="conflicting_provisions",
                     citations=[c for c in (a, b) if c],
                     competing_provisions=["rbi_circular_2017 7(i)", "rbi_circular_2017 Table 2"],
                     basis="Customer negligence (para 7(i)) and a third-party breach reported beyond seven "
                           "working days (Table 2) both apply.",
                     note="Whether sharing an OTP with a caller impersonating the bank is customer negligence "
                          "is not settled by the circular; an analyst must decide which provision applies.",
                     claims=d["claims"] + [
                         "Where the loss is due to customer negligence such as sharing payment credentials, "
                         "the customer bears the entire loss until reporting.",
                         "For a third-party breach reported beyond 7 working days, liability is as per the "
                         "bank's Board approved policy."])
        elif kind == "negligence":
            a = cite("rbi_circular_2017", "7(i)", "entire loss")
            d.update(route="decline", citations=[a] if a else [],
                     basis="Para 7(i): loss due to customer negligence (shared payment credentials).",
                     liability_position="Customer bears the entire loss until the transaction was reported.")
        elif kind == "zero":
            a = cite("rbi_circular_2017", "6(ii)", "within three working days")
            d.update(route="provisional_credit", citations=[a] if a else [],
                     basis="Para 6(ii): third-party breach notified within three working days.",
                     liability_position="Zero liability for the customer.")
        elif kind == "limited":
            a = cite("rbi_circular_2017", "7(ii)", "four to seven working days")
            d.update(route="provisional_credit", citations=[a] if a else [],
                     basis="Para 7(ii): third-party breach notified in four to seven working days.",
                     liability_position="Limited to the transaction value or the Table 1 amount, whichever is lower.")
        else:  # late
            a = cite("sbi_policy_2021", "Table 2", "customer liability shall be full")
            d.update(route="decline", citations=[a] if a else [],
                     basis="SBI policy Table 2: reported beyond seven working days.",
                     liability_position="Full liability for the customer under the bank's Board-approved policy.")
        return d


def scenario_script(name: str):
    """The FakeLLM script for ``LLM_FAKE_SCENARIO``."""
    if name == "provider_down":
        return lambda messages, tools: TimeoutError("provider unreachable")
    scenarios = {
        "default": ScriptedAgents(),
        "fabricate_citation": ScriptedAgents(fabricate="first_attempt"),
        "fabricate_always": ScriptedAgents(fabricate="always"),
        "verifier_down": ScriptedAgents(verifier_down=True),
    }
    if name not in scenarios:
        raise ValueError(f"unknown LLM_FAKE_SCENARIO {name!r}; choose from {sorted(scenarios) + ['provider_down']}")
    return scenarios[name]
