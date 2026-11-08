"""The single interface to the language model (requirements A5, L5, D4).

Every model call in the agents goes through ``LLMClient.call``.  One call does:

    circuit breaker check
      -> primary provider, up to LLM_MAX_ATTEMPTS attempts,
         each with a timeout, exponential backoff between attempts
      -> if the primary fails (or its breaker is open): fallback provider, if configured
      -> otherwise raise ProviderUnavailable
         (the graph turns that into an escalation with reason "provider_unavailable")

and records prompt tokens, completion tokens, cost and latency for the call,
attributed to a step name ("analysis", "verification", ...).

Configuration comes ONLY from environment variables (D4): no model name,
endpoint or key is written anywhere in the repository.  See .env.example.

``FakeLLM`` is a provider that replays scripted responses.  Tests and demos use
it, so they never make a real API call.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .observability import log_event, span


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #
class LLMError(RuntimeError):
    """A model call failed and will not be retried."""


class ProviderUnavailable(LLMError):
    """No provider could answer: retries exhausted, breaker open, no fallback."""


class CircuitOpen(ProviderUnavailable):
    """The breaker is open, so the provider was not even called."""


# --------------------------------------------------------------------------- #
# data types
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class UsageRecord:
    """Cost of one model call, recorded at the time of the call (L5)."""
    step: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float


class UsageLedger:
    """All model calls of one run.  Gives per-step and per-run totals."""

    def __init__(self) -> None:
        self.records: list[UsageRecord] = []

    def add(self, record: UsageRecord) -> None:
        self.records.append(record)

    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.records)

    def by_step(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for r in self.records:
            s = out.setdefault(r.step, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})
            s["calls"] += 1
            s["prompt_tokens"] += r.prompt_tokens
            s["completion_tokens"] += r.completion_tokens
            s["cost_usd"] += r.cost_usd
        return out


# --------------------------------------------------------------------------- #
# configuration (environment only)
# --------------------------------------------------------------------------- #
def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


@dataclass(frozen=True)
class LLMConfig:
    provider: str                 # "openai" (any OpenAI-compatible endpoint) or "fake"
    base_url: str | None
    model: str | None
    api_key: str | None
    fallback_base_url: str | None
    fallback_model: str | None
    fallback_api_key: str | None
    timeout_s: float              # per attempt
    max_attempts: int             # per provider, per call
    backoff_base_s: float
    backoff_max_s: float
    breaker_threshold: int        # consecutive failures before the breaker opens
    breaker_reset_s: float        # how long it stays open before a trial call
    cost_per_1k_prompt: float     # USD
    cost_per_1k_completion: float
    fake_scenario: str

    @classmethod
    def from_env(cls) -> "LLMConfig":
        get = lambda n: (os.environ.get(n) or "").strip() or None  # noqa: E731
        return cls(
            provider=(get("LLM_PROVIDER") or "openai").lower(),
            base_url=get("LLM_BASE_URL"),
            model=get("LLM_MODEL"),
            api_key=get("LLM_API_KEY"),
            fallback_base_url=get("LLM_FALLBACK_BASE_URL"),
            fallback_model=get("LLM_FALLBACK_MODEL"),
            fallback_api_key=get("LLM_FALLBACK_API_KEY"),
            timeout_s=_env_float("LLM_TIMEOUT_S", 20.0),
            max_attempts=int(_env_float("LLM_MAX_ATTEMPTS", 3)),
            backoff_base_s=_env_float("LLM_BACKOFF_BASE_S", 0.5),
            backoff_max_s=_env_float("LLM_BACKOFF_MAX_S", 8.0),
            breaker_threshold=int(_env_float("LLM_BREAKER_THRESHOLD", 5)),
            breaker_reset_s=_env_float("LLM_BREAKER_RESET_S", 30.0),
            cost_per_1k_prompt=_env_float("LLM_COST_PER_1K_PROMPT", 0.0),
            cost_per_1k_completion=_env_float("LLM_COST_PER_1K_COMPLETION", 0.0),
            fake_scenario=(get("LLM_FAKE_SCENARIO") or "default").lower(),
        )

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.cost_per_1k_prompt + completion_tokens * self.cost_per_1k_completion) / 1000


# --------------------------------------------------------------------------- #
# circuit breaker
# --------------------------------------------------------------------------- #
class CircuitBreaker:
    """closed --(N consecutive failures)--> open --(reset time)--> half-open
    half-open --(trial succeeds)--> closed;  half-open --(trial fails)--> open

    While open, calls fail immediately without touching the provider, so an
    outage costs nothing and the provider gets time to recover.
    """

    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, threshold: int, reset_s: float, clock: Callable[[], float] = time.monotonic):
        self.threshold = threshold
        self.reset_s = reset_s
        self.clock = clock
        self.state = self.CLOSED
        self.failures = 0
        self.opened_at = 0.0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self.state == self.OPEN and self.clock() - self.opened_at >= self.reset_s:
                self.state = self.HALF_OPEN          # let one trial call through
                return True
            return self.state == self.CLOSED

    def record_success(self) -> None:
        with self._lock:
            self.state, self.failures = self.CLOSED, 0

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.state == self.HALF_OPEN or self.failures >= self.threshold:
                if self.state != self.OPEN:
                    log_event("circuit_breaker_open", failures=self.failures)
                self.state, self.opened_at = self.OPEN, self.clock()


# --------------------------------------------------------------------------- #
# providers: one raw attempt each, no retry logic
# --------------------------------------------------------------------------- #
class Provider(Protocol):
    name: str

    def complete(self, messages: list[dict], tools: list[dict] | None, timeout_s: float) -> LLMResponse: ...


class TransientError(RuntimeError):
    """A failure worth retrying (timeout, connection, 429, 5xx)."""


class OpenAICompatProvider:
    """Any endpoint that speaks the OpenAI chat-completions API."""

    def __init__(self, name: str, base_url: str | None, model: str | None, api_key: str | None):
        missing = [n for n, v in (("base URL", base_url), ("model", model), ("API key", api_key)) if not v]
        if missing:
            raise LLMError(f"{name} LLM provider is missing {', '.join(missing)}; set the LLM_* variables (.env.example)")
        import openai

        self.name = name
        self.model = model
        self._openai = openai
        # max_retries=0: retries are ours (bounded, with backoff, visible to the breaker)
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=0)

    def complete(self, messages: list[dict], tools: list[dict] | None, timeout_s: float) -> LLMResponse:
        oa = self._openai
        kwargs = {"model": self.model, "messages": messages, "timeout": timeout_s, "temperature": 0}
        if tools:
            kwargs["tools"] = tools
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except (oa.APITimeoutError, oa.APIConnectionError, oa.RateLimitError, oa.InternalServerError) as exc:
            raise TransientError(f"{type(exc).__name__}: {exc}") from exc
        except oa.OpenAIError as exc:                       # bad request, auth, ...: retrying will not help
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        msg = resp.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable": tc.function.arguments}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        usage = resp.usage
        return LLMResponse(
            content=msg.content, tool_calls=calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


Script = list | Callable[[list[dict], list[dict] | None], "LLMResponse | Exception"]


class FakeLLM:
    """Replays scripted responses; never touches the network.

    ``script`` is either a list (one entry per call, in order) or a function
    ``(messages, tools) -> response``.  An entry may be an ``LLMResponse``, a
    plain string (returned as content), or an exception instance (raised, to
    simulate a provider failure; ``TimeoutError``/``ConnectionError`` count as
    transient and are retried).
    """

    def __init__(self, script: Script, name: str = "fake"):
        self.name = name
        self.script = script
        self.calls: list[list[dict]] = []          # every message list we were sent

    def complete(self, messages: list[dict], tools: list[dict] | None, timeout_s: float) -> LLMResponse:
        self.calls.append(messages)
        if callable(self.script):
            item = self.script(messages, tools)
        else:
            if not self.script:
                raise LLMError("FakeLLM script exhausted")
            item = self.script.pop(0)
        if isinstance(item, (TimeoutError, ConnectionError)):
            raise TransientError(str(item) or type(item).__name__)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            item = LLMResponse(content=item)
        if not item.prompt_tokens:                   # rough estimate so cost tracking is exercised
            item.prompt_tokens = sum(len(str(m.get("content") or "")) for m in messages) // 4
            item.completion_tokens = len(item.content or "") // 4 + 10 * len(item.tool_calls)
        return item


# --------------------------------------------------------------------------- #
# the client the agents use
# --------------------------------------------------------------------------- #
class LLMClient:
    def __init__(
        self, primary: Provider, config: LLMConfig, fallback: Provider | None = None,
        breaker: CircuitBreaker | None = None, sleep: Callable[[float], None] = time.sleep,
    ):
        self.primary = primary
        self.fallback = fallback
        self.config = config
        self.breaker = breaker or CircuitBreaker(config.breaker_threshold, config.breaker_reset_s)
        self.sleep = sleep

    def call(
        self, step: str, messages: list[dict], tools: list[dict] | None = None,
        ledger: UsageLedger | None = None, timeout_s: float | None = None,
    ) -> LLMResponse:
        """One logical model call for ``step``.  Raises ProviderUnavailable if nobody answered."""
        timeout = min(self.config.timeout_s, timeout_s) if timeout_s else self.config.timeout_s
        with span("llm.call", step=step):
            if self.breaker.allow():
                try:
                    return self._attempts(self.primary, step, messages, tools, ledger, timeout, self.breaker)
                except ProviderUnavailable as exc:
                    reason = str(exc)
            else:
                reason = "circuit breaker open"
            log_event("llm_primary_unavailable", step=step, reason=reason, breaker=self.breaker.state)
            if self.fallback is not None:                   # defined fallback path
                try:
                    return self._attempts(self.fallback, step, messages, tools, ledger, timeout, None)
                except ProviderUnavailable as exc:
                    reason = f"{reason}; fallback: {exc}"
            if self.breaker.state == CircuitBreaker.OPEN:
                raise CircuitOpen(reason)
            raise ProviderUnavailable(reason)

    def _attempts(self, provider, step, messages, tools, ledger, timeout, breaker) -> LLMResponse:
        last = ""
        for attempt in range(self.config.max_attempts):
            if attempt:
                delay = min(self.config.backoff_max_s, self.config.backoff_base_s * 2 ** (attempt - 1))
                self.sleep(delay + random.uniform(0, self.config.backoff_base_s))   # jitter
            if breaker is not None and attempt and not breaker.allow():
                break                                     # opened while we were retrying
            start = time.perf_counter()
            try:
                resp = provider.complete(messages, tools, timeout)
            except TransientError as exc:
                last = str(exc)
                if breaker is not None:
                    breaker.record_failure()
                log_event("llm_attempt_failed", step=step, provider=provider.name, attempt=attempt + 1, error=last)
                continue
            latency_ms = (time.perf_counter() - start) * 1000
            if breaker is not None:
                breaker.record_success()
            record = UsageRecord(
                step=step, provider=provider.name, prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
                cost_usd=self.config.cost(resp.prompt_tokens, resp.completion_tokens), latency_ms=latency_ms,
            )
            if ledger is not None:
                ledger.add(record)
            log_event("llm_call", **record.__dict__)
            return resp
        raise ProviderUnavailable(f"{provider.name}: {last or 'not attempted'}")


# --------------------------------------------------------------------------- #
# process-wide client
# --------------------------------------------------------------------------- #
_client: LLMClient | None = None
_client_lock = threading.Lock()


def build_client(config: LLMConfig | None = None) -> LLMClient:
    """Build a client from the environment (or ``config``)."""
    config = config or LLMConfig.from_env()
    if config.provider == "fake":
        from .fake_scenarios import scenario_script      # demo scripts (no network)

        return LLMClient(FakeLLM(scenario_script(config.fake_scenario)), config)
    primary = OpenAICompatProvider("primary", config.base_url, config.model, config.api_key)
    fallback = None
    if config.fallback_base_url:
        fallback = OpenAICompatProvider("fallback", config.fallback_base_url, config.fallback_model,
                                        config.fallback_api_key)
    return LLMClient(primary, config, fallback=fallback)


def get_llm() -> LLMClient:
    """The shared client: one circuit breaker per worker process."""
    global _client
    with _client_lock:
        if _client is None:
            _client = build_client()
        return _client


def reset_llm() -> None:
    """Forget the shared client (tests, or after changing LLM_* variables)."""
    global _client
    with _client_lock:
        _client = None
