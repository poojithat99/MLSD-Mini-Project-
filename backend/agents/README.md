# Agent layer (`backend/agents/`)

The reasoning layer of the dispute triage service. It takes one validated dispute and returns one
`DisputeResult`: either a **recommendation** (route + liability position + clause-level citations, all
verified against retrieved text) or an **escalation** with a reason code. It never issues a credit or
closes a dispute. A human analyst decides.

Entry point: `run_dispute_graph(request, run_id, correlation_id) -> DisputeResult` in `graph.py`. Schemas are in
`state.py`. It never raises: every failure becomes an escalation.

## Topology: two agents, one hand-off

```
                 +-------------+     +------------------+  draft (claims, case facts,  +-----------+
 request ------> | guard_input | --> |  analysis agent  | ---- citations, outcome) ---> |  verifier |
                 +-------------+     |  tool loop over  |                              +-----------+
                   |  injection      |  4 read-only     | <-- retry: new strategy --------+   |   |
                   |  scan, delay    |  tools           |     (max 2: rewrite_query,          |   |
                   |  (code)         +------------------+      neighbour_clause, wider_k)    |   |
                   |                                                                   pass  |   | fail
                   v                                                                         v   v
              +----------+                                                          +----------+ +----------+
              | escalate | <-------- any node: budget / timeout / provider down ---- | finalize | | escalate |
              +----------+                                                          +----------+ +----------+
```

LangGraph (`graph.py`) with hand-written nodes: `guard_input -> analysis_agent -> verifier -> finalize | analysis_agent (retry) | escalate`.

- **Analysis agent** (`analysis_agent.py`, prompt `analysis_v1.txt`): classifies the dispute, chooses which
  tools to call, and drafts the outcome as JSON. The **modifier check** (brief section 5, step 2) is enforced in code:
  an answer is not accepted until the agent has made a second, different search, built from the passages it read
  and the facts of the case. Out-of-scope drafts skip this check.
- **Verifier** (`verification_agent.py`, prompt `verification_v1.txt`): a separate agent with its own prompt.
  It receives **only** the draft's claims and case facts, the cited passages, the request's own fields as
  delimited untrusted data, and code-computed facts. It never sees the analyst's messages, tool calls or
  reasoning. It returns `pass`, `retry` (with a strategy hint) or `escalate`.
- **Hand-off**: the draft (`AnalysisDraft`) goes to the verifier, and the verdict (`Verdict`) comes back to the graph. On `retry`, the
  graph re-enters the analyst with a strategy it has not used yet. On `pass`, `finalize` builds the result and runs
  the output filter.

## Tools (all read-only, `tools.py`)

| Tool | What it does |
|---|---|
| `search_corpus(query, k, expand_refs)` | Semantic search over the three corpus documents (Member 1's `Retriever.search`) |
| `get_clause(doc_id, clause_id)` | Exact lookup of a clause or table by id, e.g. `7(ii)`, `Table 2` |
| `get_request_fields()` | The dispute's own fields (free text wrapped as untrusted data) |
| `compute_reporting_delay()` | Working days from the bank's alert to the report, **computed in code**. Assumption: count from the day after the communication up to and including the report day, excluding Sundays and the 2nd/4th Saturdays |

Every tool call is charged to the run budget, runs under a 10 s timeout, records the passages it returns (the
verifier only accepts citations from these), and adds an entry to the trace.

## Loop limits

| Limit | Env var | Default |
|---|---|---|
| Model calls per run (both agents, all retries) | `AGENT_MAX_MODEL_CALLS` | 14 |
| Tool calls per run | `AGENT_MAX_TOOL_CALLS` | 16 |
| Wall-clock per run (s) | `AGENT_WALL_CLOCK_S` | 60 |
| Model turns per analysis attempt | constant `MAX_TURNS` | 6 |
| Verifier-driven retries | constant `MAX_RETRIES` | 2 |
| Per tool call timeout (s), hard (`future.result(timeout=...)`) | constant `TOOL_TIMEOUT_S` | 10 |
| Per model call timeout (s): SDK timeout, plus a hard backstop 5 s later (`future.result`) | `LLM_TIMEOUT_S` | 20 |
| Per evaluation scenario (s), watchdog in `evaluation/evaluate.py` | `EVAL_SCENARIO_TIMEOUT_S` | 300 |

The budget is shared by both agents and every retry, so a retry cannot extend it. In the latest real-model
runs, each of the three demo cases used 5 model calls.

## When an agent fails mid-run (requirement A2)

**Rule: any run that does not complete cleanly returns an escalation with a reason code, never a confident
answer.** `DisputeResult` enforces this in its own type: `status="recommended"` is impossible without a route,
citations and passed verification.

| Failure | What happens |
|---|---|
| Analyst's model call fails (timeout, 5xx, 429 after retries, breaker open) | Node wrapper catches it, escalation `provider_unavailable` |
| Analyst runs out of model calls / tool calls / turns | Escalation `budget_exhausted` |
| Wall-clock deadline passes | Escalation `timeout` |
| Analyst output is malformed | One repair turn, then escalation: `verification_failed` if the JSON does not fit the schema; `provider_unavailable` if the reply has no JSON at all (a known imprecision; see limitations) |
| Verifier's model is down | The deterministic citation check still runs, but the run is **forced** to escalate as `provider_unavailable`; claims are never released unchecked |
| Verifier rejects the draft | Retry with a new strategy (max 2), then escalation `verification_failed` |
| Any unexpected exception | Escalation (`verification_failed`: nothing was verified) |

An **unverified** escalation has empty `basis`, `liability_position`, `citations` and `competing_provisions`.
The model's draft is kept only in the trace entry `escalate.unverified_draft`, labelled **NOT VERIFIED**, for the
analyst's reference. **Verified** escalations (the section 4.1 conflict, out-of-corpus) keep their verified basis
and citations.

## Escalation reason codes

| Code | Used when |
|---|---|
| `conflicting_provisions` | Two provisions apply with different outcomes. Typically customer negligence (RBI 7(i)) against the Table 2 row for the delay; both are cited and listed in `competing_provisions` |
| `contested_fault` | The facts do not settle who was at fault (analyst's judgement) |
| `out_of_corpus` | Not an unauthorised electronic transaction covered by the corpus (e.g. a payment the customer made themselves). No route, citations or classification |
| `verification_failed` | Citations or claims still not supported after 2 retries, output filter blocked the answer, or an unexpected error |
| `budget_exhausted` | Model-call, tool-call or per-attempt turn limit reached |
| `timeout` | Wall-clock limit reached |
| `provider_unavailable` | Model provider down, circuit breaker open, or the verifier's model unavailable |
| `injection_detected` | Prompt injection found in a free-text field; escalated before any model call |

## Verification

1. **Deterministic citation check (no model, always first).** Every citation's `clause_id` must belong to a
   passage actually retrieved in this run (same `doc_id`), and its quote (at least 20 characters) must be a
   normalised substring of that passage's text. A conflict/contested escalation must cite at least two
   distinct provisions. A recommendation needs a route and at least one citation.
2. **Model check, split in two:**
   - `case_facts` (statements about this dispute) are compared with the **request's own fields**. `consistent` or
     `not_stated` passes. `contradicts` fails, and so does `not_a_case_fact` (a regulatory statement filed as a fact).
   - `claims` (statements about what the provisions say) must be **supported by the cited passages**.
   Anything the verifier did not rule on counts as unverified.

## Guardrails (`guardrails.py`)

- **Prompt injection:** every string field is scanned with regex patterns (ignore-instructions, role change, fake role
  tags, our own delimiters, tool smuggling, outcome steering). A hit in a free-text field (`narrative`,
  `merchant`) **escalates immediately as `injection_detected`, before any model call**. The note names the field and
  never echoes the text. Free text that reaches a model is wrapped in `<<<UNTRUSTED …>>>` delimiters.
- **Output filter:** only `chargeback`, `provisional_credit`, `decline` or `escalate`. Any wording claiming a credit was
  issued, money refunded or a dispute closed blocks the answer, and the run escalates instead.
- **Toxicity:** interface only. `ToxicityDetector` is wired into input and output, but the default
  `NoopToxicityDetector` always returns 0. No real classifier yet.

## LLM configuration (`llm.py`)

Every model call goes through `LLMClient.call`: circuit breaker check, then the primary provider (attempts with a timeout and
exponential backoff that honours `Retry-After`), then the optional fallback provider, otherwise `provider_unavailable`.
Tokens, cost and latency are recorded per call and per step.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` = any OpenAI-compatible endpoint; `fake` = scripted, no network |
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` | none | Provider endpoint, model id, key (only from env; never committed) |
| `LLM_FALLBACK_BASE_URL` / `_MODEL` / `_API_KEY` | empty | Optional second provider |
| `LLM_TIMEOUT_S` | 20 | Timeout per attempt |
| `LLM_MAX_ATTEMPTS` | 3 | Attempts per provider per call |
| `LLM_BACKOFF_BASE_S`, `LLM_BACKOFF_MAX_S` | 0.5, 8 | Exponential backoff with jitter (a `Retry-After` up to 60 s wins) |
| `LLM_BREAKER_THRESHOLD`, `LLM_BREAKER_RESET_S` | 5, 30 | Consecutive failures that open the breaker; seconds before a half-open trial |
| `LLM_MIN_INTERVAL_S` | 0 | Minimum gap between calls per worker process |
| `LLM_COST_PER_1K_PROMPT`, `LLM_COST_PER_1K_COMPLETION` | 0 | USD per 1,000 tokens |
| `LLM_FAKE_SCENARIO` | `default` | With `fake`: `default`, `fabricate_citation`, `fabricate_always`, `verifier_down`, `provider_down` |
| `AGENT_MAX_MODEL_CALLS`, `AGENT_MAX_TOOL_CALLS`, `AGENT_WALL_CLOCK_S` | 14, 16, 60 | Run budget (see above) |
| `AGENT_SEED_PROVISIONS` | `on` | Pre-fetch the Table 2 row + RBI 7(i) (see design decisions) |

**Check the provider** (one plain call, one tool call, one two-turn tool loop; `--agent-tools` also sends our real
tool definitions):

```bash
python -m backend.agents.llm_check --agent-tools
```

**Free-tier pacing:** set `LLM_MIN_INTERVAL_S = 60 / requests-per-minute × number of workers`. Pacing is per
process, so `--scale 3` triples the total rate. With pacing on, raise `AGENT_WALL_CLOCK_S`, since a normal run makes about 5 calls.
429 responses are retried, waiting at least as long as the provider asks (`Retry-After`, or a `retryDelay` in the body).
400 responses are not retried, and are logged with the response body.

**Provider quirk: thought signatures.** Some reasoning models attach an opaque signature to each tool call
(`tool_calls[i].extra_content.google.thought_signature`). When the conversation is sent back with that tool call in
the history, the signature must be echoed back unchanged, or the provider rejects the request with a 400 ("function
call is missing a thought_signature"). `llm.py` stores `extra_content` and the raw argument string on every
tool call and replays both verbatim (`assistant_tool_call_message`). If a call has no stored signature (e.g. a
FakeLLM history), it sends the provider's documented placeholder `skip_thought_signature_validator`.

## Design decisions and trade-offs

**(a) Seeded provisions (`AGENT_SEED_PROVISIONS`, default on).** Before the analyst starts, code fetches three
provisions by id: the RBI Table 2 row for the computed reporting delay, RBI para 7(i) (customer negligence) and
RBI para 6(i) (deficiency on the part of the bank: zero liability whenever reported). Why: the corpus says
"payment credentials" where customers say "OTP", and in real runs the model's own searches repeatedly missed 7(i);
in the baseline evaluation it also anchored on the seeded table row and missed 6(i). The agent still chooses its
own tools for everything else. The flag exists so evaluation can measure with and without it; the mode is
recorded as the first trace entry (`run.config`) of every result.

**(b) Shared credentials escalate, including clear cases.** If the analyst reports `credentials_shared: true`
(a load-bearing fact the verifier must confirm against the request), code escalates as `conflicting_provisions`
citing RBI 7(i) and the Table 2 row: under Table 2 every delay row gives a different outcome from the negligence
rule, and whether the conduct is negligence is not settled by the circular (brief section 4.1). Citing 7(i), or a
missed alert, no longer triggers it (the baseline over-escalated S07 and S13 that way). A clear case (a PIN given
to a friend) still goes to a human: the safe default.

**(c) Route consistency is checked, not imposed.** The verifier's deterministic `route_problems` check compares the
proposed route with the liability rules, in order: card dispute -> chargeback; bank at fault -> provisional_credit
citing 6(i); otherwise RBI Table 2 (up to 7 working days -> provisional_credit; beyond -> decline under SBI's policy).
A mismatch sends the analyst back with the problem stated; code never rewrites the route. When the date the
customer received the bank's communication is unknown, the delay rests only on our calculator's fallback (the
transaction date) and Table 2 cannot be applied as written, so no recommendation is accepted and the analyst must
escalate `contested_fault`: whether the bank sent the mandatory SMS alert (RBI para 5) decides who was at fault.
For card disputes, basis and liability position may only restate SBI 5(i) (the complaint follows card-network
procedures); the corpus has nothing specific about duplicate charges or non-delivery.

**Other choices.** Working days are computed in code. Out-of-corpus is not decided by retrieval score, because the
retrieval plan measured an out-of-corpus query scoring above a real hit; it is decided by the analyst's `in_scope` judgement and the scope rule. Prompts are versioned
files, and the version (currently `analysis_v1.4+verification_v1.1`) is recorded in every result.

## How to run

After the embedding model has been downloaded once (the first real retriever run fetches it), set
`HF_HUB_OFFLINE=1` so later runs load it from the local cache and make no network calls to the model hub at
startup (fewer stalls, works offline).

```bash
python -m pytest backend/tests/agents -q          # offline: FakeLLM + keyword retriever over the real chunks

# demos: cases  ok (ordinary, 1 working day) | 41 (section 4.1) | ooc (out of corpus) | injection
LLM_PROVIDER=fake python -m backend.agents.demo --case 41                       # scripted model, no network
LLM_PROVIDER=fake LLM_FAKE_SCENARIO=fabricate_citation python -m backend.agents.demo --case ok
python -m backend.agents.demo --retriever real --case ok  --trace --logs        # real model (.env) + real index
python -m backend.agents.demo --retriever real --case 41  --trace
python -m backend.agents.demo --retriever real --case ooc --trace
AGENT_SEED_PROVISIONS=off python -m backend.agents.demo --retriever real --case 41 --trace   # agent alone
```

Use `python -m pytest`, not bare `pytest`: `backend/` has no `__init__.py`. The real retriever needs the root
`requirements.txt`, and embedding settings that match `data/index/manifest.json`.

## Known limitations and what did not work

- **Derived amounts are not checked (seen in S05, baseline pass 1).** The model wrote that the bank "must
  provide provisional credit for the remaining balance (Rs 35,000)": an amount it computed, stated in no
  passage. The verifier let it through. The deterministic checks verify quotes and routes, not arithmetic in
  free text. Not fixed deliberately: an analyst sees the cited Table 1 cap and recomputes.
- **Card-dispute evidence is thin.** Only SBI 5(i) (complaints follow card-network guidelines) supports the
  chargeback route; nothing in the corpus addresses duplicate charges or non-delivery specifically.

- **Model-only modifier search did not find 7(i).** Real runs searched with "OTP" wording and never retrieved the
  negligence clause. This led to seeding (a). With `AGENT_SEED_PROVISIONS=off` the 4.1 case depends on the model's own search.
- **The verifier first rejected true facts about the case** ("customer did not share details"), because it only saw
  passages, so correct answers failed and burned the budget. Fixed by the case-fact / claim split.
- **Real-model results vary between runs.** One correct ok run lost its 6(ii) citation because the model
  stitched two sentences into one quote. The deterministic check rightly rejected it, and only the Table 2 row
  survived the retry.
- **Negligence always escalates** (b), so clear-negligence declines are not automated.
- **The conflict rule and seeding are corpus-specific:** the ids `rbi_circular_2017` `Table 2` and `7(i)` are hard-coded.
- **Injection detection is regex-based:** it can be evaded and can false-positive (a false positive escalates).
  Toxicity detection is a stub.
- **Correlated errors:** the verifier is the same kind of model as the analyst, so it can share the analyst's blind spots. The
  deterministic citation check is the only model-independent guarantee.
- **Thin chargeback support:** `chargeback` for merchant/duplicate disputes has little support in this corpus, so expect escalations.
- **Process-level limits:** pacing and the circuit breaker are per worker process, not shared across `--scale`d workers.
  A timed-out tool call's thread cannot be killed; the agent just stops waiting for it.
- **Rough fake token counts:** FakeLLM estimates tokens as characters / 4. Costs from real providers are only as accurate as the
  `LLM_COST_PER_1K_*` you set.
- **Tests don't measure retrieval quality:** tests use a keyword retriever over the real chunks, so they test agent logic, not the embedding model.
- **Unexpected crashes report `verification_failed`:** no dedicated reason code fits an internal error.
- **Reply with no JSON reports `provider_unavailable`:** a model reply with no JSON at all, even after the repair turn, is reported this way, although the provider is up.
