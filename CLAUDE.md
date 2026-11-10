# CLAUDE.md — Member 2 (agents) working notes

Project brief: `MiniProject_02_Card_Dispute_Triage.pdf` (sections 4, 4.1, 5, 6 A1–A5, 7 L2–L5, 9).
Team plan: `Card Dispute Triage Service Project Plan.pdf`.

## My scope (Member 2) — only touch these

- `backend/agents/*`, `backend/prompts/*`, `evaluation/*`, agent tests (`backend/tests/agents/`)
- May edit my sections of `.env.example` (`LLM_*`, `AGENT_*`, and the `EMBEDDING_*` block on request) and add
  dependencies to `backend/requirements.txt`. In the root `README.md` only the short `## Agent layer` section
  (details live in `backend/agents/README.md`, to avoid merge conflicts).
- Do NOT edit: `backend/retrieval/*`, `corpus/`, `data/` (Member 1); `backend/app/*`, `backend/worker.py`,
  `frontend/`, `docker-compose.yml`, `monitoring/` (Member 3).
- If something I need does not exist, build against a fake and report what is missing. Never edit another
  member's files to make mine work.

## Retrieval interface (Member 1, as actually implemented)

`backend/retrieval/search.py`:

```python
r = Retriever()                       # loads data/index/*; needs EMBEDDING_MODEL (+ faiss, sentence-transformers)
r.search(query, k=None, *, doc_ids=None, chunk_types=None, expand_refs=False, min_score=None) -> list[Passage]
r.get_clause(doc_id, clause_id, part=None) -> list[Passage]   # "7(ii)", "Table 2", "4.6.2.b (ii)" all normalise
r.get_chunk(chunk_id) -> Passage | None
r.quote_resolves(chunk_id, quote) -> bool                       # normalised substring of text + lead_in
r.by_id[chunk_id]["clause_ids"]                                 # raw chunk dict: every clause id inside the chunk
```

`normalize_for_match(s)` (in `ingest.py`): NFKC, lower-case, curly quotes folded, whitespace collapsed.

`Passage` fields: `rank, score, chunk_id, doc_id, doc_title, short_title, part, chunk_type ("clause"|"table"|"text"),
clause, supporting_clause, page, page_start, page_end, section_heading, heading_path, text, lead_in,
supporting_text, citation, via, refs, referenced_by`.

- **doc_ids:** `rbi_circular_2017`, `sbi_policy_2021`, `rbi_ios_2026`.
- **chunk_id format:** `<doc_id>:<clause>` e.g. `rbi_circular_2017:7(i)`, `rbi_circular_2017:table-2`,
  `sbi_policy_2021:compensation-policy:4.6.3`; split chunks get `#n`, repeats `~n`.
- **clause_id format:** `7(i)`, `7(ii)`, `4.6.2(b)(i)`, `3(1)(a)`; tables are `Table 1` / `Table 2`.
  One chunk can hold several clauses (`rbi_circular_2017:7(i)` holds `7(i)` and `7(ii)`).
  SBI numbers its three sub-policies from 1, so a bare clause id can be ambiguous there (`part` disambiguates).
- **Score scale:** cosine similarity in [−1, 1]; real hits ~0.4–0.6. Uncalibrated (`SEARCH_MIN_SCORE=0`).
  An out-of-corpus query scored 0.535, above a real hit (0.427): **never escalate on score alone.**
- The corpus never says "OTP"; it says "payment credentials". Query rewriting matters.

Key provisions for the section 4.1 case:
- RBI `7(i)` (negligence, shared payment credentials → customer bears entire loss until reported)
- RBI `Table 1` note / `Table 2` (beyond 7 working days → bank's Board-approved policy); `Table 2` note:
  working days counted per home-branch schedule, excluding the date of receiving the communication.
- SBI `4.6.2(b)(i)` (same negligence rule) and SBI `Table 2` (beyond 7 working days → full liability).

## How to run (agents)

```bash
python3 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt
.venv/bin/python -m pytest backend/tests/agents -q        # use "python -m pytest": backend/ has no __init__.py
LLM_PROVIDER=fake .venv/bin/python -m backend.agents.demo --case 41          # 4.1 -> escalation citing both
LLM_PROVIDER=fake LLM_FAKE_SCENARIO=fabricate_citation .venv/bin/python -m backend.agents.demo --case ok
LLM_BASE_URL=http://127.0.0.1:9 LLM_MODEL=x LLM_API_KEY=x LLM_BREAKER_THRESHOLD=3 \
  .venv/bin/python -m backend.agents.demo --case ok --repeat 3 --logs       # breaker opens

# real model (LLM_* from local .env) + real index (pip install -r requirements.txt; EMBEDDING_* must match manifest)
.venv/bin/python -m backend.agents.llm_check --agent-tools                  # provider check incl. two-turn tool loop
.venv/bin/python -m backend.agents.demo --retriever real --case ok  --trace --logs
.venv/bin/python -m backend.agents.demo --retriever real --case 41  --trace
.venv/bin/python -m backend.agents.demo --retriever real --case ooc --trace
AGENT_SEED_PROVISIONS=off .venv/bin/python -m backend.agents.demo --retriever real --case 41 --trace
```

Real calls use free-tier quota (~500/day): at most one real run per case per iteration; diagnose from saved
`--trace --logs` output, reproduce with FakeLLM first.

Agent env vars (all in `.env.example`; full table in `backend/agents/README.md`):
`LLM_PROVIDER, LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, LLM_FALLBACK_*, LLM_TIMEOUT_S, LLM_MAX_ATTEMPTS,
LLM_BACKOFF_BASE_S, LLM_BACKOFF_MAX_S, LLM_BREAKER_THRESHOLD, LLM_BREAKER_RESET_S, LLM_MIN_INTERVAL_S,
LLM_COST_PER_1K_PROMPT, LLM_COST_PER_1K_COMPLETION, LLM_FAKE_SCENARIO, AGENT_MAX_MODEL_CALLS (14),
AGENT_MAX_TOOL_CALLS (16), AGENT_WALL_CLOCK_S (60), AGENT_SEED_PROVISIONS (on)`. `.env` is read only by the CLI
entry points (`llm_check`, `demo`), never by library code or tests.

Agent module map: `llm.py` (only model interface) · `state.py` (types, Budget, reason codes) · `tools.py` ·
`analysis_agent.py` · `verification_agent.py` · `guardrails.py` · `graph.py` (entry: `run_dispute_graph`) ·
`prompts.py` + `backend/prompts/*.txt` · `observability.py` · `llm_check.py` (provider check) ·
test/demo only: `fake_retriever.py`, `fake_scenarios.py`, `demo.py`. Docs: `backend/agents/README.md`.

## Rules I follow

- Every model call goes through `backend/agents/llm.py`. Base URL, model and key come only from env
  (`LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`).
- **D4: no real keys, endpoints or model identifiers in any tracked file** (code, docs, `.env.example`, commit
  messages). Use placeholders like `<your-model-id>` / `<your-provider-base-url>`; tests use obviously fake
  values (`http://llm.test/v1`, `k`, `m`). `.env` stays untracked. Check before committing:
  `git grep -nEi "gemini[-]|google[a]pis|AI[z]a|AQ[.]" -- . ':!*.lock'` (same regex, written so this line
  does not match itself; add `--untracked` before committing new files).
- Provider quirk: tool calls may carry `extra_content.google.thought_signature`; it must be replayed unchanged on
  every tool call in later requests (`assistant_tool_call_message`), else 400 "missing a thought_signature".
  Missing signature -> documented dummy `skip_thought_signature_validator`. Never rebuild assistant tool-call
  messages by hand.
- Every model and tool call has a HARD timeout (`future.result(timeout=...)`: tools 10 s; model calls the SDK timeout
  + 5 s backstop); `evaluation/evaluate.py` has a per-scenario watchdog (`EVAL_SCENARIO_TIMEOUT_S`, 300 s) that records
  `timeout` and moves on. Resuming a results file from different prompt or code versions is refused (code fingerprint).
  Set `HF_HUB_OFFLINE=1` once the embedding model is cached.
- Every model and tool call has a timeout. LLM calls: bounded retry with exponential backoff, circuit breaker,
  defined fallback. Token counts, cost and latency recorded per call, attributed to a step name.
- Agent loops bounded by steps, tool calls and wall-clock (`Budget`).
- **Any run that does not complete cleanly returns an escalation with a reason code, never a confident answer.**
  Reason codes (8): `contested_fault, conflicting_provisions, out_of_corpus, verification_failed,
  budget_exhausted, timeout, provider_unavailable, injection_detected`.
- Verifier never sees the analyst's reasoning. It gets: regulatory `claims` + cited passages (must be supported),
  `case_facts` + the request's own fields as delimited untrusted data (a contradiction is a failure; a regulatory
  statement filed as a case fact is a failure). Deterministic citation check first and unchanged; if the
  verifier's LLM is down, run only that check and force escalation.
- `AGENT_SEED_PROVISIONS` (default on): code fetches the RBI Table 2 row for the delay, RBI 7(i) and RBI 6(i) by id
  before the analyst starts, and the conflict rule applies: ONLY `credentials_shared=true` (a load-bearing fact the
  verifier must confirm against the request) -> `conflicting_provisions` citing RBI 7(i) + the Table 2 row.
  Citing 7(i), or a missed alert, does not trigger it. Off: no pre-fetch, no conflict rule; the agent finds
  provisions itself. The mode is recorded as the first trace entry `run.config` of every result.
- Load-bearing facts (`CREDENTIALS_FACT`, `BANK_FAULT_FACT` in `state.py`) are added to case_facts when the model
  asserts them and must be "consistent" with the request ("not_stated" fails).
- Deterministic route check in the verifier (`route_problems`, both modes): duplicate/merchant_dispute ->
  chargeback; bank_fault -> provisional_credit citing RBI 6(i) whatever the delay; alert (communication) date
  unknown -> no recommendation, escalate contested_fault (RBI para 5 + Table 2); otherwise a third-party breach
  -> provisional_credit up to 7 working days, decline beyond (RBI Table 2 + SBI Table 2). A mismatch is a retry,
  never a silent override.
- Scope rule (both modes): `in_scope=false` -> `out_of_corpus`, no route/citations/classification; such drafts
  skip the modifier check (saves a model call).
- An UNVERIFIED escalation (verification_failed, budget, timeout, provider, injection) has empty `basis`,
  `liability_position`, `citations`, `competing_provisions`; the model's draft is kept only in the trace entry
  `escalate.unverified_draft`, labelled NOT VERIFIED. Verified escalations (4.1, out_of_corpus) keep them.
- The system never issues a credit or closes a dispute. Routes allowed: chargeback, provisional_credit,
  decline, escalate.
- Untrusted free text is injection-scanned and wrapped in delimiters before it reaches a model.
  An injection in a free-text field (`narrative`, `merchant`) escalates immediately with `injection_detected`,
  before any model call; the note names the field and never echoes the injected text.
- Working days are computed in code, not by the model.
- Prompts are versioned files in `backend/prompts/`; the version comes from the file's `version:` header and is
  recorded in every `DisputeResult`. Current: `analysis_v1.4` (file `analysis_v1.txt`), `verification_v1.1`
  (file `verification_v1.txt`) and `analysis_v2.2` (file `analysis_v2.txt` = v1.4 + quote-first, for L2).
  The pre-fix baseline (v1.2) is kept in `evaluation/results/archive/`; v1.2 and v1.4 results are not comparable.
  Bump the header whenever a
  prompt's meaning changes.
- Tests use `FakeLLM` and a fake retriever only — no real API calls.
- Work on branch `member2/agents`. Never push to main. Keep code simple: I must explain every line in a viva.
