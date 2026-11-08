# CLAUDE.md — Member 2 (agents) working notes

Project brief: `MiniProject_02_Card_Dispute_Triage.pdf` (sections 4, 4.1, 5, 6 A1–A5, 7 L2–L5, 9).
Team plan: `Card Dispute Triage Service Project Plan.pdf`.

## My scope (Member 2) — only touch these

- `backend/agents/*`, `backend/prompts/*`, `evaluation/*`, agent tests (`backend/tests/agents/`)
- May append `LLM_*` variables to `.env.example` and add dependencies to `backend/requirements.txt`
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

## Rules I follow

- Every model call goes through `backend/agents/llm.py`. Base URL, model and key come only from env
  (`LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`). No model names, endpoints or keys in the repo (D4).
- Every model and tool call has a timeout. LLM calls: bounded retry with exponential backoff, circuit breaker,
  defined fallback. Token counts, cost and latency recorded per call, attributed to a step name.
- Agent loops bounded by steps, tool calls and wall-clock (`Budget`).
- **Any run that does not complete cleanly returns an escalation with a reason code, never a confident answer.**
  Reason codes: `contested_fault, conflicting_provisions, out_of_corpus, verification_failed,
  budget_exhausted, timeout, provider_unavailable`.
- Verifier sees only claims + retrieved passages, never the analyst's reasoning. Deterministic citation check
  first; if the verifier's LLM is down, run only that check and force escalation.
- The system never issues a credit or closes a dispute. Routes allowed: chargeback, provisional_credit,
  decline, escalate.
- Untrusted free text is injection-scanned and wrapped in delimiters before it reaches a model.
- Working days are computed in code, not by the model.
- Prompts are versioned files in `backend/prompts/`; the version is recorded in every `DisputeResult`.
- Tests use `FakeLLM` and a fake retriever only — no real API calls.
- Work on branch `member2/agents`. Never push to main. Keep code simple: I must explain every line in a viva.
