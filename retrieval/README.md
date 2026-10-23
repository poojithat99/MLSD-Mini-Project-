# Retrieval layer (steps 2–5)

```
data/corpus/*.pdf + corpus.json
        │  ingest.py     PyMuPDF, page by page → PageRecord(doc_id, page, section_heading, text, lines)
        ▼
   chunker.py            clause tree → chunks (text, clause ids, page spans, lead-in, cross-refs)
        ▼
   build_index.py        embed (incrementally) → FAISS + chunks.jsonl + vectors.npy + manifest.json
        ▼
   search.py             Retriever.search / get_clause / get_chunk → Passage(citation, supporting_text, …)
```

## Run

```bash
pip install -r requirements.txt
cp .env.example .env                       # set EMBEDDING_MODEL (not hardcoded anywhere: D4)
python -m backend.retrieval.ingest --doc rbi_circular_2017 --page 4 --lines   # inspect extraction
python -m backend.retrieval.build_index --dry-run                              # inspect chunks
python -m backend.retrieval.build_index                                        # build (no-op if unchanged)
python -m backend.retrieval.search "customer shared an OTP" -k 5 --expand
python -m backend.retrieval.search --clause rbi_circular_2017 "7(ii)"
EMBEDDING_BACKEND=hashing pytest backend/tests/test_retrieval.py -q            # offline, ~30 s (PowerShell: $env:EMBEDDING_BACKEND='hashing')
```

## Quick start (Windows PowerShell or Linux/macOS)

```
pip install -r requirements.txt
cp .env.example .env                        # Windows PowerShell: cp works too
python -m backend.retrieval.build_index --force
python -m backend.retrieval.probe           # retrieval smoke test + out-of-corpus scores
python -m backend.retrieval.search "customer shared payment credentials" -k 5 --expand
python -m backend.retrieval.search --clause rbi_circular_2017 "7(ii)" --full
python -m backend.retrieval.diagnose        # only if extracted text looks wrong
```

The build prints `WARNING: extraction integrity check failed` if words are being fused
(it compares our token count per page against PyMuPDF's own word list).

## Chunking strategy and how it answers the section 3 difficulty

The liability table is meaningless without who-is-at-fault definitions, the working-day
counting rule and the "beyond seven days" exception. A fixed-size chunker separates them.

| Mechanism | What it prevents |
|---|---|
| Clause tree from numbered markers, validated by sequence | cutting mid-rule; mistaking a wrapped "(a)"/"10" for a clause; `(i)` as roman vs. letter |
| Largest subtree that fits `CHUNK_MAX_WORDS`; split only at sub-clause boundaries | a rule separated from its conditions |
| `lead_in`: parent text repeated into each child of a split clause | "(ii) …" losing "in the following cases:" |
| Unrecognised lines are continuation text | "Further, …" / "However, …" provisos detached from their rule |
| Table = caption + rows + trailing note + introducing clause, one chunk; rows also embedded as sentences | Table 2 retrieved without the working-day rule or para 8's definition of third-party breach |
| Cross-references extracted and resolved both ways (`refs`, `referenced_by`); `expand_refs=True` | a provision the query never mentions (the 4.1 failure mode) |
| `page_spans` / `clause_spans` per chunk | citing "para 7" when the sentence is 7(ii) on p. 4 |
| Per-profile boilerplate rules + statistical header/footer removal | index/TOC pages and page furniture outranking real clauses |

`text` is source wording; breadcrumb/lead-in/sentence-form rows exist only in `embed_text`.

## Embedding strategy

* Local `sentence-transformers` model chosen by `EMBEDDING_MODEL`; vectors L2-normalised; FAISS
  `IndexFlatIP` (exact cosine; ~250 chunks, so approximate search would only add error).
* Each chunk embeds `breadcrumb | citation label` + `Context: <lead-in>` + body.
* **Re-embedding without a full rebuild:** every chunk has a `content_hash` of exactly what is
  embedded (plus the embedder identity). Unchanged chunks reuse stored vectors; `--only DOC_ID`
  forces one document; `--force` redoes all. A fingerprint of (document hashes, chunker config,
  embedder, parser source) makes an unchanged run a no-op (idempotent).
* The index records build time, document versions + SHA-256, chunk counts and throughput
  (`manifest.json`; `Retriever.index_info()` is shaped for Prometheus gauges, D2).
* The retriever refuses to start if the embedder differs from the one that built the index, or if
  `chunks.jsonl` no longer matches the manifest.

## Known limits (read before relying on it)

* MiniLM-class models truncate at 256 tokens. The builder warns when chunks exceed it; lower
  `CHUNK_MAX_WORDS` or set `EMBEDDING_MAX_SEQ_LENGTH`.
* Structure detection relies on bold headings and numbering. Annex/form pages and the SBI Rights
  policy's long lists are parsed less cleanly (`manifest.json → warnings`); they are not on the
  liability path.
* `supporting_text` is chosen by lexical overlap with the query. Treat it as a pointer; the
  agent's verifier (A3) should still check any quote with `Retriever.quote_resolves`.
* `search_min_score` is uncalibrated (default 0). Use the evaluation set to pick the threshold
  that drives escalation for out-of-corpus questions (F7).
