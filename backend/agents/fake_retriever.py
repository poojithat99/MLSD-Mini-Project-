"""An offline stand-in for Member 1's Retriever, for tests and demos.

It loads the REAL chunks (``data/index/chunks.jsonl``), so tests run against the
actual clause text of the corpus, but ranks by keyword overlap instead of
embeddings, so it needs neither FAISS nor the sentence-transformers model.

Everything except ranking (``get_clause``, ``get_chunk``, ``quote_resolves`` and
the construction of ``Passage`` objects) is Member 1's own code, inherited.
Results say nothing about the real model's ranking quality.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from backend.retrieval.search import Passage, Retriever, _terms

INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "index"


class KeywordRetriever(Retriever):
    def __init__(self, index_dir: Path = INDEX_DIR, max_related: int = 4):
        # Deliberately NOT calling Retriever.__init__: it loads FAISS and the embedding model.
        with open(index_dir / "chunks.jsonl", encoding="utf-8") as fh:
            self.chunks = [json.loads(line) for line in fh if line.strip()]
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        self._terms = {c["chunk_id"]: _terms(c["text"] + " " + c["lead_in"]) for c in self.chunks}
        self.max_related = max_related

    def search(self, query: str, k: int | None = None, *, expand_refs: bool = False, **_ignored) -> list[Passage]:
        k = k or 5
        q = _terms(query)
        if not q:
            return []
        scored = []
        for c in self.chunks:
            t = self._terms[c["chunk_id"]]
            overlap = len(q & t)
            if overlap:
                scored.append((overlap / math.sqrt(len(q) * len(t)), c))
        scored.sort(key=lambda sc: -sc[0])
        hits = [self._passage(c, s, i + 1, query=query) for i, (s, c) in enumerate(scored[:k])]
        if expand_refs:
            seen = {p.chunk_id for p in hits}
            for p in list(hits):
                for rid in p.refs + p.referenced_by:
                    if rid not in seen and len(hits) < k + self.max_related:
                        seen.add(rid)
                        hits.append(self._passage(self.by_id[rid], 0.0, len(hits) + 1, query=query,
                                                  via=f"cross-reference of {p.chunk_id}"))
        return hits
