"""Step 5: retrieval with citations.

    from backend.retrieval.search import Retriever
    r = Retriever()
    for p in r.search("customer shared OTP with a caller", k=5, expand_refs=True):
        print(p.citation, p.score); print(p.supporting_text)

    python -m backend.retrieval.search "reported after seven working days" -k 5 --expand
    python -m backend.retrieval.search --clause rbi_circular_2017 "7(ii)"

Three entry points, intended to be wrapped as agent tools (requirement A1):

* ``search(query, ...)``     semantic top-k; optional cross-reference expansion
* ``get_clause(doc, id)``    exact lookup of a clause / table by identifier
* ``get_chunk(chunk_id)``    re-read a chunk the agent has already seen

Every :class:`Passage` carries what a citation needs: document title, page,
clause, the section heading, the full passage text, and ``supporting_text``: the
sentence(s) of the passage that best match the query, copied verbatim so a
verifier can check them as a substring (requirement A3; see
:func:`Retriever.quote_resolves`).

What ``expand_refs`` is for
---------------------------
A query describing a customer who shared an OTP and reported late will retrieve
the negligence clause.  It will not necessarily retrieve the "beyond seven working
days -> board-approved policy" rule, because the query never mentions it
(project brief section 5).  The chunker recorded which clauses and tables each
passage cites, and the reverse.  ``expand_refs=True`` appends those neighbours
(flagged ``via=...``) so the definition, the table and the exception travel
together.  It is a retrieval aid, not a substitute for the agent's own
second-hop search.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .chunker import clause_at, page_at, sentence_spans
from .config import Settings
from .embeddings import Embedder, get_embedder
from .ingest import normalize_for_match


class IndexError_(RuntimeError):
    """The on-disk index is missing or inconsistent."""


@dataclass
class Passage:
    rank: int
    score: float                     # cosine similarity to the query (not a probability)
    chunk_id: str
    doc_id: str
    doc_title: str
    short_title: str
    part: str
    chunk_type: str
    clause: str                      # label of the whole chunk, e.g. "7" or "3(1)(a)-(d)"
    supporting_clause: str           # innermost clause containing supporting_text, e.g. "7(ii)"
    page: int                        # page of supporting_text
    page_start: int
    page_end: int
    section_heading: str
    heading_path: list[str]
    text: str                        # full passage, source wording
    lead_in: str                     # parent text this passage depends on
    supporting_text: str             # verbatim excerpt of `text`
    citation: str
    via: str = ""                    # "" for direct hits; otherwise why it was added
    refs: list[str] = field(default_factory=list)
    referenced_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


_STOP = frozenset(
    "a an the of to in on for and or by with as at is are was were be been it its this that these those from "
    "which who than then there any all not no but if so shall may must will can has have had do does did their "
    "his her he she they them we our you your i my me what when how where whether about after before".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


def _stem(t: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(t) > len(suf) + 3 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def _terms(s: str) -> set[str]:
    return {_stem(t) for t in _TOKEN.findall(s.lower()) if t not in _STOP}


def _norm_clause_id(raw: str) -> str:
    """'Para 7 (ii)' / '4.6.2.b (ii)' / '7(II)' -> '7(ii)' / '4.6.2(b)(ii)'."""
    s = re.sub(r"(?i)\b(paragraphs?|paras?\.?|clauses?|sub-?clauses?)\b", "", raw).strip().lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"^(\d+(?:\.\d+)*)\.([a-z])(?=\(|$)", r"\1(\2)", s)     # 4.6.2.b -> 4.6.2(b)
    return s


class Retriever:
    def __init__(self, settings: Settings | None = None, embedder: Embedder | None = None):
        self.settings = settings or Settings.from_env()
        d = self.settings.index_dir
        for name in ("manifest.json", "chunks.jsonl", "index.faiss", "vectors.npy"):
            if not (d / name).exists():
                raise IndexError_(f"{d / name} not found; run: python -m backend.retrieval.build_index")
        self.manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        h = hashlib.sha256((d / "chunks.jsonl").read_bytes()).hexdigest()
        if h != self.manifest["integrity"]["chunks_sha256"]:
            raise IndexError_("chunks.jsonl does not match manifest.json (partial or mixed build); rebuild the index")
        with open(d / "chunks.jsonl", encoding="utf-8") as fh:
            self.chunks: list[dict] = [json.loads(line) for line in fh if line.strip()]
        self.vectors = np.load(d / "vectors.npy")
        if len(self.chunks) != len(self.vectors):
            raise IndexError_("chunks and vectors are out of step; rebuild the index")
        import faiss

        self.index = faiss.read_index(str(d / "index.faiss"))
        if self.index.ntotal != len(self.chunks):
            raise IndexError_("FAISS index size differs from chunk count; rebuild the index")
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        self.embedder = embedder or get_embedder(self.settings)
        if self.embedder.name != self.manifest["embedder"]["name"]:
            raise IndexError_(
                f"index was built with embedder {self.manifest['embedder']['name']!r} but the current "
                f"configuration is {self.embedder.name!r}; rebuild the index or fix the environment"
            )
        self._qcache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def _embed_query(self, query: str) -> np.ndarray:
        with self._lock:
            hit = self._qcache.get(query)
            if hit is not None:
                self._qcache.move_to_end(query)
                return hit
        vec = self.embedder.embed_query(query)
        with self._lock:
            self._qcache[query] = vec
            while len(self._qcache) > self.settings.query_cache_size:
                self._qcache.popitem(last=False)
        return vec

    def _passage(self, c: dict, score: float, rank: int, query: str = "", via: str = "",
                 clause_hint: str = "") -> Passage:
        text = c["text"]
        sup_a, sup_b = 0, min(len(text), 400)
        if clause_hint:                                   # exact clause lookup: quote that clause's span
            for sp in c["clause_spans"]:
                if sp["clause"] == clause_hint:
                    sup_a, sup_b = sp["start"], sp["end"]
                    break
        elif query:
            q = _terms(query)
            best, best_s = None, -1.0
            for a, b in sentence_spans(text):
                st = _terms(text[a:b])
                if not st:
                    continue
                s = len(q & st) / (len(st) ** 0.5)
                if s > best_s:
                    best, best_s = (a, b), s
            if best:
                sup_a, sup_b = best
        supporting = text[sup_a:sup_b].strip()
        sup_clause = clause_hint or clause_at(c["clause_spans"], sup_a) or (c["clause_ids"][0] if c["clause_ids"] else "")
        page = page_at(c["page_spans"], sup_a)
        word = c.get("citation_word", "clause")
        if c["chunk_type"] == "table":
            where = c["clause"]
        elif sup_clause:
            where = f"{word} {sup_clause}"
        else:
            where = c["section_heading"] or "preamble"
        part = f"{c['part']}, " if c["part"] else ""
        pages = f"p. {page}" if c["page_start"] == c["page_end"] else f"p. {page} (passage spans pp. {c['page_start']}-{c['page_end']})"
        return Passage(
            rank=rank, score=float(score), chunk_id=c["chunk_id"], doc_id=c["doc_id"], doc_title=c["doc_title"],
            short_title=c["short_title"], part=c["part"], chunk_type=c["chunk_type"], clause=c["clause"],
            supporting_clause=sup_clause, page=page, page_start=c["page_start"], page_end=c["page_end"],
            section_heading=c["section_heading"], heading_path=c["heading_path"], text=text,
            lead_in=c["lead_in"], supporting_text=supporting,
            citation=f"{c['short_title']}, {part}{where}, {pages}",
            via=via, refs=c["refs"], referenced_by=c["referenced_by"],
        )

    # ------------------------------------------------------------------ #
    def search(
        self, query: str, k: int | None = None, *, doc_ids: list[str] | None = None,
        chunk_types: list[str] | None = None, expand_refs: bool = False, min_score: float | None = None,
    ) -> list[Passage]:
        """Top-k passages for ``query``; optionally add cross-referenced neighbours."""
        if not query or not query.strip():
            return []
        k = k or self.settings.search_top_k
        floor = self.settings.search_min_score if min_score is None else min_score
        q = self._embed_query(query)
        filtered = bool(doc_ids or chunk_types)
        fetch = self.index.ntotal if filtered else min(self.index.ntotal, max(k * 3, k))
        scores, ids = self.index.search(q[None, :].astype(np.float32), fetch)
        hits: list[Passage] = []
        for s, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            c = self.chunks[int(i)]
            if doc_ids and c["doc_id"] not in doc_ids:
                continue
            if chunk_types and c["chunk_type"] not in chunk_types:
                continue
            if s < floor:
                continue
            hits.append(self._passage(c, s, len(hits) + 1, query=query))
            if len(hits) == k:
                break
        if expand_refs and hits:
            seen = {p.chunk_id for p in hits}
            extra: list[Passage] = []
            for p in list(hits):
                for rid, why in [(r, "cites it") for r in p.refs] + [(r, "is cited by") for r in p.referenced_by]:
                    if rid in seen or len(extra) >= self.settings.search_max_related:
                        continue
                    seen.add(rid)
                    c = self.by_id[rid]
                    extra.append(self._passage(
                        c, float(self.vectors[c["vector_id"]] @ q), len(hits) + len(extra) + 1,
                        query=query, via=f"added because {p.chunk_id} {why}" if why == "cites it" else f"added because it cites {p.chunk_id}",
                    ))
            hits += extra
        return hits

    def get_clause(self, doc_id: str, clause_id: str, part: str | None = None) -> list[Passage]:
        """Exact lookup by clause id ("7(ii)", "4.6.3", "3(1)(a)") or table ("Table 2").

        Returns the passage(s) containing that clause; ``supporting_text`` is that
        clause's own text (including its sub-clauses).  Several results mean the id
        is ambiguous across parts (SBI numbers each of its three policies from 1);
        pass ``part`` to disambiguate.
        """
        raw = clause_id.strip()
        tm = re.fullmatch(r"(?i)(?:table[:\s]*)(\d+)", raw)
        out: list[Passage] = []
        if tm:
            for c in self.chunks:
                if c["doc_id"] == doc_id and c["table_no"] == tm.group(1) and (not part or c["part"] == part):
                    out.append(self._passage(c, 1.0, len(out) + 1, via="exact"))
            return out
        cid = _norm_clause_id(raw)
        for c in self.chunks:
            if c["doc_id"] != doc_id or (part and c["part"] != part):
                continue
            if cid in c["clause_ids"]:
                out.append(self._passage(c, 1.0, len(out) + 1, via="exact", clause_hint=cid))
        if not out:                                       # parent id: gather descendants
            for c in self.chunks:
                if c["doc_id"] != doc_id or (part and c["part"] != part):
                    continue
                if any(x.startswith(cid + "(") or x.startswith(cid + ".") for x in c["clause_ids"]):
                    out.append(self._passage(c, 1.0, len(out) + 1, via="descendants"))
        return out

    def get_chunk(self, chunk_id: str) -> Passage | None:
        c = self.by_id.get(chunk_id)
        return self._passage(c, 1.0, 1, via="exact") if c else None

    def quote_resolves(self, chunk_id: str, quote: str) -> bool:
        """True if ``quote`` occurs in the chunk (whitespace/case/quote-style tolerant)."""
        c = self.by_id.get(chunk_id)
        if not c or not quote.strip():
            return False
        hay = normalize_for_match(c["text"] + " " + c["lead_in"])
        return normalize_for_match(quote) in hay

    # ------------------------------------------------------------------ #
    def index_info(self) -> dict:
        """Freshness + throughput, shaped for exposure as Prometheus gauges (requirement D2)."""
        m = self.manifest
        return {
            "built_at": m["built_at"], "built_at_unix": m["built_at_unix"], "n_chunks": m["n_chunks"],
            "documents": [{k: d[k] for k in ("doc_id", "version", "sha256", "chunks")} for d in m["documents"]],
            "embedder": m["embedder"]["name"], "stats": m["stats"],
        }


# --------------------------------------------------------------------------- #
_default: Retriever | None = None


def search(question: str, k: int = 5, **kw) -> list[Passage]:
    """Module-level convenience: lazily creates one shared Retriever."""
    global _default
    if _default is None:
        _default = Retriever()
    return _default.search(question, k, **kw)


def _main() -> None:
    ap = argparse.ArgumentParser(description="Query the corpus index.")
    ap.add_argument("query", nargs="?", help="question text (or clause id with --clause)")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--doc", action="append", help="restrict to doc_id (repeatable)")
    ap.add_argument("--expand", action="store_true", help="add cross-referenced passages")
    ap.add_argument("--clause", metavar="DOC_ID", help="exact clause lookup in DOC_ID; query is the clause id")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--full", action="store_true", help="print full passage text")
    args = ap.parse_args()
    r = Retriever()
    if args.clause:
        res = r.get_clause(args.clause, args.query or "")
    else:
        res = r.search(args.query or "", args.k, doc_ids=args.doc, expand_refs=args.expand)
    if args.json:
        print(json.dumps([p.to_dict() for p in res], indent=2, ensure_ascii=False))
        return
    for p in res:
        tag = f"  [{p.via}]" if p.via else ""
        print(f"\n#{p.rank} score={p.score:.3f}  {p.citation}{tag}")
        if p.lead_in and args.full:
            print(f"   context: {p.lead_in[:200]}")
        print("   > " + (p.text if args.full else p.supporting_text).replace("\n", "\n   > "))


if __name__ == "__main__":
    _main()
