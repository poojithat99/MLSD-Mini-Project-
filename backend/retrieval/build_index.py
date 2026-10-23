"""Step 4: embed every chunk and build the FAISS index.

    python -m backend.retrieval.build_index            # build / refresh
    python -m backend.retrieval.build_index --force    # re-embed everything
    python -m backend.retrieval.build_index --only sbi_policy_2021
    python -m backend.retrieval.build_index --dry-run  # parse + chunk, print stats, write nothing

Artifacts written to INDEX_DIR (all linked by ``vector_id``):

    index.faiss    FAISS IndexIDMap2(IndexFlatIP); the id of each vector is the chunk's vector_id
    vectors.npy    the same vectors, kept so re-embedding can be incremental
    chunks.jsonl   one JSON object per chunk: text + all metadata (page, clause, citation fields...)
    pages.jsonl    the cleaned page text each chunk came from (for audit / citation checking)
    manifest.json  build time, document versions + sha256, embedder identity, counts, throughput

Design points tied to the brief
-------------------------------
* D2 idempotent: a *fingerprint* of (document hashes, chunker settings, embedder
  identity, parser source) is stored.  If nothing changed the run is a no-op;
  running twice yields the same index.
* L1 "re-embedding without rebuilding the whole index": every chunk carries a
  ``content_hash`` of exactly what is embedded.  Vectors whose hash (and embedder
  identity) are unchanged are reused; only new/changed chunks are embedded.  FAISS
  flat-index construction from stored vectors takes milliseconds, so the index file
  itself is rewritten but *embedding* work is incremental.  ``--only DOC`` forces
  re-embedding of one document.
* Why IndexFlatIP: the corpus is a few hundred chunks.  Exact search is
  instantaneous at this size and removes approximate-search recall as a variable
  while evaluating retrieval.  Swap in HNSW/IVF only if the corpus grows by 100x.
* The text/metadata live in chunks.jsonl rather than inside FAISS (which stores
  only vectors); the explicit ``vector_id`` -> FAISS id mapping keeps them linked,
  and a manifest checksum lets the loader detect a mismatched pair.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .chunker import Chunk, ChunkerConfig, chunk_document, resolve_xrefs
from .config import Settings
from .embeddings import Embedder, get_embedder
from .ingest import PageRecord, file_sha256, ingest_pdf, load_manifest
from .profiles import get_profile

SCHEMA_VERSION = 1
_HERE = Path(__file__).resolve().parent


def _source_hash() -> str:
    """Hash of the parsing code: if the chunker changes, the index must be rebuilt."""
    h = hashlib.sha256()
    for name in ("ingest.py", "chunker.py", "profiles.py"):
        h.update((_HERE / name).read_bytes())
    return h.hexdigest()[:16]


def _sha_bytes(path: Path) -> str:
    return file_sha256(path)


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _read_manifest(path: Path) -> dict:
    """Read manifest.json as UTF-8; {} if missing or unreadable (e.g. written by an older build)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):          # UnicodeDecodeError is a ValueError
        return {}


def _save_npy(path: Path, arr: np.ndarray) -> None:
    with open(path, "wb") as fh:           # closed before os.replace (Windows cannot replace an open file)
        np.save(fh, arr)


def _atomic_write(path: Path, writer) -> None:
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    writer(tmp)
    os.replace(tmp, path)


def fingerprint(settings: Settings, specs, embedder_name: str, cfg: ChunkerConfig, sha: dict[str, str]) -> str:
    payload = {
        "docs": [(s.doc_id, sha[s.doc_id], s.profile, s.version) for s in specs],
        "chunker": asdict(cfg),
        "embedder": embedder_name,
        "code": _source_hash(),
        "schema": SCHEMA_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build(
    settings: Settings, *, force: bool = False, only: str | None = None, dry_run: bool = False,
    embedder: Embedder | None = None, log=print,
) -> dict:
    t_start = time.perf_counter()
    cfg = ChunkerConfig(
        max_words=settings.chunk_max_words, lead_max_words=settings.chunk_lead_max_words,
        table_min_words=settings.table_min_words, header_mode=settings.embed_header_mode,
    )
    specs = load_manifest(settings.corpus_dir)
    sha = {s.doc_id: file_sha256(settings.corpus_dir / s.file) for s in specs}
    index_dir = settings.index_dir
    manifest_path = index_dir / "manifest.json"

    if dry_run:
        embedder = None
    else:
        embedder = embedder or get_embedder(settings)
    fp = fingerprint(settings, specs, embedder.name if embedder else "dry-run", cfg, sha)

    if not (force or only or dry_run) and manifest_path.exists():
        old = _read_manifest(manifest_path)
        needed = ["index.faiss", "vectors.npy", "chunks.jsonl", "pages.jsonl"]
        if old and old.get("fingerprint") == fp and all((index_dir / n).exists() for n in needed):
            log(f"index up to date (built {old['built_at']}, {old['n_chunks']} chunks); nothing to do")
            return {**old, "skipped": True}

    # ---- 1. ingest ----------------------------------------------------------- #
    t0 = time.perf_counter()
    pages: dict[str, list[PageRecord]] = {}
    for s in specs:
        pages[s.doc_id] = ingest_pdf(settings.corpus_dir / s.file, s.doc_id, s.title, get_profile(s.profile))
    t_ingest = time.perf_counter() - t0
    extraction_notes = [f"{d}: {n}" for d, recs in pages.items() for r in recs for n in r.notes]
    if extraction_notes:
        log(f"WARNING: extraction integrity check failed on {len(extraction_notes)} page(s); words are being "
            f"fused, which degrades retrieval and citations:")
        for n in extraction_notes[:6]:
            log(f"  {n}")
        log("  run: python -m backend.retrieval.diagnose   and send the output")

    # ---- 2. chunk ------------------------------------------------------------ #
    t0 = time.perf_counter()
    chunks: list[Chunk] = []
    warnings: list[str] = []
    for s in specs:
        ch, w = chunk_document(pages[s.doc_id], s, get_profile(s.profile), cfg)
        chunks += ch
        warnings += w
    unresolved = resolve_xrefs(chunks)
    seen: set[str] = set()
    for c in chunks:                                  # ids must be unique: they are the join key
        if c.chunk_id in seen:
            raise RuntimeError(f"duplicate chunk_id {c.chunk_id}")
        seen.add(c.chunk_id)
    t_chunk = time.perf_counter() - t0
    log(f"ingested {sum(len(p) for p in pages.values())} pages -> {len(chunks)} chunks "
        f"({len(warnings)} parse warnings, {len(unresolved)} unresolved cross-references)")

    if dry_run:
        for c in chunks:
            log(f"{c.chunk_id:60} p{c.page_start}-{c.page_end} {c.chunk_type:6} {c.word_count:4}w")
        return {"dry_run": True, "n_chunks": len(chunks), "warnings": warnings, "unresolved_xrefs": unresolved}

    assert embedder is not None

    # ---- 3. embed (incrementally) ------------------------------------------- #
    for c in chunks:
        c.content_hash = hashlib.sha1((embedder.name + "\n" + c.embed_text).encode()).hexdigest()
    cache: dict[str, np.ndarray] = {}
    if (index_dir / "chunks.jsonl").exists() and (index_dir / "vectors.npy").exists() and not force:
        try:
            old_chunks = _read_jsonl(index_dir / "chunks.jsonl")
            old_vecs = np.load(index_dir / "vectors.npy")
            old_man = _read_manifest(manifest_path)
            if len(old_chunks) == len(old_vecs) and old_man.get("embedder", {}).get("name") == embedder.name:
                for oc in old_chunks:
                    if only and oc["doc_id"] == only:
                        continue                       # forced re-embed of this document
                    cache[oc["content_hash"]] = old_vecs[oc["vector_id"]]
        except Exception as exc:                       # a corrupt cache must never block a build
            log(f"ignoring unreadable previous index ({exc}); embedding everything")
            cache = {}

    todo = [i for i, c in enumerate(chunks) if c.content_hash not in cache]
    vectors = np.zeros((len(chunks), embedder.dim), dtype=np.float32)
    for i, c in enumerate(chunks):
        if c.content_hash in cache:
            vectors[i] = cache[c.content_hash]
    t0 = time.perf_counter()
    if todo:
        if hasattr(embedder, "token_lengths"):
            lens = embedder.token_lengths([chunks[i].embed_text for i in todo])
            over = sorted(((n, chunks[i].chunk_id) for n, i in zip(lens, todo) if n > embedder.max_seq_length), reverse=True)
            if over:
                log(f"WARNING: {len(over)} chunk(s) exceed the model's {embedder.max_seq_length}-token limit; the "
                    f"excess is silently cut. Worst: " + "; ".join(f"{cid} ({n} tok)" for n, cid in over[:5]))
                log("  fix: set EMBEDDING_MAX_SEQ_LENGTH=512, or lower CHUNK_MAX_WORDS / CHUNK_LEAD_MAX_WORDS")
        new = embedder.embed_documents([chunks[i].embed_text for i in todo])
        for k, i in enumerate(todo):
            vectors[i] = new[k]
    t_embed = time.perf_counter() - t0
    for i, c in enumerate(chunks):
        c.vector_id = i
    log(f"embedded {len(todo)} chunk(s), reused {len(chunks) - len(todo)} cached vector(s) in {t_embed:.2f}s")

    # ---- 4. write ------------------------------------------------------------ #
    import faiss

    index_dir.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(embedder.dim))
    index.add_with_ids(vectors, np.arange(len(chunks), dtype=np.int64))

    def w_chunks(p: Path) -> None:
        with open(p, "w", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    def w_pages(p: Path) -> None:
        with open(p, "w", encoding="utf-8") as fh:
            for recs in pages.values():
                for r in recs:
                    fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    _atomic_write(index_dir / "chunks.jsonl", w_chunks)
    _atomic_write(index_dir / "pages.jsonl", w_pages)
    _atomic_write(index_dir / "vectors.npy", lambda p: _save_npy(p, vectors))
    _atomic_write(index_dir / "index.faiss", lambda p: faiss.write_index(index, str(p)))

    total = time.perf_counter() - t_start
    per_doc = []
    for s in specs:
        per_doc.append({
            "doc_id": s.doc_id, "file": s.file, "sha256": sha[s.doc_id], "version": s.version,
            "title": s.title, "pages_extracted": len(pages[s.doc_id]),
            "chunks": sum(1 for c in chunks if c.doc_id == s.doc_id),
        })
    manifest = {
        "schema": SCHEMA_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "built_at_unix": int(time.time()),
        "fingerprint": fp,
        "embedder": {"name": embedder.name, "dim": embedder.dim, "backend": settings.embedding_backend,
                     "model": settings.embedding_model},
        "chunker": asdict(cfg),
        "documents": per_doc,
        "n_chunks": len(chunks),
        "integrity": {"chunks_sha256": _sha_bytes(index_dir / "chunks.jsonl")},
        "stats": {
            "ingest_seconds": round(t_ingest, 3), "chunk_seconds": round(t_chunk, 3),
            "embed_seconds": round(t_embed, 3), "total_seconds": round(total, 3),
            "chunks_embedded": len(todo), "chunks_reused": len(chunks) - len(todo),
            "embed_chunks_per_second": round(len(todo) / t_embed, 2) if todo and t_embed > 0 else 0.0,
            "pipeline_chunks_per_second": round(len(chunks) / total, 2) if total > 0 else 0.0,
        },
        "unresolved_xrefs": unresolved,
        "warnings": warnings,
        "extraction_notes": extraction_notes,
    }
    _atomic_write(manifest_path, lambda p: p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"))
    log(f"wrote {index_dir} in {total:.2f}s ({manifest['stats']['pipeline_chunks_per_second']} chunks/s end to end)")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the FAISS retrieval index.")
    ap.add_argument("--force", action="store_true", help="re-embed every chunk")
    ap.add_argument("--only", metavar="DOC_ID", help="force re-embedding of one document")
    ap.add_argument("--dry-run", action="store_true", help="parse and chunk only; write nothing")
    args = ap.parse_args(argv)
    try:
        settings = Settings.from_env()
    except RuntimeError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    build(settings, force=args.force, only=args.only, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
