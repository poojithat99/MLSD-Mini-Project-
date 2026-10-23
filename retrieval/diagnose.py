"""Trace one phrase through every extraction stage and show where a space is lost.

    python -m backend.retrieval.diagnose
    python -m backend.retrieval.diagnose --phrase "customer protection" --page 1

For each stage it prints a short window of the text around the phrase (as repr, plus the
code points of the gap), so a lost or replaced space is visible.  The first stage that
differs from the stage before it is the culprit.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import unicodedata

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz  # type: ignore

from . import ingest
from .config import REPO_ROOT, _load_dotenv, _path
from .profiles import get_profile


def _find_line_spans(page, needle: str, **kw) -> list[dict] | None:
    for block in page.get_text("dict", **kw)["blocks"]:
        for ln in block.get("lines", []):
            if needle.lower() in "".join(s["text"] for s in ln["spans"]).lower():
                return ln["spans"]
    return None


def _window(text: str, needle: str) -> str:
    i = text.lower().find(needle.lower())
    if i < 0:
        return "(phrase not in this text)"
    seg = text[i: i + len(needle) + 6]
    k = len(needle.split()[0])
    gap = " ".join(f"{ord(c):04x}" for c in seg[k: k + 2])
    return f"{seg!r}   gap-codepoints: {gap}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="rbi_circular_2017")
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--phrase", default="Limiting Liability")
    a = ap.parse_args()
    _load_dotenv()
    corpus = _path("CORPUS_DIR", REPO_ROOT / "data" / "corpus")
    index = _path("INDEX_DIR", REPO_ROOT / "data" / "index")
    spec = next(s for s in ingest.load_manifest(corpus) if s.doc_id == a.doc)
    path = corpus / spec.file
    P = a.phrase
    first_word = P.split()[0]
    print(f"python {sys.version.split()[0]} | {platform.platform()} | pymupdf {fitz.VersionBind}")
    print(f"phrase: {P!r}  (doc {a.doc}, page {a.page})\n")

    def show(label: str, text: str | None) -> None:
        print(f"{label:<46} {_window(text, P) if text is not None else '(line not found)'}")

    # A. raw extraction, fresh handle
    d1 = fitz.open(path)
    pg = d1[a.page - 1]
    sp = _find_line_spans(pg, first_word) or []
    raw = "".join(s["text"] for s in sp) if sp else None
    show("A1 raw dict, default flags, fresh", raw)
    sp2 = _find_line_spans(pg, first_word, flags=ingest._TEXT_FLAGS) or []
    show("A2 raw dict, explicit flags", "".join(s["text"] for s in sp2) if sp2 else None)

    # B. does running find_tables first change raw extraction?
    d2 = fitz.open(path)
    pg2 = d2[a.page - 1]
    pg2.find_tables()
    sp3 = _find_line_spans(pg2, first_word) or []
    show("B1 raw dict AFTER find_tables (same page)", "".join(s["text"] for s in sp3) if sp3 else None)
    d3 = fitz.open(path)
    d3[a.page - 1].find_tables()
    sp4 = _find_line_spans(fitz.open(path)[a.page - 1], first_word) or []
    show("B2 raw dict, other handle ran find_tables", "".join(s["text"] for s in sp4) if sp4 else None)

    # C. our string functions applied to the raw text
    if raw:
        lt = ingest._line_text(sp)
        show("C1 after _line_text", lt)
        s = unicodedata.normalize("NFKC", lt)
        show("C2 after NFKC", s)
        s = s.replace("\u200b", "").replace("\u00ad", "")
        show("C3 after zero-width removal", s)
        s = ingest._LEADERS.sub(" ", s)
        show("C4 after leader removal", s)
        s = ingest._SPACES.sub(" ", s).strip()
        show("C5 after space collapse", s)
        show("C6 clean_text(_line_text(spans))", ingest.clean_text(lt))

    # D. MuPDF's own word list
    ws = [w[4] for w in pg.get_text("words")]
    second = P.split()[-1].lower()
    near = [w for w in ws if first_word.lower() in w.lower() or second in w.lower()]
    print(f"{'D  words list containing the phrase words':<46} {near[:6]}")

    # E. ingest output and stored chunks
    recs = ingest.ingest_pdf(path, spec.doc_id, spec.title, get_profile(spec.profile))
    rec = next(r for r in recs if r.page_number == a.page)
    row = next((ln.text for ln in rec.lines if first_word.lower() in ln.text.lower()), None)
    show("E1 row produced by ingest_pdf", row)
    cj = index / "chunks.jsonl"
    if cj.exists():
        for line in cj.read_text(encoding="utf-8").splitlines():
            c = json.loads(line)
            if c["doc_id"] == a.doc and first_word.lower() in c["text"].lower():
                show(f"E2 stored chunk {c['chunk_id']}", c["text"])
                break
        else:
            print(f"{'E2 stored chunk':<46} (phrase not found in any stored chunk text)")
    print("\nThe first line whose window differs from the line above it is where the space is lost.")


if __name__ == "__main__":
    main()
