"""Step 2: extract text page by page with PyMuPDF.

For every PDF page we produce a :class:`PageRecord` preserving

* ``doc_id``           stable identifier from ``data/corpus/corpus.json``
* ``page_number``      1-based PDF page (equals the printed page number in all three
                       corpus documents; checked by the tests)
* ``section_heading``  the heading in effect at the first content line of the page
* ``text``             the cleaned source text of the page

and, for the chunker, the structured :class:`Line` objects the text came from.

What "cleaned" means here, and why (each item fixes a real problem found in the
corpus, not a hypothetical one):

1. **Visual rows.**  Several PDFs emit a clause number ("7.") and its text as
   separate lines on one baseline.  Lines whose baselines are within a few points
   are joined back into one row, left to right, so "7." and "A customer ..." are
   one line again.
2. **Boilerplate.**  Repeated page furniture ("29 | P a g e", the policy title
   footer, "RBI,DBR,CO - 4 - continuation sheet") is detected *statistically*:
   a margin line whose digits-normalised text recurs on many pages is dropped.
   The circular's bilingual letterhead (its Hindi font is mis-encoded and
   extracts as U+FFFD) is dropped by character class plus a profile pattern.
3. **Tables.**  ``Page.find_tables()`` locates ruled tables; their cells are
   extracted row-wise (so "Within 3 working days" stays attached to "Zero
   liability") and the table region is removed from the running text so the cells
   are not also emitted as a jumbled paragraph.
4. **Headings and captions.**  All three documents set headings in bold.  Bold
   rows become ``heading`` lines (multi-line headings merged); a bold "Table N"
   row and the bold title under it become ``caption`` lines.

Run ``python -m backend.retrieval.ingest --doc rbi_circular_2017 --page 4`` to
look at what the extractor produced for a page.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

try:  # PyMuPDF >= 1.24 exposes the new name; the old "fitz" alias is deprecated
    import pymupdf as fitz
except ImportError:  # pragma: no cover - older PyMuPDF
    import fitz  # type: ignore

from .config import Settings
from .profiles import DocProfile, get_profile

# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #

KIND_BODY = "body"
KIND_HEADING = "heading"
KIND_CAPTION = "caption"
KIND_TABLE = "table"


@dataclass
class Line:
    """One visual row of a page (or one whole table)."""

    text: str
    page: int
    y0: float
    y1: float
    x0: float
    size: float
    bold: bool
    kind: str = KIND_BODY
    new_para: bool = False                      # blank-line gap before this row
    rows: list[list[str]] | None = None         # only for kind == "table"
    x1: float = 0.0                             # right edge (used for table-region tests)

    def to_dict(self) -> dict:
        d = {"text": self.text, "kind": self.kind, "page": self.page}
        if self.rows is not None:
            d["rows"] = self.rows
        return d


@dataclass
class PageRecord:
    doc_id: str
    doc_title: str
    page_number: int
    section_heading: str
    headings: list[str]
    text: str
    lines: list[Line] = field(default_factory=list, repr=False)
    notes: list[str] = field(default_factory=list)   # extraction-integrity warnings for this page

    def to_dict(self, with_lines: bool = False) -> dict:
        d = {
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "page_number": self.page_number,
            "section_heading": self.section_heading,
            "headings": self.headings,
            "text": self.text,
            "n_lines": len(self.lines),
        }
        if with_lines:
            d["lines"] = [ln.to_dict() for ln in self.lines]
        return d


@dataclass(frozen=True)
class DocSpec:
    doc_id: str
    file: str
    title: str
    short_title: str
    issuer: str
    version: str
    profile: str


# --------------------------------------------------------------------------- #
# manifest helpers
# --------------------------------------------------------------------------- #

def load_manifest(corpus_dir: Path) -> list[DocSpec]:
    raw = json.loads((corpus_dir / "corpus.json").read_text(encoding="utf-8"))
    return [DocSpec(**{k: v for k, v in d.items() if not k.startswith("_")}) for d in raw["documents"]]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# text cleaning
# --------------------------------------------------------------------------- #

_SPACES = re.compile(r"[ \t\u00a0\u2009\u202f]+")
_BROKEN_OR_DEVANAGARI = re.compile(r"[\u0900-\u097F\ufffd]")
_LEADERS = re.compile(r"(?:\.\s*){4,}|_{3,}|\u2026{2,}")
_RULE_LINE = re.compile(r"^[\*\-_=\s]{3,}$")


def clean_text(s: str) -> str:
    """NFKC-normalise, drop soft hyphens / zero-width spaces, collapse blanks."""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u200b", "").replace("\u00ad", "")
    # dot / underscore leaders in fill-in forms: ~10 characters per token for a tokenizer,
    # zero information (the complaint-form annex was 121 words but >1,000 tokens)
    s = _LEADERS.sub(" ", s)
    return _SPACES.sub(" ", s).strip()


def normalize_for_match(s: str) -> str:
    """Canonical form used to check that a quoted citation occurs in a passage.

    Whitespace is collapsed and case/curly quotes are folded.  The agent-side
    verifier (requirement A3) should compare ``normalize_for_match(quote)`` as a
    substring of ``normalize_for_match(passage_text)``.
    """
    s = clean_text(s).lower()
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", s)


# --------------------------------------------------------------------------- #
# per-page extraction
# --------------------------------------------------------------------------- #

def _is_bold(span: dict) -> bool:
    return bool(span["flags"] & 16) or "bold" in span["font"].lower()


def _in_any(rects: list[tuple], cx: float, cy: float, pad: float = 2.0) -> bool:
    return any(r[0] - pad <= cx <= r[2] + pad and r[1] - pad <= cy <= r[3] + pad for r in rects)


def _line_text(spans: list[dict]) -> str:
    """Join a PDF line's spans, inserting a space wherever two spans are visibly apart.

    ``"".join(span texts)`` is only right if every inter-word space is *inside* some span.
    Some PyMuPDF versions / PDFs put it nowhere (the gap is just positioning), or merge a
    wrapped line into one line object; the words then fuse ("protectionand").  So:
    a gap wider than ~a fifth of the font size, or a change of baseline, means a space.
    """
    out: list[str] = []
    prev: dict | None = None
    for sp in spans:
        t = sp["text"].replace("\n", " ")
        if not t:
            continue
        if prev is not None and out and not out[-1].endswith(" ") and not t.startswith(" "):
            size = max(sp.get("size", 0) or 0, 1.0)
            gap = sp["bbox"][0] - prev["bbox"][2]
            rebased = abs(sp["bbox"][1] - prev["bbox"][1]) > 0.6 * size
            if gap > 0.2 * size or rebased:
                out.append(" ")
        out.append(t)
        prev = sp
    return "".join(out)


# Explicit text-extraction flags: never rely on library defaults or on state left by another call.
_TEXT_FLAGS = (
    getattr(fitz, "TEXT_PRESERVE_WHITESPACE", 2) | getattr(fitz, "TEXT_PRESERVE_LIGATURES", 1)
    | getattr(fitz, "TEXT_MEDIABOX_CLIP", 64)
)


def _raw_pieces(page, page_no: int) -> list[Line]:
    """Every text line of the page as a Line (no table filtering yet)."""
    pieces: list[Line] = []
    for block in page.get_text("dict", flags=_TEXT_FLAGS)["blocks"]:
        if block.get("type") != 0:
            continue
        for ln in block["lines"]:
            spans = [s for s in ln["spans"] if s["text"].strip()]
            if not spans:
                continue
            x0, y0, x1, y1 = ln["bbox"]
            text = clean_text(_line_text(ln["spans"]))
            if not text:
                continue
            pieces.append(
                Line(
                    text=text, page=page_no, y0=y0, y1=y1, x0=x0, x1=x1,
                    size=max(s["size"] for s in spans),
                    bold=all(_is_bold(s) for s in spans),
                )
            )
    return pieces


def _page_rows(pieces: list[Line], table_rects: list[tuple]) -> list[Line]:
    """Visual rows of a page, excluding anything inside a detected table."""
    pieces = [p for p in pieces if not _in_any(table_rects, (p.x0 + p.x1) / 2, (p.y0 + p.y1) / 2)]
    pieces = sorted(pieces, key=lambda p: (p.y0, p.x0))

    rows: list[list[Line]] = []
    for p in pieces:
        if rows and abs(p.y0 - rows[-1][0].y0) <= 3.0:
            rows[-1].append(p)
        else:
            rows.append([p])

    merged: list[Line] = []
    for group in rows:
        group.sort(key=lambda p: p.x0)
        merged.append(
            Line(
                text=clean_text(" ".join(g.text for g in group)),
                page=group[0].page,
                y0=min(g.y0 for g in group), y1=max(g.y1 for g in group), x0=group[0].x0,
                x1=max(g.x1 for g in group), size=max(g.size for g in group),
                bold=all(g.bold for g in group),
            )
        )
    return merged


def _fusion_check(page, pieces: list[Line], table_rects: list[tuple], page_no: int) -> str | None:
    """Compare our token count with MuPDF's independent word list for the page.

    Fused words ("protectionand") reduce our token count below MuPDF's.  The word list
    is requested separately, so a mismatch means our extraction lost spaces."""
    words = page.get_text("words", flags=_TEXT_FLAGS)
    # apply the same cleaning we apply to text (dot leaders become nothing) before counting
    n_words = sum(1 for w in words
                  if clean_text(w[4]) and not _in_any(table_rects, (w[0] + w[2]) / 2, (w[1] + w[3]) / 2))
    n_ours = sum(len(p.text.split()) for p in pieces
                 if not _in_any(table_rects, (p.x0 + p.x1) / 2, (p.y0 + p.y1) / 2))
    if n_words and n_ours < n_words - max(2, int(0.01 * n_words)):
        return (f"page {page_no}: extracted {n_ours} tokens but PyMuPDF lists {n_words} words "
                f"(~{n_words - n_ours} fused word(s)); text quality is degraded")
    return None


def _table_lines(page, page_no: int) -> tuple[list[tuple], list[Line]]:
    """Locate ruled tables; return their rectangles and one ``table`` Line each."""
    rects: list[tuple] = []
    lines: list[Line] = []
    try:
        found = page.find_tables().tables
    except Exception:  # pragma: no cover - defensive: table finder must never kill ingest
        return rects, lines
    for t in found:
        grid = []
        for row in t.extract():
            cells = [clean_text((c or "").replace("\n", " ")) for c in row]
            if any(cells):
                grid.append(cells)
        if not grid:
            continue
        x0, y0, x1, y1 = t.bbox
        rects.append((x0, y0, x1, y1))
        lines.append(
            Line(
                text="\n".join(" | ".join(r) for r in grid),
                page=page_no, y0=y0, y1=y1, x0=x0, size=0.0, bold=False,
                kind=KIND_TABLE, rows=grid,
            )
        )
    return rects, lines


def _digits_key(text: str) -> str:
    return re.sub(r"\d+", "#", text.strip().lower())


def _drop_boilerplate(
    per_page: dict[int, list[Line]], heights: dict[int, float], profile: DocProfile
) -> None:
    """Remove letterhead, broken-encoding rows, and repeated page furniture (in place)."""
    patterns = profile.compiled_drop()
    n_pages = len(per_page)
    threshold = max(3, math.ceil(0.3 * n_pages)) if n_pages > 2 else 2

    def in_margin(ln: Line, pno: int) -> bool:
        rel = ln.y0 / heights[pno]
        # Measured on the corpus: running headers sit at ~0.075 of page height,
        # footers at ~0.92+.  Body text also reaches into these bands, which is
        # safe because a line is only dropped if its digits-normalised text
        # *repeats* on many pages.
        return rel < 0.095 or rel > 0.905

    seen: dict[str, set[int]] = {}
    for pno, lines in per_page.items():
        for ln in lines:
            if ln.kind == KIND_BODY and in_margin(ln, pno):
                seen.setdefault(_digits_key(ln.text), set()).add(pno)
    furniture = {k for k, pages in seen.items() if len(pages) >= threshold}

    for pno, lines in per_page.items():
        keep: list[Line] = []
        for ln in lines:
            if ln.kind == KIND_TABLE:
                keep.append(ln)
                continue
            if _BROKEN_OR_DEVANAGARI.search(ln.text):
                continue
            if _RULE_LINE.match(ln.text):
                continue
            if any(p.search(ln.text) for p in patterns):
                continue
            if in_margin(ln, pno) and _digits_key(ln.text) in furniture:
                continue
            keep.append(ln)
        per_page[pno] = keep


MAX_HEADING_CHARS = 150
MAX_HEADING_WORDS = 24

_CAPTION = re.compile(r"^Table\s+\d+\b", re.IGNORECASE)
_NEW_ITEM = re.compile(r"^(\d+(\.\d+)*\.?\s|\([a-z0-9]{1,4}\)|[a-z]\)\s|CHAPTER\b)", re.IGNORECASE)


def _median_pitch(rows: list[Line]) -> float:
    pitches = [
        b.y0 - a.y0 for a, b in zip(rows, rows[1:])
        if a.kind != KIND_TABLE and b.kind != KIND_TABLE and 4.0 < b.y0 - a.y0 < 40.0
    ]
    if pitches:
        return statistics.median(pitches)
    sizes = [r.size for r in rows if r.size]
    return 1.25 * (statistics.median(sizes) if sizes else 12.0)


def _classify(rows: list[Line]) -> None:
    """Tag rows as heading / caption / body and mark paragraph starts (in place)."""
    pitch = _median_pitch(rows)

    i = 0
    while i < len(rows):
        row = rows[i]
        if row.kind == KIND_TABLE:
            i += 1
            continue
        if row.bold and _CAPTION.match(row.text):
            row.kind = KIND_CAPTION
            j = i + 1
            while (
                j < len(rows) and rows[j].kind == KIND_BODY and rows[j].bold
                and abs(rows[j].size - row.size) < 0.8
                and rows[j].y0 - rows[j - 1].y0 <= 1.5 * pitch
            ):
                rows[j].kind = KIND_CAPTION
                j += 1
            i = j
            continue
        if row.bold:
            row.kind = KIND_HEADING
        i += 1

    # Merge wrapped multi-line headings ("4.6 Limiting Liability ... Banking" +
    # "transactions:").  A merged run that is too long to be a heading is a bold
    # *paragraph* (the SBI Customer Rights section sets whole paragraphs in bold)
    # and is demoted back to body rows.
    out: list[Line] = []
    run: list[Line] = []

    def flush() -> None:
        if not run:
            return
        text = clean_text(" ".join(r.text for r in run))
        if len(text) > MAX_HEADING_CHARS or len(text.split()) > MAX_HEADING_WORDS:
            for r in run:
                r.kind = KIND_BODY
                out.append(r)
        else:
            first = run[0]
            first.text, first.y1 = text, run[-1].y1
            out.append(first)
        run.clear()

    for row in rows:
        if row.kind == KIND_HEADING:
            prev = run[-1] if run else None
            mergeable = (
                prev is not None and abs(row.size - prev.size) < 0.6
                and row.y0 - prev.y0 <= 1.15 * pitch and not _NEW_ITEM.match(row.text)
            )
            if not mergeable:
                flush()
            run.append(row)
        else:
            flush()
            out.append(row)
    flush()
    rows[:] = out

    prev_body: Line | None = None
    for row in rows:
        if row.kind == KIND_BODY:
            row.new_para = prev_body is not None and (row.y0 - prev_body.y0) > 1.3 * pitch
            prev_body = row
        elif row.kind in (KIND_HEADING, KIND_TABLE, KIND_CAPTION):
            prev_body = None



def _merge_cross_page_headings(per_page: dict[int, list[Line]]) -> None:
    """A bold heading wrapped over a page break leaves a lower-case fragment at the
    top of the next page ("... UPI Services / INB" | "acknowledgement of request ...").
    Append such fragments to the heading they continue.  Fragments that continue
    nothing are demoted to body text so they cannot masquerade as headings."""
    prev_last: Line | None = None
    for pno in sorted(per_page):
        lines = per_page[pno]
        first = lines[0] if lines else None
        if first is not None and first.kind == KIND_HEADING and first.text[:1].islower():
            if prev_last is not None and prev_last.kind == KIND_HEADING:
                prev_last.text = clean_text(f"{prev_last.text} {first.text}")
                lines.pop(0)
            else:
                first.kind = KIND_BODY
        for ln in lines[1:]:  # lower-case heading fragments elsewhere on the page
            if ln.kind == KIND_HEADING and ln.text[:1].islower():
                ln.kind = KIND_BODY
        prev_last = lines[-1] if lines else None

def _render(lines: list[Line]) -> str:
    return "\n".join(ln.text for ln in lines)


def ingest_pdf(path: Path, doc_id: str, doc_title: str, profile: DocProfile) -> list[PageRecord]:
    """Extract one PDF into PageRecords (skipped pages are omitted, numbering is unchanged)."""
    # Two independent handles: text is read FIRST, from a handle that table detection never
    # touches, with explicit flags.  (On one Windows machine, reading text from a page after
    # find_tables() had run on it produced fused words; this ordering removes the dependency.)
    text_doc, table_doc = fitz.open(path), fitz.open(path)
    per_page: dict[int, list[Line]] = {}
    heights: dict[int, float] = {}
    notes: dict[int, list[str]] = {}
    try:
        for idx in range(text_doc.page_count):
            pno = idx + 1
            if pno in profile.skip_pages:
                continue
            page = text_doc[idx]
            heights[pno] = float(page.rect.height)
            pieces = _raw_pieces(page, pno)
            rects, tables = _table_lines(table_doc[idx], pno)
            rows = _page_rows(pieces, rects)
            problem = _fusion_check(page, pieces, rects, pno)
            notes[pno] = [problem] if problem else []
            per_page[pno] = sorted(rows + tables, key=lambda ln: (ln.y0, ln.x0))
    finally:
        text_doc.close()
        table_doc.close()

    _drop_boilerplate(per_page, heights, profile)
    for rows in per_page.values():
        _classify(rows)
    _merge_cross_page_headings(per_page)

    records: list[PageRecord] = []
    carried = ""
    for pno in sorted(per_page):
        lines = per_page[pno]
        headings = [ln.text for ln in lines if ln.kind == KIND_HEADING]
        current, at_first_body = carried, None
        for ln in lines:
            if ln.kind == KIND_HEADING:
                current = ln.text
            elif at_first_body is None:
                at_first_body = current
        records.append(
            PageRecord(
                doc_id=doc_id, doc_title=doc_title, page_number=pno,
                section_heading=at_first_body if at_first_body is not None else current,
                headings=headings, text=_render(lines), lines=lines, notes=notes.get(pno, []),
            )
        )
        carried = current
    return records


def ingest_corpus(settings: Settings, only: str | None = None) -> dict[str, list[PageRecord]]:
    """Ingest every document in the manifest (or just ``only``)."""
    out: dict[str, list[PageRecord]] = {}
    for spec in load_manifest(settings.corpus_dir):
        if only and spec.doc_id != only:
            continue
        out[spec.doc_id] = ingest_pdf(
            settings.corpus_dir / spec.file, spec.doc_id, spec.title, get_profile(spec.profile)
        )
    return out


# --------------------------------------------------------------------------- #
# CLI: inspect what the extractor produced
# --------------------------------------------------------------------------- #

def _main() -> None:
    ap = argparse.ArgumentParser(description="Show extracted, cleaned page text.")
    ap.add_argument("--doc", help="doc_id from corpus.json (default: all)")
    ap.add_argument("--page", type=int, help="only this 1-based page")
    ap.add_argument("--lines", action="store_true", help="show line kinds (heading/caption/table/body)")
    args = ap.parse_args()

    # Inspection needs only the corpus location, not an embedding model, so we avoid
    # Settings.from_env() (which insists EMBEDDING_MODEL is set).
    from .config import REPO_ROOT, _load_dotenv, _path
    _load_dotenv()
    corpus_dir = _path("CORPUS_DIR", REPO_ROOT / "data" / "corpus")
    for spec in load_manifest(corpus_dir):
        if args.doc and spec.doc_id != args.doc:
            continue
        pages = ingest_pdf(corpus_dir / spec.file, spec.doc_id, spec.title, get_profile(spec.profile))
        for rec in pages:
            if args.page and rec.page_number != args.page:
                continue
            print(f"\n=== {rec.doc_id} | page {rec.page_number} | section: {rec.section_heading!r} ===")
            if args.lines:
                for ln in rec.lines:
                    print(f"  [{ln.kind:7}{'¶' if ln.new_para else ' '}] {ln.text[:110]}")
            else:
                print(rec.text)


if __name__ == "__main__":
    _main()
