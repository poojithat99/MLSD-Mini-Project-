"""Step 3: split extracted pages into meaningful, citable passages.

The problem this module exists to solve (project brief, section 3)
-------------------------------------------------------------------
The liability rule is a table indexed by *who was at fault* and *how many working
days elapsed*; it means nothing without (a) the paragraphs that define "third
party breach" and "customer negligence", (b) the sentence that says how working
days are counted, and (c) the exception that sends delays beyond seven days to
the bank's board-approved policy.  A fixed-size chunker separates all of these.

Strategy
--------
1. **Parse structure, don't slice characters.**  Lines are folded into a tree of
   clauses using numbered markers (``7.``, ``4.6.2``, ``(1)``, ``(a)``, ``(ii)``).
   Marker candidates are validated by sequence (``(ii)`` must follow ``(i)``) so
   a wrapped line that merely *starts* with "(a)" or "10" is not mistaken for a
   clause.  Unrecognised lines are continuation text, so provisos such as
   "Further, ..." or "However, ..." stay attached to the rule they qualify.
2. **Largest subtree that fits.**  A clause is kept whole, with all its
   sub-clauses, if it fits ``max_words``.  Only an oversize clause is split, at
   sub-clause boundaries; its introductory text is then repeated into every
   child as ``lead_in`` so that "(ii) ..." never loses the "in the following
   cases:" it depends on.  Tiny sibling sub-clauses are packed together.
3. **A table is one chunk with its context**: caption + rows + the note that
   follows it (where "working days" counting and the beyond-7-days exception
   live) + ``lead_in`` = the clause that introduced the table.  Rows are also
   rendered as sentences for embedding ("Time taken ...: Within 3 working days;
   Customer's liability: Zero liability") because a row like "Within 3 working
   days | Zero liability" embeds poorly on its own.
4. **Cross-references are data.**  "paragraph 6 (ii) and paragraph 7 (ii)",
   "paragraphs 6 to 9", "Table 2" are extracted and resolved to chunk ids, with
   the reverse links.  search.py uses them to pull in a definition that a query
   never mentioned, which is exactly the failure described in section 5.
5. **Every chunk remembers where its words came from**: ``page_spans`` and
   ``clause_spans`` map character offsets back to PDF pages and to sub-clause
   ids, so a citation can say "para 7(ii), p. 4", not just "para 7".

``text`` is the source wording (whitespace-normalised; line-wrap hyphens
rejoined; table cells joined with " | ").  Context that was *added* for
embedding (heading breadcrumb, lead-in, sentence-form table rows) lives in
``embed_text`` and never in ``text``, so what we cite is what the PDF says.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Iterable, Iterator

from .ingest import (
    KIND_BODY, KIND_CAPTION, KIND_HEADING, KIND_TABLE,
    DocSpec, Line, PageRecord, clean_text,
)
from .profiles import DocProfile

# --------------------------------------------------------------------------- #
# configuration and output record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChunkerConfig:
    max_words: int = 140
    lead_max_words: int = 70
    table_min_words: int = 10
    #: tables may exceed max_words by this factor before rows are split, because
    #: splitting a table from its note is worse than a long chunk
    table_slack: float = 1.6
    #: sub-clauses are packed together until a group reaches this share of max_words
    pack_target: float = 0.6
    #: top-level clauses smaller than this are merged with the following sibling
    min_clause_words: int = 12
    #: what is prepended to each chunk before embedding.
    #:   "full"  -> document title | part | heading breadcrumb | clause label
    #:   "light" -> part | heading breadcrumb | clause label (no document title)
    #: The document title is identical for every chunk of a document and swamps short chunks.
    #: Measured on the probe set (MiniLM): "light" MRR 0.717 / hit@5 0.90 vs "full" 0.68 / 0.80.
    #: Changing it needs a rebuild.
    header_mode: str = "light"


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    short_title: str
    issuer: str
    version: str
    chunk_type: str                 # clause | text | table
    part: str                       # e.g. "Compensation Policy" (SBI); "" otherwise
    clause: str                     # display label: "7", "3(1)(a)-(e)", "Table 2"
    clause_ids: list[str]           # every clause id fully contained in `text`
    clause_spans: list[dict]        # [{"clause": "7(ii)", "start": 140, "end": 330}]
    section_heading: str
    heading_path: list[str]
    page_start: int
    page_end: int
    page_spans: list[dict]          # [{"page": 3, "start": 0, "end": 210}]
    lead_in: str                    # parent text this chunk depends on ("" if none)
    text: str                       # source wording; what citations quote
    embed_text: str = ""            # text actually embedded (adds breadcrumb etc.)
    table_no: str = ""              # "2" for "Table 2"
    part_no: int = 1                # >1 when one long clause is split by sentences
    part_of: int = 1
    xrefs: list[str] = field(default_factory=list)          # raw normalised ids
    refs: list[str] = field(default_factory=list)           # resolved chunk ids
    referenced_by: list[str] = field(default_factory=list)  # inverse of refs
    word_count: int = 0
    content_hash: str = ""          # sha1 of embed_text; drives incremental re-embedding
    vector_id: int = -1             # row in the FAISS index / vectors.npy
    citation_word: str = "clause"   # "para" for circular/bank policy, "clause" for the Scheme

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# tiny text helpers
# --------------------------------------------------------------------------- #

def words(s: str) -> int:
    return len(s.split())


_ABBREV = ("No", "Rs", "Sr", "Dr", "Mr", "Mrs", "vs", "etc", "viz", "Sl", "approx", "Ltd")
_SENT_END = re.compile(
    r"(?<=[.;:?!])(?<!\b\d\.)(?<!\b\d\d\.)(?<!\b\d\d\d\.)(?<!\bNo\.)(?<!\bRs\.)(?<!\bSr\.)(?<!\bSl\.)(?<!\bDr\.)(?<!\bvs\.)\s+(?=[A-Z(\u201c\"'\u2018\u2022])"
)
_DEPENDENT = re.compile(
    r"^(Further|However|Provided|Except|Unless|Notwithstanding|Explanation|Note|Such|This|These|Thereafter)\b"
)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) offsets of sentences; newlines are always boundaries.

    Sentences that open with a dependent word ("Further, ...", "However, ...")
    are glued to the previous sentence so a rule is never separated from its
    proviso.
    """
    raw: list[tuple[int, int]] = []
    pos = 0
    for line in text.split("\n"):
        if line.strip():
            offset = pos
            last = 0
            for m in _SENT_END.finditer(line):
                raw.append((offset + last, offset + m.start()))
                last = m.end()
            raw.append((offset + last, offset + len(line.rstrip())))
        pos += len(line) + 1
    out: list[tuple[int, int]] = []
    for a, b in raw:
        if out and _DEPENDENT.match(text[a:b]) and "\n" not in text[out[-1][1]:a]:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return [(a, b) for a, b in out if text[a:b].strip()]


def trim_tail_sentences(text: str, max_words: int) -> str:
    """Keep the last sentences of ``text`` totalling <= max_words (never cut mid-sentence
    unless a single sentence is itself too long)."""
    if words(text) <= max_words:
        return text
    spans = sentence_spans(text)
    kept: list[str] = []
    total = 0
    for a, b in reversed(spans):
        w = words(text[a:b])
        if total + w > max_words and kept:
            break
        kept.append(text[a:b])
        total += w
    return " ".join(reversed(kept))


def page_at(page_spans: list[dict], pos: int) -> int:
    for sp in page_spans:
        if sp["start"] <= pos < sp["end"]:
            return sp["page"]
    return page_spans[-1]["page"] if page_spans else 0


def clause_at(clause_spans: list[dict], pos: int) -> str:
    """Innermost clause id whose span contains ``pos`` ('' if none)."""
    best, best_len = "", 10**9
    for sp in clause_spans:
        if sp["start"] <= pos < sp["end"] and sp["end"] - sp["start"] < best_len:
            best, best_len = sp["clause"], sp["end"] - sp["start"]
    return best


# --------------------------------------------------------------------------- #
# marker recognition
# --------------------------------------------------------------------------- #

_ROMAN = re.compile(r"^(?:x{0,3})(?:ix|iv|v?i{0,3})$")
_UP = r"(?=[A-Z\u201c\"(\u2018'])"
_N_MULTI = re.compile(rf"^(\d{{1,3}}(?:\.\d{{1,3}})+)\.?\s+{_UP}(.*)$")
_N_SINGLE = re.compile(rf"^(\d{{1,3}})\.\s+{_UP}(.*)$")
_N_ONLY = re.compile(r"^(\d{1,3})\.\s*$")
_P_NUM = re.compile(r"^\((\d{1,2})\)\s*(.*)$")
_P_ALPHA = re.compile(r"^\(([a-z]{1,4})\)\s*(.*)$")
_P_ALPHA_CLOSE = re.compile(r"^([a-z]{1,4})\)\s+(.*)$")

RANK_P, RANK_A, RANK_R = 10, 11, 12


def _roman_to_int(tok: str) -> int:
    vals = {"i": 1, "v": 5, "x": 10}
    total, prev = 0, 0
    for ch in reversed(tok):
        v = vals[ch]
        total += -v if v < prev else v
        prev = max(prev, v)
    return total


@dataclass
class Marker:
    kind: str        # "N" dotted number | "P" (1) | "A" (a) | "R" (i)
    label: str       # normalised: "7", "4.6.2", "1", "a", "ii"
    rest: str        # text after the marker
    marker: str      # as printed, e.g. "7." or "(ii)"
    rank: int
    idx: int = 0     # ordinal for sequence checking
    dots: int = 0


def _match_n(text: str) -> Marker | None:
    for rx in (_N_MULTI, _N_SINGLE):
        m = rx.match(text)
        if m:
            label = m.group(1)
            dots = label.count(".")
            printed = text[: m.start(2)].strip()
            return Marker("N", label, m.group(2).strip(), printed, dots, int(label.split(".")[-1]), dots)
    m = _N_ONLY.match(text)
    if m:
        return Marker("N", m.group(1), "", text.strip(), 0, int(m.group(1)), 0)
    return None


def _paren_candidates(text: str) -> list[Marker]:
    """All readings of a "(x)"-style marker; the caller picks by sequence."""
    out: list[Marker] = []
    m = _P_NUM.match(text)
    if m:
        out.append(Marker("P", m.group(1), m.group(2).strip(), f"({m.group(1)})", RANK_P, int(m.group(1))))
        return out
    m = _P_ALPHA.match(text)
    closing_only = False
    if not m:
        m = _P_ALPHA_CLOSE.match(text)
        closing_only = True
    if not m:
        return out
    tok, rest = m.group(1), m.group(2).strip()
    printed = f"{tok})" if closing_only else f"({tok})"
    if len(tok) == 1:
        out.append(Marker("A", tok, rest, printed, RANK_A, ord(tok) - 96))
    if _ROMAN.match(tok) and tok:
        out.append(Marker("R", tok, rest, printed, RANK_R, _roman_to_int(tok)))
    return out


# --------------------------------------------------------------------------- #
# clause tree
# --------------------------------------------------------------------------- #

@dataclass
class Seg:
    text: str
    page: int
    new_para: bool = False


@dataclass(eq=False)
class Node:
    kind: str                                   # root | clause | implicit | table
    marker: str = ""
    label: str = ""
    cid: str = ""
    rank: int = -1
    title: str = ""
    segs: list[Seg] = field(default_factory=list)
    children: list["Node"] = field(default_factory=list)
    parent: "Node | None" = None
    part: str = ""
    heading_path: tuple[str, ...] = ()
    seq: dict[str, int] = field(default_factory=dict)   # last child index per marker kind
    # table-only
    caption: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    note: list[Seg] = field(default_factory=list)
    table_no: str = ""
    lead_node: "Node | None" = None
    page: int = 0

    # -- derived text ------------------------------------------------------- #
    def own_text(self) -> str:
        return _join_segs(self.segs)

    def subtree_words(self) -> int:
        n = words(self.own_text())
        for c in self.children:
            if c.kind != "table":
                n += c.subtree_words()
        return n


def _join_segs(segs: list[Seg]) -> str:
    out = ""
    for s in segs:
        if not out:
            out = s.text
        elif s.new_para:
            out += "\n" + s.text
        elif out.endswith("-"):
            out += s.text          # soft-wrapped hyphenation / compound ("Non-" "Banking")
        else:
            out += " " + s.text
    return out


class _Parser:
    def __init__(self, spec: DocSpec, profile: DocProfile, warnings: list[str]):
        self.spec, self.profile, self.warn = spec, profile, warnings
        self.root = Node("root")
        self.stack: list[Node] = []
        self.headings: list[tuple[int, str]] = []
        self.part = ""
        self.cur_table: Node | None = None
        self.pending_caption: list[str] = []
        self.pending_chapter: str | None = None
        self.n0_last: int | None = None
        self.chapter_rx = re.compile(profile.chapter_pattern) if profile.chapter_pattern else None
        self.caption_page = 0

    # -- helpers ------------------------------------------------------------- #
    def _heading_path(self) -> tuple[str, ...]:
        path = [h for _, h in self.headings]
        path += [f"{n.marker} {n.title}".strip() for n in self.stack if n.title]
        return tuple(path)

    def _container(self, page: int) -> Node:
        if self.stack:
            return self.stack[-1]
        node = Node("implicit", part=self.part, heading_path=self._heading_path())
        self.root.children.append(node)
        node.parent = self.root
        self.stack.append(node)
        return node

    def _new_node(self, mk: Marker, title: str, first_text: str, page: int) -> Node:
        while self.stack and self.stack[-1].rank >= mk.rank:
            self.stack.pop()
        parent = self.stack[-1] if self.stack else self.root
        if mk.kind == "N":
            cid = mk.label
        else:
            base = parent.cid if parent.kind == "clause" else ""
            cid = f"{base}({mk.label})"
        node = Node(
            "clause", marker=mk.marker, label=mk.label, cid=cid, rank=mk.rank,
            title=title, part=self.part, heading_path=self._heading_path(),
        )
        node.parent = parent
        parent.children.append(node)
        if first_text:
            node.segs.append(Seg(first_text, page, True))
        self.stack.append(node)
        self.cur_table = None
        return node

    # -- marker acceptance ---------------------------------------------------- #
    def _accept_n(self, mk: Marker, bold: bool) -> bool:
        if mk.dots > 0:
            return True
        if self.n0_last is None:
            ok = mk.idx <= 3 or bold
        else:
            ok = mk.idx == self.n0_last + 1 or (mk.idx == 1) or (bold and mk.idx > self.n0_last)
        if ok:
            self.n0_last = mk.idx
        return ok

    def _pick_paren(self, cands: list[Marker]) -> Marker | None:
        """Choose among readings by sequence: (i) after (h) is a letter, else a numeral."""
        # parent that would result after popping to each candidate's rank
        for mk in cands:
            stack = list(self.stack)
            while stack and stack[-1].rank >= mk.rank:
                stack.pop()
            parent = stack[-1] if stack else self.root
            last = parent.seq.get(mk.kind)
            if mk.idx == 1 or (last is not None and mk.idx == last + 1):   # idx 1 = (re)start of a list
                return mk
        return None

    def _commit_paren(self, mk: Marker) -> None:
        stack = list(self.stack)
        while stack and stack[-1].rank >= mk.rank:
            stack.pop()
        parent = stack[-1] if stack else self.root
        parent.seq[mk.kind] = mk.idx
        # a new sibling restarts the numbering of anything deeper
        parent.seq = {k: v for k, v in parent.seq.items() if k == mk.kind or _KIND_RANK[k] < mk.rank}

    # -- line handlers -------------------------------------------------------- #
    def feed(self, ln: Line) -> None:
        if ln.kind == KIND_CAPTION:
            self.pending_caption.append(ln.text)
            self.caption_page = ln.page
            return
        if ln.kind == KIND_TABLE:
            self._table(ln)
            return
        if self.pending_caption:                       # caption that had no table after it
            for t in self.pending_caption:
                self._body(Line(t, self.caption_page, 0, 0, 0, 0, False), as_text=True)
            self.pending_caption = []
        if ln.kind == KIND_HEADING:
            self._heading(ln)
        else:
            self._body(ln)

    def _table(self, ln: Line) -> None:
        parent = self._container(ln.page)
        node = Node("table", part=self.part, heading_path=self._heading_path(),
                    caption=self.pending_caption, rows=ln.rows or [], page=ln.page)
        m = re.match(r"^Table\s+(\d+)", node.caption[0], re.IGNORECASE) if node.caption else None
        node.table_no = m.group(1) if m else ""
        node.lead_node, node.parent = parent, parent
        parent.children.append(node)
        self.pending_caption = []
        self.cur_table = node

    def _heading(self, ln: Line) -> None:
        text = ln.text
        mk = _match_n(text)
        if mk and self._accept_n(mk, bold=True):
            self._new_node(mk, mk.rest, f"{mk.marker} {mk.rest}".strip(), ln.page)
            return
        if not self.profile.headings_close_clauses and not self.profile.match_part(text):
            self._body(ln)                             # bold used for emphasis, not structure
            return
        # a bold "(b) Title" is a clause marker when the sequence says so (SBI
        # sub-heads), otherwise a sub-heading (circular: "(a) Zero Liability ...")
        cands = _paren_candidates(text)
        if cands and self.stack:
            picked = self._pick_paren(cands)
            if picked is not None:
                self._commit_paren(picked)
                self._new_node(picked, picked.rest, f"{picked.marker} {picked.rest}".strip(), ln.page)
                return
        part_name = self.profile.match_part(text)
        self.stack, self.cur_table = [], None
        if part_name:
            self.part, self.headings, self.n0_last, self.pending_chapter = part_name, [(0, part_name)], None, None
            self.root.seq = {}
            return
        if self.chapter_rx and self.chapter_rx.match(text):
            self.pending_chapter = text
            self.headings = [(0, text)]
            return
        if self.pending_chapter:
            self.headings = [(0, f"{self.pending_chapter} - {text}")]
            self.pending_chapter = None
            return
        level = 2 if re.match(r"^\([a-z]\)\s", text) else 1
        while self.headings and self.headings[-1][0] >= level:
            self.headings.pop()
        self.headings.append((level, text))

    def _body(self, ln: Line, as_text: bool = False) -> None:
        text = ln.text
        if not as_text:
            mk = _match_n(text)
            if mk and self._accept_n(mk, bold=False):
                self._new_node(mk, "", f"{mk.marker} {mk.rest}".strip(), ln.page)
                return
            cands = _paren_candidates(text)
            if cands:
                picked = self._pick_paren(cands)
                if picked is not None and (self.stack or picked.idx == 1):
                    self._commit_paren(picked)
                    self._new_node(picked, "", f"{picked.marker} {picked.rest}".strip(), ln.page)
                    return
                if cands and self.stack:
                    self.warn.append(f"{self.spec.doc_id} p{ln.page}: ignored marker-like line {text[:50]!r}")
        if self.cur_table is not None:
            self.cur_table.note.append(Seg(text, ln.page, ln.new_para))
            return
        self._container(ln.page).segs.append(Seg(text, ln.page, ln.new_para))


_KIND_RANK = {"N": 0, "P": RANK_P, "A": RANK_A, "R": RANK_R}


def parse_document(pages: list[PageRecord], spec: DocSpec, profile: DocProfile) -> tuple[Node, list[str]]:
    warnings: list[str] = []
    p = _Parser(spec, profile, warnings)
    for rec in pages:
        for ln in rec.lines:
            p.feed(ln)
    for t in p.pending_caption:
        p._body(Line(t, p.caption_page, 0, 0, 0, 0, False), as_text=True)
    return p.root, warnings


# --------------------------------------------------------------------------- #
# text assembly with offset bookkeeping
# --------------------------------------------------------------------------- #

class _Builder:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.pos = 0
        self.pages: list[dict] = []

    def add(self, text: str, page: int, sep: str) -> None:
        if self.parts:
            if sep == " " and self.parts[-1].endswith("-"):
                sep = ""
            self.parts.append(sep)
            self.pos += len(sep)
        start = self.pos
        self.parts.append(text)
        self.pos += len(text)
        if self.pages and self.pages[-1]["page"] == page:
            self.pages[-1]["end"] = self.pos
        else:
            self.pages.append({"page": page, "start": start, "end": self.pos})

    def text(self) -> str:
        return "".join(self.parts)


def _write_node(b: _Builder, node: Node, spans: list[dict]) -> None:
    start = b.pos if not b.parts else b.pos + 1
    span = {"clause": node.cid, "start": start, "end": start}
    if node.cid:
        spans.append(span)            # appended before children => parents precede descendants
    for i, seg in enumerate(node.segs):
        b.add(seg.text, seg.page, "\n" if (i == 0 or seg.new_para) else " ")
    for c in node.children:
        if c.kind != "table":
            _write_node(b, c, spans)
    span["end"] = b.pos


def _slice_pages(pages: list[dict], a: int, b: int) -> list[dict]:
    out = []
    for p in pages:
        lo, hi = max(a, p["start"]), min(b, p["end"])
        if lo < hi:
            out.append({"page": p["page"], "start": lo - a, "end": hi - a})
    return out or [{"page": pages[-1]["page"], "start": 0, "end": b - a}]


def _tables_in(node: Node) -> Iterator[Node]:
    for c in node.children:
        if c.kind == "table":
            yield c
        else:
            yield from _tables_in(c)


def range_label(first: str, last: str) -> str:
    if first == last:
        return first
    i = max(first.rfind("("), 0)
    if i and last.startswith(first[:i]):
        return f"{first}-{last[i:]}"
    return f"{first}-{last}"


# --------------------------------------------------------------------------- #
# cross references
# --------------------------------------------------------------------------- #

_REF_INTRO = re.compile(r"\b(?:paragraphs?|paras?\.?|sub-?clauses?|clauses?)\s+", re.IGNORECASE)
_REF_ITEM = re.compile(r"(\d{1,3}(?:\.\d{1,3})*)(?:\.([a-z])\b)?((?:\s*\([a-z0-9]{1,4}\))*)")
_REF_CONN = re.compile(r"\s*(,|and|&|or|to)\s+(?:(?:paragraphs?|paras?\.?|sub-?clauses?|clauses?)\s+)?", re.IGNORECASE)
_TABLE_REF = re.compile(r"\bTable\s+(\d+)\b", re.IGNORECASE)


def _norm_ref(core: str, letter: str | None, tail: str) -> str:
    out = core
    if letter:
        out += f"({letter})"
    out += "".join(f"({t})" for t in re.findall(r"\(([a-z0-9]{1,4})\)", tail or ""))
    return out


def extract_xrefs(text: str, own_table_no: str = "") -> list[str]:
    """Normalised references found in ``text``: clause ids ("7(ii)", "4.6.2(b)(ii)")
    and tables ("table:2").  Integer/dotted ranges ("paragraphs 6 to 9") are expanded."""
    found: list[str] = []
    for intro in _REF_INTRO.finditer(text):
        pos, prev = intro.end(), None
        connector = ""
        while True:
            im = _REF_ITEM.match(text, pos)
            if not im:
                break
            ref = _norm_ref(*im.groups())
            simple = im.group(2) is None and not im.group(3)
            if connector.lower() == "to" and prev and simple and re.fullmatch(r"[\d.]+", prev) and re.fullmatch(r"[\d.]+", ref):
                a, b = prev.split("."), ref.split(".")
                if len(a) == len(b) and a[:-1] == b[:-1] and int(a[-1]) < int(b[-1]) <= int(a[-1]) + 12:
                    found += [".".join(a[:-1] + [str(n)]) for n in range(int(a[-1]) + 1, int(b[-1]))]
            found.append(ref)
            prev = ref if simple else None
            pos = im.end()
            cm = _REF_CONN.match(text, pos)
            if not cm:
                break
            connector, pos = cm.group(1), cm.end()
    for m in _TABLE_REF.finditer(text):
        if m.group(1) != own_table_no:
            found.append(f"table:{m.group(1)}")
    seen, out = set(), []
    for r in found:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def resolve_xrefs(chunks: list[Chunk]) -> list[str]:
    """Fill ``refs`` / ``referenced_by``; return unresolved references (for the build report)."""
    by_cid: dict[tuple[str, str, str], list[int]] = {}
    by_table: dict[tuple[str, str, str], int] = {}
    for i, c in enumerate(chunks):
        for cid in c.clause_ids:
            by_cid.setdefault((c.doc_id, c.part, cid), []).append(i)
        if c.table_no:
            by_table[(c.doc_id, c.part, c.table_no)] = i
    unresolved: list[str] = []
    for i, c in enumerate(chunks):
        targets: list[int] = []
        for ref in c.xrefs:
            hits: list[int] = []
            if ref.startswith("table:"):
                j = by_table.get((c.doc_id, c.part, ref[6:]))
                hits = [j] if j is not None else []
            else:
                hits = by_cid.get((c.doc_id, c.part, ref), [])
                if not hits:                           # ref to a parent clause: take its descendants
                    for (d, p, cid), idxs in by_cid.items():
                        if d == c.doc_id and p == c.part and (cid.startswith(ref + "(") or cid.startswith(ref + ".")):
                            hits = hits + idxs
            hits = [j for j in dict.fromkeys(hits) if j != i][:3]
            if not hits:
                unresolved.append(f"{c.chunk_id} -> {ref}")
            targets += hits
        c.refs = [chunks[j].chunk_id for j in dict.fromkeys(targets)]
    index = {c.chunk_id: c for c in chunks}
    for c in chunks:
        for r in c.refs:
            if c.chunk_id not in index[r].referenced_by:
                index[r].referenced_by.append(c.chunk_id)
    return unresolved


# --------------------------------------------------------------------------- #
# chunk construction
# --------------------------------------------------------------------------- #

def _table_nl(rows: list[list[str]]) -> str:
    """Rows as sentences: 'Header A: cell; Header B: cell.' (for embedding only)."""
    if len(rows) < 2:
        return "\n".join(" | ".join(r) for r in rows)
    head = rows[0]
    lines = []
    for r in rows[1:]:
        pairs = [f"{h}: {c}" for h, c in zip(head, r) if c]
        lines.append("; ".join(pairs) + ".")
    return "\n".join(lines)


class Chunker:
    def __init__(self, spec: DocSpec, profile: DocProfile, cfg: ChunkerConfig):
        self.spec, self.profile, self.cfg = spec, profile, cfg
        self.chunks: list[Chunk] = []
        self._ids: dict[str, int] = {}

    # -- public --------------------------------------------------------------- #
    def run(self, root: Node) -> list[Chunk]:
        kids = [c for c in root.children if c.kind != "table"]
        i = 0
        while i < len(kids):
            node = kids[i]
            # merge a tiny top-level clause into its successor (no lonely stubs)
            group = [node]
            while (
                words(" ".join(_flat_text(n) for n in group)) < self.cfg.min_clause_words
                and i + 1 < len(kids) and kids[i + 1].part == node.part
                and words(" ".join(_flat_text(n) for n in group)) + kids[i + 1].subtree_words() <= self.cfg.max_words
            ):
                i += 1
                group.append(kids[i])
            if len(group) == 1:
                self._emit(node, "")
            else:
                self._group_chunk(group, "")
                for n in group:
                    self._emit_tables(n)
            i += 1
        for t in root.children:
            if t.kind == "table":
                self._table_chunks(t)
        return self.chunks

    # -- emission ------------------------------------------------------------- #
    def _emit(self, node: Node, parent_lead: str) -> None:
        cfg = self.cfg
        kids = [c for c in node.children if c.kind != "table"]
        direct_tables = [c for c in node.children if c.kind == "table"]
        own = node.own_text()
        total = node.subtree_words()

        if total <= cfg.max_words:
            stub = (not kids) and direct_tables and words(own) <= cfg.lead_max_words
            if total and not stub:
                self._group_chunk([node], parent_lead)
            for t in _tables_in(node):
                self._table_chunks(t, extra_cid=node.cid if stub and t.lead_node is node else "")
            return

        lead = trim_tail_sentences(own, cfg.lead_max_words) if own and (words(own) <= cfg.lead_max_words or kids) else ""
        if not kids or words(own) > cfg.lead_max_words:
            self._text_parts(node, parent_lead)        # substantial own text -> its own chunk(s)
            lead = trim_tail_sentences(own, cfg.lead_max_words) if kids else ""
        group: list[Node] = []

        def flush() -> None:
            if group:
                self._group_chunk(list(group), lead or parent_lead)
                for n in group:
                    self._emit_tables(n)
                group.clear()

        is_list = bool(kids) and all(k.rank >= RANK_P for k in kids)
        for k in kids:
            w = k.subtree_words()
            if w > cfg.max_words:
                flush()
                self._emit(k, lead or parent_lead)
                continue
            gw = sum(g.subtree_words() for g in group)
            limit = cfg.pack_target * cfg.max_words if is_list else cfg.min_clause_words
            if group and (gw + w > cfg.max_words or gw >= limit):
                flush()
            group.append(k)
        flush()
        for t in direct_tables:
            self._table_chunks(t)

    def _emit_tables(self, node: Node) -> None:
        for t in _tables_in(node):
            self._table_chunks(t)

    # -- chunk makers ---------------------------------------------------------- #
    def _group_chunk(self, nodes: list[Node], lead: str) -> None:
        b, spans = _Builder(), []
        for n in nodes:
            _write_node(b, n, spans)
        text = b.text()
        if not text.strip():
            return
        ids = [s["clause"] for s in spans]
        first = nodes[0]
        label = range_label(first.cid, nodes[-1].cid) if first.cid else ""
        self._add(
            chunk_type="clause" if first.kind == "clause" else "text",
            node=first, clause=label, clause_ids=_ordered(ids), clause_spans=spans,
            text=text, pages=b.pages, lead_in=lead,
        )

    def _text_parts(self, node: Node, lead: str) -> None:
        """Split a node's *own* text by sentences into <= max_words parts."""
        b = _Builder()
        for i, seg in enumerate(node.segs):
            b.add(seg.text, seg.page, "\n" if (i == 0 or seg.new_para) else " ")
        full = b.text()
        sents = sentence_spans(full)
        parts: list[tuple[int, int]] = []
        cur_a = cur_b = None
        for a, e in sents:
            if cur_a is not None and words(full[cur_a:e]) > self.cfg.max_words:
                parts.append((cur_a, cur_b))
                cur_a = None
            if cur_a is None:
                cur_a = a
            cur_b = e
        if cur_a is not None:
            parts.append((cur_a, cur_b))
        for n, (a, e) in enumerate(parts, 1):
            seg_text = full[a:e]
            pages = _slice_pages(b.pages, a, e)
            self._add(
                chunk_type="clause" if node.kind == "clause" else "text", node=node,
                clause=node.cid, clause_ids=[node.cid] if node.cid else [],
                clause_spans=[{"clause": node.cid, "start": 0, "end": len(seg_text)}] if node.cid else [],
                text=seg_text, pages=pages, lead_in=lead, part_no=n, part_of=len(parts),
            )

    def _table_chunks(self, t: Node, extra_cid: str = "") -> None:
        cfg = self.cfg
        rows = [r for r in t.rows if any(r)]
        cap = list(t.caption)
        note = _join_segs(t.note)
        flat = " ".join(cap + [" ".join(r) for r in rows])
        if words(flat) < cfg.table_min_words:
            return                                   # form tick-boxes etc.
        lead_text = t.lead_node.own_text() if t.lead_node is not None else ""
        lead = trim_tail_sentences(lead_text, cfg.lead_max_words) if lead_text else ""
        header, body = rows[:1], rows[1:]
        groups: list[list[list[str]]] = []
        cur: list[list[str]] = []
        limit = int(cfg.max_words * cfg.table_slack)
        base = words(" ".join(cap)) + words(note) + words(" ".join(header[0] if header else []))
        for r in body or rows:
            if cur and base + words(" ".join(" ".join(x) for x in cur + [r])) > limit:
                groups.append(cur)
                cur = []
            cur.append(r)
        if cur:
            groups.append(cur)
        for gi, g in enumerate(groups, 1):
            use = (header + g) if body else g
            b = _Builder()
            for c in cap:
                b.add(c, t.page, "\n")
            for r in use:
                b.add(" | ".join(r), t.page, "\n")
            if note and gi == len(groups):
                for i, s in enumerate(t.note):
                    b.add(s.text, s.page, "\n" if (i == 0 or s.new_para) else " ")
            text = b.text()
            cids = _ordered(([extra_cid] if extra_cid else []) + ([t.lead_node.cid] if (t.lead_node is not None and t.lead_node.cid and extra_cid) else []))
            label = f"Table {t.table_no}" if t.table_no else "Table"
            if extra_cid:
                label += f" ({self.profile.citation_word} {extra_cid})"
            self._add(
                chunk_type="table", node=t, clause=label, clause_ids=cids,
                clause_spans=[{"clause": cids[0], "start": 0, "end": len(text)}] if cids else [],
                text=text, pages=b.pages, lead_in=lead, table_no=t.table_no,
                part_no=gi, part_of=len(groups),
                embed_body="\n".join(cap + [_table_nl(use)] + ([note] if note and gi == len(groups) else [])),
            )

    # -- common finishing ------------------------------------------------------ #
    def _add(
        self, *, chunk_type: str, node: Node, clause: str, clause_ids: list[str], clause_spans: list[dict],
        text: str, pages: list[dict], lead_in: str, part_no: int = 1, part_of: int = 1,
        table_no: str = "", embed_body: str = "",
    ) -> None:
        spec = self.spec
        heading_path = list(node.heading_path)
        section = heading_path[-1] if heading_path else node.part
        slug = (node.part.lower().replace(" ", "-") + ":" if node.part else "")
        base = f"{spec.doc_id}:{slug}" + (f"table-{table_no}" if chunk_type == "table" and table_no
                                             else (clause_ids[0] if clause_ids else f"p{pages[0]['page']}"))
        if part_of > 1:
            base += f"#{part_no}"
        n = self._ids.get(base, 0)
        self._ids[base] = n + 1
        chunk_id = base if n == 0 else f"{base}~{n + 1}"

        crumbs = " > ".join(heading_path[-3:])
        cite_label = f"{self.profile.citation_word} {clause}" if (clause and chunk_type != "table") else clause
        title = spec.short_title if self.cfg.header_mode == "full" else ""
        header = " | ".join(x for x in (title, node.part, crumbs, cite_label) if x)
        body = embed_body or text
        embed = header + "\n" + (f"Context: {lead_in}\n" if lead_in else "") + body
        chunk = Chunk(
            chunk_id=chunk_id, doc_id=spec.doc_id, doc_title=spec.title, short_title=spec.short_title,
            issuer=spec.issuer, version=spec.version, chunk_type=chunk_type, part=node.part,
            clause=clause, clause_ids=clause_ids, clause_spans=clause_spans,
            section_heading=section, heading_path=heading_path,
            page_start=pages[0]["page"], page_end=pages[-1]["page"], page_spans=pages,
            lead_in=lead_in, text=text, embed_text=embed, table_no=table_no,
            part_no=part_no, part_of=part_of, word_count=words(text),
            citation_word=self.profile.citation_word,
        )
        refs = extract_xrefs(text, own_table_no=table_no)
        for r in extract_xrefs(lead_in, own_table_no=table_no) if lead_in else []:
            if r not in refs:
                refs.append(r)
        own_ids = set(clause_ids)
        chunk.xrefs = [r for r in refs if r not in own_ids]
        self.chunks.append(chunk)


def _ordered(items: Iterable[str]) -> list[str]:
    return [x for x in dict.fromkeys(items) if x]


def _flat_text(node: Node) -> str:
    parts = [node.own_text()]
    parts += [_flat_text(c) for c in node.children if c.kind != "table"]
    return " ".join(p for p in parts if p)


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #

def chunk_document(
    pages: list[PageRecord], spec: DocSpec, profile: DocProfile, cfg: ChunkerConfig | None = None
) -> tuple[list[Chunk], list[str]]:
    """Parse + chunk one document.  Returns (chunks, warnings)."""
    cfg = cfg or ChunkerConfig()
    root, warnings = parse_document(pages, spec, profile)
    chunks = Chunker(spec, profile, cfg).run(root)
    for c in chunks:
        c.content_hash = ""        # filled by build_index once the embedder prefix is known
    return chunks, warnings
