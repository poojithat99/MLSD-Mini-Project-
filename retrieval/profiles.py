"""Per-document parsing profiles.

The three corpus documents are laid out differently (letter-style circular,
three-policies-in-one bank document, chapter/clause statute-style scheme).
Rather than one brittle set of regexes, each document names a *profile* in
``data/corpus/corpus.json`` and the profile carries only what genuinely differs:
boilerplate to drop, pages to skip, and how the top-level "part"/"chapter"
boundaries are recognised.

Generic structure (bold headings, numbered clauses, (a)/(i) sub-clauses, tables)
is handled once, in ingest.py and chunker.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DocProfile:
    name: str
    #: word used when rendering citations: "para 7(ii)" vs "clause 15(2)"
    citation_word: str = "clause"
    #: 1-based PDF pages that carry no citable content (cover, table of contents)
    skip_pages: frozenset[int] = frozenset()
    #: regexes (matched against a cleaned line) for letterhead/boilerplate lines
    drop_patterns: tuple[str, ...] = ()
    #: ((regex, canonical name), ...) matched against bold heading text; a match
    #: starts a new "part" (resets clause numbering; becomes metadata `part`)
    part_patterns: tuple[tuple[str, str], ...] = ()
    #: regex for a bare chapter label such as "CHAPTER II" (next heading = its title)
    chapter_pattern: str | None = None
    #: True  -> an un-numbered bold heading ends the open clause (circular, scheme:
    #:          headings separate sections).
    #: False -> only numbered clauses define structure; stray bold lines are text
    #:          (SBI policy: bold is also used for emphasis, whole paragraphs, formulas).
    headings_close_clauses: bool = True

    def compiled_drop(self) -> list[re.Pattern]:
        return [re.compile(p, re.IGNORECASE) for p in self.drop_patterns]

    def match_part(self, heading: str) -> str | None:
        for pattern, name in self.part_patterns:
            if re.search(pattern, heading, re.IGNORECASE):
                return name
        return None


PROFILES: dict[str, DocProfile] = {
    # 9-page RBI letter-style circular. Page 1 has a bilingual letterhead whose
    # Hindi half has a broken font encoding (extracts as U+FFFD); dropped in ingest.
    "rbi_circular": DocProfile(
        name="rbi_circular",
        citation_word="para",
        drop_patterns=(
            r"^_{5,}$",
            r"^Department of Banking Regulation",
            r"^Tel(ephone)?\s*No",
            r"^Fax\s*No",
        ),
    ),
    # SBI file = three policies concatenated, each numbering its paras from 1.
    # PDF pages 1-4 are the cover and the index (their lines would out-rank the
    # real clauses for any query mentioning "limiting liability").
    "bank_policy": DocProfile(
        name="bank_policy",
        citation_word="para",
        skip_pages=frozenset({1, 2, 3, 4}),
        part_patterns=(
            (r"^CUSTOMER RIGHTS POLICY", "Customer Rights Policy"),
            (r"^CUSTOMER GRIEVANCES? REDRESSAL POLICY", "Customer Grievances Redressal Policy"),
            (r"^COMPENSATION POLICY", "Compensation Policy"),
        ),
        headings_close_clauses=False,
    ),
    # Statute-style: CHAPTER I..N, "N. Title" clauses, (1) / (a) / (i) sub-clauses.
    "rbi_scheme": DocProfile(
        name="rbi_scheme",
        citation_word="clause",
        chapter_pattern=r"^CHAPTER\s+[IVXLC]+$",
    ),
}


def get_profile(name: str) -> DocProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown profile {name!r}; known: {sorted(PROFILES)}") from exc
