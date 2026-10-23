"""Retrieval-layer tests.  Fully offline: they use the hashing embedder.

    EMBEDDING_BACKEND=hashing pytest backend/tests/test_retrieval.py -q

These test *structure and fidelity* (what the chunker preserves, whether citations
resolve), not semantic ranking quality; ranking quality depends on the real
embedding model and belongs in the evaluation set (requirement L3).
"""
from __future__ import annotations

import dataclasses
import json
import re

import pytest

from backend.retrieval.build_index import build
from backend.retrieval.config import Settings
from backend.retrieval.ingest import ingest_pdf, load_manifest, normalize_for_match
from backend.retrieval.profiles import get_profile
from backend.retrieval.search import IndexError_, Retriever

CIRC, SBI, OMB = "rbi_circular_2017", "sbi_policy_2021", "rbi_ios_2026"


@pytest.fixture(scope="module")
def settings(tmp_path_factory) -> Settings:
    import os

    os.environ["EMBEDDING_BACKEND"] = "hashing"
    base = Settings.from_env()
    return dataclasses.replace(base, index_dir=tmp_path_factory.mktemp("index"))


@pytest.fixture(scope="module")
def manifest(settings):
    return build(settings, log=lambda *_: None)


@pytest.fixture(scope="module")
def retriever(settings, manifest) -> Retriever:
    return Retriever(settings)


@pytest.fixture(scope="module")
def pages(settings):
    out = {}
    for s in load_manifest(settings.corpus_dir):
        out[s.doc_id] = ingest_pdf(settings.corpus_dir / s.file, s.doc_id, s.title, get_profile(s.profile))
    return out


# ------------------------------------------------------------------ ingest
def test_page_counts_and_numbering(pages):
    assert [p.page_number for p in pages[CIRC]] == list(range(1, 10))
    assert pages[SBI][0].page_number == 5            # cover + index (pp. 1-4) skipped
    assert len(pages[OMB]) == 24


def test_no_broken_encoding_or_furniture(pages):
    for recs in pages.values():
        for r in recs:
            assert not re.search(r"[\u0900-\u097F\ufffd]", r.text), (r.doc_id, r.page_number)
            assert "continuation sheet" not in r.text
            assert not re.search(r"\d+ \| P a g e", r.text)


def test_headings_and_section_heading(pages):
    heads = [h for r in pages[CIRC] for h in r.headings]
    assert "Burden of Proof" in heads
    assert pages[CIRC][3].section_heading.startswith("(b) Limited Liability")   # page 4


def test_tables_extracted_as_rows(pages):
    p5 = pages[CIRC][4].text
    assert "Within 3 working days | Zero liability" in p5
    assert "Beyond 7 working days | As per bank’s Board approved policy" in p5


# ------------------------------------------------------------------ chunk fidelity (section 3 problem)
def _chunk(retriever, cid):
    return retriever.by_id[cid]


def test_table2_travels_with_its_definitions(retriever):
    c = _chunk(retriever, f"{CIRC}:table-2")
    assert "Beyond 7 working days" in c["text"] and "Board approved policy" in c["text"]
    assert "excluding the date of receiving the communication" in c["text"]       # how working days count
    assert "third party breaches" in c["lead_in"]                                  # para 8 introduces it
    assert "8" in c["clause_ids"]
    ref_ids = " ".join(c["refs"])
    assert f"{CIRC}:6" in ref_ids and f"{CIRC}:7(i)" in ref_ids and f"{CIRC}:table-1" in ref_ids


def test_table1_keeps_the_beyond_seven_days_exception(retriever):
    c = _chunk(retriever, f"{CIRC}:table-1")
    assert "Further, if the delay in reporting is beyond seven working days" in c["text"]
    assert "25,000" in c["text"] and "5,000" in c["text"]


def test_negligence_and_third_party_limbs_kept_with_lead_in(retriever):
    c = _chunk(retriever, f"{CIRC}:7(i)")
    assert {"7(i)", "7(ii)"} <= set(c["clause_ids"])
    assert "negligence" in c["text"] and "four to seven working days" in c["text"]
    assert "A customer shall be liable" in c["lead_in"]


def test_sbi_beyond_seven_days_is_full_liability(retriever):
    c = _chunk(retriever, f"{SBI}:compensation-policy:table-2")
    assert "Full liability" in c["text"]
    assert c["part"] == "Compensation Policy"
    assert any("4.6.2" in r for r in retriever.by_id[f"{SBI}:compensation-policy:4.6.3"]["refs"])


def test_burden_of_proof_is_not_merged_away(retriever):
    hits = [c for c in retriever.chunks if c["doc_id"] == CIRC and "12" in c["clause_ids"]]
    assert hits and "burden of proving customer liability" in hits[0]["text"]


def test_ombudsman_letter_i_is_a_definition_not_a_roman_numeral(retriever):
    res = retriever.get_clause(OMB, "3(1)(i)")
    assert res and res[0].supporting_text.startswith("(i) “Deficiency in service”")


def test_chunk_invariants(retriever, pages):
    ids = set()
    page_text = {d: normalize_for_match(" ".join(r.text for r in recs)).replace(" ", "") for d, recs in pages.items()}
    for c in retriever.chunks:
        assert c["chunk_id"] not in ids
        ids.add(c["chunk_id"])
        assert c["text"].strip() and c["page_start"] <= c["page_end"]
        assert all(c["page_start"] <= s["page"] <= c["page_end"] for s in c["page_spans"])
        assert all(0 <= s["start"] <= s["end"] <= len(c["text"]) for s in c["clause_spans"])
        assert c["vector_id"] == retriever.chunks.index(c)
        if c["chunk_type"] != "table":
            # Source wording must be findable in the PDF text.  Checked per paragraph:
            # a clause that contains a table is split around it (the table is its own chunk).
            for para in c["text"].split("\n"):
                if para.strip():
                    assert normalize_for_match(para).replace(" ", "") in page_text[c["doc_id"]], (c["chunk_id"], para[:60])
    for c in retriever.chunks:              # reference graph is symmetric and resolvable
        for r in c["refs"]:
            assert r in retriever.by_id and c["chunk_id"] in retriever.by_id[r]["referenced_by"]


# ------------------------------------------------------------------ search
def test_search_returns_citable_passages(retriever):
    res = retriever.search("customer shared payment credentials negligence bears entire loss", k=3)
    assert any(p.doc_id == CIRC and p.supporting_clause == "7(i)" for p in res)
    for p in res:
        assert p.supporting_text in p.text
        assert retriever.quote_resolves(p.chunk_id, p.supporting_text)
        assert p.page_start <= p.page <= p.page_end
        assert p.citation.startswith(p.short_title)


def test_quote_verification_rejects_fabrication(retriever):
    assert retriever.quote_resolves(f"{CIRC}:7(i)", "the customer will bear the entire loss until he reports")
    assert not retriever.quote_resolves(f"{CIRC}:7(i)", "the customer bears no liability for shared OTPs")
    assert not retriever.quote_resolves("no-such-chunk", "anything")


def test_expand_refs_pulls_in_unmentioned_provisions(retriever):
    plain = retriever.search("customer shared payment credentials negligence bears entire loss", k=2)
    expanded = retriever.search("customer shared payment credentials negligence bears entire loss", k=2, expand_refs=True)
    assert len(expanded) > len(plain)
    assert any(p.via and p.chunk_type == "table" for p in expanded)


def test_filters(retriever):
    res = retriever.search("liability of customer", k=5, doc_ids=[SBI])
    assert res and all(p.doc_id == SBI for p in res)
    res = retriever.search("liability of customer", k=5, chunk_types=["table"])
    assert res and all(p.chunk_type == "table" for p in res)
    assert retriever.search("   ") == []


def test_get_clause_variants(retriever):
    a = retriever.get_clause(CIRC, "7(ii)")
    b = retriever.get_clause(CIRC, "para 7 (II)")
    assert a and a[0].chunk_id == b[0].chunk_id
    assert a[0].supporting_text.startswith("(ii) In cases where the responsibility")
    assert "Within 3 working days" in retriever.get_clause(CIRC, "Table 2")[0].text
    assert retriever.get_clause(SBI, "4.6.2.b (ii)")[0].supporting_clause == "4.6.2(b)(ii)"
    assert retriever.get_clause(CIRC, "99") == []


# ------------------------------------------------------------------ build pipeline (D2, L1)
def test_build_is_idempotent_and_incremental(settings, manifest):
    again = build(settings, log=lambda *_: None)
    assert again.get("skipped") is True
    part = build(settings, only=SBI, log=lambda *_: None)
    assert part["stats"]["chunks_reused"] > 0
    assert part["stats"]["chunks_embedded"] == sum(1 for d in part["documents"] if d["doc_id"] == SBI for _ in range(d["chunks"]))
    assert part["n_chunks"] == manifest["n_chunks"]


def test_manifest_records_freshness(manifest):
    assert manifest["built_at"] and manifest["n_chunks"] > 100
    assert {d["doc_id"] for d in manifest["documents"]} == {CIRC, SBI, OMB}
    assert all(len(d["sha256"]) == 64 for d in manifest["documents"])
    assert manifest["stats"]["pipeline_chunks_per_second"] > 0
    assert manifest["unresolved_xrefs"] == []


def test_tampered_index_is_rejected(settings, manifest):
    p = settings.index_dir / "chunks.jsonl"
    original = p.read_text(encoding="utf-8")
    try:
        p.write_text(original + json.dumps({"x": 1}) + "\n", encoding="utf-8")
        with pytest.raises(IndexError_):
            Retriever(settings)
    finally:
        p.write_text(original, encoding="utf-8")


# ------------------------------------------------------------------ config (D4)
def test_model_identifier_must_come_from_environment(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "sentence-transformers")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="EMBEDDING_MODEL"):
        Settings.from_env()


# ------------------------------------------------------------------ regressions found on a real Windows run
def test_manifest_is_utf8_and_stale_cp1252_manifest_does_not_crash(settings, manifest):
    raw = (settings.index_dir / "manifest.json").read_bytes()
    raw.decode("utf-8")                                    # must be valid UTF-8 whatever the OS locale
    # a manifest written by the old build on Windows (cp1252: curly apostrophe = byte 0x92)
    (settings.index_dir / "manifest.json").write_bytes('{"x": "bank’s"}'.encode("cp1252"))
    out = build(settings, log=lambda *_: None)             # must rebuild, not raise UnicodeDecodeError
    assert out.get("skipped") is not True and out["n_chunks"] > 100
    Retriever(settings)                                    # and the result must load


def test_truncation_report_names_the_offending_chunks(settings):
    from backend.retrieval.embeddings import HashingEmbedder

    class Tiny(HashingEmbedder):
        max_seq_length = 20
        name = "tiny-test"
        def token_lengths(self, texts):
            return [len(t.split()) for t in texts]

    msgs: list[str] = []
    s2 = dataclasses.replace(settings, index_dir=settings.index_dir.parent / "trunc")
    build(s2, embedder=Tiny(), log=msgs.append)
    warn = [m for m in msgs if m.startswith("WARNING") and "token limit" in m]
    assert warn and "tok)" in warn[0] and ":" in warn[0]


# ------------------------------------------------------------------ regressions from the first real-model run
def test_clause_number_is_not_a_sentence_boundary():
    from backend.retrieval.chunker import sentence_spans

    t = "14. The instructions supersede some. Yours faithfully"
    assert [t[a:b] for a, b in sentence_spans(t)] == ["14. The instructions supersede some.", "Yours faithfully"]


def test_form_dot_leaders_are_removed(retriever):
    assert all("...." not in c["text"] and "____" not in c["text"] for c in retriever.chunks)
    assert max(c["word_count"] for c in retriever.chunks if c["chunk_type"] != "table") <= 160


def test_supporting_text_never_a_bare_clause_number(retriever):
    for q in ("customer shared an OTP", "bank customer relations policy", "ombudsman complaint"):
        for p in retriever.search(q, k=8):
            assert len(p.supporting_text.split()) > 2, (q, p.chunk_id, p.supporting_text)


def test_header_mode_changes_embedded_text_but_not_cited_text(settings, tmp_path):
    light = dataclasses.replace(settings, embed_header_mode="light", index_dir=tmp_path / "light")
    full = dataclasses.replace(settings, embed_header_mode="full", index_dir=tmp_path / "full")
    build(light, log=lambda *_: None)
    build(full, log=lambda *_: None)
    r_light, r_full = Retriever(light), Retriever(full)
    a, b = r_full.by_id[f"{CIRC}:table-2"], r_light.by_id[f"{CIRC}:table-2"]
    assert a["text"] == b["text"]                         # what is cited is identical
    assert "RBI Circular on Limiting Customer Liability" in a["embed_text"]
    assert "RBI Circular on Limiting Customer Liability" not in b["embed_text"]


def test_probe_runs_and_reports_out_of_corpus_scores(retriever):
    from backend.retrieval.probe import run

    res = run(retriever)
    assert len(res["probes"]) >= 10 and 0 <= res["summary"]["mrr"] <= 1
    s = res["summary"]
    assert {"in_corpus_top_score_min", "out_of_corpus_top_score_max", "threshold_separable", "margin"} <= set(s)
    assert len(res["out_of_corpus"]) >= 8        # includes near-domain misses, the hard cases for F7


def test_line_text_never_fuses_words_across_a_wrap_or_gap():
    from backend.retrieval.ingest import _line_text

    # one line object that actually contains a wrapped line: second span sits on a lower baseline
    wrapped = [
        {"text": "customer protection", "bbox": (72, 100, 190, 112), "size": 11.0},
        {"text": "and considering", "bbox": (72, 114, 160, 126), "size": 11.0},
    ]
    assert _line_text(wrapped) == "customer protection and considering"
    # same baseline, space only expressed as positioning
    gapped = [
        {"text": "dated", "bbox": (72, 100, 100, 112), "size": 11.0},
        {"text": "July 1, 2015", "bbox": (104, 100, 160, 112), "size": 11.0},
    ]
    assert _line_text(gapped) == "dated July 1, 2015"
    # kerning / bold first letter inside a word must NOT gain a space
    kerned = [
        {"text": "C", "bbox": (72, 100, 80, 112), "size": 11.0},
        {"text": "ustomer", "bbox": (80.2, 100, 120, 112), "size": 11.0},
    ]
    assert _line_text(kerned) == "Customer"
    # spaces already present are not doubled
    assert _line_text([{"text": "a ", "bbox": (0, 0, 5, 10), "size": 11.0}, {"text": "b", "bbox": (20, 0, 25, 10), "size": 11.0}]) == "a b"


def test_no_fused_words_in_any_chunk(retriever):
    import re

    fused = re.compile(r"\b(?:protection|unauthorised|dated|liability|customer)(?:and|transactions|July|the)\b")
    assert not [c["chunk_id"] for c in retriever.chunks if fused.search(c["text"])]


# ------------------------------------------------------------------ extraction integrity (fused words seen on one Windows machine)
def test_clean_corpus_passes_the_extraction_integrity_check(pages):
    assert [n for recs in pages.values() for r in recs for n in r.notes] == []


def test_integrity_check_detects_fused_words(settings, monkeypatch):
    from backend.retrieval import ingest as ing

    monkeypatch.setattr(ing, "_line_text", lambda spans: "".join(sp["text"] for sp in spans).replace(" ", "", 1))
    spec = next(s for s in load_manifest(settings.corpus_dir) if s.doc_id == CIRC)
    recs = ing.ingest_pdf(settings.corpus_dir / spec.file, spec.doc_id, spec.title, get_profile(spec.profile))
    assert sum(1 for r in recs if r.notes) >= 7        # of 9 pages


def test_get_clause_quotes_only_that_clause(retriever):
    p = retriever.get_clause(CIRC, "2")[0]
    assert p.supporting_text.startswith("2. With the increased thrust")
    assert "Please refer to our circular" not in p.supporting_text
