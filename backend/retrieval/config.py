"""Runtime configuration for the retrieval layer.

Requirement D4: no credentials, endpoints or model identifiers live in the
repository. Everything environment-specific is read from environment variables
(optionally loaded from a local ``.env`` file; see ``.env.example``).

Nothing here has a model-name default on purpose: if EMBEDDING_MODEL is missing
while the sentence-transformers backend is selected, we fail loudly rather than
silently using a model nobody chose.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    """Load REPO_ROOT/.env if python-dotenv is installed. Existing env vars win."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional
        return
    load_dotenv(REPO_ROOT / ".env", override=False)


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _header_mode() -> str:
    mode = os.environ.get("EMBED_HEADER_MODE", "light").strip().lower()
    if mode not in {"full", "light"}:
        raise ValueError(f"EMBED_HEADER_MODE must be 'full' or 'light', got {mode!r}")
    return mode


def _path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    p = Path(raw) if raw else default
    return p if p.is_absolute() else (REPO_ROOT / p)


@dataclass(frozen=True)
class Settings:
    # --- locations ---------------------------------------------------------
    corpus_dir: Path
    index_dir: Path
    # --- embeddings --------------------------------------------------------
    embedding_backend: str          # "sentence-transformers" | "hashing"
    embedding_model: str | None     # required for sentence-transformers
    embedding_device: str | None    # e.g. "cpu", "cuda"; None = library default
    embedding_query_prefix: str     # some models want "query: " / "passage: "
    embedding_doc_prefix: str
    embedding_batch_size: int
    # --- chunking ----------------------------------------------------------
    chunk_max_words: int            # soft cap on words per chunk (see chunker.py)
    chunk_lead_max_words: int       # cap on parent "lead-in" repeated into children
    table_min_words: int            # drop form-field tables smaller than this
    embed_header_mode: str          # "full" | "light" (see ChunkerConfig.header_mode)
    # --- search ------------------------------------------------------------
    search_top_k: int
    search_min_score: float         # 0 disables the floor
    search_max_related: int         # cap on cross-reference expansions
    query_cache_size: int

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        backend = os.environ.get("EMBEDDING_BACKEND", "sentence-transformers").strip().lower()
        if backend not in {"sentence-transformers", "hashing"}:
            raise ValueError(
                f"EMBEDDING_BACKEND must be 'sentence-transformers' or 'hashing', got {backend!r}"
            )
        model = (os.environ.get("EMBEDDING_MODEL") or "").strip() or None
        if backend == "sentence-transformers" and model is None:
            raise RuntimeError(
                "EMBEDDING_MODEL is not set. Copy .env.example to .env and set it "
                "(model identifiers are deliberately not hardcoded; requirement D4). "
                "For an offline smoke test you can set EMBEDDING_BACKEND=hashing."
            )
        return cls(
            corpus_dir=_path("CORPUS_DIR", REPO_ROOT / "data" / "corpus"),
            index_dir=_path("INDEX_DIR", REPO_ROOT / "data" / "index"),
            embedding_backend=backend,
            embedding_model=model,
            embedding_device=(os.environ.get("EMBEDDING_DEVICE") or "").strip() or None,
            embedding_query_prefix=os.environ.get("EMBEDDING_QUERY_PREFIX", ""),
            embedding_doc_prefix=os.environ.get("EMBEDDING_DOC_PREFIX", ""),
            embedding_batch_size=_int("EMBEDDING_BATCH_SIZE", 32),
            chunk_max_words=_int("CHUNK_MAX_WORDS", 140),
            chunk_lead_max_words=_int("CHUNK_LEAD_MAX_WORDS", 70),
            table_min_words=_int("TABLE_MIN_WORDS", 10),
            embed_header_mode=_header_mode(),
            search_top_k=_int("SEARCH_TOP_K", 5),
            search_min_score=_float("SEARCH_MIN_SCORE", 0.0),
            search_max_related=_int("SEARCH_MAX_RELATED", 4),
            query_cache_size=_int("QUERY_CACHE_SIZE", 512),
        )
