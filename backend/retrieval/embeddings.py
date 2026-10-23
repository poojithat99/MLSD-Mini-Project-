"""Embedding backends behind one small interface.

* ``sentence-transformers``: the real backend.  The model identifier comes from
  the EMBEDDING_MODEL environment variable (requirement D4: never hardcoded).
  Runs locally, no API cost.
* ``hashing``: a deterministic, dependency-free bag-of-words/bigram embedder.  It
  is NOT a semantic model; it exists so the pipeline, tests and CI can run
  offline, and as a lexical baseline.  Results obtained with it say nothing about
  the quality of the real model.

Vectors are L2-normalised float32, so inner product == cosine similarity (the
FAISS index uses inner product).
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol, Sequence

import numpy as np

from .config import Settings


class Embedder(Protocol):
    name: str          # identifies the vector space; changing it invalidates cached vectors
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _l2(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


# --------------------------------------------------------------------------- #
class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str | None, batch_size: int,
                 query_prefix: str = "", doc_prefix: str = ""):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install sentence-transformers (see requirements.txt)") from exc
        self._model = SentenceTransformer(model_name, device=device)
        override = os.environ.get("EMBEDDING_MAX_SEQ_LENGTH", "").strip()
        if override:
            self._model.max_seq_length = int(override)
        self.batch_size = batch_size
        self.query_prefix, self.doc_prefix = query_prefix, doc_prefix
        get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())
        self.max_seq_length = int(self._model.max_seq_length)
        # prefixes change the vector space, so they are part of the identity
        self.name = f"st:{model_name}|q={query_prefix}|d={doc_prefix}|L={self.max_seq_length}"

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        """Token count of each text as the model will see it (before silent truncation)."""
        tok = self._model.tokenizer
        return [len(tok.encode(self.doc_prefix + t, add_special_tokens=True, truncation=False)) for t in texts]

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(
            [self.doc_prefix + t for t in texts], batch_size=self.batch_size,
            normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        v = self._model.encode(
            [self.query_prefix + text], normalize_embeddings=True,
            show_progress_bar=False, convert_to_numpy=True,
        )
        return np.asarray(v, dtype=np.float32)[0]


# --------------------------------------------------------------------------- #
_STOP = frozenset(
    "a an the of to in on for and or by with as at is are was were be been it its this that these those "
    "from which who whom whose than then there such any all not no nor but if so shall may must will can "
    "has have had do does did their his her he she they them we our you your".split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")


class HashingEmbedder:
    """Feature hashing of unigrams + bigrams, sublinear tf (log1p), L2-normalised."""

    def __init__(self, dim: int = 4096):
        self.dim = dim
        self.name = f"hashing:{dim}"

    def _vec(self, text: str) -> np.ndarray:
        toks = [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0
        return np.log1p(v)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return _l2(np.stack([self._vec(t) for t in texts])) if len(texts) else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return _l2(self._vec(text)[None, :])[0]


# --------------------------------------------------------------------------- #
def get_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "hashing":
        return HashingEmbedder()
    assert settings.embedding_model  # enforced by Settings.from_env
    return SentenceTransformerEmbedder(
        settings.embedding_model, settings.embedding_device, settings.embedding_batch_size,
        settings.embedding_query_prefix, settings.embedding_doc_prefix,
    )
