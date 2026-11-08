"""Prompts are versioned files (requirement L2).

Each file in ``backend/prompts/`` starts with a header line

    version: analysis_v1

and the rest is the prompt.  The version is read from the file, not from the
code, and is recorded in every DisputeResult, so a result can always be traced
to the exact prompt that produced it.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


@lru_cache(maxsize=None)
def load_prompt(name: str) -> tuple[str, str]:
    """Return ``(text, version)`` for ``backend/prompts/<name>.txt``."""
    raw = (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")
    header, _, body = raw.partition("\n")
    key, _, version = header.partition(":")
    if key.strip().lower() != "version" or not version.strip():
        raise ValueError(f"{name}.txt must start with a 'version: <id>' line")
    return body.strip(), version.strip()
