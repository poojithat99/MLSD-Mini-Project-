"""Analysis agent (skeleton: one model call, no tools; the tool loop comes next)."""
from __future__ import annotations

import json

from .llm import parse_json_object
from .state import AnalysisDraft, DisputeInput


def run_analysis(run, request: DisputeInput, strategy: str | None = None, verdict=None) -> AnalysisDraft:
    messages = [
        {"role": "system", "content": "Return a JSON liability draft."},
        {"role": "user", "content": json.dumps(request.model_dump(mode="json"))},
    ]
    resp = run.call_llm("analysis", messages)
    return AnalysisDraft.model_validate(parse_json_object(resp.content))
