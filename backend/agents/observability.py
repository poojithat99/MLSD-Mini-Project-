"""Tracing and structured logging shared by every agent module (requirement D5).

* ``run_context(run_id, correlation_id)`` stores the two IDs in context variables,
  so every span and log line inside a run carries them without threading them
  through every function call.
* ``span(name, **attrs)`` opens an OpenTelemetry span.  We depend only on
  ``opentelemetry-api``: until the service (Member 3) configures an SDK and an
  exporter (Jaeger/Tempo), spans are no-ops and cost nothing.
* ``log_event(event, **fields)`` writes one JSON line per event.
"""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from opentelemetry import trace

logger = logging.getLogger("dispute.agents")
tracer = trace.get_tracer("dispute.agents")

_run_id: ContextVar[str | None] = ContextVar("run_id", default=None)
_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


@contextmanager
def run_context(run_id: str, correlation_id: str) -> Iterator[None]:
    """Everything executed inside this block is attributed to ``run_id``."""
    t1 = _run_id.set(run_id)
    t2 = _correlation_id.set(correlation_id)
    try:
        yield
    finally:
        _run_id.reset(t1)
        _correlation_id.reset(t2)


def current_ids() -> dict[str, str | None]:
    return {"run_id": _run_id.get(), "correlation_id": _correlation_id.get()}


def log_event(event: str, **fields) -> None:
    """One structured (JSON) log line, always tagged with run_id and correlation_id."""
    record = {"event": event, **current_ids(), **fields}
    logger.info(json.dumps(record, default=str))


@contextmanager
def span(name: str, **attrs) -> Iterator[trace.Span]:
    """An OpenTelemetry span tagged with run_id / correlation_id plus ``attrs``."""
    with tracer.start_as_current_span(name) as s:
        for key, value in {**current_ids(), **attrs}.items():
            if value is not None:
                s.set_attribute(key, value if isinstance(value, (str, bool, int, float)) else str(value))
        yield s
