"""
api/monitoring.py

Structured request logging with correlation IDs.

Provides:
  - request_id_var  — a contextvar holding the current request's UUID
  - RequestIDFilter — a logging filter that injects request_id into log records
  - setup_logging() — applies the filter to the root logger
"""
from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar

# ── Context variable ──────────────────────────────────────────────────────────

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


# ── Logging filter ────────────────────────────────────────────────────────────

class RequestIDFilter(logging.Filter):
    """Inject the current request_id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get("-")  # type: ignore[attr-defined]
        return True


# ── Setup helper ──────────────────────────────────────────────────────────────

def setup_logging() -> None:
    """
    Reconfigure the root logger to include request_id in every message.
    Call once at application startup (inside lifespan or module-level).
    """
    root = logging.getLogger()

    # Add the filter to every existing handler
    rid_filter = RequestIDFilter()
    for handler in root.handlers:
        handler.addFilter(rid_filter)

    # Update formatters to include request_id
    fmt = "%(asctime)s %(name)s %(levelname)s [%(request_id)s] %(message)s"
    formatter = logging.Formatter(fmt)
    for handler in root.handlers:
        handler.setFormatter(formatter)


def generate_request_id() -> str:
    """Return a new UUID4 hex string for use as a request correlation ID."""
    return uuid.uuid4().hex
