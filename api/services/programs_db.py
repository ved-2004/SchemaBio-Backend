"""
api/services/programs_db.py

Supabase-backed persistence for DrugProgram data (the `programs` table).

Graceful fallback: if Supabase is not configured, all operations use an
in-memory dict so the app still works in development.

Run once in Supabase SQL Editor to create the table:
──────────────────────────────────────────────────────────────────────
  create table if not exists programs (
    id           uuid primary key default gen_random_uuid(),
    program_id   text not null unique,
    user_id      text,
    data         jsonb not null default '{}'::jsonb,
    created_at   timestamptz not null default now(),
    updated_at   timestamptz not null default now()
  );
  create index if not exists programs_program_id_idx on programs(program_id);
  create index if not exists programs_user_id_idx   on programs(user_id);
──────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_TABLE = "programs"

# In-memory fallback when Supabase is not configured
_fallback_store: dict[str, dict] = {}


def _client():
    """Return a Supabase client, or None if not configured."""
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    if not (url and key):
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception as exc:
        logger.warning("Supabase client init failed: %s", exc)
        return None


def is_configured() -> bool:
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"))


# ── CRUD operations ──────────────────────────────────────────────────────────


def save_program(program_id: str, data: dict, user_id: Optional[str] = None) -> bool:
    """
    Upsert a program into the Supabase `programs` table.
    Falls back to in-memory dict if Supabase is not configured.
    Returns True on success.
    """
    pid = program_id.upper()

    sb = _client()
    if sb is None:
        _fallback_store[pid] = data
        logger.info("Program %s saved to in-memory store (Supabase not configured)", pid)
        return True

    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        sb.table(_TABLE).upsert(
            {
                "program_id": pid,
                "user_id": user_id,
                "data": json.loads(json.dumps(data, default=str)),
                "updated_at": now_iso,
            },
            on_conflict="program_id",
        ).execute()
        logger.info("Program %s upserted to Supabase", pid)
        return True
    except Exception as exc:
        logger.error("save_program failed for %s: %s", pid, exc)
        # Fall back to memory so the request doesn't lose data entirely
        _fallback_store[pid] = data
        return False


def get_program(program_id: str) -> Optional[dict]:
    """
    Fetch a program by its program_id.
    Returns the program data dict or None if not found.
    """
    pid = program_id.upper()

    sb = _client()
    if sb is None:
        return _fallback_store.get(pid)

    try:
        resp = (
            sb.table(_TABLE)
            .select("data")
            .eq("program_id", pid)
            .single()
            .execute()
        )
        if resp.data:
            return resp.data["data"]
        return None
    except Exception as exc:
        logger.error("get_program failed for %s: %s", pid, exc)
        # Try in-memory fallback
        return _fallback_store.get(pid)


def get_user_programs(user_id: str) -> list[dict]:
    """
    List all programs belonging to a user, most recent first.
    Returns a list of program data dicts.
    """
    sb = _client()
    if sb is None:
        # In-memory: return all programs that have matching user_id in data
        return [
            prog for prog in _fallback_store.values()
            if prog.get("user_id") == user_id
        ]

    try:
        resp = (
            sb.table(_TABLE)
            .select("program_id, data, created_at, updated_at")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .execute()
        )
        return [row["data"] for row in (resp.data or [])]
    except Exception as exc:
        logger.error("get_user_programs failed for user %s: %s", user_id, exc)
        return []
