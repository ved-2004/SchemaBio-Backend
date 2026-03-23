"""
api/services/rag_meta_db.py

Tracks when each RAG collection was last indexed.
Uses Supabase `rag_index_metadata` table with fallback to a local JSON file.

Supabase table (run in SQL editor):
──────────────────────────────────────────────────────────────────────
  create table if not exists rag_index_metadata (
    collection_name  text primary key,
    last_indexed_at  timestamptz not null default now(),
    document_count   int not null default 0
  );
──────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_TABLE = "rag_index_metadata"
_LOCAL_PATH = Path(__file__).parent.parent / "data" / "rag_index_meta.json"

# Re-index if data is older than this
STALE_DAYS = 7


def _client():
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    if not (url and key):
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception:
        return None


def _read_local() -> dict[str, dict]:
    """Read the local JSON fallback file."""
    if _LOCAL_PATH.exists():
        try:
            return json.loads(_LOCAL_PATH.read_text())
        except Exception:
            pass
    return {}


def _write_local(data: dict[str, dict]) -> None:
    _LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LOCAL_PATH.write_text(json.dumps(data, default=str))


def is_fresh(collection_name: str) -> bool:
    """Return True if the collection was indexed within the last STALE_DAYS."""
    ts = get_last_indexed(collection_name)
    if ts is None:
        return False
    cutoff = datetime.now(timezone.utc) - timedelta(days=STALE_DAYS)
    return ts > cutoff


def get_last_indexed(collection_name: str) -> Optional[datetime]:
    """Return the last_indexed_at timestamp for a collection, or None."""
    sb = _client()
    if sb is not None:
        try:
            resp = (
                sb.table(_TABLE)
                .select("last_indexed_at")
                .eq("collection_name", collection_name)
                .single()
                .execute()
            )
            if resp.data:
                raw = resp.data["last_indexed_at"]
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception as exc:
            logger.debug("rag_meta_db Supabase read failed: %s", exc)

    # Fallback to local JSON
    local = _read_local()
    entry = local.get(collection_name)
    if entry and "last_indexed_at" in entry:
        return datetime.fromisoformat(entry["last_indexed_at"].replace("Z", "+00:00"))
    return None


def mark_indexed(collection_name: str, document_count: int) -> None:
    """Record that a collection was just indexed."""
    now_iso = datetime.now(timezone.utc).isoformat()

    sb = _client()
    if sb is not None:
        try:
            sb.table(_TABLE).upsert(
                {
                    "collection_name": collection_name,
                    "last_indexed_at": now_iso,
                    "document_count": document_count,
                },
                on_conflict="collection_name",
            ).execute()
            logger.info("rag_meta_db: marked %s indexed (%d docs)", collection_name, document_count)
        except Exception as exc:
            logger.warning("rag_meta_db Supabase write failed: %s — using local fallback", exc)

    # Always write local fallback too (cheap insurance)
    local = _read_local()
    local[collection_name] = {
        "last_indexed_at": now_iso,
        "document_count": document_count,
    }
    _write_local(local)
