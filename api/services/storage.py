"""
api/services/storage.py

Supabase-backed file storage + user_uploads metadata table.

Graceful fallback: if SUPABASE_URL / SUPABASE_SERVICE_KEY are not set,
all operations are no-ops and the caller falls back to local persistence.

Supabase table (run once in the Supabase SQL editor):
──────────────────────────────────────────────────────
  create table if not exists user_uploads (
    id            uuid primary key default gen_random_uuid(),
    upload_id     text not null unique,
    user_id       text not null,
    filename      text not null,
    file_size_bytes bigint not null default 0,
    bucket_path   text not null,
    program_id    text,
    uploaded_at   timestamptz not null default now(),
    expires_at    timestamptz not null
  );
  create index if not exists user_uploads_user_id_idx on user_uploads(user_id);
  create index if not exists user_uploads_expires_at_idx on user_uploads(expires_at);
──────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from api.models.upload import UserUpload

logger = logging.getLogger(__name__)

_BUCKET = os.environ.get("SUPABASE_BUCKET", "schemabio-uploads")
_UPLOAD_TTL_DAYS = int(os.environ.get("UPLOAD_TTL_DAYS", "30"))

_TABLE = "user_uploads"


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


# ── File storage ──────────────────────────────────────────────────────────────


def upload_file(user_id: str, upload_id: str, filename: str, content: bytes) -> Optional[str]:
    """
    Upload bytes to Supabase Storage.
    Returns the bucket_path on success, None on failure / not configured.
    Path format: {user_id}/{upload_id}/{filename}
    """
    sb = _client()
    if sb is None:
        return None

    bucket_path = f"{user_id}/{upload_id}/{filename}"
    try:
        sb.storage.from_(_BUCKET).upload(
            path=bucket_path,
            file=content,
            file_options={"content-type": "application/octet-stream", "upsert": "true"},
        )
        logger.info("Uploaded %s to bucket %s", bucket_path, _BUCKET)
        return bucket_path
    except Exception as exc:
        logger.error("Storage upload failed for %s: %s", bucket_path, exc)
        return None


def get_presigned_url(bucket_path: str, expires_in: int = 3600) -> Optional[str]:
    """Return a signed URL valid for `expires_in` seconds."""
    sb = _client()
    if sb is None:
        return None
    try:
        resp = sb.storage.from_(_BUCKET).create_signed_url(bucket_path, expires_in)
        return resp.get("signedURL") or resp.get("signedUrl")
    except Exception as exc:
        logger.error("Presigned URL failed for %s: %s", bucket_path, exc)
        return None


def delete_file(bucket_path: str) -> bool:
    """Delete a file from the bucket. Returns True on success."""
    sb = _client()
    if sb is None:
        return False
    try:
        sb.storage.from_(_BUCKET).remove([bucket_path])
        logger.info("Deleted %s from bucket %s", bucket_path, _BUCKET)
        return True
    except Exception as exc:
        logger.error("Storage delete failed for %s: %s", bucket_path, exc)
        return False


def download_file(bucket_path: str) -> Optional[bytes]:
    """Download file bytes from the bucket (for re-ingestion)."""
    sb = _client()
    if sb is None:
        return None
    try:
        return sb.storage.from_(_BUCKET).download(bucket_path)
    except Exception as exc:
        logger.error("Storage download failed for %s: %s", bucket_path, exc)
        return None


# ── Metadata DB ───────────────────────────────────────────────────────────────


def save_upload_metadata(upload: UserUpload) -> bool:
    """Insert a row into user_uploads. Returns True on success."""
    sb = _client()
    if sb is None:
        return False
    try:
        sb.table(_TABLE).insert(upload.to_db_row()).execute()
        return True
    except Exception as exc:
        logger.error("DB insert failed for upload %s: %s", upload.upload_id, exc)
        return False


def get_uploads_for_user(user_id: str) -> list[UserUpload]:
    """Return all non-expired uploads for a user, most recent first."""
    sb = _client()
    if sb is None:
        return []
    try:
        now_iso = datetime.now(timezone.utc).isoformat()
        resp = (
            sb.table(_TABLE)
            .select("*")
            .eq("user_id", user_id)
            .gt("expires_at", now_iso)
            .order("uploaded_at", desc=True)
            .execute()
        )
        return [UserUpload.from_db_row(row) for row in (resp.data or [])]
    except Exception as exc:
        logger.error("DB query failed for user %s: %s", user_id, exc)
        return []


def get_expired_uploads() -> list[UserUpload]:
    """Return all rows where expires_at has passed."""
    sb = _client()
    if sb is None:
        return []
    try:
        now_iso = datetime.now(timezone.utc).isoformat()
        resp = sb.table(_TABLE).select("*").lte("expires_at", now_iso).execute()
        return [UserUpload.from_db_row(row) for row in (resp.data or [])]
    except Exception as exc:
        logger.error("DB query for expired uploads failed: %s", exc)
        return []


def delete_upload_metadata(upload_id: str) -> bool:
    """Delete a metadata row by upload_id."""
    sb = _client()
    if sb is None:
        return False
    try:
        sb.table(_TABLE).delete().eq("upload_id", upload_id).execute()
        return True
    except Exception as exc:
        logger.error("DB delete failed for upload %s: %s", upload_id, exc)
        return False


# ── TTL helper ────────────────────────────────────────────────────────────────


def make_expires_at() -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=_UPLOAD_TTL_DAYS)
