"""
api/routers/uploads.py

  GET /uploads         — List current user's unexpired uploads
  DELETE /uploads/{id} — Delete a specific upload (owner only)
  cleanup_expired()    — Called by APScheduler daily
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from api.models.user import User
from api.models.upload import UserUpload
from api.routers.auth import get_current_user
import api.services.storage as storage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/uploads", tags=["uploads"])


class UploadListItem(UserUpload):
    """UserUpload extended with a short-lived presigned download URL."""
    presigned_url: Optional[str] = None


@router.get("", response_model=list[UploadListItem])
async def list_uploads(current_user: User = Depends(get_current_user)):
    """Return the authenticated user's active (non-expired) uploads."""
    uploads = storage.get_uploads_for_user(current_user.id)
    items: list[UploadListItem] = []
    for u in uploads:
        url = storage.get_presigned_url(u.bucket_path)
        items.append(UploadListItem(**u.model_dump(), presigned_url=url))
    return items


@router.delete("/{upload_id}", status_code=204)
async def delete_upload(upload_id: str, current_user: User = Depends(get_current_user)):
    """Delete an upload by ID. Only the owner can delete their own files."""
    uploads = storage.get_uploads_for_user(current_user.id)
    match = next((u for u in uploads if u.upload_id == upload_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="Upload not found")
    storage.delete_file(match.bucket_path)
    storage.delete_upload_metadata(upload_id)


# ── Scheduled cleanup ─────────────────────────────────────────────────────────


async def cleanup_expired() -> None:
    """
    Delete files from storage and DB rows where expires_at has passed.
    Called daily by APScheduler (registered in main.py).
    """
    expired = storage.get_expired_uploads()
    if not expired:
        logger.info("Cleanup: no expired uploads.")
        return

    logger.info("Cleanup: found %d expired upload(s).", len(expired))
    for upload in expired:
        storage.delete_file(upload.bucket_path)
        storage.delete_upload_metadata(upload.upload_id)
        logger.info("Cleaned up expired upload: %s (%s)", upload.upload_id, upload.filename)
