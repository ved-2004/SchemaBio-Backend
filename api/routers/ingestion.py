"""
routers/ingestion.py

SchemaBio ingestion API routes.
- POST /api/upload-and-parse — upload files, run ingestion, return IngestionResponse
- POST /api/program-state   — optional: return program state from provided test input
"""

from __future__ import annotations
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException

from api.schemas.ingestion import IngestionResponse
from api.ingestion.service import run_ingestion
from api.models.user import User
from api.models.upload import UserUpload
from api.routers.auth import get_current_user, get_optional_user
import api.services.storage as storage
import api.services.runs_db as runs_db
import api.services.programs_db as programs_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["ingestion"])


# Local fallback directory (used when Supabase is not configured)
_LOCAL_UPLOADS_DIR = Path(__file__).resolve().parent.parent / "uploads"


def _persist_local(program_id: str, path: Path) -> None:
    """Fallback: save a file to the local uploads directory."""
    dest_dir = _LOCAL_UPLOADS_DIR / program_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    try:
        dest.write_bytes(path.read_bytes())
        logger.debug("Local fallback persist: %s", dest)
    except Exception as exc:
        logger.warning("Local persist failed for %s: %s", path.name, exc)


async def _store_uploaded_files(
    program_id: str,
    paths: list[Path],
    file_sizes: dict[str, int],
    user: Optional[User],
) -> list[str]:
    """
    Upload each file to Supabase Storage and save metadata to the DB.
    Falls back to local disk if Supabase is not configured.
    """
    user_id = user.id if user else None
    logger.info(
        "_store_uploaded_files: storage.is_configured()=%s user_id=%s paths=%s",
        storage.is_configured(),
        user_id,
        [str(p) for p in paths],
    )
    if not storage.is_configured():
        # Supabase not configured — development fallback to local disk
        logger.warning("Supabase not configured — writing files to local disk instead")
        for p in paths:
            _persist_local(program_id, p)
        return []
    if user_id is None:
        # Should not happen now that the endpoint requires auth, but guard just in case
        logger.error("_store_uploaded_files called with no authenticated user — skipping storage")
        return []

    upload_id = program_id  # reuse program_id as the upload session identifier
    expires_at = storage.make_expires_at()
    now = datetime.now(timezone.utc)

    saved_upload_ids: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        content = path.read_bytes()
        bucket_path = storage.upload_file(user_id, upload_id, path.name, content)
        if bucket_path is None:
            _persist_local(program_id, path)
            continue

        uid = f"{upload_id}_{path.stem}"
        upload = UserUpload(
            upload_id=uid,
            user_id=user_id,
            filename=path.name,
            file_size_bytes=file_sizes.get(path.name, len(content)),
            bucket_path=bucket_path,
            program_id=program_id,
            uploaded_at=now,
            expires_at=expires_at,
        )
        storage.save_upload_metadata(upload)
        saved_upload_ids.append(uid)

    return saved_upload_ids


@router.post("/upload-and-parse", response_model=IngestionResponse)
async def upload_and_parse(
    files: list[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
) -> IngestionResponse:
    """
    Accept uploaded files, run the ingestion pipeline, return IngestionResponse.
    Files are saved to uploads/<program_id>/ for context (Experiment Design / Execution layers).
    Supported: resistance assay CSV, compound screen CSV, VCF, PDF, TXT/MD.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    # ── Input validation ────────────────────────────────────────────────────
    MAX_FILE_SIZE = 50 * 1024 * 1024   # 50 MB
    MAX_FILES = 20
    ALLOWED_EXT = {".csv", ".tsv", ".vcf", ".bcf", ".pdf", ".txt", ".md"}

    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Too many files ({len(files)}). Max {MAX_FILES}.")

    for f in files:
        if not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(ALLOWED_EXT))}",
            )
        # Sanitize filename — strip path traversal characters
        sanitized = Path(f.filename).name.replace("..", "").replace("/", "").replace("\\", "")
        if not sanitized:
            raise HTTPException(status_code=400, detail="Invalid filename")
        f.filename = sanitized

    paths: list[Path] = []
    file_sizes: dict[str, int] = {}
    tmp_dir = Path(tempfile.mkdtemp(prefix="schemabio_ingest_"))
    try:
        for f in files:
            if not f.filename:
                continue
            content = await f.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {f.filename} exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit",
                )
            if len(content) == 0:
                continue  # skip empty files
            path = tmp_dir / f.filename
            path.write_bytes(content)
            file_sizes[f.filename] = len(content)
            paths.append(path)
        if not paths:
            raise HTTPException(status_code=400, detail="No valid files to process")
        response = run_ingestion(paths)
        program_id = response.program_state.program_id

        # Save program row FIRST so user_uploads and experiment_runs can reference it
        # via program_id without a FK violation. Uses upsert so re-uploads are safe.
        programs_db.save_program(
            program_id,
            response.model_dump(),
            user_id=current_user.id,
        )

        upload_ids = await _store_uploaded_files(program_id, paths, file_sizes, current_user) or []

        run_id = runs_db.create_run(
            user_id=current_user.id,
            program_id=program_id,
            upload_ids=upload_ids,
        )
        response.run_id = run_id

        return response
    finally:
        for p in paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        try:
            tmp_dir.rmdir()
        except Exception:
            pass


