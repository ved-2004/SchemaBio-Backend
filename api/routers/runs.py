"""
api/routers/runs.py

GET /runs — Return the authenticated user's experiment run history.

Response shape (one item per run):
  {
    "id":         "uuid",
    "program_id": "PRG_00001",
    "status":     "complete",
    "created_at": "2026-03-15T12:00:00Z",
    "filenames":  ["resistance_data.csv", "compound_screen.csv"]
  }
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from api.models.user import User
from api.routers.auth import get_current_user
import api.services.runs_db as runs_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("", response_model=list[dict])
async def list_runs(current_user: User = Depends(get_current_user)):
    """
    Return the current user's past experiment runs, most recent first.
    Each item includes the run ID, program ID, status, creation timestamp,
    and the filenames of the files uploaded in that run.
    """
    return runs_db.get_runs_for_user(current_user.id)
