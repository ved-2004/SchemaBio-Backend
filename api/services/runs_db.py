"""
api/services/runs_db.py

Supabase DB operations for experiment_runs, experiment_results, execution_plans.

All three tables have user_id uuid references users(id) on delete cascade.
If Supabase is not configured every function is a no-op / returns a safe default
so the app still runs locally without a DB.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

def _client():
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


def _jsonable(data: dict) -> dict:
    """Round-trip through JSON to make all values serialisable."""
    return json.loads(json.dumps(data, default=str))


# ── experiment_runs ───────────────────────────────────────────────────────────


def create_run(
    user_id: str,
    program_id: str,
    upload_ids: list[str],
) -> Optional[str]:
    """
    Insert a row into experiment_runs.
    Returns the DB-generated run UUID (str) on success, None on failure / no DB.
    """
    sb = _client()
    if sb is None:
        return None
    try:
        resp = (
            sb.table("experiment_runs")
            .insert({
                "user_id":    user_id,
                "program_id": program_id,
                "upload_ids": upload_ids,
                "status":     "complete",
            })
            .execute()
        )
        rows = resp.data or []
        if not rows:
            logger.error("create_run returned no rows for program_id=%s", program_id)
            return None
        return str(rows[0]["id"])
    except Exception as exc:
        logger.error("create_run failed for program_id=%s: %s", program_id, exc)
        return None


def get_runs_for_user(user_id: str) -> list[dict]:
    """
    Return all experiment_runs for a user, most recent first.
    Enriches each run with the filenames of the associated uploads.
    """
    sb = _client()
    if sb is None:
        return []
    try:
        runs_resp = (
            sb.table("experiment_runs")
            .select("id, program_id, status, created_at, upload_ids")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        runs = runs_resp.data or []
        if not runs:
            return []

        # Gather all upload_ids across all runs in one query
        all_upload_ids: list[str] = []
        for r in runs:
            all_upload_ids.extend(r.get("upload_ids") or [])

        filename_map: dict[str, str] = {}
        if all_upload_ids:
            uploads_resp = (
                sb.table("user_uploads")
                .select("upload_id, filename")
                .in_("upload_id", list(set(all_upload_ids)))
                .execute()
            )
            for row in (uploads_resp.data or []):
                filename_map[row["upload_id"]] = row["filename"]

        result = []
        for r in runs:
            filenames = [
                filename_map.get(uid, uid)
                for uid in (r.get("upload_ids") or [])
            ]
            result.append({
                "id":         str(r["id"]),
                "program_id": r.get("program_id"),
                "status":     r.get("status", "complete"),
                "created_at": r.get("created_at"),
                "filenames":  filenames,
            })
        return result

    except Exception as exc:
        logger.error("get_runs_for_user failed for user %s: %s", user_id, exc)
        return []


# ── experiment_results ────────────────────────────────────────────────────────


def save_experiment_result(
    run_id:     str,
    user_id:    str,
    program_id: str,
    data:       dict,
) -> bool:
    """
    Upsert Layer 2 output into experiment_results (keyed by run_id).
    Returns True on success.
    """
    sb = _client()
    if sb is None:
        return False
    try:
        sb.table("experiment_results").upsert(
            {
                "run_id":     run_id,
                "user_id":    user_id,
                "program_id": program_id,
                "data":       _jsonable(data),
            },
            on_conflict="run_id",
        ).execute()
        logger.info("Saved experiment_result for run_id=%s", run_id)
        return True
    except Exception as exc:
        logger.error("save_experiment_result failed for run_id=%s: %s", run_id, exc)
        return False


# ── execution_plans ───────────────────────────────────────────────────────────


def save_execution_plan(
    run_id:     str,
    user_id:    str,
    program_id: str,
    data:       dict,
) -> bool:
    """
    Upsert Layer 3 output into execution_plans (keyed by run_id).
    Returns True on success.
    """
    sb = _client()
    if sb is None:
        return False
    try:
        sb.table("execution_plans").upsert(
            {
                "run_id":     run_id,
                "user_id":    user_id,
                "program_id": program_id,
                "data":       _jsonable(data),
            },
            on_conflict="run_id",
        ).execute()
        logger.info("Saved execution_plan for run_id=%s", run_id)
        return True
    except Exception as exc:
        logger.error("save_execution_plan failed for run_id=%s: %s", run_id, exc)
        return False
