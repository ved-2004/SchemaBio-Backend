"""
routers/execution_planning.py

Layer 3 — Execution / Translational Planning API

POST /api/execution-planning/run
  Accepts ExecutionPlanningInput (from Layer 1) + optional Layer 2 output.
  Runs the Drug-to-Market engine and returns a response shaped for Execution.tsx.

Response shape (matches Execution.tsx data contracts):
  {
    "readinessItems":    [{ label, value }],          # 0.0–1.0 floats
    "croTypes":          [{ type, desc, urgency }],
    "grants":            [{ name, focus, stage, fit }],
    "evidenceChecklist": [{ item, done }],
    "manufacturingFlags":[{ title, description, severity }],  # "warning"|"critical"
    # Pass-through extended fields
    "fdaPathway":        {...},
    "competitiveLandscape": [...],
    "executionBrief":    str,
    "stageTimeline":     {...},
    "probabilityOfSuccess": {...},
  }
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.schemas.ingestion import ExecutionPlanningInput
from api.execution_planning.pipeline import run_layer3
import api.services.runs_db as runs_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/execution-planning", tags=["execution-planning"])


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class ExecutionPlanningRequest(BaseModel):
    execution_planning_input: ExecutionPlanningInput
    # Layer 2 output dict (from _layer2_output field in experiment-design response)
    experiment_design_output: Optional[dict[str, Any]] = None
    # Optional — when present, Layer 3 results are persisted to execution_plans
    run_id:     Optional[str] = None
    user_id:    Optional[str] = None
    program_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/run")
async def run_execution_planning(req: ExecutionPlanningRequest) -> dict:
    """
    Run the Layer 3 Drug-to-Market execution engine.

    Accepts ExecutionPlanningInput from Layer 1 and optionally the Layer 2
    ExperimentDesignOutput so that partner routing and next-steps are informed
    by the recommended experiments.
    """
    try:
        raw = await run_layer3(
            req.execution_planning_input,
            layer2_output=req.experiment_design_output,
        )
        result = _shape_for_frontend(raw, req.execution_planning_input, req.experiment_design_output)

        logger.info(
            "Layer 3 save check: run_id=%r user_id=%r program_id=%r",
            req.run_id, req.user_id, req.program_id,
        )
        if req.run_id and req.user_id and req.program_id:
            logger.info("Layer 3 saving result for run_id=%s", req.run_id)
            runs_db.save_execution_plan(
                run_id=req.run_id,
                user_id=req.user_id,
                program_id=req.program_id,
                data=result,
            )
            logger.info("Layer 3 save complete for run_id=%s", req.run_id)
        else:
            logger.warning(
                "Layer 3 result NOT saved — run_id, user_id, or program_id missing "
                "(run_id=%r, user_id=%r, program_id=%r)",
                req.run_id, req.user_id, req.program_id,
            )

        return result
    except Exception as exc:
        logger.exception("Layer 3 execution planning failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Execution planning failed: {exc}")


# ---------------------------------------------------------------------------
# Response shaper — maps Layer 3 raw output → Execution.tsx contracts
# ---------------------------------------------------------------------------

def _shape_for_frontend(
    raw:  dict[str, Any],
    epi:  ExecutionPlanningInput,
    l2:   Optional[dict[str, Any]],
) -> dict:
    return {
        "readinessItems":       _make_readiness_items(raw),
        "croTypes":             _make_cro_types(raw),
        "grants":               _make_grants(raw, epi),
        "evidenceChecklist":    _make_evidence_checklist(raw, l2),
        "manufacturingFlags":   _make_manufacturing_flags(raw),
        # Pass-through extended fields for future use
        "fdaPathway":           raw.get("fda_pathway", {}),
        "competitiveLandscape": raw.get("competitive_landscape", []),
        "executionBrief":       raw.get("execution_brief", ""),
        "stageTimeline":        raw.get("stage_timeline", {}),
        "probabilityOfSuccess": raw.get("probability_of_success", {}),
        "grantStacking":        raw.get("grant_stacking", []),
    }


def _make_readiness_items(raw: dict) -> list[dict]:
    ra  = raw.get("readiness_assessment") or {}
    ev  = _pct(ra.get("evidence_completeness_pct"))
    gmp = _pct(ra.get("gmp_readiness_pct"))

    # Derive ADMET and regulatory scores from signals and blockers
    blockers = [b.get("blocker", "").lower() for b in raw.get("translational_blockers", [])]
    admet_done    = not any("admet" in b or "toxicity" in b for b in blockers)
    repro_done    = not any("repro" in b or "replicat" in b for b in blockers)
    reg_signals   = raw.get("fda_pathway", {})
    reg_score     = 0.35 if reg_signals.get("qidp_eligible") else 0.10

    return [
        {"label": "Evidence Package",       "value": ev},
        {"label": "ADMET Completion",        "value": 0.65 if admet_done else 0.20},
        {"label": "Reproducibility",         "value": 0.70 if repro_done else 0.35},
        {"label": "Regulatory Alignment",    "value": reg_score},
        {"label": "Manufacturing Readiness", "value": gmp},
    ]


def _make_cro_types(raw: dict) -> list[dict]:
    cros = []
    for pr in raw.get("partner_recommendations") or []:
        ptype    = pr.get("partner_type", "CRO")
        desc     = pr.get("rationale", "")
        urgency  = pr.get("readiness_required") or "Needed now"
        cros.append({"type": ptype, "desc": desc, "urgency": urgency})
    return cros or [{"type": "Microbiology CRO", "desc": "MIC / time-kill assays", "urgency": "Needed now"}]


def _make_grants(raw: dict, epi: ExecutionPlanningInput) -> list[dict]:
    grants = []
    for fo in (raw.get("funding_opportunities") or [])[:5]:
        fit_score = fo.get("fit_score", 0.0)
        fit_label = "High" if fit_score >= 0.7 else ("Medium" if fit_score >= 0.4 else "Low")
        focus     = fo.get("fit_rationale", fo.get("agency", ""))[:80]
        stage     = epi.stage.replace("_", " ").title() if epi.stage else "Early"
        grants.append({
            "name":  fo.get("name", fo.get("program_name", "")),
            "focus": focus,
            "stage": stage,
            "fit":   fit_label,
        })
    return grants


_IND_CHECKLIST: list[tuple[str, list[str]]] = [
    # (item label, signal kinds that indicate completion)
    ("MIC data across ≥3 resistant strains",           ["resistance_fold_shift"]),
    ("Compound selectivity index",                      ["selectivity_index"]),
    ("ADMET / cytotoxicity panel",                      ["admet_complete", "cytotoxicity"]),
    ("Independent lab replication",                     ["reproducibility_confirmed"]),
    ("Mechanism of action evidence",                    ["resistance_associated_variant", "mechanism_hint"]),
    ("In vivo efficacy (mouse model)",                  ["in_vivo_efficacy"]),
    ("Scale-up synthesis feasibility",                  ["synthesis_steps"]),
    ("Regulatory pre-IND meeting",                      ["regulatory_meeting"]),
]

def _make_evidence_checklist(raw: dict, l2: Optional[dict]) -> list[dict]:
    ra = raw.get("readiness_assessment") or {}
    present_kinds = {s.get("kind", "") for s in ra.get("signals") or []}
    missing_elements = {
        (m.get("element") or "").lower()
        for m in raw.get("missing_evidence_package_elements") or []
    }

    checklist = []
    for item_label, signal_kinds in _IND_CHECKLIST:
        done = (
            any(k in present_kinds for k in signal_kinds)
            and not any(item_label.lower()[:20] in me for me in missing_elements)
        )
        checklist.append({"item": item_label, "done": done})

    # If Layer 2 has experiments marked as blocking, add them as not-yet-done items
    if l2:
        for exp in l2.get("ranked_experiments", []):
            if exp.get("blocking") and exp.get("title"):
                label = f"Layer 2: {exp['title']}"
                if not any(c["item"] == label for c in checklist):
                    checklist.append({"item": label, "done": False})

    return checklist


def _make_manufacturing_flags(raw: dict) -> list[dict]:
    flags = []
    for blocker in raw.get("translational_blockers") or []:
        b_text   = blocker.get("blocker", "")
        severity = blocker.get("severity", "warning")
        # Normalise severity to "warning" | "critical"
        if severity in ("high", "critical"):
            severity = "critical"
        else:
            severity = "warning"
        # Split long blocker text into title + description
        parts = b_text.split(".", 1)
        title = parts[0].strip()[:60]
        desc  = parts[1].strip() if len(parts) > 1 else b_text
        flags.append({"title": title, "description": desc, "severity": severity})
    return flags


def _pct(val: Any) -> float:
    """Normalise a percentage value (0–100 or 0.0–1.0) to a 0.0–1.0 float."""
    if val is None:
        return 0.0
    v = float(val)
    return v / 100.0 if v > 1.0 else v


# ---------------------------------------------------------------------------
# SSE streaming endpoint
# ---------------------------------------------------------------------------

def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/run-stream")
async def run_execution_planning_stream(req: ExecutionPlanningRequest):
    """
    SSE streaming version of /run. Yields progress events during the pipeline,
    then a final 'complete' event with the shaped response.
    """
    async def generate():
        try:
            yield _sse_event("progress", {"step": "planning_start", "message": "Starting execution planning..."})

            yield _sse_event("progress", {"step": "llm_reasoning", "message": "Running drug-to-market analysis..."})
            raw = await run_layer3(
                req.execution_planning_input,
                layer2_output=req.experiment_design_output,
            )
            yield _sse_event("progress", {"step": "llm_complete", "message": "Analysis complete."})

            yield _sse_event("progress", {"step": "shaping", "message": "Formatting execution brief..."})
            result = _shape_for_frontend(raw, req.execution_planning_input, req.experiment_design_output)

            if req.run_id and req.user_id and req.program_id:
                runs_db.save_execution_plan(
                    run_id=req.run_id,
                    user_id=req.user_id,
                    program_id=req.program_id,
                    data=result,
                )

            yield _sse_event("complete", result)

        except Exception as exc:
            logger.exception("Layer 3 SSE stream failed: %s", exc)
            yield _sse_event("error", {"message": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
