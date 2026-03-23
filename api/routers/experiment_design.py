"""
routers/experiment_design.py

Layer 2 — Experiment Design API

POST /api/experiment-design/run
  Accepts ExperimentDesignInput + ProgramState (for RAG).
  Fetches CARD / AlphaFold / IMGT context, runs the LLM pipeline,
  and returns a response shaped exactly for Experiments.tsx.

Response shape (matches Experiments.tsx data contracts):
  {
    "recommendations":    [{ title, rationale, confidence, urgency, sources, expectedValue }],
    "hypotheses":         [{ title, evidence, status }],
    "bioinfTasks":        [str],
    "controlSuggestions": [{ name, type }],
    "stageConfirmed":     str,
    "keyHypothesis":      str,
    "literatureQueries":  [str],
    "pipelineNotes":      [str],
    "status":             str,
  }
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.schemas.ingestion import ExperimentDesignInput, ProgramState
from api.rag.service import ensure_indexed_and_query
from api.experiment_design.pipeline import ExperimentDesignPipeline
from api.schemas.layer2 import ExperimentDesignOutput, RankedExperiment
import api.services.runs_db as runs_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/experiment-design", tags=["experiment-design"])

_pipeline = ExperimentDesignPipeline()   # singleton — reuses cached prompts


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class ExperimentDesignRequest(BaseModel):
    experiment_design_input: ExperimentDesignInput
    program_state: ProgramState   # passed through for RAG query generation
    # Optional — returned by /api/upload-and-parse. When present, Layer 2 results
    # are persisted to experiment_results so the user can retrieve them later.
    run_id: Optional[str] = None
    user_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/run")
async def run_experiment_design(req: ExperimentDesignRequest) -> dict:
    """
    Run the Layer 2 Experiment Design pipeline.

    1. Fetch RAG context (CARD / AlphaFold / IMGT) using program_state
    2. Run async LLM pipeline with prompt caching + speculative execution
    3. Return response shaped for Experiments.tsx
    """
    try:
        # Step 1 — RAG retrieval
        rag_bundle = await ensure_indexed_and_query(req.program_state.model_dump(), top_k=6)
        rag_docs   = (
            rag_bundle.get("card_documents", [])
            + rag_bundle.get("alphafold_documents", [])
            + rag_bundle.get("imgt_documents", [])
        )

        # Step 2 — Layer 2 LLM pipeline
        edi_dict = req.experiment_design_input.model_dump()
        output   = await _pipeline.run(edi_dict, rag_docs)

        # Step 3 — Shape response for frontend
        result = _shape_for_frontend(output, req.experiment_design_input)

        # Step 4 — Persist to DB if run_id was supplied
        logger.info(
            "Layer 2 save check: run_id=%r user_id=%r program_id=%r",
            req.run_id, req.user_id, req.program_state.program_id,
        )
        if req.run_id and req.user_id:
            program_id = req.program_state.program_id or ""
            logger.info("Layer 2 saving result for run_id=%s", req.run_id)
            runs_db.save_experiment_result(
                run_id=req.run_id,
                user_id=req.user_id,
                program_id=program_id,
                data=result,
            )
            logger.info("Layer 2 save complete for run_id=%s", req.run_id)
        else:
            logger.warning(
                "Layer 2 result NOT saved — run_id or user_id missing (run_id=%r, user_id=%r)",
                req.run_id, req.user_id,
            )

        return result

    except Exception as exc:
        logger.exception("Layer 2 experiment design failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Experiment design failed: {exc}")


# ---------------------------------------------------------------------------
# Response shaper — maps ExperimentDesignOutput → Experiments.tsx contracts
# ---------------------------------------------------------------------------

def _shape_for_frontend(
    output: ExperimentDesignOutput,
    edi:    ExperimentDesignInput,
) -> dict:
    return {
        "recommendations":    _make_recommendations(output),
        "hypotheses":         _make_hypotheses(output),
        "bioinfTasks":        _make_bioinf_tasks(output),
        "controlSuggestions": _make_control_suggestions(output),
        "stageConfirmed":     getattr(output, "stage_confirmed", ""),
        "keyHypothesis":      getattr(output, "key_hypothesis", ""),
        "literatureQueries":  getattr(output, "literature_queries", []) or [],
        "pipelineNotes":      getattr(output, "pipeline_notes", []) or [],
        "status":             getattr(output, "status", "final"),
        "_layer2_output":     _serialise(output),
    }


def _make_recommendations(output: ExperimentDesignOutput) -> list[dict]:
    """Map ranked_experiments → RecommendationCard props."""
    recs = []
    experiments = getattr(output, "ranked_experiments", None) or []
    for exp in sorted(experiments, key=lambda e: getattr(e, "rank", 0)):
        urgency = _urgency(exp)
        rank = getattr(exp, "rank", 1)
        blocking = getattr(exp, "blocking", False)
        confidence = round(
            1.0 if blocking else max(0.60, 0.95 - (rank - 1) * 0.05), 2
        )
        sources = [s for s in [getattr(exp, "cro_type", "")] if s]
        recs.append({
            "title":         getattr(exp, "title", ""),
            "rationale":     getattr(exp, "rationale", ""),
            "confidence":    confidence,
            "urgency":       urgency,
            "sources":       sources,
            "expectedValue": getattr(exp, "expected_outcome", ""),
        })
    return recs


def _urgency(exp: RankedExperiment) -> str:
    if getattr(exp, "blocking", False):
        return "high"
    if getattr(exp, "stage_gate", False):
        return "medium"
    return "low"


def _make_hypotheses(output: ExperimentDesignOutput) -> list[dict]:
    """Derive hypothesis cards from key_hypothesis + top blocking experiments."""
    hypotheses = []
    key_hyp = getattr(output, "key_hypothesis", "") or ""
    reasoning = getattr(output, "reasoning_steps", None) or []
    experiments = getattr(output, "ranked_experiments", None) or []

    if key_hyp:
        evidence = reasoning[0] if reasoning else "Based on integrated analysis of program data."
        hypotheses.append({
            "title":    key_hyp,
            "evidence": evidence,
            "status":   "testing",
        })

    for exp in experiments:
        if getattr(exp, "blocking", False) and getattr(exp, "expected_outcome", "") and len(hypotheses) < 3:
            hypotheses.append({
                "title":    f"{getattr(exp, 'experiment_type', 'experiment').replace('_', ' ').title()} hypothesis",
                "evidence": getattr(exp, "expected_outcome", ""),
                "status":   "untested",
            })

    return hypotheses


def _make_bioinf_tasks(output: ExperimentDesignOutput) -> list[str]:
    analyses = getattr(output, "bioinformatics_analyses", None) or []
    return [
        f"{getattr(t, 'analysis', '')} ({getattr(t, 'tool', '')})" if getattr(t, "tool", None) else getattr(t, "analysis", "")
        for t in analyses
    ]


def _make_control_suggestions(output: ExperimentDesignOutput) -> list[dict]:
    """Parse controls from experiments + missing_controls into typed suggestion cards."""
    seen:     set[str]   = set()
    controls: list[dict] = []
    missing_controls = getattr(output, "missing_controls", None) or []
    experiments = getattr(output, "ranked_experiments", None) or []

    for ctrl in missing_controls:
        ctype = _classify_control(ctrl)
        key   = ctrl.lower().strip()
        if key not in seen:
            seen.add(key)
            controls.append({"name": ctrl, "type": ctype})

    for exp in experiments:
        for ctrl in getattr(exp, "controls", []) or []:
            ctype = _classify_control(ctrl)
            # Strip "positive control: " prefix etc.
            name = ctrl.split(":", 1)[-1].strip() if ":" in ctrl else ctrl
            key  = name.lower().strip()
            if key not in seen:
                seen.add(key)
                controls.append({"name": name, "type": ctype})

    return controls[:8]   # cap for UI


def _classify_control(ctrl_str: str) -> str:
    s = ctrl_str.lower()
    if any(k in s for k in ("positive", "atcc", "reference", "known")):
        return "Positive"
    if any(k in s for k in ("negative", "vehicle", "dmso", "solvent", "untreated")):
        return "Negative"
    if any(k in s for k in ("comparator", "ciproflox", "standard", "existing")):
        return "Comparator"
    if any(k in s for k in ("baseline", "growth kinetic", "kinetic", "uninduced")):
        return "Baseline"
    return "Positive"   # safe default


def _serialise(output: ExperimentDesignOutput) -> dict:
    """Serialise ExperimentDesignOutput to a plain dict for JSON transport."""
    import dataclasses
    return dataclasses.asdict(output)


# ---------------------------------------------------------------------------
# SSE streaming endpoint
# ---------------------------------------------------------------------------

def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/run-stream")
async def run_experiment_design_stream(req: ExperimentDesignRequest):
    """
    SSE streaming version of /run. Yields progress events during the pipeline,
    then a final 'complete' event with the shaped response.
    """
    async def generate():
        try:
            yield _sse_event("progress", {"step": "rag_retrieval", "message": "Fetching RAG context from CARD, AlphaFold, IMGT..."})
            rag_bundle = await ensure_indexed_and_query(req.program_state.model_dump(), top_k=6)
            rag_docs = (
                rag_bundle.get("card_documents", [])
                + rag_bundle.get("alphafold_documents", [])
                + rag_bundle.get("imgt_documents", [])
            )
            yield _sse_event("progress", {"step": "rag_complete", "message": f"Retrieved {len(rag_docs)} RAG documents."})

            yield _sse_event("progress", {"step": "llm_reasoning", "message": "Running experiment design reasoning..."})
            edi_dict = req.experiment_design_input.model_dump()
            output = await _pipeline.run(edi_dict, rag_docs)
            yield _sse_event("progress", {"step": "llm_complete", "message": f"Reasoning complete. Confidence: {output.overall_confidence:.0%}"})

            yield _sse_event("progress", {"step": "shaping", "message": "Formatting response..."})
            result = _shape_for_frontend(output, req.experiment_design_input)

            if req.run_id and req.user_id:
                program_id = req.program_state.program_id or ""
                runs_db.save_experiment_result(
                    run_id=req.run_id,
                    user_id=req.user_id,
                    program_id=program_id,
                    data=result,
                )

            yield _sse_event("complete", result)

        except Exception as exc:
            logger.exception("Layer 2 SSE stream failed: %s", exc)
            yield _sse_event("error", {"message": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
