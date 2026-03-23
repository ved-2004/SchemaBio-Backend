"""
agents/orchestrator.py

AIDEN's 9-stage pipeline.
Takes any combination of uploaded files → streams a complete drug program analysis.

Stage order:
 1. parse_files          → Universal parser → DrugProgram
 2. classify_stage       → T1–T8 taxonomy
 3. run_auditor          → Assumption Auditor (heuristic + LLM)
 4. retrieve_literature  → PubMed + claim extraction
 5. detect_contradictions→ IC50/MIC vs literature values
 6. map_epistemic_gaps   → Knowledge frontier mapping
 7. translational_analysis → GMP/CDMO/regulatory/funding
 8. generate_actions     → Ranked evidence-linked action plan
 9. finalize             → Complete program with all outputs

Each stage yields SSE events that stream to the frontend in real-time.
The DrugProgram object is the shared state that flows through all stages.
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)


async def run_pipeline(
    vcf_path:    Optional[Path] = None,
    csv_paths:   Optional[list[Path]] = None,
    pdf_path:    Optional[Path] = None,
    text_paths:  Optional[list[Path]] = None,
    use_demo:    bool = False,
) -> AsyncGenerator[dict, None]:
    """
    Main async generator — runs full AIDEN pipeline.

    Yields SSE events:
      {"event": "phase",          "data": {"phase": "parse", "label": "..."}}
      {"event": "drug_program",   "data": {DrugProgram partial}}
      {"event": "stage",          "data": {"label": "...", "confidence": 0.91}}
      {"event": "audit_flags",    "data": [...AuditFlag]}
      {"event": "literature",     "data": [...LiteratureResult]}
      {"event": "contradictions", "data": [...Contradiction]}
      {"event": "epistemic_gaps", "data": [...EpistemicGap]}
      {"event": "translational",  "data": {manufacturing, execution, regulatory}}
      {"event": "actions",        "data": [...DrugProgramAction]}
      {"event": "trace",          "data": {step, agent, action, finding}}
      {"event": "complete",       "data": {full DrugProgram}}
      {"event": "error",          "data": {"message": "..."}}
    """
    from api.legacy.models.drug_program import DrugProgram

    program = DrugProgram()

    try:
        # ── Stage 1: Parse files ────────────────────────────────────────
        yield _phase("parse", "Parsing uploaded files deterministically")

        if use_demo:
            from api.data.demo_program import build_antibiotic_demo_program
            program = build_antibiotic_demo_program()
        else:
            from api.legacy.parsers.universal_parser import build_drug_program_from_files
            program = build_drug_program_from_files(
                vcf_path=vcf_path,
                csv_paths=csv_paths or [],
                pdf_path=pdf_path,
                text_paths=text_paths or [],
            )

        # Stream initial DrugProgram (before any LLM — shows deterministic parsing)
        yield _phase("schema", "Building DrugProgram object")
        yield {"event": "drug_program", "data": _safe_dump(program)}
        for t in program.agent_trace:
            yield {"event": "trace", "data": t}

        # ── Stage 2: Classify stage ─────────────────────────────────────
        yield _phase("classify", "Detecting program stage (T1–T8)")
        from api.legacy.agents.stage_classifier import classify_program_stage
        classify_program_stage(program)
        yield {"event": "stage", "data": {
            "label": program.stage_label,
            "t_num": program.stage_t_num,
            "confidence": program.stage_confidence,
            "rationale": program.stage_rationale,
            "color": program.stage_color,
        }}
        yield _last_trace(program)

        # ── Stage 3: Assumption Auditor ─────────────────────────────────
        yield _phase("audit", "Running Assumption Auditor")
        from api.legacy.agents.assumption_auditor import run_assumption_auditor
        run_assumption_auditor(program)
        yield {"event": "audit_flags", "data": [f.model_dump() for f in program.audit_flags]}
        yield _last_trace(program)

        # ── Stage 4: Literature retrieval ───────────────────────────────
        yield _phase("lit", "Retrieving evidence literature")
        from api.legacy.agents.literature_agent import retrieve_literature
        await retrieve_literature(program)
        yield {"event": "literature", "data": [p.model_dump() for p in program.literature]}
        yield _last_trace(program)

        # ── Stage 5: Contradiction Detector ────────────────────────────
        yield _phase("contra", "Detecting contradictions vs. literature")
        from api.legacy.agents.contradiction_detector import run_contradiction_detector
        run_contradiction_detector(program)
        yield {"event": "contradictions", "data": [c.model_dump() for c in program.contradictions]}
        yield _last_trace(program)

        # ── Stage 6: Epistemic Gap Map ──────────────────────────────────
        yield _phase("gap", "Mapping knowledge frontier")
        from api.legacy.agents.epistemic_gap_mapper import run_epistemic_gap_mapper
        run_epistemic_gap_mapper(program)
        yield {"event": "epistemic_gaps", "data": [g.model_dump() for g in program.epistemic_gaps]}
        yield _last_trace(program)

        # ── Stage 7: Translational Analysis ────────────────────────────
        yield _phase("translational", "Analyzing translational feasibility")
        from api.legacy.agents.translational_agent import run_translational_agent
        run_translational_agent(program)
        yield {"event": "translational", "data": {
            "manufacturing": program.manufacturing.model_dump(),
            "execution": program.execution.model_dump(),
            "regulatory": program.regulatory.model_dump(),
            "evidence_gaps": {
                "blocking": program.evidence.blocking_gaps,
                "critical": program.evidence.critical_gaps,
            },
            "gmp_readiness_pct": program.gmp_readiness_pct,
            "completeness_pct": program.completeness_pct,
        }}
        yield _last_trace(program)

        # ── Stage 8: Action Generator ───────────────────────────────────
        yield _phase("reason", "Generating evidence-linked action plan")
        from api.legacy.agents.action_generator import generate_actions
        generate_actions(program)
        yield {"event": "actions", "data": [a.model_dump() for a in program.ranked_actions]}
        yield {"event": "key_finding", "data": {
            "finding": program.key_finding,
            "blocking_question": program.blocking_question,
        }}
        yield _last_trace(program)

        # ── Stage 9: Complete ───────────────────────────────────────────
        yield _phase("complete", "Analysis complete")
        yield {"event": "complete", "data": _safe_dump(program)}

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        yield {"event": "error", "data": {
            "message": str(e),
            "stage": program.current_stage.value if program else "unknown",
            "partial_program": _safe_dump(program) if program else {},
        }}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _phase(phase: str, label: str) -> dict:
    return {"event": "phase", "data": {"phase": phase, "label": label}}


def _last_trace(program) -> dict:
    if program.agent_trace:
        return {"event": "trace", "data": program.agent_trace[-1]}
    return {"event": "trace", "data": {}}


def _safe_dump(program) -> dict:
    try:
        return program.model_dump()
    except Exception as e:
        logger.warning(f"Program dump failed: {e}")
        return {"error": str(e)}
