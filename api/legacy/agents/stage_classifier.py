"""
agents/stage_classifier.py

Classifies drug program stage from the assembled DrugProgram object.

Heuristic-first approach:
  - Resistance mutations detected → Resistance Characterization
  - GMP/synthesis data present → GMP Readiness
  - ADMET data → ADMET Profiling
  - IC50 + compound screen → Hit Discovery / Hit Triage
  etc.

LLM escalation only when confidence < 0.80.
All LLM calls are compact (program summary, not raw files).
"""

from __future__ import annotations
import json
import logging
import os
from typing import Optional

import anthropic

from api.legacy.models.drug_program import DrugProgram, ProgramStage, STAGE_META

logger = logging.getLogger(__name__)

# ─── Stage gate requirements ─────────────────────────────────────────────────

STAGE_GATE_REQUIREMENTS: dict[ProgramStage, list[str]] = {
    ProgramStage.HIT_DISCOVERY: [
        "Confirmed IC50 with n≥3 replicates",
        "Selectivity ratio vs. ≥2 off-targets",
        "Initial solubility + stability flags",
    ],
    ProgramStage.RESISTANCE_CHAR: [
        "Resistance mechanism identified (target mutation vs. efflux vs. enzyme)",
        "Cross-resistance panel completed (≥3 compounds)",
        "Frequency of resistance measured (≤1×10⁻⁸ acceptable)",
        "Structural hypothesis for gyrA/parC mutation impact",
    ],
    ProgramStage.HIT_TRIAGE: [
        "Lead scaffold selected with SAR rationale",
        "Lipinski/Ro5 compliance checked",
        "In vitro ADMET panel initiated (Caco-2, microsomal stability, hERG)",
    ],
    ProgramStage.VALIDATION_PLANNING: [
        "Mechanism of action confirmed biochemically",
        "In vivo proof-of-concept (murine infection model or equivalent)",
        "Dose fractionation data",
    ],
    ProgramStage.PRECLINICAL_PACKAGE: [
        "GLP toxicology study designed",
        "PK/PD model established",
        "CMC section drafted for IND",
        "Safety pharmacology studies: hERG, respiratory, CNS",
    ],
    ProgramStage.GMP_READINESS: [
        "Synthetic route finalized (≤8 steps preferred)",
        "Analytical methods: HPLC identity, purity, assay",
        "Forced degradation / impurity profile",
        "Accelerated stability (40°C/75% RH, 6 months)",
    ],
}

_CLASSIFIER_PROMPT = """You are a drug development expert classifying a program stage.

Taxonomy (respond with exact ID):
  target_id               — Gene/target discovery from omics
  hit_discovery           — Primary screen, IC50 triage
  hit_triage              — Narrowing hits to lead scaffold
  resistance_characterization — MIC assays, mutation profiling, mechanism  
  hit_to_lead             — SAR analysis, scaffold optimization
  lead_optimization       — ADMET + potency/selectivity balance
  admet_profiling         — Dedicated ADMET studies
  validation_planning     — Mechanism confirmed, designing validation
  preclinical_package     — Building IND package, in vivo data
  ind_enabling            — IND-enabling studies in progress
  gmp_readiness           — CMC/synthesis scale-up evaluation
  cdmo_evaluation         — CDMO selection and manufacturing feasibility
  phase1_design           — Phase 1 clinical trial design
  unknown                 — Insufficient data to classify

Respond ONLY with valid JSON (no markdown):
{
  "stage": "resistance_characterization",
  "confidence": 0.91,
  "rationale": "1-2 sentences citing specific data — e.g. gyrA D87N mutation + MIC fold-shift",
  "next_stage_blockers": ["what must happen before advancing"]
}"""


def _heuristic_stage(program: DrugProgram) -> Optional[tuple[ProgramStage, float]]:
    """Rule-based stage detection. Returns (stage, confidence) or None."""
    ev = program.evidence
    r  = program.resistance

    # Very specific signals — high confidence
    if r.resistance_mutations or (r.resistant_strains and ev.has_mic_data):
        return ProgramStage.RESISTANCE_CHAR, 0.91

    if ev.has_synthesis_route and ev.has_analytical_methods:
        return ProgramStage.GMP_READINESS, 0.87

    if ev.has_in_vivo_efficacy and ev.has_acute_toxicity:
        return ProgramStage.PRECLINICAL_PACKAGE, 0.85

    if ev.has_metabolic_stability and ev.has_solubility and ev.has_permeability:
        return ProgramStage.ADMET_PROFILING, 0.84

    if ev.has_mechanism_confirmed and ev.has_dose_response:
        return ProgramStage.VALIDATION_PLANNING, 0.82

    if ev.has_metabolic_stability and ev.has_dose_response:
        return ProgramStage.LEAD_OPTIMIZATION, 0.80

    if ev.has_dose_response and ev.has_selectivity_data:
        return ProgramStage.HIT_TRIAGE, 0.78

    if ev.has_dose_response or ev.has_mic_data or (program.compound.ic50_nm is not None):
        return ProgramStage.HIT_DISCOVERY, 0.75

    if ev.has_target_validation:
        return ProgramStage.TARGET_ID, 0.72

    return None


def classify_program_stage(program: DrugProgram) -> ProgramStage:
    """
    Classify stage. Updates program in place. Returns detected stage.
    """
    heuristic = _heuristic_stage(program)

    if heuristic and heuristic[1] >= 0.80:
        stage, conf = heuristic
        meta = STAGE_META.get(stage, {})
        program.current_stage = stage
        program.stage_confidence = conf
        program.stage_rationale = (
            f"Detected from: "
            + (f"resistance mutations {program.resistance.resistance_mutations[:2]}" if program.resistance.resistance_mutations
               else f"evidence flags (dose_response={program.evidence.has_dose_response})")
        )
        program.add_trace(
            len(program.agent_trace)+1, "StageClassifier",
            "Stage classification",
            f"{meta.get('label','?')} ({round(conf*100)}% confidence, heuristic)",
            "evidence",
        )
        return stage

    # LLM fallback
    try:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY",""))
        ctx = {
            "target_gene": program.target.gene,
            "compound": program.compound.name,
            "ic50_nm": program.compound.ic50_nm,
            "mic_ugml": program.compound.mic_ugml,
            "resistance_mutations": program.resistance.resistance_mutations[:3],
            "resistant_strains": program.resistance.resistant_strains[:3],
            "fold_shift": program.resistance.fold_shift,
            "evidence_present": [k for k,v in program.evidence.model_dump().items()
                                  if v is True],
            "uploaded_files": program.uploaded_files,
        }
        msg = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=400,
            messages=[{"role":"user","content":f"{_CLASSIFIER_PROMPT}\n\n{json.dumps(ctx)}"}],
        )
        raw = msg.content[0].text.strip().replace("```json","").replace("```","").strip()
        data = json.loads(raw)
        try:
            stage = ProgramStage(data["stage"])
        except ValueError:
            stage = ProgramStage.HIT_DISCOVERY
        conf = float(data.get("confidence", 0.65))
        rationale = data.get("rationale", "LLM classification")
    except Exception as e:
        logger.error(f"Stage classifier LLM failed: {e}")
        stage = heuristic[0] if heuristic else ProgramStage.HIT_DISCOVERY
        conf = 0.55
        rationale = "Fallback classification"

    program.current_stage = stage
    program.stage_confidence = conf
    program.stage_rationale = rationale
    program.add_trace(
        len(program.agent_trace)+1, "StageClassifier",
        "Stage classification",
        f"{STAGE_META.get(stage,{}).get('label','?')} ({round(conf*100)}% confidence)",
        "evidence",
    )
    return stage


def get_stage_gate_requirements(stage: ProgramStage) -> list[str]:
    return STAGE_GATE_REQUIREMENTS.get(stage, [])
