"""
agents/action_generator.py

Generates 6 ranked, evidence-linked next actions for the drug program.

Each action:
  - Stage-specific (references actual detected stage)
  - Compound-specific (names actual compound, mutation, IC50 value)
  - Evidence-referenced (audit_id | contra_id | gap_id | field)
  - Annotated with CRO type, cost estimate, timeline
  - Stratified by urgency: blocking | high | medium | low

Ordering rules (enforced via system prompt):
  Rank 1 → BLOCKING gate: must-do before advancing to next stage
  Rank 2 → Critical evidence gap for current stage
  Rank 3 → Contradiction resolution or audit flag
  Rank 4 → Specific experiment recommendation  
  Rank 5 → Translational / manufacturing step
  Rank 6 → Funding / partnership path

Biomni-compatible suggestions:
  For relevant actions, suggests specific Biomni tools:
  - Resistance docking → vina + autosite
  - CRISPR validation → perform_crispr_cas9_genome_editing
  - Literature search → query_pubmed
  - Sequence alignment → DIAMOND / mafft
"""

from __future__ import annotations
import json
import logging
import os

import anthropic

from api.legacy.models.drug_program import DrugProgram, DrugProgramAction

logger = logging.getLogger(__name__)

_ACTION_SYSTEM_PROMPT = """You are AIDEN, an AI principal investigator for drug development programs.
You receive a complete DrugProgram summary.

Your job: generate exactly 6 ranked next actions that will move this specific drug program forward.

MANDATORY ordering:
  Rank 1: A BLOCKING action — without this, the program CANNOT advance. Mark stage_gate=true.
  Rank 2: The critical evidence gap required for this stage gate.
  Rank 3: Address the top contradiction OR highest-severity audit flag.
  Rank 4: A specific experiment with named target/compound/condition.
  Rank 5: A manufacturing, CDMO, or regulatory step.
  Rank 6: A funding path or CRO partnership recommendation.

Critical rules:
  - Name ACTUAL compounds: "Compound-14", "gyrA D87N", "12.4 nM", "72× fold-shift"
  - Reference ACTUAL evidence: audit_001, contra_001, gap_001
  - Never write "consider running" or "may want to" — write direct instructions
  - CRO type must be specific: "microbiology CRO (e.g. IHMA)", not just "CRO"
  - stage_gate: true ONLY for rank 1 and rank 2

Respond ONLY with valid JSON (no markdown, no preamble):
{
  "ranked_actions": [
    {
      "rank": 1,
      "category": "experiment|control|analysis|regulatory|manufacturing|funding",
      "action": "specific instruction ≤15 words",
      "rationale": "cite actual data — compound name, mutation, fold-shift value",
      "evidence_ref": "audit_001|contra_001|gap_001|field_name",
      "urgency": "blocking|high|medium|low",
      "estimated_cost_usd": integer or null,
      "estimated_weeks": integer or null,
      "cro_type": "specific CRO type + example",
      "stage_gate": true or false
    }
  ],
  "key_finding": "single most important insight ≤25 words — cite actual values",
  "blocking_question": "the single most critical unanswered scientific question ≤20 words",
  "missing_controls": ["specific missing control 1", "specific missing control 2"],
  "next_stage_requirements": ["what MUST happen before stage advancement"],
  "biomni_tools": ["relevant Biomni tool 1: use case", "tool 2: use case"]
}"""


def _build_action_context(program: DrugProgram) -> dict:
    return {
        "program_name": program.program_name,
        "stage": program.stage_label,
        "stage_confidence": program.stage_confidence,
        "target": {
            "gene": program.target.gene,
            "organism": program.target.organism,
            "mechanism": program.target.mechanism_of_action,
            "indication": program.target.indication,
        },
        "lead_compound": {
            "name": program.compound.name,
            "ic50_nm": program.compound.ic50_nm,
            "mic_ugml": program.compound.mic_ugml,
            "synthesis_steps": program.compound.synthesis_steps,
            "logp": program.compound.logp,
            "lipinski": program.compound.lipinski_pass,
        },
        "top_hits": [
            {"name": c.get("name"), "ic50_nm": c.get("ic50_nm"),
             "mic_ugml": c.get("mic_ugml"), "flag": c.get("flag")}
            for c in program.all_compounds
            if c.get("flag") in ("TOP_HIT","FOLLOW_UP","CONTRADICTION")
        ][:5],
        "resistance": {
            "mutations": program.resistance.resistance_mutations[:5],
            "resistant_strains": program.resistance.resistant_strains[:5],
            "fold_shift": program.resistance.fold_shift,
            "mechanism": program.resistance.mechanism,
            "characterized": program.resistance.characterized,
        },
        "blocking_evidence_gaps": program.evidence.blocking_gaps[:4],
        "critical_evidence_gaps": program.evidence.critical_gaps[:3],
        "audit_flags": [
            {"id": f.id, "title": f.title, "severity": f.severity}
            for f in program.audit_flags[:3]
        ],
        "contradictions": [
            {"compound": c.compound, "your_value": c.your_value,
             "lit_median": c.lit_median, "fold_diff": c.fold_difference}
            for c in program.contradictions[:2]
        ],
        "epistemic_gaps": [
            {"query": g.query, "intersection_count": g.intersection_paper_count,
             "classification": g.classification}
            for g in program.epistemic_gaps
            if g.classification == "white_space"
        ][:3],
        "manufacturing": {
            "cdmo_readiness": program.manufacturing.cdmo_readiness,
            "synthesis_tractability": program.manufacturing.synthesis_tractability,
            "scale_up_risks": program.manufacturing.scale_up_risks[:3],
        },
        "gmp_readiness_pct": program.gmp_readiness_pct,
        "completeness_pct": program.completeness_pct,
        "funding_stage": program.stage_label,
        "lit_count": len(program.literature),
        "blocking_question": program.blocking_question,
    }


def generate_actions(program: DrugProgram) -> DrugProgram:
    """
    Generate ranked action plan. Updates program.ranked_actions in place.
    """
    try:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY",""))
        context = _build_action_context(program)
        msg = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=1400,
            system=_ACTION_SYSTEM_PROMPT,
            messages=[{"role":"user","content":json.dumps(context, indent=2)}],
        )
        raw = msg.content[0].text.strip().replace("```json","").replace("```","").strip()
        data = json.loads(raw)

        program.ranked_actions = [DrugProgramAction(**a) for a in data.get("ranked_actions",[])]
        program.key_finding = data.get("key_finding","")
        if data.get("blocking_question") and not program.blocking_question:
            program.blocking_question = data["blocking_question"]

        program.add_trace(
            len(program.agent_trace)+1, "ActionGenerator",
            "Action plan generated",
            f"{len(program.ranked_actions)} ranked actions — blocking: {program.ranked_actions[0].action if program.ranked_actions else 'none'}",
            "reason",
        )

    except Exception as e:
        logger.error(f"Action generator failed: {e}")
        _fallback_actions(program)

    return program


def _fallback_actions(program: DrugProgram) -> None:
    """Rule-based fallback actions when LLM fails."""
    from api.legacy.agents.stage_classifier import get_stage_gate_requirements
    stage_reqs = get_stage_gate_requirements(program.current_stage)
    comp = program.compound.name or "lead compound"
    target = program.target.gene or "target"
    muts = program.resistance.resistance_mutations[:2]

    actions = []

    if program.resistance.resistant_strains and not program.resistance.mechanism:
        actions.append(DrugProgramAction(
            rank=1, category="experiment",
            action=f"Characterize {target} resistance mechanism via WGS",
            rationale=f"{len(program.resistance.resistant_strains)} resistant strains; mechanism unknown",
            evidence_ref="audit_005", urgency="blocking", stage_gate=True,
            estimated_weeks=8, estimated_cost_usd=12000,
            cro_type="Microbiology / genomics CRO",
        ))
    elif muts:
        actions.append(DrugProgramAction(
            rank=1, category="experiment",
            action=f"Enzyme inhibition assay: {comp} vs {muts[0]} mutant",
            rationale=f"Resistance mutation {muts[0]} detected; target engagement unconfirmed",
            evidence_ref="resistance.resistance_mutations", urgency="blocking", stage_gate=True,
            estimated_weeks=4, estimated_cost_usd=8000,
            cro_type="Biochemistry CRO (e.g. Reaction Biology)",
        ))
    elif program.compound.ic50_nm:
        actions.append(DrugProgramAction(
            rank=1, category="control",
            action=f"Add vehicle control and replicate {comp} IC50 (n=3)",
            rationale="Single replicate detected; n=1 screens have 20-40% false positive rate",
            evidence_ref="audit_002", urgency="blocking", stage_gate=True,
            estimated_weeks=3, estimated_cost_usd=5000,
            cro_type="Internal or contract biochemistry lab",
        ))

    actions.append(DrugProgramAction(
        rank=2, category="experiment",
        action=f"Run selectivity counter-screen for {comp}",
        rationale="No selectivity data; required before lead nomination",
        evidence_ref="audit_006", urgency="high", stage_gate=True,
        estimated_weeks=6, estimated_cost_usd=15000,
        cro_type="Selectivity profiling CRO (e.g. Eurofins Cerep)",
    ))
    actions.append(DrugProgramAction(
        rank=3, category="analysis",
        action="Initiate in vitro ADMET panel (solubility, stability, permeability)",
        rationale="ADMET required before any in vivo work or lead nomination",
        evidence_ref="evidence.has_metabolic_stability", urgency="high", stage_gate=False,
        estimated_weeks=6, estimated_cost_usd=12000,
        cro_type="ADME/DMPK CRO (e.g. Cyprotex)",
    ))

    program.ranked_actions = actions[:len(actions)]
    program.key_finding = (
        f"Program at {program.stage_label} — "
        + (f"{program.resistance.resistance_mutations[0]} resistance mutation"
           if program.resistance.resistance_mutations
           else f"{comp} showing promising activity")
        + " — critical gaps block stage advancement"
    )
    program.add_trace(
        len(program.agent_trace)+1, "ActionGenerator",
        "Action plan (rule-based fallback)",
        f"{len(program.ranked_actions)} actions generated via fallback rules",
        "reason",
    )
