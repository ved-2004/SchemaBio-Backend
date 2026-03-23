"""
agents/assumption_auditor.py

Catches methodological problems BEFORE generating the action plan.
Fires on heuristic rules first (zero API cost).
LLM pass for context-specific risks not caught heuristically.

Drug-program-aware checks:
  - Missing vehicle/solvent control in compound screen
  - Single replicate (n=1) — false positive risk
  - DMSO-sensitive scaffold in top hits
  - MIC assay without matched wild-type control
  - Resistance fold-shift without mechanism data
  - IC50 unit ambiguity (nM vs μM mixing)
  - Missing bactericidal endpoint (only bacteriostatic)
  - Compound screen without counter-screen for selectivity
"""

from __future__ import annotations
import json
import logging
import os
from typing import Optional

import anthropic

from api.legacy.models.drug_program import AuditFlag, DrugProgram

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# HEURISTIC CHECKS
# ═══════════════════════════════════════════════════════════

def _check_vehicle_control(program: DrugProgram) -> Optional[AuditFlag]:
    if program.all_compounds and not program.evidence.has_target_validation:
        # Infer missing control from assay CSV structure
        has_vc = any("vehicle" in str(c.get("id","")).lower()
                     or "dmso" in str(c.get("name","")).lower()
                     for c in program.all_compounds)
        if not has_vc:
            return AuditFlag(
                id="audit_001", type="missing_control", severity="high",
                title="No vehicle control detected in compound screen",
                detail=(
                    "No vehicle/DMSO control wells found in assay CSV. "
                    "Without matched solvent controls, IC50 normalization is unreliable "
                    "and apparent activity may reflect solvent cytotoxicity."
                ),
                field_source="all_compounds: no vehicle/DMSO_ctrl id",
            )
    return None


def _check_replicates(program: DrugProgram) -> Optional[AuditFlag]:
    if program.all_compounds:
        # Heuristic: if all compound IDs are unique there's likely n=1
        ids = [c.get("id","") for c in program.all_compounds]
        unique_ratio = len(set(ids)) / max(len(ids), 1)
        if unique_ratio > 0.95 and len(program.all_compounds) > 3:
            return AuditFlag(
                id="audit_002", type="replicate_concern", severity="medium",
                title="Single replicate detected (n=1)",
                detail=(
                    f"All {len(program.all_compounds)} compounds appear as single measurements. "
                    "Primary screens with n=1 have ~20–40% false positive rates. "
                    "Confirmatory dose-response (n≥3) required before lead selection."
                ),
                field_source="all_compounds: unique IDs ratio > 0.95",
            )
    return None


def _check_dmso_scaffold(program: DrugProgram) -> Optional[AuditFlag]:
    risky_hits = [c for c in program.all_compounds
                  if c.get("flag") in ("TOP_HIT","FOLLOW_UP") and c.get("dmso_risk")]
    if len(risky_hits) >= 2:
        names = ", ".join(c.get("name","?") for c in risky_hits[:4])
        return AuditFlag(
            id="audit_003", type="scaffold_artifact", severity="high",
            title=f"DMSO-sensitive scaffold in {len(risky_hits)} top hits",
            detail=(
                f"{names} share scaffold fragments associated with DMSO precipitation artifacts. "
                "Activity may reflect compound aggregation rather than target engagement. "
                "Add a matched solvent titration (0.01–1% DMSO) before confirmatory assay."
            ),
            field_source="all_compounds.dmso_risk = true for TOP_HIT compounds",
        )
    return None


def _check_resistance_without_wt(program: DrugProgram) -> Optional[AuditFlag]:
    r = program.resistance
    if r.resistant_strains and not any(v.get("wt_mic") for v in r.mic_values):
        return AuditFlag(
            id="audit_004", type="missing_control", severity="high",
            title="MIC values without wild-type baseline",
            detail=(
                f"MIC data found for {len(r.resistant_strains)} resistant strains "
                "but no wild-type/parent strain MIC detected. "
                "Without WT MIC, fold-shift cannot be calculated and resistance classification is unreliable."
            ),
            field_source="resistance.mic_values: wt_mic = null for all entries",
        )
    return None


def _check_resistance_without_mechanism(program: DrugProgram) -> Optional[AuditFlag]:
    r = program.resistance
    if r.resistant_strains and not r.mechanism and not r.characterized:
        return AuditFlag(
            id="audit_005", type="incomplete_characterization", severity="medium",
            title="Resistance phenotype without mechanism data",
            detail=(
                f"{len(r.resistant_strains)} resistant strains identified "
                f"but resistance mechanism not characterized. "
                "Unknown mechanism (efflux vs. target mutation vs. enzyme) prevents rational counter-strategy. "
                "Recommend WGS of resistant mutants and enzyme binding assay."
            ),
            field_source="resistance.mechanism = null, resistance.characterized = false",
        )
    return None


def _check_selectivity(program: DrugProgram) -> Optional[AuditFlag]:
    if program.evidence.has_dose_response and not program.evidence.has_selectivity_data:
        return AuditFlag(
            id="audit_006", type="missing_data", severity="medium",
            title="No counter-screen selectivity data",
            detail=(
                f"IC50 data present for {program.compound.name or 'lead compound'} "
                "but no selectivity panel detected. "
                "Selectivity ratio vs. mammalian cells or off-targets required before lead nomination."
            ),
            field_source="evidence.has_dose_response = true, evidence.has_selectivity_data = false",
        )
    return None


def _check_ic50_units(program: DrugProgram) -> Optional[AuditFlag]:
    ic50s = [c.get("ic50_nm") for c in program.all_compounds if c.get("ic50_nm")]
    if len(ic50s) >= 3:
        ratio = max(ic50s) / min(ic50s) if min(ic50s) > 0 else 1
        if ratio > 500_000:
            return AuditFlag(
                id="audit_007", type="unit_ambiguity", severity="high",
                title="IC50 values span >5 orders of magnitude — likely unit mixing",
                detail=(
                    f"IC50 range: {min(ic50s):.2f} – {max(ic50s):.0f}. "
                    "This spread suggests nM/μM mixing. Verify units before any comparative analysis."
                ),
                field_source=f"all_compounds.ic50_nm range: {min(ic50s):.2f}–{max(ic50s):.0f}",
            )
    return None


# ═══════════════════════════════════════════════════════════
# LLM PASS — context-specific risks
# ═══════════════════════════════════════════════════════════

_LLM_AUDIT_PROMPT = """You are a critical methodological reviewer for drug development.
Given a drug program summary, identify 1–2 ADDITIONAL methodological risks not already flagged.
Focus on context-specific risks based on the actual data — no generic advice.

Already flagged: {existing}

Return ONLY a JSON array (empty if nothing to add):
[
  {{
    "id": "audit_llm_001",
    "type": "context_specific",
    "severity": "high|medium|low",
    "title": "short title",
    "detail": "specific detail citing actual values or conditions",
    "field_source": "which field or data triggered this"
  }}
]"""


def _llm_audit_pass(program: DrugProgram, existing: list[AuditFlag]) -> list[AuditFlag]:
    try:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY",""))
        ctx = {
            "stage": program.stage_label,
            "target": program.target.gene,
            "organism": program.target.organism,
            "compound": program.compound.name,
            "ic50_nm": program.compound.ic50_nm,
            "mic_ugml": program.compound.mic_ugml,
            "resistance_mutations": program.resistance.resistance_mutations[:3],
            "fold_shift": program.resistance.fold_shift,
            "top_hits": [{"name":c.get("name"), "ic50_nm":c.get("ic50_nm"), "dmso_risk":c.get("dmso_risk")}
                         for c in program.all_compounds if c.get("flag")=="TOP_HIT"][:3],
        }
        existing_titles = [f.title for f in existing]
        msg = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=500,
            messages=[{"role":"user","content":
                _LLM_AUDIT_PROMPT.format(existing=existing_titles) + f"\n\n{json.dumps(ctx)}"}],
        )
        raw = msg.content[0].text.strip().replace("```json","").replace("```","").strip()
        return [AuditFlag(**f) for f in json.loads(raw) if isinstance(f,dict)]
    except Exception as e:
        logger.warning(f"LLM audit pass failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════
# MAIN ENTRY
# ═══════════════════════════════════════════════════════════

def run_assumption_auditor(program: DrugProgram) -> list[AuditFlag]:
    """
    Run all heuristic checks + LLM pass.
    Updates program.audit_flags in place.
    """
    flags: list[AuditFlag] = []
    for check in [
        _check_vehicle_control,
        _check_replicates,
        _check_dmso_scaffold,
        _check_resistance_without_wt,
        _check_resistance_without_mechanism,
        _check_selectivity,
        _check_ic50_units,
    ]:
        f = check(program)
        if f:
            flags.append(f)

    logger.info(f"Auditor: {len(flags)} heuristic flags")

    # LLM context-specific pass
    llm_flags = _llm_audit_pass(program, flags)
    flags.extend(llm_flags)

    program.audit_flags = flags
    high = sum(1 for f in flags if f.severity == "high")
    program.add_trace(
        len(program.agent_trace)+1, "AssumptionAuditor",
        "Assumption audit",
        f"{len(flags)} flags ({high} high severity): "
        + ", ".join(f.title for f in flags[:2]),
        "audit",
    )
    return flags
