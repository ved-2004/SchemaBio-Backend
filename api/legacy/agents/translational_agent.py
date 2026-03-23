"""
agents/translational_agent.py

Translational Feasibility Agent.
Answers: "What stands between this compound and the clinic?"

Analyzes:
  1. Evidence gaps for current stage gate
  2. Manufacturing feasibility (synthesis tractability, CDMO type)
  3. Regulatory readiness (IND package gaps)
  4. CRO recommendations (what external help is needed)
  5. Bioinformatics needs
  6. Funding / grant opportunities for this stage
  7. The single BLOCKING question

Biomni-compatible CRO/tool suggestions:
  - Recommends specific assay types that map to Biomni tools
    e.g. "Docking simulations for gyrA D87N" → vina + autosite tools
"""

from __future__ import annotations
import json
import logging
import os
from typing import Optional

import anthropic

from api.legacy.models.drug_program import (
    DrugProgram, ProgramStage, ManufacturingProfile,
    EvidencePackage, ExecutionGuidance, CRONeed, FundingOpportunity,
)

logger = logging.getLogger(__name__)


# ─── Stage gate evidence requirements ────────────────────────────────────────

STAGE_EVIDENCE_MAP: dict[ProgramStage, list[tuple[str, str, str]]] = {
    # (field, human_label, blocking_level)
    ProgramStage.HIT_DISCOVERY: [
        ("has_dose_response",      "Dose-response curve (n≥3 replicates)",           "blocking"),
        ("has_selectivity_data",   "Selectivity counter-screen vs. off-targets",      "high"),
        ("has_solubility",         "Kinetic solubility check",                        "medium"),
    ],
    ProgramStage.RESISTANCE_CHAR: [
        ("has_resistance_profiling","Resistance mechanism characterized",              "blocking"),
        ("has_dose_response",       "MIC across ≥5 strains including WT",             "blocking"),
        ("has_target_validation",   "Target engagement confirmed (enzyme inhibition)", "blocking"),
        ("has_selectivity_data",    "Mammalian cytotoxicity counter-screen",           "high"),
    ],
    ProgramStage.HIT_TRIAGE: [
        ("has_dose_response",       "Replicated IC50 (n≥3)",                          "blocking"),
        ("has_selectivity_data",    "Selectivity ratio vs. ≥2 off-targets",           "blocking"),
        ("has_solubility",          "Solubility in PBS + FaSSIF",                     "high"),
        ("has_metabolic_stability", "Microsomal metabolic stability (t½)",            "high"),
    ],
    ProgramStage.VALIDATION_PLANNING: [
        ("has_mechanism_confirmed", "Biochemical MoA confirmed",                      "blocking"),
        ("has_dose_response",       "Replicated dose-response in disease-relevant model","blocking"),
        ("has_in_vivo_efficacy",    "In vivo proof-of-concept experiment",            "high"),
        ("has_time_kill_data",      "Time-kill kinetics (bactericidal vs -static)",   "high"),
    ],
    ProgramStage.PRECLINICAL_PACKAGE: [
        ("has_in_vivo_efficacy",    "In vivo efficacy with PK/PD modeling",           "blocking"),
        ("has_acute_toxicity",      "Acute toxicity (rat, 14-day repeat dose)",       "blocking"),
        ("has_metabolic_stability", "Human microsomal stability + CYP panel",         "blocking"),
        ("has_genotoxicity",        "Ames test + clastogenicity (ICH S2(R1))",        "blocking"),
        ("has_synthesis_route",     "Synthesis route for toxicology batch",           "blocking"),
    ],
    ProgramStage.GMP_READINESS: [
        ("has_synthesis_route",      "GMP synthesis route (≤8 steps preferred)",       "blocking"),
        ("has_analytical_methods",   "Identity + purity + assay methods (HPLC/NMR)",   "blocking"),
        ("has_forced_degradation",   "Forced degradation / ICH Q1B photostability",    "blocking"),
        ("has_gmp_batch",            "First GMP batch manufactured",                   "high"),
    ],
}

CRO_RECOMMENDATIONS = {
    "has_dose_response":       CRONeed(need="Dose-response (IC50)", assay_type="biochemical_cell",
        cro_examples=["Eurofins Discovery","Charles River Labs","Reaction Biology"],
        urgency="blocking", estimated_cost_usd=8000, estimated_weeks=4),
    "has_selectivity_data":    CRONeed(need="Selectivity panel", assay_type="selectivity_profiling",
        cro_examples=["Eurofins Cerep","DiscoverX","Reaction Biology"],
        urgency="high", estimated_cost_usd=15000, estimated_weeks=6),
    "has_mic_data":            CRONeed(need="MIC/AST panel", assay_type="microbiology",
        cro_examples=["IHMA","Eurofins DPMK","Micromyx"],
        urgency="blocking", estimated_cost_usd=5000, estimated_weeks=3),
    "has_metabolic_stability": CRONeed(need="Metabolic stability", assay_type="adme_dmpk",
        cro_examples=["Cyprotex","Sekisui XenoTech","QPS"],
        urgency="high", estimated_cost_usd=3000, estimated_weeks=3),
    "has_solubility":          CRONeed(need="Kinetic + thermodynamic solubility", assay_type="adme",
        cro_examples=["Cyprotex","Pion Inc","SGS"],
        urgency="medium", estimated_cost_usd=1500, estimated_weeks=2),
    "has_permeability":        CRONeed(need="Caco-2 / PAMPA permeability", assay_type="adme",
        cro_examples=["Cyprotex","QPS","Absorption Systems"],
        urgency="high", estimated_cost_usd=4000, estimated_weeks=3),
    "has_cyp_inhibition":      CRONeed(need="CYP3A4/2D6/2C9 inhibition", assay_type="adme",
        cro_examples=["Cyprotex","XenoTech","BD Biosciences"],
        urgency="high", estimated_cost_usd=5000, estimated_weeks=3),
    "has_herg_data":           CRONeed(need="hERG patch-clamp", assay_type="cardiac_safety",
        cro_examples=["ChanTest (Charles River)","Sophion","IonWorks"],
        urgency="high", estimated_cost_usd=6000, estimated_weeks=4),
    "has_genotoxicity":        CRONeed(need="Ames test + clastogenicity", assay_type="safety",
        cro_examples=["Covance","Charles River","MPI Research"],
        urgency="blocking", estimated_cost_usd=20000, estimated_weeks=8),
    "has_acute_toxicity":      CRONeed(need="14-day repeat dose tox (rat GLP)", assay_type="toxicology",
        cro_examples=["Charles River","Covance","MPI Research"],
        urgency="blocking", estimated_cost_usd=150000, estimated_weeks=20),
    "has_in_vivo_efficacy":    CRONeed(need="In vivo efficacy (infection model)", assay_type="in_vivo",
        cro_examples=["WuXi AppTec","Champions Oncology","Xenograft models"],
        urgency="high", estimated_cost_usd=40000, estimated_weeks=12),
    "has_synthesis_route":     CRONeed(need="Medicinal chemistry / synthesis", assay_type="chemistry",
        cro_examples=["WuXi Chemistry","Pharmaron","Enamine"],
        urgency="blocking", estimated_cost_usd=25000, estimated_weeks=8),
    "has_analytical_methods":  CRONeed(need="Analytical method development", assay_type="analytical",
        cro_examples=["SGS","Intertek","Charles River Analytical"],
        urgency="blocking", estimated_cost_usd=15000, estimated_weeks=6),
    "has_gmp_batch":           CRONeed(need="GMP synthesis batch", assay_type="cdmo",
        cro_examples=["Lonza","Patheon (Thermo)","Albany Molecular (AMRI)"],
        urgency="high", estimated_cost_usd=200000, estimated_weeks=26),
}

FUNDING_OPPORTUNITIES: dict[ProgramStage, list[FundingOpportunity]] = {
    ProgramStage.HIT_DISCOVERY: [
        FundingOpportunity(name="NIH SBIR Phase I (R43)", amount="$300K", mechanism="SBIR",
            fit_rationale="Early-stage discovery and mechanism proof of concept",
            url="https://sbir.nih.gov"),
        FundingOpportunity(name="BARDA CARB-X", amount="Up to $2M", mechanism="BARDA",
            fit_rationale="Antimicrobial resistance drug discovery — hit-to-lead funding",
            url="https://carb-x.org"),
        FundingOpportunity(name="Wellcome Trust Drug Discovery", amount="£0.5–2M", mechanism="foundation",
            fit_rationale="Global health/AMR early-stage programs"),
    ],
    ProgramStage.RESISTANCE_CHAR: [
        FundingOpportunity(name="BARDA CARB-X", amount="Up to $4M", mechanism="BARDA",
            fit_rationale="Resistance mechanism characterization and AMR drug development"),
        FundingOpportunity(name="NIH R01 (R21 exploratory)", amount="$275K/2yr", mechanism="NIH",
            fit_rationale="Mechanism of resistance studies — NIAID Study Section",
            url="https://grants.nih.gov"),
        FundingOpportunity(name="NIAID DMID contract", amount="Variable", mechanism="NIAID",
            fit_rationale="Antimicrobial drug development contracts"),
    ],
    ProgramStage.PRECLINICAL_PACKAGE: [
        FundingOpportunity(name="NIH SBIR Phase II (R44)", amount="$1.5–2M", mechanism="SBIR",
            fit_rationale="IND-enabling preclinical studies"),
        FundingOpportunity(name="BARDA BioShield", amount="Up to $10M", mechanism="BARDA",
            fit_rationale="Medical countermeasure preclinical development"),
        FundingOpportunity(name="DoD CDMRP", amount="$500K–3M", mechanism="DoD",
            fit_rationale="Disease-specific preclinical funding programs"),
    ],
    ProgramStage.GMP_READINESS: [
        FundingOpportunity(name="NIH SBIR Phase II Bridge", amount="$3M", mechanism="SBIR",
            fit_rationale="CMC, GMP manufacturing, IND filing"),
        FundingOpportunity(name="FDA Orphan Drug Designation", amount="Tax credit + waiver", mechanism="FDA",
            fit_rationale="Rare disease or unmet need programs"),
        FundingOpportunity(name="BARDA Contract", amount="$10–50M", mechanism="BARDA",
            fit_rationale="Biodefense or pandemic preparedness manufacturing"),
    ],
}

BIOINFORMATICS_NEEDS: dict[ProgramStage, list[str]] = {
    ProgramStage.RESISTANCE_CHAR: [
        "WGS of resistant mutants → variant calling (tools: samtools + GATK4)",
        "Structural docking for resistance mutations (tools: Vina + AutoSite)",
        "Cross-strain phylogenetics: conservation of target across pathogen strains (MAFFT + FastTree)",
        "Efflux pump gene expression profiling (RNA-seq differential expression)",
    ],
    ProgramStage.HIT_DISCOVERY: [
        "Target pathway enrichment (GSEA: gseapy tool)",
        "Off-target similarity search (DIAMOND sequence alignment)",
        "Compound scaffold clustering (RDKit fingerprints + k-means)",
    ],
    ProgramStage.PRECLINICAL_PACKAGE: [
        "PK/PD modeling (population PK analysis: Monolix or NONMEM)",
        "Toxicogenomics: expression profiling of tox endpoints",
        "Biomarker identification for clinical monitoring",
    ],
}

_TRANSLATIONAL_PROMPT = """You are a drug development expert assessing translational feasibility.

Given this drug program, identify:
1. The single most critical BLOCKING question that must be answered to advance
2. Top manufacturing/chemistry risks (specific to the compound, not generic)
3. Regulatory strategy recommendation

Respond ONLY as valid JSON:
{
  "blocking_question": "The single most critical unanswered question (≤25 words, cite compound/target name)",
  "manufacturing_risks": ["specific risk 1", "risk 2"],
  "regulatory_recommendation": "1-2 sentence specific strategy",
  "synthesis_assessment": "tractable|moderate|challenging + rationale",
  "cdmo_type": "small_molecule|peptide|biologic|oligo",
  "key_insight": "one sentence — where does this program stand today?"
}"""


def run_translational_agent(program: DrugProgram) -> DrugProgram:
    """Full translational analysis. Updates program in place."""

    # 1. Evidence gaps
    stage_reqs = STAGE_EVIDENCE_MAP.get(program.current_stage, [])
    blocking, critical = [], []
    for field, label, level in stage_reqs:
        if not getattr(program.evidence, field, False):
            (blocking if level == "blocking" else critical).append(label)
    program.evidence.blocking_gaps = blocking
    program.evidence.critical_gaps = critical

    # 2. Manufacturing analysis
    mfg = ManufacturingProfile()
    comp = program.compound
    if comp.synthesis_steps:
        mfg.synthesis_tractability = (
            "tractable" if comp.synthesis_steps <= 5
            else "moderate" if comp.synthesis_steps <= 10
            else "challenging"
        )
    elif comp.molecular_weight:
        mfg.synthesis_tractability = (
            "tractable" if comp.molecular_weight < 400
            else "moderate" if comp.molecular_weight < 600
            else "challenging"
        )
    else:
        mfg.synthesis_tractability = "unknown"

    risks = []
    if comp.logp and comp.logp > 4:
        risks.append(f"High cLogP ({comp.logp:.1f}) — solubility challenges expected at scale")
    if comp.synthesis_steps and comp.synthesis_steps > 8:
        risks.append(f"{comp.synthesis_steps}-step synthesis — yield loss and COGS concern")
    if not program.evidence.has_synthesis_route:
        risks.append("No synthesis route established — CMC work not started")
    if comp.psa and comp.psa > 140:
        risks.append(f"High PSA ({comp.psa:.0f} Å²) — poor oral bioavailability likely")
    mfg.scale_up_risks = risks
    mfg.cdmo_readiness = (
        "ready" if (program.evidence.has_synthesis_route and program.evidence.has_analytical_methods and program.evidence.has_gmp_batch)
        else "partially_ready" if (program.evidence.has_synthesis_route and program.evidence.has_analytical_methods)
        else "not_ready"
    )
    mfg.recommended_cdmo_type = "small_molecule"
    program.manufacturing = mfg

    # 3. CRO recommendations
    cro_needs = []
    for field, cro in CRO_RECOMMENDATIONS.items():
        if not getattr(program.evidence, field, False) and len(cro_needs) < 5:
            cro_needs.append(cro)
    program.execution.cro_needs = cro_needs

    # 4. Bioinformatics
    program.execution.bioinformatics_needs = BIOINFORMATICS_NEEDS.get(program.current_stage, [])

    # 5. Funding
    program.execution.grant_opportunities = FUNDING_OPPORTUNITIES.get(program.current_stage, [])

    # 6. LLM synthesis
    try:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY",""))
        ctx = {
            "stage": program.stage_label,
            "compound": comp.name,
            "target": program.target.gene,
            "organism": program.target.organism,
            "ic50_nm": comp.ic50_nm,
            "mic_ugml": comp.mic_ugml,
            "synthesis_steps": comp.synthesis_steps,
            "logp": comp.logp,
            "blocking_gaps": blocking[:3],
            "resistance_mutations": program.resistance.resistance_mutations[:3],
            "fold_shift": program.resistance.fold_shift,
        }
        msg = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=500,
            messages=[{"role":"user","content":f"{_TRANSLATIONAL_PROMPT}\n\n{json.dumps(ctx)}"}],
        )
        raw = msg.content[0].text.strip().replace("```json","").replace("```","").strip()
        data = json.loads(raw)
        program.blocking_question = data.get("blocking_question")
        if data.get("manufacturing_risks"):
            mfg.scale_up_risks = (mfg.scale_up_risks + data["manufacturing_risks"])[:5]
        program.add_trace(
            len(program.agent_trace)+1, "TranslationalAgent",
            "Translational feasibility",
            data.get("key_insight", f"CDMO readiness: {mfg.cdmo_readiness}, {len(blocking)} blocking gaps"),
            "manufacturing",
        )
    except Exception as e:
        logger.warning(f"Translational LLM failed: {e}")
        program.blocking_question = (
            f"What evidence is needed to advance {comp.name or 'lead compound'} "
            f"past {program.stage_label}?"
        )
        program.add_trace(
            len(program.agent_trace)+1, "TranslationalAgent",
            "Translational feasibility",
            f"CDMO readiness: {mfg.cdmo_readiness}, {len(blocking)} blocking gaps, "
            f"{len(program.execution.cro_needs)} CRO needs identified",
            "manufacturing",
        )

    return program
