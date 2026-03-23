"""
models/drug_program.py  ★ THE ANCHOR OBJECT

The DrugProgram is the central object AIDEN reasons around.
Not a molecule. A structured scientific program with three dimensions:

  Scientific:     target, mechanism, resistance, efficacy
  Development:    stage, evidence gaps, regulatory readiness
  Manufacturing:  synthesis tractability, CDMO suitability

Every parser writes TO this object.
Every agent reads FROM and writes TO this object.
The LLM reasons OVER this object — never over raw files.

"Given the data we have today — what should we do next
 to turn this into a real drug?"
"""

from __future__ import annotations
from typing import Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import uuid
from datetime import datetime


# ═══════════════════════════════════════════════════════════
# STAGE TAXONOMY — T1 through T8
# ═══════════════════════════════════════════════════════════

class ProgramStage(str, Enum):
    # Discovery
    TARGET_ID           = "target_id"
    HIT_DISCOVERY       = "hit_discovery"
    HIT_TRIAGE          = "hit_triage"
    # Characterization
    RESISTANCE_CHAR     = "resistance_characterization"
    HIT_TO_LEAD         = "hit_to_lead"
    LEAD_OPTIMIZATION   = "lead_optimization"
    # Preclinical
    ADMET_PROFILING     = "admet_profiling"
    VALIDATION_PLANNING = "validation_planning"
    PRECLINICAL_PACKAGE = "preclinical_package"
    # Translational
    IND_ENABLING        = "ind_enabling"
    GMP_READINESS       = "gmp_readiness"
    CDMO_EVALUATION     = "cdmo_evaluation"
    PHASE1_DESIGN       = "phase1_design"
    UNKNOWN             = "unknown"


STAGE_META: dict[ProgramStage, dict] = {
    ProgramStage.TARGET_ID:           {"label": "Target Identification",          "t_num": "T1", "color": "#818cf8"},
    ProgramStage.HIT_DISCOVERY:       {"label": "Hit Discovery",                   "t_num": "T2", "color": "#34d399"},
    ProgramStage.HIT_TRIAGE:          {"label": "Hit Triage",                      "t_num": "T3", "color": "#34d399"},
    ProgramStage.RESISTANCE_CHAR:     {"label": "Resistance Characterization",     "t_num": "T3", "color": "#fbbf24"},
    ProgramStage.HIT_TO_LEAD:         {"label": "Hit-to-Lead",                     "t_num": "T4", "color": "#f97316"},
    ProgramStage.LEAD_OPTIMIZATION:   {"label": "Lead Optimization",               "t_num": "T5", "color": "#f97316"},
    ProgramStage.ADMET_PROFILING:     {"label": "ADMET Profiling",                 "t_num": "T5", "color": "#fb923c"},
    ProgramStage.VALIDATION_PLANNING: {"label": "Experimental Validation Planning","t_num": "T6", "color": "#e879f9"},
    ProgramStage.PRECLINICAL_PACKAGE: {"label": "Preclinical Package Gap Analysis","t_num": "T6", "color": "#e879f9"},
    ProgramStage.IND_ENABLING:        {"label": "IND-Enabling Studies",            "t_num": "T7", "color": "#f43f5e"},
    ProgramStage.GMP_READINESS:       {"label": "GMP Readiness Review",            "t_num": "T7", "color": "#f43f5e"},
    ProgramStage.CDMO_EVALUATION:     {"label": "CDMO/Manufacturing Evaluation",   "t_num": "T8", "color": "#ef4444"},
    ProgramStage.PHASE1_DESIGN:       {"label": "Phase 1 Design",                  "t_num": "T8", "color": "#dc2626"},
    ProgramStage.UNKNOWN:             {"label": "Classifying…",                    "t_num": "?",  "color": "#6b7280"},
}


# ═══════════════════════════════════════════════════════════
# SCIENTIFIC DIMENSION
# ═══════════════════════════════════════════════════════════

class TargetProfile(BaseModel):
    gene: Optional[str] = None
    protein: Optional[str] = None
    mechanism_of_action: Optional[str] = None
    pathway: Optional[str] = None
    indication: Optional[str] = None
    organism: Optional[str] = None                  # "S. aureus", "E. coli", "H. sapiens"
    target_class: Optional[str] = None              # kinase | gyrase | protease | GPCR
    druggability_score: Optional[float] = None
    validated_in_disease: bool = False
    known_resistance_mechanisms: list[str] = Field(default_factory=list)


class CompoundProfile(BaseModel):
    name: Optional[str] = None
    smiles: Optional[str] = None
    chembl_id: Optional[str] = None
    ic50_nm: Optional[float] = None
    mic_ugml: Optional[float] = None                # antibiotics
    selectivity_ratio: Optional[float] = None
    structural_class: Optional[str] = None
    synthesis_steps: Optional[int] = None
    synthesis_complexity: Optional[str] = None      # simple | moderate | complex
    molecular_weight: Optional[float] = None
    logp: Optional[float] = None
    psa: Optional[float] = None
    hbd: Optional[int] = None
    hba: Optional[int] = None
    lipinski_pass: Optional[bool] = None
    ro3_pass: Optional[bool] = None                 # fragment-like


class ResistanceProfile(BaseModel):
    resistant_strains: list[str] = Field(default_factory=list)
    sensitive_strains: list[str] = Field(default_factory=list)
    resistance_mutations: list[str] = Field(default_factory=list)   # e.g. ["gyrA D87N", "parC S80I"]
    fold_shift: Optional[float] = None
    mechanism: Optional[str] = None                  # efflux | target_mutation | enzyme_degradation | bypass
    cross_resistance_compounds: list[str] = Field(default_factory=list)
    collateral_sensitive_compounds: list[str] = Field(default_factory=list)
    frequency_of_resistance: Optional[float] = None  # mutations per cell per generation
    mic_values: list[dict] = Field(default_factory=list)
    characterized: bool = False


class EfficacySignals(BaseModel):
    in_vitro_confirmed: bool = False
    in_vivo_confirmed: bool = False
    cell_line: Optional[str] = None
    organism_model: Optional[str] = None
    animal_model: Optional[str] = None
    efficacy_endpoint: Optional[str] = None
    primary_data_points: list[dict] = Field(default_factory=list)
    time_kill_data: Optional[dict] = None


# ═══════════════════════════════════════════════════════════
# DEVELOPMENT DIMENSION
# ═══════════════════════════════════════════════════════════

class EvidencePackage(BaseModel):
    """
    Tracks what evidence exists and what is missing.
    Set deterministically by parsers — never by LLM guess.
    """
    # Target validation
    has_target_validation: bool = False
    has_mechanism_confirmed: bool = False
    # Efficacy
    has_dose_response: bool = False
    has_selectivity_data: bool = False
    has_mic_data: bool = False
    has_time_kill_data: bool = False
    has_resistance_profiling: bool = False
    has_in_vivo_efficacy: bool = False
    # ADMET
    has_solubility: bool = False
    has_permeability: bool = False
    has_metabolic_stability: bool = False
    has_cyp_inhibition: bool = False
    has_herg_data: bool = False
    has_genotoxicity: bool = False
    has_acute_toxicity: bool = False
    has_repeat_dose_tox: bool = False
    # Manufacturing
    has_synthesis_route: bool = False
    has_analytical_methods: bool = False
    has_forced_degradation: bool = False
    has_gmp_batch: bool = False
    # Regulatory
    has_ind_filed: bool = False
    has_clinical_protocol: bool = False
    # Gaps (computed by translational agent)
    blocking_gaps: list[str] = Field(default_factory=list)
    critical_gaps: list[str] = Field(default_factory=list)


class RegulatoryReadiness(BaseModel):
    ind_filed: bool = False
    ind_target_date: Optional[str] = None
    breakthrough_therapy: bool = False
    fast_track: bool = False
    orphan_drug: bool = False
    fda_guidance_docs: list[str] = Field(default_factory=list)
    missing_ind_components: list[str] = Field(default_factory=list)
    regulatory_strategy: Optional[str] = None


# ═══════════════════════════════════════════════════════════
# MANUFACTURING DIMENSION
# ═══════════════════════════════════════════════════════════

class ManufacturingProfile(BaseModel):
    synthesis_tractability: Optional[str] = None       # tractable | moderate | challenging | unknown
    estimated_steps: Optional[int] = None
    key_starting_materials: list[str] = Field(default_factory=list)
    scale_up_risks: list[str] = Field(default_factory=list)
    cdmo_readiness: Optional[str] = None               # ready | partially_ready | not_ready | unknown
    recommended_cdmo_type: Optional[str] = None        # small_molecule | peptide | biologic
    estimated_cogs_per_gram_usd: Optional[float] = None
    known_blockers: list[str] = Field(default_factory=list)
    gmp_batch_needed_g: Optional[float] = None
    stability_concerns: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════
# EXECUTION GUIDANCE
# ═══════════════════════════════════════════════════════════

class CRONeed(BaseModel):
    need: str
    assay_type: str
    cro_examples: list[str] = Field(default_factory=list)
    urgency: str = "high"
    estimated_cost_usd: Optional[int] = None
    estimated_weeks: Optional[int] = None


class FundingOpportunity(BaseModel):
    name: str
    amount: str
    mechanism: str               # SBIR | BARDA | NIH_R01 | DoD | private
    fit_rationale: str
    deadline: Optional[str] = None
    url: Optional[str] = None


class ExecutionGuidance(BaseModel):
    cro_needs: list[CRONeed] = Field(default_factory=list)
    bioinformatics_needs: list[str] = Field(default_factory=list)
    cdmo_next_steps: list[str] = Field(default_factory=list)
    grant_opportunities: list[FundingOpportunity] = Field(default_factory=list)
    estimated_next_milestone_weeks: Optional[int] = None
    estimated_cost_next_phase_usd: Optional[int] = None
    team_gaps: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════
# RANKED ACTION
# ═══════════════════════════════════════════════════════════

class DrugProgramAction(BaseModel):
    rank: int
    category: str        # experiment | control | analysis | regulatory | manufacturing | funding
    action: str          # Specific actionable instruction ≤15 words
    rationale: str       # Cite actual data values — no generic advice
    evidence_ref: str    # audit_001 | contra_001 | gap_001 | field_name | pmid
    urgency: str         # blocking | high | medium | low
    estimated_cost_usd: Optional[int] = None
    estimated_weeks: Optional[int] = None
    cro_type: Optional[str] = None
    stage_gate: bool = False    # MUST complete before advancing to next stage


# ═══════════════════════════════════════════════════════════
# AUDIT FLAGS + CONTRADICTIONS + GAPS
# ═══════════════════════════════════════════════════════════

class AuditFlag(BaseModel):
    id: str
    type: str            # missing_control | scaffold_artifact | replicate_concern | unit_ambiguity
    severity: str        # high | medium | low
    title: str
    detail: str
    field_source: str


class Contradiction(BaseModel):
    id: str
    compound: str
    your_value: float
    your_unit: str
    your_condition: str
    lit_range_low: float
    lit_range_high: float
    lit_median: float
    fold_difference: float
    pmids: list[str]
    explanations: list[str]
    recommended_action: str
    is_likely_artifact: bool = False
    is_potentially_novel: bool = False


class EpistemicGap(BaseModel):
    id: str
    query: str
    gene: str
    compound: str
    condition: Optional[str] = None
    intersection_paper_count: int
    gene_paper_count: int
    classification: str  # white_space | emerging | well_studied
    novelty_score: float
    viability_signal: Optional[str] = None
    guidance: str


class LiteratureResult(BaseModel):
    pmid: str
    title: str
    authors: str
    journal: str
    year: int
    abstract: str
    relevance_score: float
    triggered_by: str
    quantitative_claims: list[dict] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════
# THE DRUG PROGRAM OBJECT  ★
# ═══════════════════════════════════════════════════════════

class DrugProgram(BaseModel):
    """
    The central anchor object AIDEN reasons around.

    Not a molecule — a structured scientific program.
    Tracks scientific, development, and manufacturing dimensions.

    Every parser writes TO this.
    Every agent reads FROM and writes TO this.
    The LLM reasons OVER this — never over raw files.
    """
    # Identity
    program_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8].upper())
    program_name: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    uploaded_files: list[str] = Field(default_factory=list)

    # ── Scientific dimension ──────────────────────────────
    target: TargetProfile = Field(default_factory=TargetProfile)
    compound: CompoundProfile = Field(default_factory=CompoundProfile)
    all_compounds: list[dict] = Field(default_factory=list)      # full compound screen
    all_variants: list[dict] = Field(default_factory=list)       # full variant list
    resistance: ResistanceProfile = Field(default_factory=ResistanceProfile)
    efficacy: EfficacySignals = Field(default_factory=EfficacySignals)

    # ── Development dimension ─────────────────────────────
    current_stage: ProgramStage = ProgramStage.UNKNOWN
    stage_confidence: float = 0.0
    stage_rationale: str = ""
    previous_stages: list[str] = Field(default_factory=list)
    evidence: EvidencePackage = Field(default_factory=EvidencePackage)
    regulatory: RegulatoryReadiness = Field(default_factory=RegulatoryReadiness)

    # ── Manufacturing dimension ───────────────────────────
    manufacturing: ManufacturingProfile = Field(default_factory=ManufacturingProfile)
    execution: ExecutionGuidance = Field(default_factory=ExecutionGuidance)

    # ── Agent outputs ─────────────────────────────────────
    audit_flags: list[AuditFlag] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    epistemic_gaps: list[EpistemicGap] = Field(default_factory=list)
    literature: list[LiteratureResult] = Field(default_factory=list)
    ranked_actions: list[DrugProgramAction] = Field(default_factory=list)
    key_finding: Optional[str] = None
    blocking_question: Optional[str] = None

    # ── Agent trace (streamed to UI) ──────────────────────
    agent_trace: list[dict] = Field(default_factory=list)

    # ─── Methods ─────────────────────────────────────────

    def add_trace(self, step: int, agent: str, action: str, finding: str, source: str = ""):
        self.agent_trace.append({
            "step": step, "agent": agent,
            "action": action, "finding": finding, "source": source,
        })
        self.updated_at = datetime.utcnow().isoformat()

    @property
    def stage_label(self) -> str:
        return STAGE_META.get(self.current_stage, {}).get("label", "Unknown")

    @property
    def stage_color(self) -> str:
        return STAGE_META.get(self.current_stage, {}).get("color", "#6b7280")

    @property
    def stage_t_num(self) -> str:
        return STAGE_META.get(self.current_stage, {}).get("t_num", "?")

    @property
    def completeness_pct(self) -> int:
        ev = self.evidence
        fields = [
            ev.has_target_validation, ev.has_mechanism_confirmed,
            ev.has_dose_response, ev.has_selectivity_data,
            ev.has_resistance_profiling, ev.has_in_vivo_efficacy,
            ev.has_solubility, ev.has_metabolic_stability,
            ev.has_synthesis_route, ev.has_gmp_batch,
        ]
        return round(sum(1 for f in fields if f) / len(fields) * 100)

    @property
    def gmp_readiness_pct(self) -> int:
        score = 0
        ev = self.evidence
        if ev.has_synthesis_route:     score += 25
        if ev.has_analytical_methods:  score += 20
        if ev.has_forced_degradation:  score += 20
        if ev.has_gmp_batch:           score += 35
        return min(score, 100)

    @property
    def top_compound_name(self) -> str:
        return self.compound.name or "Lead compound"

    @property
    def has_high_severity_flags(self) -> bool:
        return any(f.severity == "high" for f in self.audit_flags)

    @property
    def blocking_action(self) -> Optional[DrugProgramAction]:
        for a in self.ranked_actions:
            if a.urgency == "blocking":
                return a
        return self.ranked_actions[0] if self.ranked_actions else None
