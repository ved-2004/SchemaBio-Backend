"""
LAYER 3 — Drug-to-Market Execution Engine (SchemaBio)
=========================================================
Adapted from the legacy AIDEN-AMP Layer 3 engine.

Accepts ExecutionPlanningInput from the SchemaBio ingestion layer and returns
a structured execution roadmap matching the schema defined in:
  docs/execution-planning-layer-handoff.md

Maps scientific evidence to:
  - Funding/grant strategy (12 programs, scored with stage mismatch penalty)
  - CRO/lab partner routing (all stages)
  - CDMO/GMP readiness score (data-aware bonuses)
  - FDA pathway (QIDP, Fast Track, LPAD, Breakthrough, PRV)
  - EMA/MHRA international pathway signals
  - IND-enabling study checklist (data-aware completion, not linear ordering)
  - Stage timeline + budget estimates
  - Structured competitive landscape per pathogen
  - IP/patent landscape per gene family
  - Probability of success by stage
  - Grant stacking guidance
  - Live ClinicalTrials.gov + openFDA lookups
"""
from __future__ import annotations

import asyncio
import httpx
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any

from api.legacy.agents.fda_agent import enrich_fda_intelligence
from api.schemas.ingestion import ExecutionPlanningInput, ExtractedSignal


# ─────────────────────────────────────────────
# Internal model types (self-contained — no external dependency)
# ─────────────────────────────────────────────

class WorkflowStage(str, Enum):
    UNKNOWN = "unknown"
    TARGET_IDENTIFICATION = "target_identification"
    HIT_DISCOVERY = "hit_discovery"
    RESISTANCE_MECHANISM_CHARACTERIZATION = "resistance_mechanism_characterization"
    EXPERIMENTAL_VALIDATION = "experimental_validation"
    PRECLINICAL_GAP_ANALYSIS = "preclinical_gap_analysis"
    MANUFACTURING_FEASIBILITY = "manufacturing_feasibility"


class EvidenceStrength(str, Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


@dataclass
class MICEntry:
    compound: str
    organism: str
    mic_value: float
    unit: str = "ug/mL"
    method: Optional[str] = None
    source: Optional[str] = None


@dataclass
class ResistanceGene:
    gene_name: str
    mechanism: Optional[str] = None
    drug_class: Optional[str] = None
    card_accession: Optional[str] = None
    prevalence: Optional[str] = None


@dataclass
class Compound:
    name: str
    cid: Optional[str] = None
    smiles: Optional[str] = None
    molecular_weight: Optional[float] = None
    logp: Optional[float] = None
    hbd: Optional[int] = None
    hba: Optional[int] = None
    drug_likeness_score: Optional[float] = None
    mic_entries: list = field(default_factory=list)
    toxicity_flags: list = field(default_factory=list)


@dataclass
class FundingTarget:
    program_name: str
    agency: str
    fit_score: float
    stage_match: bool = False
    eligibility_gaps: list = field(default_factory=list)
    url: Optional[str] = None
    award_size: Optional[str] = None
    next_deadline: Optional[str] = None


@dataclass
class TranslationalReadiness:
    cdmo_readiness_score: float = 0.0
    evidence_completeness_score: float = 0.0
    scale_up_blockers: list = field(default_factory=list)
    missing_ind_studies: list = field(default_factory=list)
    qidp_eligible: Optional[bool] = None
    fast_track_eligible: Optional[bool] = None
    lpad_eligible: Optional[bool] = None
    breakthrough_eligible: Optional[bool] = None
    prv_eligible: Optional[bool] = None
    fda_pathway: Optional[str] = None
    cro_partner_type: Optional[str] = None
    formulation_data_present: bool = False
    in_vivo_data_present: bool = False


@dataclass
class LegacyProgramState:
    """Internal state object used by the Layer 3 execution engine."""
    session_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    stage: WorkflowStage = WorkflowStage.UNKNOWN
    organism: Optional[str] = None
    target_gene: Optional[str] = None
    project_summary: Optional[str] = None

    resistance_genes: list = field(default_factory=list)   # List[ResistanceGene]
    compounds: list = field(default_factory=list)           # List[Compound]
    mic_data: list = field(default_factory=list)            # List[MICEntry]

    has_assay_csv: bool = False
    has_variant_file: bool = False
    has_paper_pdf: bool = False
    has_compound_screen: bool = False
    has_free_text: bool = False

    evidence_strength: EvidenceStrength = EvidenceStrength.WEAK
    missing_flags: list = field(default_factory=list)
    data_quality_notes: list = field(default_factory=list)

    experiment_recommendations: list = field(default_factory=list)
    hypothesis_card: Optional[str] = None
    literature_context: list = field(default_factory=list)

    funding_targets: list = field(default_factory=list)              # List[FundingTarget]
    translational_readiness: Optional[TranslationalReadiness] = None
    execution_brief: Optional[str] = None

    competitive_signals: list = field(default_factory=list)
    market_intelligence: dict = field(default_factory=dict)
    ip_landscape: dict = field(default_factory=dict)
    fda_notes: list = field(default_factory=list)
    stage_timeline: dict = field(default_factory=dict)
    cro_details: dict = field(default_factory=dict)
    probability_of_success: dict = field(default_factory=dict)
    grant_stacks: list = field(default_factory=list)
    international_regulatory: dict = field(default_factory=dict)
    live_trials: list = field(default_factory=list)
    fda_intelligence: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# Funding / Grant Database (12 programs)
# next_deadline: recurring or approximate — users must verify
# ─────────────────────────────────────────────

FUNDING_DATABASE = [
    {
        "program_name": "CARB-X Explorer Award",
        "agency": "CARB-X (Wellcome / BARDA / Gates Foundation)",
        "eligible_stages": [WorkflowStage.HIT_DISCOVERY, WorkflowStage.RESISTANCE_MECHANISM_CHARACTERIZATION],
        "award_size": "Up to $2M",
        "url": "https://carb-x.org/apply/",
        "next_deadline": "Quarterly rounds — check carb-x.org for current call dates",
        "focus": ["novel antibiotics", "non-traditional therapies", "antibiotic resistance", "gram-negative"],
        "eligibility_requirements": [
            "Novel mechanism of action or differentiated spectrum",
            "Demonstrated in vitro activity (MIC data required)",
            "Not a me-too compound",
            "Early-stage company or academic spinout preferred",
        ],
        "stackable_with": ["NIH NIAID R01 — Drug Development for Drug-Resistant Bacteria"],
    },
    {
        "program_name": "CARB-X Development Award",
        "agency": "CARB-X",
        "eligible_stages": [WorkflowStage.EXPERIMENTAL_VALIDATION, WorkflowStage.PRECLINICAL_GAP_ANALYSIS],
        "award_size": "Up to $20M",
        "url": "https://carb-x.org/apply/",
        "next_deadline": "By invitation after Explorer Award or direct application — check carb-x.org",
        "focus": ["lead optimization", "preclinical package", "gram-negative bacteria", "ESKAPE"],
        "eligibility_requirements": [
            "Completed hit-to-lead optimization",
            "In vivo efficacy data in at least one infection model",
            "Clear ADME/tox profile initiated",
        ],
        "stackable_with": ["BARDA CARB BAA — Advanced Development"],
    },
    {
        "program_name": "NIH NIAID R01 — Drug Development for Drug-Resistant Bacteria",
        "agency": "NIH NIAID",
        "eligible_stages": [
            WorkflowStage.TARGET_IDENTIFICATION,
            WorkflowStage.HIT_DISCOVERY,
            WorkflowStage.RESISTANCE_MECHANISM_CHARACTERIZATION,
            WorkflowStage.EXPERIMENTAL_VALIDATION,
        ],
        "award_size": "Up to $500K/year x 5 years",
        "url": "https://grants.nih.gov/grants/guide/pa-files/PAR-22-179.html",
        "next_deadline": "Standard R01 deadlines: Feb 5 / Jun 5 / Oct 5 (recurring annually)",
        "focus": ["mechanism research", "drug discovery", "antimicrobial resistance", "bacteria"],
        "eligibility_requirements": [
            "Academic or nonprofit institution (lead PI)",
            "Novel hypothesis with preliminary data",
            "Clear translational relevance to human disease",
        ],
        "stackable_with": [
            "CARB-X Explorer Award",
            "DARPA PREPARE — Prophylactic/Therapeutic AMR",
            "Wellcome Trust Innovator Award — AMR",
        ],
    },
    {
        "program_name": "BARDA CARB BAA — Advanced Development",
        "agency": "BARDA (Biomedical Advanced Research and Development Authority)",
        "eligible_stages": [WorkflowStage.PRECLINICAL_GAP_ANALYSIS, WorkflowStage.MANUFACTURING_FEASIBILITY],
        "award_size": "$10M-$200M",
        "url": "https://www.medicalcountermeasures.gov/barda/",
        "next_deadline": "Rolling BAA — check SAM.gov for open solicitations",
        "focus": ["ESKAPE pathogens", "gram-negative MDR", "Phase 1/2 ready", "carbapenem"],
        "eligibility_requirements": [
            "IND-enabling studies completed or actively in progress",
            "Phase 1 safety data (if Phase 2+ candidate)",
            "US manufacturing capability preferred",
            "QIDP designation advantageous",
        ],
        "stackable_with": ["AMR Action Fund — Series A/B Investment"],
    },
    {
        "program_name": "DARPA PREPARE — Prophylactic/Therapeutic AMR",
        "agency": "DARPA",
        "eligible_stages": [WorkflowStage.HIT_DISCOVERY, WorkflowStage.EXPERIMENTAL_VALIDATION],
        "award_size": "Up to $15M",
        "url": "https://www.darpa.mil/program/prepare",
        "next_deadline": "Check SAM.gov for PREPARE BAA open periods",
        "focus": ["novel modalities", "phage therapy", "CRISPR antimicrobials", "rapid development"],
        "eligibility_requirements": [
            "High-risk, high-reward approach",
            "Rapid deployment potential (field-deployable focus)",
            "Non-traditional antibiotic modality preferred",
        ],
        "stackable_with": ["NIH NIAID R01 — Drug Development for Drug-Resistant Bacteria"],
    },
    {
        "program_name": "Wellcome Trust Innovator Award — AMR",
        "agency": "Wellcome Trust",
        "eligible_stages": [
            WorkflowStage.TARGET_IDENTIFICATION,
            WorkflowStage.HIT_DISCOVERY,
            WorkflowStage.RESISTANCE_MECHANISM_CHARACTERIZATION,
        ],
        "award_size": "Up to GBP500K",
        "url": "https://wellcome.org/grant-funding/schemes/innovator-awards-health-innovation",
        "next_deadline": "Annual open call — typically Q1/Q2, check wellcome.org",
        "focus": ["diagnostics", "research tools", "global health", "low-middle income countries", "resistance"],
        "eligibility_requirements": [
            "Clear global health impact statement required",
            "Feasibility demonstrated with preliminary data",
            "Academic or early-stage company (pre-Series A)",
        ],
        "stackable_with": ["NIH NIAID R01 — Drug Development for Drug-Resistant Bacteria", "JPIAMR — REPAIR Impact Fund"],
    },
    {
        "program_name": "AMR Action Fund — Series A/B Investment",
        "agency": "AMR Action Fund (private; backed by 20+ pharma companies)",
        "eligible_stages": [
            WorkflowStage.EXPERIMENTAL_VALIDATION,
            WorkflowStage.PRECLINICAL_GAP_ANALYSIS,
            WorkflowStage.MANUFACTURING_FEASIBILITY,
        ],
        "award_size": "$25M-$100M (equity investment)",
        "url": "https://www.amractionfund.com/",
        "next_deadline": "Rolling — submit expression of interest via amractionfund.com",
        "focus": ["Phase 2/3 antibiotics", "gram-negative MDR", "ESKAPE pathogens", "carbapenem-resistant"],
        "eligibility_requirements": [
            "Phase 1 safety data completed",
            "Clear unmet medical need for target pathogen-indication",
            "QIDP-eligible pathogen-indication pair",
            "Commercial-stage company or spinout with management team",
        ],
        "stackable_with": ["BARDA CARB BAA — Advanced Development"],
    },
    {
        "program_name": "GARDP — R&D Partnership Grant",
        "agency": "GARDP (Global Antibiotic R&D Partnership)",
        "eligible_stages": [
            WorkflowStage.HIT_DISCOVERY,
            WorkflowStage.RESISTANCE_MECHANISM_CHARACTERIZATION,
            WorkflowStage.EXPERIMENTAL_VALIDATION,
        ],
        "award_size": "Up to $5M (+ co-development support)",
        "url": "https://gardp.org/partnerships/",
        "next_deadline": "Rolling — contact partnerships@gardp.org for current opportunities",
        "focus": ["global health", "neglected infections", "neonatal sepsis", "gonorrhea", "gram-negative"],
        "eligibility_requirements": [
            "Clear global access and equitable pricing commitments",
            "Partnerships with LMIC institutions preferred",
            "Non-profit or public-private model",
        ],
        "stackable_with": ["Wellcome Trust Innovator Award — AMR", "USAID GHSA — Antimicrobial Resistance Activity"],
    },
    {
        "program_name": "EU Horizon Europe — AMR Cluster (Innovative Health Initiative)",
        "agency": "European Commission / IHI JU",
        "eligible_stages": [
            WorkflowStage.TARGET_IDENTIFICATION,
            WorkflowStage.HIT_DISCOVERY,
            WorkflowStage.RESISTANCE_MECHANISM_CHARACTERIZATION,
            WorkflowStage.EXPERIMENTAL_VALIDATION,
        ],
        "award_size": "EUR2M-EUR6M per project",
        "url": "https://www.ihi.europa.eu/",
        "next_deadline": "Annual calls — check ec.europa.eu/info/funding-tenders for open AMR calls",
        "focus": ["antibacterial discovery", "diagnostics", "resistance surveillance", "gram-positive", "gram-negative"],
        "eligibility_requirements": [
            "EU-based consortium (>=3 legal entities from >=3 EU/associated countries)",
            "Industry in-kind or cash co-funding required",
            "Open science and open access commitment",
        ],
        "stackable_with": ["Innovate UK — Biomedical Catalyst (AMR Priority)"],
    },
    {
        "program_name": "USAID GHSA — Antimicrobial Resistance Activity",
        "agency": "USAID (Global Health Security Agenda)",
        "eligible_stages": [
            WorkflowStage.RESISTANCE_MECHANISM_CHARACTERIZATION,
            WorkflowStage.HIT_DISCOVERY,
            WorkflowStage.EXPERIMENTAL_VALIDATION,
        ],
        "award_size": "Up to $2M",
        "url": "https://www.usaid.gov/global-health/health-areas/amr",
        "next_deadline": "Check grants.gov for open GHSA/AMR solicitations",
        "focus": ["surveillance", "diagnostics", "low-resource settings", "global health", "resistance"],
        "eligibility_requirements": [
            "US-registered organization (or subgrant to non-US partner)",
            "LMIC implementation component required",
            "Tied to GHSA action package priorities",
        ],
        "stackable_with": ["GARDP — R&D Partnership Grant"],
    },
    {
        "program_name": "Innovate UK — Biomedical Catalyst (AMR Priority)",
        "agency": "UK Research and Innovation (UKRI)",
        "eligible_stages": [
            WorkflowStage.HIT_DISCOVERY,
            WorkflowStage.RESISTANCE_MECHANISM_CHARACTERIZATION,
            WorkflowStage.EXPERIMENTAL_VALIDATION,
            WorkflowStage.PRECLINICAL_GAP_ANALYSIS,
        ],
        "award_size": "Up to GBP2M",
        "url": "https://www.ukri.org/councils/innovate-uk/",
        "next_deadline": "Biomedical Catalyst rounds typically Q2/Q3 — check iuk.ukri.org",
        "focus": ["UK SME", "translational research", "AMR", "novel antibiotics", "diagnostics"],
        "eligibility_requirements": [
            "UK-based SME as lead applicant",
            "Commercially viable project plan with route to market",
            "Match funding required (30-50% from company or partners)",
        ],
        "stackable_with": ["EU Horizon Europe — AMR Cluster (Innovative Health Initiative)"],
    },
    {
        "program_name": "JPIAMR — REPAIR Impact Fund",
        "agency": "JPIAMR / ReAct (Joint Programming Initiative AMR)",
        "eligible_stages": [
            WorkflowStage.TARGET_IDENTIFICATION,
            WorkflowStage.HIT_DISCOVERY,
            WorkflowStage.RESISTANCE_MECHANISM_CHARACTERIZATION,
        ],
        "award_size": "Up to EUR1M",
        "url": "https://www.jpiamr.eu/",
        "next_deadline": "Biennial call — check jpiamr.eu for next open call announcement",
        "focus": ["early research", "target discovery", "AMR mechanisms", "academia"],
        "eligibility_requirements": [
            "Lead applicant from JPIAMR member country institution",
            "Transnational collaborative project (>=2 countries)",
            "Open data commitment and FAIR data plan",
        ],
        "stackable_with": ["Wellcome Trust Innovator Award — AMR", "NIH NIAID R01 — Drug Development for Drug-Resistant Bacteria"],
    },
]


# ─────────────────────────────────────────────
# CRO / Lab Partner Routing (all stages)
# ─────────────────────────────────────────────

CRO_ROUTING = {
    WorkflowStage.UNKNOWN: {
        "partner_type": "Scientific Advisory / Feasibility Consultant",
        "capabilities_needed": ["Program scoping", "Assay feasibility assessment", "Target validation review"],
        "example_partners": ["Evotec Scientific Advisory", "Charles River Consulting", "Quotient Sciences"],
        "biosafety_level": "N/A",
        "timeline_estimate": "2-4 weeks",
    },
    WorkflowStage.TARGET_IDENTIFICATION: {
        "partner_type": "Structural Biology / Target Validation CRO",
        "capabilities_needed": ["Protein expression & purification", "Biochemical assay development", "Cryo-EM / X-ray crystallography"],
        "example_partners": ["Reaction Biology (Malvern, PA)", "SGC (Structural Genomics Consortium)", "Wuxi Biology"],
        "biosafety_level": "BSL-1/2",
        "timeline_estimate": "6-12 weeks",
    },
    WorkflowStage.RESISTANCE_MECHANISM_CHARACTERIZATION: {
        "partner_type": "Clinical Microbiology CRO (BSL-2 certified)",
        "capabilities_needed": ["MIC testing (CLSI/EUCAST)", "PCR genotyping", "WGS with AMR annotation"],
        "example_partners": ["Micromyx (Kalamazoo, MI)", "JMI Laboratories (North Liberty, IA)", "IHMA (Schaumburg, IL)"],
        "biosafety_level": "BSL-2",
        "timeline_estimate": "4-8 weeks",
    },
    WorkflowStage.HIT_DISCOVERY: {
        "partner_type": "HTS / Drug Discovery CRO",
        "capabilities_needed": ["384-well MIC screening", "Echo liquid handling", "Data normalization"],
        "example_partners": ["Evotec (Hamburg)", "Jubilant Biosys (Bangalore)", "Eurofins Discovery"],
        "biosafety_level": "BSL-2",
        "timeline_estimate": "6-10 weeks",
    },
    WorkflowStage.EXPERIMENTAL_VALIDATION: {
        "partner_type": "In Vivo Pharmacology CRO (AAALAC-accredited)",
        "capabilities_needed": ["Murine infection models", "PKPD sampling", "CFU counting"],
        "example_partners": ["Bioduro-Sundia (Beijing/San Diego)", "Wuxi AppTec", "Charles River Laboratories"],
        "biosafety_level": "BSL-2/ABSL-2",
        "timeline_estimate": "8-16 weeks",
    },
    WorkflowStage.PRECLINICAL_GAP_ANALYSIS: {
        "partner_type": "Full-Service Preclinical CRO (IND-enabling)",
        "capabilities_needed": ["GLP toxicology", "ADME/DMPK", "hERG safety", "Formulation development"],
        "example_partners": ["Covance (Labcorp Drug Development)", "Pacific BioLabs", "BioDuro"],
        "biosafety_level": "BSL-1/2",
        "timeline_estimate": "12-24 months",
    },
    WorkflowStage.MANUFACTURING_FEASIBILITY: {
        "partner_type": "CDMO (Contract Development and Manufacturing Organization)",
        "capabilities_needed": ["API synthesis scale-up", "GMP manufacturing", "ICH stability studies"],
        "example_partners": ["Lonza (Basel)", "CARBOGEN AMCIS", "Hovione", "Recipharm"],
        "biosafety_level": "GMP facility required",
        "timeline_estimate": "18-36 months to GMP batch",
    },
}


# ─────────────────────────────────────────────
# Stage Timeline Roadmap
# ─────────────────────────────────────────────

STAGE_TIMELINES = {
    WorkflowStage.UNKNOWN: {
        "months_to_ind": "60-84+",
        "months_to_phase1": "72-96+",
        "next_milestone": "Define target organism and resistance mechanism",
        "stage_cost_estimate": "N/A — scope not defined",
    },
    WorkflowStage.TARGET_IDENTIFICATION: {
        "months_to_ind": "48-72",
        "months_to_phase1": "54-78",
        "next_milestone": "Complete target validation + identify chemical starting points",
        "stage_cost_estimate": "$500K-$2M",
    },
    WorkflowStage.HIT_DISCOVERY: {
        "months_to_ind": "36-60",
        "months_to_phase1": "42-66",
        "next_milestone": "Confirm MIC activity + selectivity across >=2 organism strains",
        "stage_cost_estimate": "$1M-$4M",
    },
    WorkflowStage.RESISTANCE_MECHANISM_CHARACTERIZATION: {
        "months_to_ind": "30-54",
        "months_to_phase1": "36-60",
        "next_milestone": "Genotypic confirmation + MIC panel across >=30 clinical isolates",
        "stage_cost_estimate": "$500K-$2M",
    },
    WorkflowStage.EXPERIMENTAL_VALIDATION: {
        "months_to_ind": "18-36",
        "months_to_phase1": "24-42",
        "next_milestone": "Complete murine infection model + PKPD target identification",
        "stage_cost_estimate": "$3M-$8M",
    },
    WorkflowStage.PRECLINICAL_GAP_ANALYSIS: {
        "months_to_ind": "6-18",
        "months_to_phase1": "12-24",
        "next_milestone": "Complete GLP tox package + IND filing with FDA CDER",
        "stage_cost_estimate": "$5M-$20M",
    },
    WorkflowStage.MANUFACTURING_FEASIBILITY: {
        "months_to_ind": "0-6",
        "months_to_phase1": "6-12",
        "next_milestone": "GMP batch release + Phase 1 site activation",
        "stage_cost_estimate": "$10M-$40M",
    },
}


# ─────────────────────────────────────────────
# Probability of Success by Stage
# Source: Pew Charitable Trusts AMR analysis; BIO pipeline report 2023
# ─────────────────────────────────────────────

PROBABILITY_OF_SUCCESS = {
    WorkflowStage.UNKNOWN:                              {"to_phase1": "N/A",    "to_approval": "N/A",    "note": "Scope undefined"},
    WorkflowStage.TARGET_IDENTIFICATION:                {"to_phase1": "~5-10%", "to_approval": "~2-5%",  "note": "Most programs fail at target validation"},
    WorkflowStage.HIT_DISCOVERY:                        {"to_phase1": "~10-15%","to_approval": "~3-7%",  "note": "Hit-to-lead attrition is high (~90%)"},
    WorkflowStage.RESISTANCE_MECHANISM_CHARACTERIZATION:{"to_phase1": "~15-20%","to_approval": "~5-10%", "note": "Mechanism clarity improves lead selection"},
    WorkflowStage.EXPERIMENTAL_VALIDATION:              {"to_phase1": "~25-35%","to_approval": "~8-14%", "note": "In vivo data significantly derisk program"},
    WorkflowStage.PRECLINICAL_GAP_ANALYSIS:             {"to_phase1": "~50-70%","to_approval": "~15-25%","note": "IND package nearly complete — late-stage tox failure risk"},
    WorkflowStage.MANUFACTURING_FEASIBILITY:            {"to_phase1": "~80-90%","to_approval": "~20-30%","note": "Phase 1/2 clinical failure remains the dominant risk"},
}


# ─────────────────────────────────────────────
# Structured Competitive Landscape
# ─────────────────────────────────────────────

COMPETITIVE_LANDSCAPE = {
    "klebsiella pneumoniae": [
        {"drug": "Ceftazidime-avibactam (Avycaz)",           "status": "Approved",  "year": 2015, "coverage": "KPC/OXA-48-producing strains"},
        {"drug": "Meropenem-vaborbactam (Vabomere)",         "status": "Approved",  "year": 2017, "coverage": "KPC-specific; limited MBL activity"},
        {"drug": "Imipenem-cilastatin-relebactam (Recarbrio)","status": "Approved",  "year": 2019, "coverage": "KPC + some OXA; no MBL coverage"},
        {"drug": "Cefiderocol (Fetroja)",                    "status": "Approved",  "year": 2019, "coverage": "Broad including MBL/OXA; siderophore cephalosporin"},
        {"drug": "Aztreonam-avibactam (Emblaveo)",           "status": "Approved",  "year": 2024, "coverage": "NDM/MBL-active; first approved MBL option"},
    ],
    "acinetobacter baumannii": [
        {"drug": "Sulbactam-durlobactam (Xacduro)",          "status": "Approved",  "year": 2023, "coverage": "First drug approved specifically for CRAB"},
        {"drug": "Cefiderocol (Fetroja)",                    "status": "Approved",  "year": 2019, "coverage": "Carbapenem-resistant A. baumannii indication"},
        {"drug": "Colistin (polymyxin E)",                   "status": "Approved",  "year": None, "coverage": "Last-resort; significant nephrotoxicity"},
        {"drug": "OXA-inhibitor combinations (multiple cos.)","status": "Phase 2",   "year": None, "coverage": "OXA-23/OXA-24 targeted — Entasis, Iterion pipeline"},
        {"drug": "Oral options",                             "status": "None",      "year": None, "coverage": "No approved oral therapy — critical unmet need"},
    ],
    "pseudomonas aeruginosa": [
        {"drug": "Ceftolozane-tazobactam (Zerbaxa)",         "status": "Approved",  "year": 2014, "coverage": "MDR P. aeruginosa; no MBL coverage"},
        {"drug": "Ceftazidime-avibactam",                    "status": "Approved",  "year": 2015, "coverage": "Limited vs. MBL-producing strains"},
        {"drug": "Imipenem-cilastatin-relebactam (Recarbrio)","status": "Approved",  "year": 2019, "coverage": "MDR P. aeruginosa indication"},
        {"drug": "Aztreonam-avibactam",                      "status": "Approved",  "year": 2024, "coverage": "MBL-active; covers VIM/NDM strains"},
        {"drug": "Phage therapy (multiple groups)",          "status": "Phase 2",   "year": None, "coverage": "Compassionate use only; not broadly approved"},
    ],
    "escherichia coli": [
        {"drug": "Ceftazidime-avibactam",                    "status": "Approved",  "year": 2015, "coverage": "ESBL + KPC-producing E. coli"},
        {"drug": "Temocillin",                               "status": "Approved",  "year": None, "coverage": "EU only; used for ESBL-E. coli UTI/bacteremia"},
        {"drug": "Gepotidacin (GSK)",                        "status": "Phase 3",   "year": 2024, "coverage": "Uncomplicated UTI (uUTI) — new mechanism (topoisomerase)"},
        {"drug": "Sulopenem-etzadroxil (Iterion)",           "status": "Phase 3",   "year": None, "coverage": "Oral ESBL-E. coli uUTI — awaiting FDA review"},
    ],
    "staphylococcus aureus": [
        {"drug": "Vancomycin",                               "status": "Approved",  "year": None, "coverage": "MRSA standard of care; increasing MIC creep"},
        {"drug": "Daptomycin (Cubicin)",                     "status": "Approved",  "year": 2003, "coverage": "MRSA bacteremia/endocarditis"},
        {"drug": "Ceftaroline (Teflaro)",                    "status": "Approved",  "year": 2010, "coverage": "MRSA skin/soft tissue + community-acquired pneumonia"},
        {"drug": "Dalbavancin (Dalvance)",                   "status": "Approved",  "year": 2014, "coverage": "Long-acting lipoglycopeptide; single-dose ABSSSI"},
        {"drug": "Exebacase (phage lysin, ContraFect)",       "status": "Phase 3",   "year": None, "coverage": "MRSA bacteremia; pivotal trial results pending"},
    ],
    "enterococcus faecium": [
        {"drug": "Linezolid (Zyvox)",                        "status": "Approved",  "year": 2000, "coverage": "VRE standard of care; resistance increasing (cfr gene)"},
        {"drug": "Daptomycin",                               "status": "Approved",  "year": 2003, "coverage": "VRE (off-label); resistance emerging"},
        {"drug": "Oritavancin (Orbactiv)",                   "status": "Approved",  "year": 2014, "coverage": "Some VRE activity (VanA); lipoglycopeptide"},
        {"drug": "Oral VRE options",                         "status": "None",      "year": None, "coverage": "No approved oral therapy for VRE — critical gap"},
    ],
    "mycobacterium tuberculosis": [
        {"drug": "Bedaquiline (Sirturo)",                    "status": "Approved",  "year": 2012, "coverage": "MDR-TB; ATP synthase inhibitor"},
        {"drug": "Delamanid (Deltyba)",                      "status": "Approved",  "year": 2014, "coverage": "EU/Japan approved; MDR-TB"},
        {"drug": "Pretomanid",                               "status": "Approved",  "year": 2019, "coverage": "BPaL regimen for XDR-TB (bedaquiline + linezolid)"},
        {"drug": "TB-PRACTECAL regimen",                     "status": "Phase 3",   "year": None, "coverage": "6-month all-oral MDR-TB cure regimen"},
    ],
}


# ─────────────────────────────────────────────
# IP / Patent Landscape Signals
# ─────────────────────────────────────────────

IP_LANDSCAPE = {
    "blakpc": {
        "landscape": "Dense — multiple beta-lactam/inhibitor combinations patented (AZ, Pfizer, Rempex/Melinta)",
        "freedom_to_operate": "Compound-specific FTO needed; inhibitor scaffolds (diazabicyclooctane, boronate) are crowded",
        "key_expiries": "Avibactam core patents 2031-2033; vaborbactam 2032-2035",
        "opportunity": "Novel inhibitor scaffolds or non-beta-lactam partners retain FTO",
    },
    "blandm": {
        "landscape": "Less crowded than KPC — MBL inhibitors emerging; aztreonam combinations now approved",
        "freedom_to_operate": "Aztreonam-avibactam approved 2024; novel standalone MBL inhibitors have broad FTO",
        "key_expiries": "Aztreonam-avibactam (Pfizer/AZ) core patents 2030-2037",
        "opportunity": "Standalone MBL inhibitor class largely unpatented — first-in-class opportunity",
    },
    "blaoxa": {
        "landscape": "Moderate — OXA-23/OXA-48 inhibitors in development; sulbactam-durlobactam approved 2023",
        "freedom_to_operate": "Broad FTO for non-OXA-A.bau OXA classes; OXA-48 inhibitors are open",
        "key_expiries": "Durlobactam (Entasis/Pfizer) core patents 2036-2040",
        "opportunity": "OXA-48 and OXA-51 class inhibitors remain largely open for novel scaffolds",
    },
    "mcr": {
        "landscape": "Sparse — colistin alternatives in early research; no targeted MCR inhibitor approved",
        "freedom_to_operate": "Broad FTO for novel colistin-sparing approaches and lipid A-targeted agents",
        "key_expiries": "No significant blocking patents for MCR-targeted agents identified",
        "opportunity": "First-in-class opportunity for MCR inhibitors or polymyxin-sparing combinations",
    },
    "vana": {
        "landscape": "Moderate — glycopeptide analogues and lipoglycopeptide landscape (vancomycin, oritavancin, dalbavancin)",
        "freedom_to_operate": "Oritavancin, dalbavancin patents expiring 2025-2026; broad FTO for non-glycopeptide VRE agents",
        "key_expiries": "Oritavancin (Melinta) 2025; dalbavancin (Allergan/AbbVie) 2026",
        "opportunity": "Non-glycopeptide VRE approaches (phage lysins, lipoteichoic acid inhibitors, type IV pilus) have broad FTO",
    },
    "default": {
        "landscape": "Patent search required before compound advancement",
        "freedom_to_operate": "Conduct FTO analysis with IP counsel before IND filing — gene-family specific",
        "key_expiries": "Consult PatSnap or Derwent World Patents Index for gene-family landscape",
        "opportunity": "Novel mechanisms of action generally retain strong FTO",
    },
}


# ─────────────────────────────────────────────
# Market Intelligence
# ─────────────────────────────────────────────

MARKET_INTELLIGENCE = {
    "klebsiella pneumoniae": {
        "global_tam": "~$3.2B (carbapenem-resistant Enterobacterales, MarketsandMarkets 2023)",
        "us_hospital_cases": "~13,000 CRE infections/year (CDC AR Threats Report 2023)",
        "mortality_rate": "~40-50% for carbapenem-resistant bloodstream infections",
        "unmet_need": "High — NDM/OXA-producing strains have very limited options post-2024",
        "payer_context": "Hospital formulary; QIDP 5-yr exclusivity + PASTEUR Act pull incentive if passed",
    },
    "acinetobacter baumannii": {
        "global_tam": "~$1.8B (carbapenem-resistant A. baumannii, GlobalData 2023)",
        "us_hospital_cases": "~8,500 CRAB infections/year (CDC 2023); ~700 deaths",
        "mortality_rate": "~45-60% for carbapenem-resistant strains in ICU",
        "unmet_need": "Critical — WHO Priority 1 pathogen; no oral options; Xacduro only recent approval",
        "payer_context": "ICU-predominant; hospital economics; GAIN Act + QIDP exclusivity critical for viability",
    },
    "pseudomonas aeruginosa": {
        "global_tam": "~$2.5B (MDR P. aeruginosa, allied market research 2023)",
        "us_hospital_cases": "~32,600 MDR P. aeruginosa cases/year (CDC 2023)",
        "mortality_rate": "~30-40% for MDR/XDR strains",
        "unmet_need": "High — MBL-producing XDR strains minimally covered even post-AZA approval",
        "payer_context": "CF patient community influential; hospital + specialty pharmacy; CF registry enables efficient trials",
    },
    "escherichia coli": {
        "global_tam": "~$4.1B (ESBL/carbapenem-resistant E. coli, WHO AMR market report 2022)",
        "us_hospital_cases": "~197,400 ESBL-producing E. coli infections/year (CDC 2023)",
        "mortality_rate": "~5-15% (significantly lower than non-fermenters)",
        "unmet_need": "Moderate — oral ESBL treatment gap remains; gepotidacin/sulopenem filling UTI gap",
        "payer_context": "Mixed inpatient/outpatient; oral agents command premium; high-volume indication favors payer acceptance",
    },
    "staphylococcus aureus": {
        "global_tam": "~$2.0B (MRSA, GlobalData 2023)",
        "us_hospital_cases": "~119,000 MRSA infections/year (CDC 2023); ~18,000 deaths",
        "mortality_rate": "~15-50% for bloodstream infections (endocarditis highest)",
        "unmet_need": "Moderate — multiple options available; biofilm/device-associated infections underserved",
        "payer_context": "Well-established MRSA drug market; premium pricing for long-acting single-dose agents",
    },
    "default": {
        "global_tam": "Global AMR drug market >$14B by 2030 (MarketsandMarkets 2022)",
        "us_hospital_cases": "~2.8M AMR infections/year, 35,000+ deaths (CDC AR Threats 2023)",
        "mortality_rate": "Variable by pathogen/resistance profile",
        "unmet_need": "Broadly high across all ESKAPE pathogens",
        "payer_context": "PASTEUR Act (pending US legislation) would provide $11B+ subscription-model pull incentive",
    },
}


# ─────────────────────────────────────────────
# IND-Enabling Study Checklist
# ─────────────────────────────────────────────

IND_ENABLING_STUDIES = [
    {"id": "in_vitro_activity", "name": "In vitro MIC panel (>=30 clinical isolates)",       "completion_check": "data"},
    {"id": "mechanism_id",      "name": "Mechanism of action identified",                     "completion_check": "data"},
    {"id": "in_vivo_efficacy",  "name": "In vivo efficacy (murine infection model)",          "completion_check": "stage",
     "stage_required": WorkflowStage.EXPERIMENTAL_VALIDATION},
    {"id": "adme",              "name": "ADME profiling (microsomal stability, PPB, CYP)",    "completion_check": "stage",
     "stage_required": WorkflowStage.PRECLINICAL_GAP_ANALYSIS},
    {"id": "herg_safety",       "name": "hERG cardiac safety assay (ICH S7B)",                "completion_check": "stage",
     "stage_required": WorkflowStage.PRECLINICAL_GAP_ANALYSIS},
    {"id": "acute_tox",         "name": "Acute toxicity study (GLP, 14-day)",                 "completion_check": "stage",
     "stage_required": WorkflowStage.PRECLINICAL_GAP_ANALYSIS},
    {"id": "repeat_dose_tox",   "name": "28-day repeat-dose toxicity study (GLP)",            "completion_check": "stage",
     "stage_required": WorkflowStage.PRECLINICAL_GAP_ANALYSIS},
    {"id": "genotox",           "name": "Genotoxicity panel (Ames + in vitro micronucleus)",  "completion_check": "stage",
     "stage_required": WorkflowStage.PRECLINICAL_GAP_ANALYSIS},
    {"id": "formulation",       "name": "Formulation development + stability data (ICH Q1)",  "completion_check": "stage",
     "stage_required": WorkflowStage.MANUFACTURING_FEASIBILITY},
    {"id": "gmp_batch",         "name": "GMP drug substance batch (>=1 batch)",                "completion_check": "stage",
     "stage_required": WorkflowStage.MANUFACTURING_FEASIBILITY},
    {"id": "cmc",               "name": "CMC documentation (ICH Q6A/Q6B)",                    "completion_check": "stage",
     "stage_required": WorkflowStage.MANUFACTURING_FEASIBILITY},
]

STAGE_ORDER = [
    WorkflowStage.UNKNOWN,
    WorkflowStage.TARGET_IDENTIFICATION,
    WorkflowStage.HIT_DISCOVERY,
    WorkflowStage.RESISTANCE_MECHANISM_CHARACTERIZATION,
    WorkflowStage.EXPERIMENTAL_VALIDATION,
    WorkflowStage.PRECLINICAL_GAP_ANALYSIS,
    WorkflowStage.MANUFACTURING_FEASIBILITY,
]


# ─────────────────────────────────────────────
# IND Study Completion (data-aware)
# ─────────────────────────────────────────────

def _is_ind_study_complete(study: dict, state: LegacyProgramState) -> bool:
    sid = study["id"]

    if study["completion_check"] == "data":
        if sid == "in_vitro_activity":
            return bool(state.mic_data)
        if sid == "mechanism_id":
            return any(g.mechanism for g in state.resistance_genes)

    required = study.get("stage_required")
    if required and required in STAGE_ORDER:
        current_idx = STAGE_ORDER.index(state.stage) if state.stage in STAGE_ORDER else 0
        return STAGE_ORDER.index(required) <= current_idx

    return False


# ─────────────────────────────────────────────
# CDMO/GMP Readiness Scoring
# ─────────────────────────────────────────────

def compute_cdmo_readiness_score(state: LegacyProgramState) -> float:
    stage_base_scores = {
        WorkflowStage.UNKNOWN: 0,
        WorkflowStage.TARGET_IDENTIFICATION: 5,
        WorkflowStage.HIT_DISCOVERY: 12,
        WorkflowStage.RESISTANCE_MECHANISM_CHARACTERIZATION: 18,
        WorkflowStage.EXPERIMENTAL_VALIDATION: 35,
        WorkflowStage.PRECLINICAL_GAP_ANALYSIS: 55,
        WorkflowStage.MANUFACTURING_FEASIBILITY: 75,
    }

    score = stage_base_scores.get(state.stage, 0)

    if state.mic_data:
        score += 5
    if state.resistance_genes and any(g.mechanism for g in state.resistance_genes):
        score += 5
    if (state.compounds
            and state.compounds[0].drug_likeness_score is not None
            and state.compounds[0].drug_likeness_score > 0.5):
        score += 5
    if state.compounds and state.compounds[0].molecular_weight:
        score += 2
    if len(state.compounds) >= 3:
        score += 3

    return min(100.0, round(score, 1))


# ─────────────────────────────────────────────
# Scale-up Blockers (data-aware)
# ─────────────────────────────────────────────

def get_scale_up_blockers(state: LegacyProgramState) -> list[str]:
    blockers = []
    stage = state.stage

    early_stages = [
        WorkflowStage.UNKNOWN,
        WorkflowStage.TARGET_IDENTIFICATION,
        WorkflowStage.HIT_DISCOVERY,
        WorkflowStage.RESISTANCE_MECHANISM_CHARACTERIZATION,
    ]

    if stage in early_stages:
        blockers.append("No in vivo efficacy data — required before any scale-up discussion")

        has_adme = any(
            c.logp is not None and c.hbd is not None
            for c in state.compounds
        )
        if not has_adme:
            blockers.append("No ADME/tox profile — PK properties unknown (logP/HBD not characterized)")

        if not state.compounds:
            blockers.append("No lead compound identified — hit-to-lead optimization not started")
        elif not any(c.molecular_weight for c in state.compounds):
            blockers.append("Lead compound physicochemistry not characterized — MW/logP required")

        blockers.append("No formulation data — route of administration undefined")

    elif stage == WorkflowStage.EXPERIMENTAL_VALIDATION:
        blockers.extend([
            "GLP toxicology studies not yet completed",
            "hERG cardiac safety data required (ICH S7B compliance)",
            "Synthetic route scalability to multi-gram scale not assessed",
        ])
        has_adme = any(c.logp is not None for c in state.compounds)
        if not has_adme:
            blockers.append("ADME package incomplete — microsomal stability and PPB data missing")

    elif stage == WorkflowStage.PRECLINICAL_GAP_ANALYSIS:
        blockers.extend([
            "CMC documentation not yet initiated (ICH Q6A/Q6B required for IND)",
            "GMP synthesis route not qualified — process chemistry scale-up needed",
            "ICH stability studies (Q1A) pending — shelf-life undefined",
            "Scale-up yield and purity data not available",
        ])

    elif stage == WorkflowStage.MANUFACTURING_FEASIBILITY:
        blockers.extend([
            "Phase 1 IND amendments and protocol finalization required",
            "Phase 1 trial site contracts and CTA submissions pending",
            "Drug product (formulated) GMP batch for clinical use not yet confirmed",
        ])

    return blockers


# ─────────────────────────────────────────────
# FDA Pathway
# ─────────────────────────────────────────────

def get_fda_pathway(state: LegacyProgramState) -> dict:
    org_lower = (state.organism or "").lower()

    has_eskape = any(
        p in org_lower for p in [
            "klebsiella", "acinetobacter", "pseudomonas", "staphylococcus aureus",
            "enterococcus", "enterobacter", "escherichia coli",
        ]
    )

    has_critical_resistance = any(
        any(k in g.gene_name.lower() for k in ["kpc", "ndm", "oxa", "vim", "imp", "mcr"])
        for g in state.resistance_genes
    )

    qidp_eligible = (has_eskape or has_critical_resistance) and bool(state.compounds)

    fast_track_eligible = qidp_eligible and state.stage in [
        WorkflowStage.EXPERIMENTAL_VALIDATION,
        WorkflowStage.PRECLINICAL_GAP_ANALYSIS,
        WorkflowStage.MANUFACTURING_FEASIBILITY,
    ]

    lpad_eligible = has_critical_resistance and bool(state.compounds)

    breakthrough_eligible = (
        qidp_eligible
        and state.stage in [WorkflowStage.PRECLINICAL_GAP_ANALYSIS, WorkflowStage.MANUFACTURING_FEASIBILITY]
        and state.evidence_strength == EvidenceStrength.STRONG
    )

    prv_eligible = any(
        p in org_lower for p in ["mycobacterium tuberculosis", "neisseria gonorrhoeae"]
    )

    if state.stage == WorkflowStage.MANUFACTURING_FEASIBILITY:
        pathway = "NDA (505(b)(1) or 505(b)(2)) — file with QIDP/Fast Track designation if eligible"
    elif state.stage in [WorkflowStage.PRECLINICAL_GAP_ANALYSIS, WorkflowStage.EXPERIMENTAL_VALIDATION]:
        pathway = "IND application (Phase 1 safety) -> Phase 2/3 efficacy — QIDP designation recommended before IND"
    else:
        pathway = "Pre-IND consultation with FDA CDER Division of Anti-Infectives — request Type B meeting"

    notes = [n for n in [
        "QIDP: 5-year market exclusivity extension + priority review voucher eligibility (GAIN Act)" if qidp_eligible else None,
        "Fast Track: rolling review + more frequent FDA interactions during development" if fast_track_eligible else None,
        "LPAD: approval based on smaller pivotal trials for serious/life-threatening infections" if lpad_eligible else None,
        "Breakthrough Therapy: intensive FDA guidance; eligible based on strong evidence + critical unmet need" if breakthrough_eligible else None,
        "GAIN Act incentives (5-yr exclusivity + fast track) apply for qualifying ESKAPE pathogens" if has_eskape else None,
        "Priority Review Voucher (PRV): eligible pathogen — PRV worth ~$100M+ at auction if granted" if prv_eligible else None,
        "PASTEUR Act (pending US legislation) would provide subscription-model pull incentive (~$1B+ per approved drug)",
    ] if n is not None]

    return {
        "qidp_eligible": qidp_eligible,
        "fast_track_eligible": fast_track_eligible,
        "lpad_eligible": lpad_eligible,
        "breakthrough_eligible": breakthrough_eligible,
        "prv_eligible": prv_eligible,
        "pathway": pathway,
        "notes": notes,
    }


# ─────────────────────────────────────────────
# International Regulatory Signals (EMA, MHRA)
# ─────────────────────────────────────────────

def get_international_regulatory_signals(state: LegacyProgramState) -> dict:
    org_lower = (state.organism or "").lower()

    has_who_priority = any(
        p in org_lower for p in [
            "klebsiella", "acinetobacter", "pseudomonas",
            "staphylococcus aureus", "enterococcus", "mycobacterium",
        ]
    )

    has_critical_resistance = any(
        any(k in g.gene_name.lower() for k in ["kpc", "ndm", "oxa", "vim", "mcr"])
        for g in state.resistance_genes
    )

    prime_eligible = has_who_priority or has_critical_resistance

    ilap_eligible = has_who_priority and state.stage in [
        WorkflowStage.EXPERIMENTAL_VALIDATION,
        WorkflowStage.PRECLINICAL_GAP_ANALYSIS,
        WorkflowStage.MANUFACTURING_FEASIBILITY,
    ]

    return {
        "ema_prime_eligible": prime_eligible,
        "ema_prime_note": (
            "EMA PRIME designation: early dialogue, co-development support, expedited assessment. "
            "Apply pre-Phase 1 or during Phase 1 with preliminary non-clinical or clinical evidence of major therapeutic advantage."
            if prime_eligible else
            "EMA PRIME eligibility requires WHO Priority pathogen or serious unmet need — assess after organism confirmation."
        ),
        "mhra_ilap_eligible": ilap_eligible,
        "mhra_ilap_note": (
            "MHRA ILAP (Innovation Passport): rolling review, Target Development Profile dialogue. "
            "UK-based approval pathway post-Brexit — consider parallel to FDA pathway."
            if ilap_eligible else
            "MHRA ILAP may apply at experimental validation stage — reassess."
        ),
        "ema_scientific_advice": "EMA Scientific Advice available pre-Phase 2 — strongly recommended for novel mechanisms",
        "japan_sakigake": "PMDA Sakigake designation applicable for novel AMR drugs targeting WHO priority pathogens in Japan",
    }


# ─────────────────────────────────────────────
# Funding Fit Scorer (with stage mismatch penalty)
# ─────────────────────────────────────────────

def score_funding_fit(grant: dict, state: LegacyProgramState) -> float:
    score = 0.0
    summary_lower = (state.project_summary or "").lower()
    org_lower = (state.organism or "").lower()

    stage_match = state.stage in grant["eligible_stages"]

    if stage_match:
        score += 0.5
    else:
        score -= 0.2

    focus_hits = set()
    for focus_term in grant.get("focus", []):
        term_words = focus_term.lower().split()
        if any(w in org_lower for w in term_words):
            focus_hits.add(focus_term)
        elif any(w in summary_lower for w in term_words):
            focus_hits.add(focus_term)
    score += min(0.25, len(focus_hits) * 0.07)

    if state.evidence_strength == EvidenceStrength.STRONG:
        score += 0.15
    elif state.evidence_strength == EvidenceStrength.MODERATE:
        score += 0.08

    if state.resistance_genes and any(
        "resistance" in f.lower() or "amr" in f.lower() or "antibiotic" in f.lower()
        for f in grant.get("focus", [])
    ):
        score += 0.05

    if state.mic_data and any(
        "in vitro" in r.lower() for r in grant.get("eligibility_requirements", [])
    ):
        score += 0.05

    return min(1.0, round(max(0.0, score), 2))


# ─────────────────────────────────────────────
# Eligibility Gap Checker
# ─────────────────────────────────────────────

def identify_eligibility_gaps(grant: dict, state: LegacyProgramState) -> list[str]:
    gaps = []
    for req in grant.get("eligibility_requirements", []):
        req_lower = req.lower()

        if "in vitro activity" in req_lower or "mic data" in req_lower:
            if not state.mic_data:
                gaps.append(f"Missing: {req}")

        if "in vivo" in req_lower and state.stage not in [
            WorkflowStage.EXPERIMENTAL_VALIDATION,
            WorkflowStage.PRECLINICAL_GAP_ANALYSIS,
            WorkflowStage.MANUFACTURING_FEASIBILITY,
        ]:
            gaps.append(f"Missing: {req}")

        if "adme" in req_lower and state.stage not in [
            WorkflowStage.PRECLINICAL_GAP_ANALYSIS,
            WorkflowStage.MANUFACTURING_FEASIBILITY,
        ]:
            gaps.append(f"Gap: {req}")

        if "phase 1" in req_lower and state.stage != WorkflowStage.MANUFACTURING_FEASIBILITY:
            gaps.append(f"Gap: {req}")

        if "novel mechanism" in req_lower and not state.resistance_genes:
            gaps.append(f"Unverified: {req} — no resistance gene mechanism detected")

        if "academic" in req_lower or "nonprofit" in req_lower:
            gaps.append(f"Verify: {req} — institution type not determined from input data")

        if "eu-based" in req_lower or "eu consortium" in req_lower:
            gaps.append(f"Verify: {req} — geographic eligibility not determined from input data")

        if "uk-based" in req_lower or "uk sme" in req_lower:
            gaps.append(f"Verify: {req} — UK jurisdiction not determined from input data")

        if "lmic" in req_lower and not any(
            k in (state.project_summary or "").lower()
            for k in ["lmic", "low-income", "global health", "developing"]
        ):
            gaps.append(f"Gap: {req} — no LMIC component detected in input data")

    return gaps


# ─────────────────────────────────────────────
# Grant Stacking Guidance
# ─────────────────────────────────────────────

def get_fundable_stacks(state: LegacyProgramState) -> list[dict]:
    if not state.funding_targets:
        return []

    matched_names = {ft.program_name for ft in state.funding_targets}
    stacks = []

    grant_map = {g["program_name"]: g for g in FUNDING_DATABASE}

    for ft in state.funding_targets:
        grant = grant_map.get(ft.program_name, {})
        for partner_name in grant.get("stackable_with", []):
            if partner_name in matched_names and partner_name != ft.program_name:
                stack_key = sorted([ft.program_name, partner_name])
                if not any(s["pair"] == stack_key for s in stacks):
                    partner = grant_map.get(partner_name, {})
                    stacks.append({
                        "pair": stack_key,
                        "grant_a": ft.program_name,
                        "grant_b": partner_name,
                        "combined_award": f"{ft.award_size} + {partner.get('award_size', 'TBD')}",
                        "note": "These programs are commonly co-funded — apply concurrently",
                    })

    return stacks


# ─────────────────────────────────────────────
# Competitive Landscape
# ─────────────────────────────────────────────

def get_competitive_signals(state: LegacyProgramState) -> list[dict]:
    org_lower = (state.organism or "").lower()
    for key, signals in COMPETITIVE_LANDSCAPE.items():
        if key in org_lower:
            return signals
    return [
        {"drug": "Competitive landscape unavailable for this organism",
         "status": "Unknown", "year": None, "coverage": "Conduct ClinicalTrials.gov + PubMed search"},
    ]


# ─────────────────────────────────────────────
# IP / Patent Landscape
# ─────────────────────────────────────────────

def get_ip_landscape_signals(state: LegacyProgramState) -> dict:
    if not state.resistance_genes:
        return IP_LANDSCAPE["default"]

    gene_lower = state.resistance_genes[0].gene_name.lower()
    for key, info in IP_LANDSCAPE.items():
        if key == "default":
            continue
        if key in gene_lower:
            return info

    return IP_LANDSCAPE["default"]


# ─────────────────────────────────────────────
# Market Intelligence
# ─────────────────────────────────────────────

def get_market_intelligence(state: LegacyProgramState) -> dict:
    org_lower = (state.organism or "").lower()
    for key, intel in MARKET_INTELLIGENCE.items():
        if key == "default":
            continue
        if key in org_lower:
            return intel
    return MARKET_INTELLIGENCE["default"]


# ─────────────────────────────────────────────
# ClinicalTrials.gov live lookup (async, graceful fallback)
# ─────────────────────────────────────────────

async def fetch_live_trials(organism: str) -> list[dict]:
    """
    Query ClinicalTrials.gov v2 API for active AMR trials for the target organism.
    Falls back to empty list on failure. Timeout is 2s to avoid blocking.
    """
    if not organism:
        return []
    try:
        query = f"antimicrobial resistance {organism}"
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(
                "https://clinicaltrials.gov/api/v2/studies",
                params={
                    "query.cond": query,
                    "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING",
                    "pageSize": 5,
                    "fields": "NCTId,BriefTitle,Phase,OverallStatus,LeadSponsorName",
                }
            )
            if r.status_code == 200:
                studies = r.json().get("studies", [])
                return [
                    {
                        "nct_id": s.get("protocolSection", {}).get("identificationModule", {}).get("nctId"),
                        "title": s.get("protocolSection", {}).get("identificationModule", {}).get("briefTitle"),
                        "phase": s.get("protocolSection", {}).get("designModule", {}).get("phases", ["Unknown"]),
                        "status": s.get("protocolSection", {}).get("statusModule", {}).get("overallStatus"),
                        "sponsor": s.get("protocolSection", {}).get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name"),
                    }
                    for s in studies
                ]
    except Exception:
        pass
    return []


# ─────────────────────────────────────────────
# Execution Brief Generator
# ─────────────────────────────────────────────

def generate_execution_brief(
    state: LegacyProgramState,
    readiness: TranslationalReadiness,
    timeline: dict,
    market: dict,
    ip_info: dict,
    pos: dict,
    stacks: list[dict],
    intl_regulatory: dict,
) -> str:
    org = state.organism or "target organism"
    genes = ", ".join(g.gene_name for g in state.resistance_genes[:2]) or "uncharacterized"
    stage = state.stage.value.replace("_", " ").title()

    top_funding = state.funding_targets[0] if state.funding_targets else None
    funding_line = (
        f"{top_funding.program_name} ({top_funding.agency}, {top_funding.award_size})"
        f" — fit score {top_funding.fit_score:.0%}"
        if top_funding else "No high-fit funding match identified"
    )

    alt_funding = ""
    if len(state.funding_targets) > 1:
        others = ", ".join(f.program_name for f in state.funding_targets[1:3])
        alt_funding = f"\n  Also consider: {others}"

    stack_text = ""
    if stacks:
        s = stacks[0]
        stack_text = f"\n  Stack opportunity: {s['grant_a']} + {s['grant_b']} ({s['combined_award']})"

    cro_info = CRO_ROUTING.get(state.stage, CRO_ROUTING[WorkflowStage.UNKNOWN])
    cro_line = f"{cro_info.get('partner_type', 'TBD')} — est. {cro_info.get('timeline_estimate', 'TBD')}"

    designations = [d for d, flag in [
        ("QIDP", readiness.qidp_eligible),
        ("Fast Track", readiness.fast_track_eligible),
        ("LPAD", readiness.lpad_eligible),
        ("Breakthrough Therapy", readiness.breakthrough_eligible),
        ("PRV-eligible", readiness.prv_eligible),
    ] if flag]

    fda_line = readiness.fda_pathway or "Pre-IND consultation recommended"
    if designations:
        fda_line += f"\n  Eligible designations: {', '.join(designations)}"

    ema_line = intl_regulatory.get("ema_prime_note", "")
    mhra_line = intl_regulatory.get("mhra_ilap_note", "")

    blockers_text = "\n  - ".join(readiness.scale_up_blockers[:3]) or "None identified at current stage"

    approved = [c["drug"] for c in (state.competitive_signals or []) if c.get("status") == "Approved"]
    pipeline = [c["drug"] for c in (state.competitive_signals or []) if c.get("status") in ("Phase 3", "Phase 2", "Phase 1")]
    competitive_summary = ""
    if approved:
        competitive_summary += f"\n  Approved competitors: {', '.join(approved[:3])}"
    if pipeline:
        competitive_summary += f"\n  Pipeline threat: {', '.join(pipeline[:2])}"

    deadline_line = (top_funding.next_deadline or "Check grant portal directly") if top_funding else "Check grant portal directly"

    return (
        f"EXECUTION BRIEF\n"
        f"{'='*60}\n"
        f"Program: {org} | Genes: {genes} | Stage: {stage}\n"
        f"CDMO Readiness: {readiness.cdmo_readiness_score:.0f}/100  |  "
        f"Evidence Completeness: {readiness.evidence_completeness_score:.0f}%\n\n"
        f"PROBABILITY OF SUCCESS\n"
        f"  To Phase 1: {pos.get('to_phase1', 'N/A')}  |  To Approval: {pos.get('to_approval', 'N/A')}\n"
        f"  Note: {pos.get('note', '')}\n\n"
        f"TIMELINE TO IND\n"
        f"  Estimated months to IND: {timeline.get('months_to_ind', 'TBD')}\n"
        f"  Estimated months to Phase 1: {timeline.get('months_to_phase1', 'TBD')}\n"
        f"  Next stage cost: {timeline.get('stage_cost_estimate', 'TBD')}\n"
        f"  Next milestone: {timeline.get('next_milestone', 'TBD')}\n\n"
        f"FUNDING STRATEGY\n"
        f"  Top match: {funding_line}\n"
        f"  Next deadline: {deadline_line}"
        f"{alt_funding}"
        f"{stack_text}\n\n"
        f"EXTERNAL EXECUTION\n"
        f"  Recommended CRO: {cro_line}\n\n"
        f"REGULATORY PATH (FDA)\n"
        f"  {fda_line}\n\n"
        f"INTERNATIONAL REGULATORY\n"
        f"  EMA: {ema_line}\n"
        f"  MHRA: {mhra_line}\n\n"
        f"MARKET CONTEXT\n"
        f"  Global TAM: {market.get('global_tam', 'N/A')}\n"
        f"  US cases/year: {market.get('us_hospital_cases', 'N/A')}\n"
        f"  Unmet need: {market.get('unmet_need', 'N/A')}\n"
        f"  Competitive landscape:{competitive_summary if competitive_summary else ' No data for this organism'}\n\n"
        f"IP LANDSCAPE\n"
        f"  {ip_info.get('landscape', 'N/A')}\n"
        f"  FTO signal: {ip_info.get('freedom_to_operate', 'N/A')}\n\n"
        f"SCALE-UP BLOCKERS\n"
        f"  - {blockers_text}\n\n"
        f"IMMEDIATE NEXT ACTIONS\n"
        f"  1. Execute top-ranked experiment (see Layer 2 recommendations)\n"
        f"  2. Apply to {top_funding.program_name if top_funding else 'matched grant program'} — deadline: {deadline_line}\n"
        f"  3. Request Pre-IND meeting with FDA CDER Division of Anti-Infectives (Type B meeting)\n"
        f"  4. Engage {cro_info.get('partner_type', 'appropriate CRO')} for next experimental phase\n"
        f"  5. IP/FTO: {ip_info.get('opportunity', 'assess compound novelty with IP counsel')}\n"
        f"  6. Consider EMA PRIME application {'— eligible at this stage' if intl_regulatory.get('ema_prime_eligible') else '— reassess after mechanism confirmation'}\n"
    )


# ─────────────────────────────────────────────
# Internal engine (operates on LegacyProgramState)
# ─────────────────────────────────────────────

async def _run_engine(state: LegacyProgramState) -> LegacyProgramState:
    """Core Layer 3 execution engine."""

    # Funding routing
    funding_targets = []
    for grant in FUNDING_DATABASE:
        fit_score = score_funding_fit(grant, state)
        if fit_score > 0.15:
            gaps = identify_eligibility_gaps(grant, state)
            funding_targets.append(FundingTarget(
                program_name=grant["program_name"],
                agency=grant["agency"],
                fit_score=fit_score,
                stage_match=state.stage in grant["eligible_stages"],
                eligibility_gaps=gaps,
                url=grant["url"],
                award_size=grant["award_size"],
                next_deadline=grant.get("next_deadline"),
            ))

    state.funding_targets = sorted(funding_targets, key=lambda x: x.fit_score, reverse=True)[:5]

    # CDMO/GMP readiness
    cdmo_score = compute_cdmo_readiness_score(state)
    scale_up_blockers = get_scale_up_blockers(state)
    fda_info = get_fda_pathway(state)

    # IND study completion (data-aware)
    missing_ind = [
        study["name"]
        for study in IND_ENABLING_STUDIES
        if not _is_ind_study_complete(study, state)
    ]

    total_ind = len(IND_ENABLING_STUDIES)
    completed_ind = total_ind - len(missing_ind)
    evidence_pct = round((completed_ind / total_ind) * 100, 1)

    # CRO routing
    cro_info = CRO_ROUTING.get(state.stage, CRO_ROUTING[WorkflowStage.UNKNOWN])

    # Timeline + PoS
    timeline = STAGE_TIMELINES.get(state.stage, STAGE_TIMELINES[WorkflowStage.UNKNOWN])
    pos = PROBABILITY_OF_SUCCESS.get(state.stage, PROBABILITY_OF_SUCCESS[WorkflowStage.UNKNOWN])

    # Competitive, market, IP, regulatory intelligence
    competitive_signals = get_competitive_signals(state)
    market_intel = get_market_intelligence(state)
    ip_info = get_ip_landscape_signals(state)
    intl_regulatory = get_international_regulatory_signals(state)

    # Grant stacking (after funding_targets is set)
    stacks = get_fundable_stacks(state)

    # Live FDA + ClinicalTrials.gov lookups
    compound_names = [c.name for c in state.compounds]
    live_trials, fda_intel = await asyncio.gather(
        fetch_live_trials(state.organism or ""),
        enrich_fda_intelligence(state.organism or "", compound_names),
    )

    readiness = TranslationalReadiness(
        cdmo_readiness_score=cdmo_score,
        evidence_completeness_score=evidence_pct,
        scale_up_blockers=scale_up_blockers,
        missing_ind_studies=missing_ind,
        qidp_eligible=fda_info["qidp_eligible"],
        fast_track_eligible=fda_info["fast_track_eligible"],
        lpad_eligible=fda_info["lpad_eligible"],
        breakthrough_eligible=fda_info["breakthrough_eligible"],
        prv_eligible=fda_info["prv_eligible"],
        fda_pathway=fda_info["pathway"],
        cro_partner_type=cro_info.get("partner_type"),
        formulation_data_present=state.stage in [
            WorkflowStage.PRECLINICAL_GAP_ANALYSIS,
            WorkflowStage.MANUFACTURING_FEASIBILITY,
        ],
        in_vivo_data_present=state.stage in [
            WorkflowStage.EXPERIMENTAL_VALIDATION,
            WorkflowStage.PRECLINICAL_GAP_ANALYSIS,
            WorkflowStage.MANUFACTURING_FEASIBILITY,
        ],
    )

    state.translational_readiness = readiness
    state.execution_brief = generate_execution_brief(
        state, readiness, timeline, market_intel, ip_info, pos, stacks, intl_regulatory
    )

    state.competitive_signals = competitive_signals
    state.market_intelligence = market_intel
    state.ip_landscape = ip_info
    state.fda_notes = fda_info["notes"]
    state.stage_timeline = timeline
    state.cro_details = cro_info
    state.probability_of_success = pos
    state.grant_stacks = stacks
    state.international_regulatory = intl_regulatory
    state.live_trials = live_trials
    state.fda_intelligence = fda_intel

    return state


# ─────────────────────────────────────────────
# SchemaBio adapter: ExecutionPlanningInput -> engine -> structured output
# ─────────────────────────────────────────────

_STAGE_MAP: dict[str, WorkflowStage] = {
    "target_identification": WorkflowStage.TARGET_IDENTIFICATION,
    "hit_discovery": WorkflowStage.HIT_DISCOVERY,
    "resistance_mechanism_characterization": WorkflowStage.RESISTANCE_MECHANISM_CHARACTERIZATION,
    "experimental_validation": WorkflowStage.EXPERIMENTAL_VALIDATION,
    "preclinical_gap_analysis": WorkflowStage.PRECLINICAL_GAP_ANALYSIS,
    "manufacturing_feasibility": WorkflowStage.MANUFACTURING_FEASIBILITY,
    "manufacturing_feasibility_review": WorkflowStage.MANUFACTURING_FEASIBILITY,
}


def _epi_to_legacy_state(epi: ExecutionPlanningInput) -> LegacyProgramState:
    """Convert SchemaBio ExecutionPlanningInput to the internal LegacyProgramState."""
    stage = _STAGE_MAP.get(epi.stage.lower(), WorkflowStage.UNKNOWN)

    # Extract entities from development_signals by kind
    organism: Optional[str] = None
    resistance_genes: list[ResistanceGene] = []
    compounds: list[Compound] = []
    mic_data: list[MICEntry] = []

    for sig in epi.development_signals:
        kind = (sig.kind or "").lower()
        val = str(sig.value) if sig.value is not None else ""

        if kind == "organism" and not organism:
            organism = val
        elif kind in ("resistance_gene", "target_gene", "gene", "resistance_associated_variant"):
            # Parse "gene:mechanism" format if present
            parts = val.split(":", 1)
            gene_name = parts[0].strip()
            mechanism = parts[1].strip() if len(parts) > 1 else None
            if gene_name:
                resistance_genes.append(ResistanceGene(gene_name=gene_name, mechanism=mechanism))
        elif kind in ("compound_hit", "compound", "lead_compound"):
            if val:
                compounds.append(Compound(name=val))
        elif kind in ("mic", "mic_value") or (sig.unit and ("ug/ml" in str(sig.unit).lower() or "µg/ml" in str(sig.unit).lower())):
            try:
                mic_data.append(MICEntry(
                    compound=str(sig.source or sig.evidence_ref or "unknown"),
                    organism=organism or "unknown",
                    mic_value=float(sig.value),  # type: ignore[arg-type]
                    unit=sig.unit or "ug/mL",
                ))
            except (ValueError, TypeError):
                pass

    # Evidence strength from stage_confidence
    if epi.stage_confidence >= 0.8:
        evidence_strength = EvidenceStrength.STRONG
    elif epi.stage_confidence >= 0.5:
        evidence_strength = EvidenceStrength.MODERATE
    else:
        evidence_strength = EvidenceStrength.WEAK

    return LegacyProgramState(
        stage=stage,
        organism=organism,
        project_summary=epi.program_summary,
        resistance_genes=resistance_genes,
        compounds=compounds,
        mic_data=mic_data,
        evidence_strength=evidence_strength,
        missing_flags=list(epi.missing_development_inputs),
    )


async def run_layer3(
    epi: ExecutionPlanningInput,
    layer2_output: Optional[dict] = None,
) -> dict:
    """
    SchemaBio Layer 3 entry point.

    Accepts ExecutionPlanningInput from the ingestion layer and (optionally) the
    Layer 2 ExperimentDesignOutput dict.  Layer 2 output is used to:
      - Surface the top recommended experiment in next_steps
      - Enrich the evidence checklist with in-plan items

    Output fields:
      partner_recommendations, funding_opportunities,
      missing_evidence_package_elements, translational_blockers,
      readiness_assessment, fda_pathway, international_regulatory,
      competitive_landscape, market_intelligence, ip_landscape,
      stage_timeline, probability_of_success, grant_stacking,
      live_trials, fda_intelligence, execution_brief
    """
    state = _epi_to_legacy_state(epi)
    state = await _run_engine(state)

    # Enrich next_steps with Layer 2 top experiment (if available)
    top_experiment_step = "Execute top-ranked experiment (see Layer 2 recommendations)"
    if layer2_output:
        ranked = layer2_output.get("ranked_experiments", [])
        if ranked:
            top = ranked[0]
            top_experiment_step = (
                f"Run Layer 2 priority experiment: {top.get('title', '')} "
                f"({top.get('cro_type', 'in-house')})"
            )

    rt = state.translational_readiness
    evidence_refs = list(epi.evidence_bundle.file_refs)

    return {
        # ── Core output schema (from handoff doc) ─────────────────────────
        "partner_recommendations": [
            {
                "partner_type": state.cro_details.get("partner_type", ""),
                "rationale": (
                    f"Stage: {state.stage.value}. "
                    f"Capabilities needed: {', '.join(state.cro_details.get('capabilities_needed', []))}."
                ),
                "readiness_required": state.cro_details.get("timeline_estimate"),
                "example_partners": state.cro_details.get("example_partners", []),
                "biosafety_level": state.cro_details.get("biosafety_level", ""),
                "evidence_refs": evidence_refs,
            }
        ],
        "funding_opportunities": [
            {
                "name": ft.program_name,
                "agency": ft.agency,
                "amount": ft.award_size or "",
                "fit_score": ft.fit_score,
                "fit_rationale": f"Fit score: {ft.fit_score:.0%}. Stage match: {ft.stage_match}.",
                "deadline": ft.next_deadline,
                "url": ft.url,
                "eligibility_gaps": ft.eligibility_gaps,
                "evidence_refs": evidence_refs,
            }
            for ft in state.funding_targets
        ],
        "missing_evidence_package_elements": [
            {
                "element": study,
                "blocking_for": "IND filing",
                "evidence_refs": [],
            }
            for study in (rt.missing_ind_studies if rt else [])
        ],
        "translational_blockers": [
            {
                "blocker": blocker,
                "severity": "high",
                "evidence_refs": evidence_refs,
            }
            for blocker in (rt.scale_up_blockers if rt else [])
        ],
        "readiness_assessment": {
            "evidence_completeness_pct": rt.evidence_completeness_score if rt else None,
            "gmp_readiness_pct": rt.cdmo_readiness_score if rt else None,
            "signals": [
                {"kind": sig.kind, "value": sig.value, "unit": sig.unit}
                for sig in epi.development_signals
            ],
            "next_steps": [
                top_experiment_step,
                f"Apply to top funding match — {state.funding_targets[0].program_name if state.funding_targets else 'check grant portals'}",
                "Request Pre-IND meeting with FDA CDER Division of Anti-Infectives (Type B meeting)",
                f"Engage {state.cro_details.get('partner_type', 'CRO')} for next experimental phase",
            ],
        },
        # ── Extended context ──────────────────────────────────────────────
        "fda_pathway": {
            "pathway": rt.fda_pathway if rt else None,
            "qidp_eligible": rt.qidp_eligible if rt else None,
            "fast_track_eligible": rt.fast_track_eligible if rt else None,
            "lpad_eligible": rt.lpad_eligible if rt else None,
            "breakthrough_eligible": rt.breakthrough_eligible if rt else None,
            "prv_eligible": rt.prv_eligible if rt else None,
            "notes": state.fda_notes,
        },
        "international_regulatory": state.international_regulatory,
        "competitive_landscape": state.competitive_signals,
        "market_intelligence": state.market_intelligence,
        "ip_landscape": state.ip_landscape,
        "stage_timeline": state.stage_timeline,
        "probability_of_success": state.probability_of_success,
        "grant_stacking": state.grant_stacks,
        "live_trials": state.live_trials,
        "fda_intelligence": state.fda_intelligence,
        "execution_brief": state.execution_brief,
    }
