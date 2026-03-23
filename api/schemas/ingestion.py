"""
models/ingestion.py

SchemaBio Ingestion Layer — request/response contracts.

All ingestion API responses use these Pydantic models.
Ingestion is the source of truth; Experiment Design and Execution layers consume these outputs.
"""

from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


# ─── Uploaded file descriptor ─────────────────────────────────────────────────

class UploadedFileDescriptor(BaseModel):
    """Single uploaded file metadata from ingestion."""
    file_id: str = ""
    filename: str = ""
    detected_type: str = ""
    schema_confidence: float = 0.0
    parse_status: str = "pending"  # pending | parsing | complete | error
    extracted_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ─── Extracted entities ─────────────────────────────────────────────────────

class ExtractedEntity(BaseModel):
    """Entity extracted from parsed files (organism, target, compound, variant, etc.)."""
    type: str = ""   # organism | target | compound | variant | assay_type | drug_class | pathway
    value: str = ""
    source: Optional[str] = None
    confidence: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── Extracted signals ───────────────────────────────────────────────────────

class ExtractedSignal(BaseModel):
    """Signal derived from data (e.g. resistance fold-shift, compound hit, mechanism hint)."""
    kind: str = ""   # resistance_associated_variant | compound_hit | target_signal | assay_pattern | mechanism_hint | ...
    value: str | int | float | bool = ""
    unit: Optional[str] = None
    source: Optional[str] = None
    evidence_ref: Optional[str] = None


# ─── Stage estimate ──────────────────────────────────────────────────────────

class StageEstimate(BaseModel):
    """Workflow/program stage estimate from ingestion (deterministic heuristics)."""
    name: str = ""
    confidence: float = 0.0
    reasoning_basis: list[str] = Field(default_factory=list)


# ─── Evidence / handoff ──────────────────────────────────────────────────────

class EvidenceRef(BaseModel):
    """Reference to a piece of evidence (file, field, claim id)."""
    id: str = ""
    type: str = "field"   # field | file | claim | audit | gap
    label: Optional[str] = None


class EvidenceBundle(BaseModel):
    """Evidence bundle for experiment design or execution planning handoff."""
    literature_refs: list[str] = Field(default_factory=list)
    quantitative_claims: list[dict[str, Any]] = Field(default_factory=list)
    audit_refs: list[str] = Field(default_factory=list)
    gap_refs: list[str] = Field(default_factory=list)
    file_refs: list[str] = Field(default_factory=list)


# ─── Program state ────────────────────────────────────────────────────────────

class ProgramState(BaseModel):
    """Program state produced by the ingestion layer (deterministic, no LLM)."""
    program_id: str = ""
    status: str = "ok"   # ok | partial | error
    uploaded_files: list[UploadedFileDescriptor] = Field(default_factory=list)
    entities: list[ExtractedEntity] = Field(default_factory=list)
    signals: list[ExtractedSignal] = Field(default_factory=list)
    stage_estimate: Optional[StageEstimate] = None
    missing_data_flags: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    evidence_index: dict[str, list[str]] = Field(default_factory=dict)  # file_id -> list of entity/signal refs


# ─── Experiment design input (handoff to Experiment Design Layer) ─────────────

class ExperimentDesignInput(BaseModel):
    """Clean handoff to the future Experiment Design Layer."""
    stage: str = ""
    stage_confidence: float = 0.0
    biological_context: str = ""
    assay_context: list[str] = Field(default_factory=list)
    priority_signals: list[ExtractedSignal] = Field(default_factory=list)
    missing_experiment_context: list[str] = Field(default_factory=list)
    evidence_bundle: EvidenceBundle = Field(default_factory=EvidenceBundle)


# ─── Execution planning input (handoff to Execution Layer) ─────────────────────

class ExecutionPlanningInput(BaseModel):
    """Clean handoff to the future Execution / Translational Planning Layer."""
    stage: str = ""
    stage_confidence: float = 0.0
    program_summary: str = ""
    development_signals: list[ExtractedSignal] = Field(default_factory=list)
    missing_development_inputs: list[str] = Field(default_factory=list)
    readiness_constraints: list[str] = Field(default_factory=list)
    evidence_bundle: EvidenceBundle = Field(default_factory=EvidenceBundle)


# ─── Top-level ingestion response ─────────────────────────────────────────────

class IngestionResponse(BaseModel):
    """Exact contract returned by the ingestion API."""
    program_state: ProgramState = Field(default_factory=ProgramState)
    experiment_design_input: ExperimentDesignInput = Field(default_factory=ExperimentDesignInput)
    execution_planning_input: ExecutionPlanningInput = Field(default_factory=ExecutionPlanningInput)
    # Set server-side after DB write; None when Supabase is not configured or user is anonymous.
    # Frontend can store this and pass it back to Layer 2 / Layer 3 requests to link results.
    run_id: Optional[str] = None
