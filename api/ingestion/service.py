"""
services/ingestion_service.py

SchemaBio ingestion orchestration.
Runs parsers per file, aggregates entities/signals, estimates stage, builds IngestionResponse.
Deterministic only; no LLM.
"""

from __future__ import annotations
import logging
import uuid
from pathlib import Path
from typing import Optional

from api.schemas.ingestion import (
    IngestionResponse,
    ProgramState,
    ExperimentDesignInput,
    ExecutionPlanningInput,
    UploadedFileDescriptor,
    ExtractedEntity,
    ExtractedSignal,
    StageEstimate,
    EvidenceBundle,
)
from api.ingestion.parser_adapter import (
    detect_file_type,
    parse_vcf_to_entities_signals,
    parse_resistance_csv_to_entities_signals,
    parse_compound_screen_to_entities_signals,
    parse_pdf_to_entities_signals,
    parse_text_to_entities_signals,
)
from api.ingestion.stage_estimator import estimate_stage

logger = logging.getLogger(__name__)


def _dedupe_entities(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
    seen: set[tuple[str, str, Optional[str]]] = set()
    out: list[ExtractedEntity] = []
    for e in entities:
        key = (e.type, e.value, e.source)
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def _dedupe_signals(signals: list[ExtractedSignal]) -> list[ExtractedSignal]:
    seen: set[tuple[str, str]] = set()
    out: list[ExtractedSignal] = []
    for s in signals:
        key = (s.kind, str(s.value))
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def run_ingestion(paths: list[Path]) -> IngestionResponse:
    """
    Run full ingestion on a list of file paths.
    Returns IngestionResponse with program_state, experiment_design_input, execution_planning_input.
    """
    program_id = str(uuid.uuid4())[:8].upper()
    all_descriptors: list[UploadedFileDescriptor] = []
    all_entities: list[ExtractedEntity] = []
    all_signals: list[ExtractedSignal] = []
    all_warnings: list[str] = []
    evidence_index: dict[str, list[str]] = {}

    for path in paths:
        if not path.exists():
            all_warnings.append(f"File not found: {path.name}")
            continue
        file_id = f"file_{path.stem}_{uuid.uuid4().hex[:6]}"
        filename = path.name
        detected, _ = detect_file_type(path)

        if detected == "vcf":
            desc, entities, signals, warns = parse_vcf_to_entities_signals(path, file_id, filename)
        elif detected == "resistance_csv":
            desc, entities, signals, warns = parse_resistance_csv_to_entities_signals(path, file_id, filename)
        elif detected in ("compound_screen_csv", "csv"):
            desc, entities, signals, warns = parse_compound_screen_to_entities_signals(path, file_id, filename)
        elif detected == "pdf":
            desc, entities, signals, warns = parse_pdf_to_entities_signals(path, file_id, filename)
        elif detected == "text":
            desc, entities, signals, warns = parse_text_to_entities_signals(path, file_id, filename)
        else:
            desc = UploadedFileDescriptor(
                file_id=file_id,
                filename=filename,
                detected_type="unknown",
                schema_confidence=0.0,
                parse_status="error",
                extracted_fields=[],
                warnings=[f"Unsupported file type: {path.suffix}"],
            )
            entities, signals, warns = [], [], desc.warnings

        all_descriptors.append(desc)
        all_entities.extend(entities)
        all_signals.extend(signals)
        all_warnings.extend(warns)
        evidence_index[file_id] = [e.value for e in entities][:20] + [f"{s.kind}:{s.value}" for s in signals][:10]

    entities = _dedupe_entities(all_entities)
    signals = _dedupe_signals(all_signals)

    # Missing data flags (deterministic from what we did not see)
    missing_flags: list[str] = []
    if not any(e.type == "compound" for e in entities) and not any(s.kind == "compound_hit" for s in signals):
        missing_flags.append("no_compound_screen_data_detected")
    if not any("admet" in f.detected_type.lower() for f in all_descriptors):
        missing_flags.append("no_admet_data_detected")
    if not any("manufactur" in w.lower() or "gmp" in w.lower() for w in all_warnings):
        missing_flags.append("no_manufacturability_data_detected")
    if not any("reproducibility" in w.lower() or "replicate" in w.lower() for w in all_warnings):
        missing_flags.append("no_reproducibility_summary_detected")
    if not any(s.kind == "target_engagement" or s.kind == "enzyme_inhibition" for s in signals):
        missing_flags.append("no_target_engagement_data_detected")
    for w in all_warnings:
        if "vehicle" in w.lower() or "control" in w.lower():
            missing_flags.append("no_vehicle_control_detected")
            break
    for w in all_warnings:
        if "n=1" in w or "single replicate" in w.lower():
            missing_flags.append("single_replicate_data_detected")
            break

    stage_estimate = estimate_stage(all_descriptors, entities, signals, missing_flags)

    program_state = ProgramState(
        program_id=program_id,
        status="ok" if not any(d.parse_status == "error" for d in all_descriptors) else "partial",
        uploaded_files=all_descriptors,
        entities=entities,
        signals=signals,
        stage_estimate=stage_estimate,
        missing_data_flags=missing_flags,
        warnings=all_warnings,
        evidence_index=evidence_index,
    )

    # Build handoff objects
    biological_context_parts = []
    for e in entities:
        if e.type == "organism":
            biological_context_parts.append(e.value)
        if e.type == "target" or e.type == "variant":
            biological_context_parts.append(e.value)
    biological_context = "; ".join(biological_context_parts[:8]) or "No biological context extracted."
    assay_context_list = list({e.value for e in entities if e.type == "assay_type"})
    if not assay_context_list and signals:
        assay_context_list = ["assay data present"]

    # Missing experiment context: flags that affect experiment design (controls, replicates, mechanism, etc.)
    exp_context = [f for f in missing_flags if any(x in f.lower() for x in ("control", "replicate", "mechanism", "target_engagement", "vehicle"))]
    if not exp_context:
        exp_context = missing_flags[:5]

    experiment_design_input = ExperimentDesignInput(
        stage=stage_estimate.name,
        stage_confidence=stage_estimate.confidence,
        biological_context=biological_context,
        assay_context=assay_context_list,
        priority_signals=signals[:10],
        missing_experiment_context=exp_context,
        evidence_bundle=EvidenceBundle(
            file_refs=list(evidence_index.keys()),
            quantitative_claims=[{"type": s.kind, "value": s.value, "unit": s.unit} for s in signals if s.unit][:10],
        ),
    )

    program_summary = f"Program {program_id}. Stage: {stage_estimate.name} (confidence {stage_estimate.confidence:.0%}). "
    program_summary += f"Entities: {len(entities)}; signals: {len(signals)}. "
    program_summary += "; ".join(stage_estimate.reasoning_basis[:2]) if stage_estimate.reasoning_basis else ""

    execution_planning_input = ExecutionPlanningInput(
        stage=stage_estimate.name,
        stage_confidence=stage_estimate.confidence,
        program_summary=program_summary,
        development_signals=signals[:10],
        missing_development_inputs=missing_flags,
        readiness_constraints=[
            "CDMO not ready until ADMET + reproducibility package complete",
            "Evidence package must be complete before CRO engagement",
        ],
        evidence_bundle=EvidenceBundle(file_refs=list(evidence_index.keys())),
    )

    return IngestionResponse(
        program_state=program_state,
        experiment_design_input=experiment_design_input,
        execution_planning_input=execution_planning_input,
    )
