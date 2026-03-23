"""
services/stage_estimator.py

Deterministic stage estimation for SchemaBio ingestion.
Uses file types and extracted signals to estimate workflow/program stage.
Stage names and logic are explicit and interpretable; reasoning_basis is returned.
"""

from __future__ import annotations
import logging
from typing import Optional

from api.schemas.ingestion import (
    StageEstimate,
    UploadedFileDescriptor,
    ExtractedEntity,
    ExtractedSignal,
)

logger = logging.getLogger(__name__)

# Supported stage names (ingestion taxonomy)
STAGE_NAMES = [
    "hit_discovery",
    "resistance_mechanism_characterization",
    "experimental_validation_planning",
    "preclinical_package_gap_analysis",
    "manufacturing_feasibility_review",
]


def _has_signal(signals: list[ExtractedSignal], kind: str) -> bool:
    return any(s.kind == kind for s in signals)


def _has_entity(entities: list[ExtractedEntity], type_: str) -> bool:
    return any(e.type == type_ for e in entities)


def _file_types(uploaded: list[UploadedFileDescriptor]) -> set[str]:
    return {f.detected_type.lower() for f in uploaded}


def _missing_flags_contain(flags: list[str], *substrings: str) -> bool:
    combined = " ".join(flags).lower()
    return any(s.lower() in combined for s in substrings)


def estimate_stage(
    uploaded_files: list[UploadedFileDescriptor],
    entities: list[ExtractedEntity],
    signals: list[ExtractedSignal],
    missing_data_flags: list[str],
) -> StageEstimate:
    """
    Deterministic stage estimation from ingestion outputs.
    Returns stage name, confidence (0–1), and reasoning_basis list.
    """
    file_types = _file_types(uploaded_files)
    reasoning: list[str] = []

    has_variant = _has_signal(signals, "resistance_associated_variant") or _has_entity(entities, "variant")
    has_compound_hit = _has_signal(signals, "compound_hit") or _has_entity(entities, "compound")
    has_resistance_assay = "resistance" in str(file_types) or _has_signal(signals, "assay_pattern")
    has_compound_screen = "compound" in str(file_types)
    has_vcf = "vcf" in str(file_types) or "genomics" in str(file_types)
    has_pdf = "pdf" in str(file_types) or "notes" in str(file_types)
    has_target_rationale = has_pdf and (_has_entity(entities, "target") or _has_entity(entities, "drug_class"))

    # Manufacturing / development terms in flags or signals
    has_manufacturing_signals = _missing_flags_contain(
        missing_data_flags,
        "admet", "manufactur", "gmp", "cdmo", "reproducibility", "analytical"
    ) or any(
        "manufactur" in str(s.kind).lower() or "gmp" in str(s.kind).lower()
        for s in signals
    )

    # Rule 1: variant + resistance assay + compound hit → resistance_mechanism_characterization
    if has_variant and has_resistance_assay and has_compound_hit:
        reasoning.append("Variant data + resistance assay + compound hit data present.")
        reasoning.append("Mechanism characterization is the natural next step.")
        return StageEstimate(
            name="resistance_mechanism_characterization",
            confidence=0.88,
            reasoning_basis=reasoning,
        )

    # Rule 2: strong compound screen + target rationale, weak validation
    if has_compound_screen and has_target_rationale and not has_resistance_assay:
        reasoning.append("Compound screen and target rationale present; no resistance assay yet.")
        return StageEstimate(
            name="hit_discovery",
            confidence=0.82,
            reasoning_basis=reasoning,
        )

    if has_compound_screen and has_target_rationale and has_resistance_assay and not has_variant:
        reasoning.append("Compound screen, target rationale, and resistance assay; no variant data.")
        return StageEstimate(
            name="experimental_validation_planning",
            confidence=0.78,
            reasoning_basis=reasoning,
        )

    # Rule 3: VCF + resistance assay only (no strong compound hit)
    if has_vcf and has_resistance_assay and not has_compound_hit:
        reasoning.append("Variant/genomics and resistance assay; no compound screen.")
        return StageEstimate(
            name="resistance_mechanism_characterization",
            confidence=0.75,
            reasoning_basis=reasoning,
        )

    # Rule 4: Manufacturing / ADMET / development package terms present → later stages
    if has_manufacturing_signals:
        reasoning.append("Development/manufacturing or ADMET terms detected in missing-data flags or signals.")
        if _missing_flags_contain(missing_data_flags, "gmp", "cdmo"):
            return StageEstimate(
                name="manufacturing_feasibility_review",
                confidence=0.70,
                reasoning_basis=reasoning,
            )
        return StageEstimate(
            name="preclinical_package_gap_analysis",
            confidence=0.72,
            reasoning_basis=reasoning,
        )

    # Rule 5: Only PDF/notes
    if has_pdf and not (has_compound_screen or has_resistance_assay or has_vcf):
        reasoning.append("Only PDF/notes uploaded; no assay or genomics data.")
        return StageEstimate(
            name="hit_discovery",
            confidence=0.55,
            reasoning_basis=reasoning,
        )

    # Default: experimental_validation_planning (middle of pipeline)
    reasoning.append("Mixed signals; defaulting to experimental validation planning.")
    return StageEstimate(
        name="experimental_validation_planning",
        confidence=0.60,
        reasoning_basis=reasoning,
    )
