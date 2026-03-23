"""
services/parser_adapter.py

Maps existing parser outputs to ingestion-layer entities and signals.
Deterministic only; no LLM. Each function returns (entities, signals, warnings).
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from api.schemas.ingestion import ExtractedEntity, ExtractedSignal, UploadedFileDescriptor
from api.ingestion.parsers import vcf_parser, assay_parser, compound_parser, pdf_parser
from api.legacy.parsers.universal_parser import _detect_csv_subtype

logger = logging.getLogger(__name__)


def _file_descriptor(
    file_id: str,
    filename: str,
    detected_type: str,
    schema_confidence: float,
    parse_status: str,
    extracted_fields: list[str],
    warnings: list[str],
) -> UploadedFileDescriptor:
    return UploadedFileDescriptor(
        file_id=file_id,
        filename=filename,
        detected_type=detected_type,
        schema_confidence=schema_confidence,
        parse_status=parse_status,
        extracted_fields=extracted_fields,
        warnings=warnings,
    )


def parse_vcf_to_entities_signals(path: Path, file_id: str, filename: str) -> tuple[UploadedFileDescriptor, list[ExtractedEntity], list[ExtractedSignal], list[str]]:
    """Run VCF parser and map to entities + signals."""
    entities: list[ExtractedEntity] = []
    signals: list[ExtractedSignal] = []
    warnings: list[str] = []
    fields: list[str] = ["chrom", "pos", "ref", "alt", "gene", "impact"]

    try:
        variants = vcf_parser.parse_vcf(path)
    except Exception as e:
        logger.exception(f"VCF parse failed: {path.name}")
        return (
            _file_descriptor(file_id, filename, "Genomics / VCF", 0.0, "error", [], [str(e)]),
            [], [], [str(e)],
        )

    for v in variants:
        if v.gene and v.gene != "UNKNOWN":
            entities.append(ExtractedEntity(
                type="target",
                value=v.gene,
                source=filename,
                confidence=0.95,
                metadata={"pos": v.pos, "ref": v.ref, "alt": v.alt},
            ))
        if v.aa_change:
            entities.append(ExtractedEntity(
                type="variant",
                value=f"{v.gene} {v.aa_change}".strip(),
                source=filename,
                confidence=0.92,
                metadata={"impact": v.impact, "clinvar": v.clinvar_sig},
            ))
        if v.is_resistance_gene and v.aa_change:
            signals.append(ExtractedSignal(
                kind="resistance_associated_variant",
                value=f"{v.gene} {v.aa_change}",
                source=filename,
                evidence_ref=file_id,
            ))

    if variants:
        signals.append(ExtractedSignal(kind="variant_count", value=len(variants), source=filename, unit="variants"))

    fields.extend(["gene", "impact", "aa_change", "clinvar"] if variants else [])
    return (
        _file_descriptor(file_id, filename, "Genomics / VCF", 0.91, "complete", list(set(fields)), warnings),
        entities, signals, warnings,
    )


def parse_resistance_csv_to_entities_signals(path: Path, file_id: str, filename: str) -> tuple[UploadedFileDescriptor, list[ExtractedEntity], list[ExtractedSignal], list[str]]:
    """Run resistance assay parser and map to entities + signals."""
    entities: list[ExtractedEntity] = []
    signals: list[ExtractedSignal] = []
    warnings: list[str] = []
    fields = ["strain_id", "isolate_id", "compound_name", "mic", "replicate", "assay_type"]

    try:
        data = assay_parser.parse_resistance_assay(path)
    except Exception as e:
        logger.exception(f"Resistance assay parse failed: {path.name}")
        return (
            _file_descriptor(file_id, filename, "Resistance Assay CSV", 0.0, "error", [], [str(e)]),
            [], [], [str(e)],
        )

    for s in data.strains[:20]:
        entities.append(ExtractedEntity(type="organism", value=s, source=filename, confidence=0.9))
    for c in data.compounds[:20]:
        entities.append(ExtractedEntity(type="compound", value=c, source=filename, confidence=0.9))
    if data.assay_type:
        entities.append(ExtractedEntity(type="assay_type", value=data.assay_type, source=filename, confidence=0.95))

    if data.max_fold_shift:
        signals.append(ExtractedSignal(
            kind="resistance_fold_shift",
            value=data.max_fold_shift,
            unit="×",
            source=filename,
            evidence_ref=file_id,
        ))
    if data.resistant_strains:
        signals.append(ExtractedSignal(kind="assay_pattern", value=f"{len(data.resistant_strains)} resistant strains", source=filename))

    if not data.has_wt_control:
        warnings.append("No wild-type control strain detected")
    if data.replicate_count <= 1:
        warnings.append("Single replicate (n=1) detected")

    extracted = list(data.compound_fold_shifts.keys())[:5] or []
    return (
        _file_descriptor(file_id, filename, "Resistance Assay CSV", 0.92, "complete", fields + extracted, warnings),
        entities, signals, warnings,
    )


def parse_compound_screen_to_entities_signals(path: Path, file_id: str, filename: str) -> tuple[UploadedFileDescriptor, list[ExtractedEntity], list[ExtractedSignal], list[str]]:
    """Run compound screen parser and map to entities + signals."""
    entities: list[ExtractedEntity] = []
    signals: list[ExtractedSignal] = []
    warnings: list[str] = []
    fields = ["compound_name", "potency metric", "activity score", "hit flag", "target"]

    try:
        data = compound_parser.parse_compound_screen(path)
    except Exception as e:
        logger.exception(f"Compound screen parse failed: {path.name}")
        return (
            _file_descriptor(file_id, filename, "Compound Screen CSV", 0.0, "error", [], [str(e)]),
            [], [], [str(e)],
        )

    for c in data.compounds[:30]:
        entities.append(ExtractedEntity(
            type="compound",
            value=c.name,
            source=filename,
            confidence=0.9,
            metadata={"ic50_nm": c.ic50_nm, "flag": c.flag},
        ))
        if c.target:
            entities.append(ExtractedEntity(type="target", value=c.target, source=filename, confidence=0.85))
    if data.lead_compound:
        lead = data.lead_compound
        entities.append(ExtractedEntity(type="compound", value=lead.name, source=filename, confidence=0.96))
        signals.append(ExtractedSignal(
            kind="compound_hit",
            value=lead.name,
            source=filename,
            evidence_ref=file_id,
        ))
        if lead.ic50_nm:
            signals.append(ExtractedSignal(kind="lead_ic50_nm", value=lead.ic50_nm, unit="nM", source=filename))
    if data.n_top_hits:
        signals.append(ExtractedSignal(kind="top_hit_count", value=data.n_top_hits, source=filename))

    if not data.has_vehicle_ctrl:
        warnings.append("No vehicle/DMSO control in compound screen")
    if data.replicate_count <= 1:
        warnings.append("Single replicate (n=1)")

    return (
        _file_descriptor(file_id, filename, "Compound Screen CSV", 0.90, "complete", fields, warnings),
        entities, signals, warnings,
    )


def parse_pdf_to_entities_signals(path: Path, file_id: str, filename: str) -> tuple[UploadedFileDescriptor, list[ExtractedEntity], list[ExtractedSignal], list[str]]:
    """Run PDF parser and map to entities + signals (lightweight / mock-friendly)."""
    entities: list[ExtractedEntity] = []
    signals: list[ExtractedSignal] = []
    warnings: list[str] = []
    fields = ["title", "target references", "organism references", "drug class", "mechanism keywords"]

    try:
        data = pdf_parser.parse_pdf(path)
    except Exception as e:
        logger.exception(f"PDF parse failed: {path.name}")
        return (
            _file_descriptor(file_id, filename, "Research Notes / PDF", 0.0, "error", [], [str(e)]),
            [], [], [str(e)],
        )

    for g in data.key_genes[:15]:
        entities.append(ExtractedEntity(type="target", value=g, source=filename, confidence=0.75))
    for o in data.organisms[:10]:
        entities.append(ExtractedEntity(type="organism", value=o, source=filename, confidence=0.75))
    for m in data.mechanisms[:5]:
        entities.append(ExtractedEntity(type="drug_class", value=m, source=filename, confidence=0.7))
    for p in data.key_pathways[:5]:
        entities.append(ExtractedEntity(type="pathway", value=p, source=filename, confidence=0.65))

    for kw in data.mechanism_keywords[:5]:
        signals.append(ExtractedSignal(kind="mechanism_hint", value=kw, source=filename))
    if data.key_genes:
        signals.append(ExtractedSignal(kind="target_signal", value=", ".join(data.key_genes[:3]), source=filename))

    if not data.key_genes and not data.organisms:
        warnings.append("No target or organism references extracted from PDF")

    return (
        _file_descriptor(file_id, filename, "Research Notes / PDF", 0.72, "complete", fields, warnings),
        entities, signals, warnings,
    )


def parse_text_to_entities_signals(path: Path, file_id: str, filename: str) -> tuple[UploadedFileDescriptor, list[ExtractedEntity], list[ExtractedSignal], list[str]]:
    """Lightweight text/notes parser: target, organism, mechanism keywords."""
    entities: list[ExtractedEntity] = []
    signals: list[ExtractedSignal] = []
    warnings: list[str] = []
    fields = ["target references", "organism references", "mechanism keywords"]
    try:
        text = path.read_text(errors="ignore").lower()
    except Exception as e:
        return (
            _file_descriptor(file_id, filename, "Text / Notes", 0.0, "error", [], [str(e)]),
            [], [], [str(e)],
        )
    known_genes = ["gyra", "gyrb", "parc", "pare", "meca", "ecoli", "s. aureus", "e. coli"]
    for g in known_genes:
        if g in text:
            entities.append(ExtractedEntity(type="target", value=g.upper(), source=filename, confidence=0.7))
    for org in ["e. coli", "e.coli", "s. aureus", "mrsa", "m. tuberculosis"]:
        if org in text:
            entities.append(ExtractedEntity(type="organism", value=org, source=filename, confidence=0.7))
    if "gyrase" in text or "topoisomerase" in text:
        signals.append(ExtractedSignal(kind="mechanism_hint", value="gyrase/topoisomerase", source=filename))
    return (
        _file_descriptor(file_id, filename, "Text / Notes", 0.65, "complete", fields, warnings),
        entities, signals, warnings,
    )


def detect_file_type(path: Path) -> tuple[str, Optional[pd.DataFrame]]:
    """Detect file type from extension and optionally CSV content. Returns (detected_type, df or None)."""
    ext = path.suffix.lower()
    if ext in (".vcf", ".vcf.gz"):
        return "vcf", None
    if ext == ".pdf":
        return "pdf", None
    if ext in (".txt", ".md"):
        return "text", None
    if ext in (".csv", ".tsv"):
        try:
            df = pd.read_csv(str(path), nrows=5)
            subtype = _detect_csv_subtype(df)
            if subtype == "resistance":
                return "resistance_csv", df
            if subtype == "compound_screen":
                return "compound_screen_csv", df
            if subtype == "admet":
                return "admet_csv", df
            return "compound_screen_csv", df
        except Exception:
            return "csv", None
    return "unknown", None
