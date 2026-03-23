"""
parsers/universal_parser.py

AIDEN's front door for all file ingestion.
Detects file type from extension + content, routes to the correct parser,
and assembles a DrugProgram from all uploaded files.

Supported inputs:
  *.vcf / *.vcf.gz     → vcf_parser  → variant data, gene list
  *.csv / *.tsv        → csv_parser  → compound screen OR resistance MIC data
  *.pdf                → pdf_parser  → abstract, methods, gene mentions
  *.txt / *.md         → text_parser → notes, target rationale
  (none)               → demo_data   → antibiotic resistance demo

CSV sub-type detection:
  - Compound screen: has IC50/zscore/logFC columns
  - Resistance MIC:  has MIC/strain/fold columns
  - ADMET data:      has solubility/permeability/stability columns
"""

from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from api.legacy.models.drug_program import (
    DrugProgram, CompoundProfile, TargetProfile, ResistanceProfile,
    EfficacySignals, EvidencePackage, ProgramStage,
)

logger = logging.getLogger(__name__)


# ─── CSV sub-type detection ───────────────────────────────────────────────────

_COMPOUND_SCREEN_SIGNALS  = ["ic50", "zscore", "z_score", "logfc", "log2fc", "neglogp", "efficacy"]
_RESISTANCE_SIGNALS       = ["mic", "fold_shift", "strain", "isolate", "resistant", "gyr", "par"]
_ADMET_SIGNALS            = ["solubility", "permeability", "caco", "stability", "clint", "cyp", "herg"]
_GENOMICS_SIGNALS         = ["gene", "chrom", "pos", "ref", "alt", "cadd", "clinvar"]


def _detect_csv_subtype(df: pd.DataFrame) -> str:
    cols = " ".join(df.columns).lower()
    if any(s in cols for s in _RESISTANCE_SIGNALS):
        return "resistance"
    if any(s in cols for s in _ADMET_SIGNALS):
        return "admet"
    if any(s in cols for s in _COMPOUND_SCREEN_SIGNALS):
        return "compound_screen"
    if any(s in cols for s in _GENOMICS_SIGNALS):
        return "genomics_tsv"
    # Inspect first row values
    for col in df.columns:
        if "mic" in col.lower() or "mhb" in col.lower():
            return "resistance"
    return "compound_screen"  # default


def _extract_col(df: pd.DataFrame, patterns: list[str]) -> Optional[str]:
    for col in df.columns:
        if any(re.search(p, col.lower()) for p in patterns):
            return col
    return None


# ─── Compound screen CSV parser ───────────────────────────────────────────────

def _parse_compound_screen(df: pd.DataFrame) -> tuple[list[dict], CompoundProfile, EvidencePackage]:
    """Parse compound screen CSV → all_compounds + lead compound profile."""
    ev = EvidencePackage()
    compounds = []

    id_col     = _extract_col(df, [r"^id$", r"cpd.?id", r"compound.?id", r"chembl"])
    name_col   = _extract_col(df, [r"^name$", r"compound$", r"drug", r"cpd.?name"])
    target_col = _extract_col(df, [r"target", r"protein"])
    ic50_col   = _extract_col(df, [r"ic50", r"ic_50"])
    mic_col    = _extract_col(df, [r"^mic$", r"mic_"])
    zsc_col    = _extract_col(df, [r"zscore", r"z_score"])
    fc_col     = _extract_col(df, [r"log2.?fc", r"logfc", r"log2fc"])
    negp_col   = _extract_col(df, [r"neg.?log", r"\-log10"])
    smiles_col = _extract_col(df, [r"smiles", r"structure"])
    veh_col    = _extract_col(df, [r"vehicle", r"dmso.?ctrl"])
    rep_col    = _extract_col(df, [r"replicate", r"n_rep", r"rep[0-9]"])

    ev.has_dose_response = ic50_col is not None or mic_col is not None
    has_vehicle = veh_col is not None
    has_replicates = rep_col is not None

    for idx, row in df.iterrows():
        def g(col, default=None):
            if col and col in row.index:
                v = row[col]
                return None if pd.isna(v) else v
            return default

        ic50 = None
        try:
            ic50 = float(g(ic50_col))
        except (TypeError, ValueError):
            pass

        mic = None
        try:
            mic = float(g(mic_col))
        except (TypeError, ValueError):
            pass

        zscore = None
        try:
            zscore = float(g(zsc_col))
        except (TypeError, ValueError):
            if ic50 and ic50 > 0:
                import numpy as np
                zscore = round(4.5 - np.log10(max(ic50, 0.01)), 2) if ic50 < 100 else round(-0.5, 2)

        log2fc = None
        try:
            log2fc = float(g(fc_col, 0))
        except (TypeError, ValueError):
            pass

        neglogp = None
        try:
            neglogp = float(g(negp_col, 1))
        except (TypeError, ValueError):
            pass

        flag = "UNCLASSIFIED"
        if zscore is not None:
            if zscore >= 3.5:   flag = "TOP_HIT"
            elif zscore >= 2.5: flag = "FOLLOW_UP"
            elif zscore < 0:    flag = "DEPRIORITIZE"
            else:               flag = "LOW"

        smiles = str(g(smiles_col, "")) or None
        dmso_risk = False
        if smiles:
            risky = ["C(=O)Nc1", "S(=O)(=O)N", "c1nc2ccccc2n1"]
            dmso_risk = any(r in smiles for r in risky)

        cpd_id = str(g(id_col, f"CPD_{str(idx+1).zfill(3)}"))
        name   = str(g(name_col, f"Compound {idx+1}"))

        compounds.append({
            "id": cpd_id,
            "name": name,
            "target": str(g(target_col, "Unknown")),
            "ic50_nm": ic50,
            "mic_ugml": mic,
            "log2fc": log2fc,
            "neg_log10p": neglogp,
            "zscore": zscore,
            "smiles": smiles,
            "dmso_risk": dmso_risk,
            "flag": flag,
        })

    # Identify lead compound (highest zscore or lowest IC50)
    lead = None
    top_hits = [c for c in compounds if c["flag"] == "TOP_HIT"]
    if top_hits:
        lead = min(top_hits, key=lambda c: c["ic50_nm"] or 9999)
    elif compounds:
        lead = compounds[0]

    compound_profile = CompoundProfile()
    if lead:
        compound_profile.name = lead["name"]
        compound_profile.ic50_nm = lead["ic50_nm"]
        compound_profile.mic_ugml = lead["mic_ugml"]
        compound_profile.smiles = lead["smiles"]

    return compounds, compound_profile, ev


# ─── Resistance CSV parser ────────────────────────────────────────────────────

def _parse_resistance_csv(df: pd.DataFrame) -> tuple[ResistanceProfile, EvidencePackage]:
    """Parse resistance assay data → ResistanceProfile."""
    ev = EvidencePackage()
    resistance = ResistanceProfile()

    strain_col = _extract_col(df, [r"strain", r"isolate", r"organism"])
    cpd_col    = _extract_col(df, [r"compound", r"drug", r"antibiotic"])
    mic_col    = _extract_col(df, [r"^mic", r"mic_"])
    wt_col     = _extract_col(df, [r"wt.?mic", r"wild.?type"])
    fold_col   = _extract_col(df, [r"fold", r"shift"])
    mut_col    = _extract_col(df, [r"mutation", r"variant", r"snp"])

    mic_values = []
    resistant_strains = set()
    sensitive_strains = set()
    mutations = set()
    fold_shifts: dict[str, float] = {}

    for _, row in df.iterrows():
        def g(col, default=None):
            if col and col in row.index:
                v = row[col]
                return None if pd.isna(v) else v
            return default

        strain   = str(g(strain_col, "Unknown"))
        compound = str(g(cpd_col, "Unknown"))
        mic = None
        try:
            mic = float(g(mic_col))
        except (TypeError, ValueError):
            pass
        wt_mic = None
        try:
            wt_mic = float(g(wt_col))
        except (TypeError, ValueError):
            pass
        fold = None
        try:
            fold = float(g(fold_col))
        except (TypeError, ValueError):
            if mic and wt_mic and wt_mic > 0:
                fold = round(mic / wt_mic, 1)

        classification = "UNKNOWN"
        if fold is not None:
            if fold >= 8:   classification = "RESISTANT"
            elif fold >= 4: classification = "INTERMEDIATE"
            elif fold < 1:  classification = "COLLATERAL_SENSITIVE"
            else:           classification = "SUSCEPTIBLE"

        mic_values.append({
            "strain": strain, "compound": compound,
            "mic": mic, "wt_mic": wt_mic, "fold": fold,
            "classification": classification,
        })

        if classification == "RESISTANT":
            resistant_strains.add(strain)
            if fold:
                fold_shifts[compound] = max(fold_shifts.get(compound, 0), fold)
        elif classification == "SUSCEPTIBLE":
            sensitive_strains.add(strain)

        mut = g(mut_col)
        if mut and str(mut) not in ("nan", "None", ""):
            mutations.add(str(mut))

    resistance.mic_values = mic_values
    resistance.resistant_strains = list(resistant_strains)
    resistance.sensitive_strains = list(sensitive_strains)
    resistance.resistance_mutations = list(mutations)
    resistance.fold_shift = max(fold_shifts.values()) if fold_shifts else None
    resistance.characterized = len(mutations) > 0

    ev.has_mic_data = len(mic_values) > 0
    ev.has_resistance_profiling = len(resistant_strains) > 0
    ev.has_dose_response = len(mic_values) >= 3

    return resistance, ev


# ─── ADMET CSV parser ─────────────────────────────────────────────────────────

def _parse_admet_csv(df: pd.DataFrame) -> EvidencePackage:
    ev = EvidencePackage()
    cols = " ".join(df.columns).lower()
    ev.has_solubility      = any(t in cols for t in ["solubility", "sol_"])
    ev.has_permeability    = any(t in cols for t in ["perm", "caco", "pampa"])
    ev.has_metabolic_stability = any(t in cols for t in ["clint", "stability", "microsomal", "t12"])
    ev.has_cyp_inhibition  = any(t in cols for t in ["cyp", "3a4", "2d6", "2c9"])
    ev.has_herg_data       = any(t in cols for t in ["herg", "ikr", "cardiac"])
    return ev


# ─── Text / notes parser ─────────────────────────────────────────────────────

def _parse_text_notes(path: Path) -> tuple[TargetProfile, list[str]]:
    """Extract target info from unstructured notes/rationale text."""
    import re as _re
    known_genes = {"GYRA", "GYRB", "PARC", "PARE", "BRCA1", "BRCA2", "TP53", "ATM",
                   "EGFR", "KRAS", "PARP1", "BRCA1", "DNAA", "GYRASE"}
    target = TargetProfile()
    context_lines = []

    try:
        text = path.read_text(errors="ignore")
        text_upper = text.upper()

        for gene in known_genes:
            if gene in text_upper:
                target.gene = gene.title()

        # Extract organism
        organisms = ["E. coli", "S. aureus", "E.coli", "MRSA", "M. tuberculosis", "H. pylori"]
        for org in organisms:
            if org.lower() in text.lower():
                target.organism = org
                break

        # Extract indication
        indications = ["antibiotic", "antibacterial", "antifungal", "oncology", "cancer", "kinase inhibitor"]
        for ind in indications:
            if ind.lower() in text.lower():
                target.indication = ind
                break

        # Extract mechanism
        mechanisms = ["gyrase", "topoisomerase", "DNA replication", "PARP trapping",
                      "kinase inhibition", "proteasome", "efflux pump"]
        for mech in mechanisms:
            if mech.lower() in text.lower():
                target.mechanism_of_action = mech
                break

        context_lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 20][:5]
    except Exception as e:
        logger.warning(f"Text parse failed: {e}")

    return target, context_lines


# ─── VCF integration (delegates to vcf_parser.py) ────────────────────────────

def _parse_vcf_file(path: Path) -> tuple[list[dict], TargetProfile, EvidencePackage]:
    """Parse VCF file and return variants, target info, evidence."""
    ev = EvidencePackage()
    try:
        from api.ingestion.parsers.vcf_parser import _parse_vcf_plaintext
        variants_raw = _parse_vcf_plaintext(path)
        variants = [v.model_dump() for v in variants_raw]
        genes = list({v["gene"] for v in variants if v.get("gene") != "UNKNOWN"})
        pathogenic = [v for v in variants if "pathogenic" in (v.get("clinvar_sig") or "").lower()]
        ev.has_target_validation = len(pathogenic) > 0
        ev.has_resistance_profiling = any("gyr" in v.get("gene","").lower() for v in variants)

        target = TargetProfile()
        if genes:
            # Prefer known antibiotic targets
            for preferred in ["GYRA", "GYRB", "PARC", "BRCA1", "TP53"]:
                if preferred in [g.upper() for g in genes]:
                    target.gene = preferred.title()
                    break
            if not target.gene:
                target.gene = genes[0]

        return variants, target, ev
    except Exception as e:
        logger.error(f"VCF parse failed: {e}")
        return [], TargetProfile(), ev


# ═══════════════════════════════════════════════════════════
# MAIN ENTRY — build DrugProgram from all uploaded files
# ═══════════════════════════════════════════════════════════

def build_drug_program_from_files(
    vcf_path: Optional[Path] = None,
    csv_paths: Optional[list[Path]] = None,
    pdf_path: Optional[Path] = None,
    text_paths: Optional[list[Path]] = None,
) -> DrugProgram:
    """
    Ingest any combination of files and return a populated DrugProgram.
    The LLM never sees raw file bytes — only the structured DrugProgram.
    """
    program = DrugProgram()
    program.uploaded_files = []
    combined_ev = EvidencePackage()

    # ── VCF ─────────────────────────────────────────────────────────────
    if vcf_path and vcf_path.exists():
        logger.info(f"Parsing VCF: {vcf_path.name}")
        variants, target_from_vcf, ev = _parse_vcf_file(vcf_path)
        program.all_variants = variants
        if target_from_vcf.gene:
            program.target.gene = target_from_vcf.gene
        _merge_evidence(combined_ev, ev)
        program.uploaded_files.append(vcf_path.name)
        program.add_trace(1, "VCF Parser", "VCF parsing",
            f"{len(variants)} variants extracted, target gene: {target_from_vcf.gene}", "vcf")

    # ── CSV files ────────────────────────────────────────────────────────
    for csv_path in (csv_paths or []):
        if not csv_path.exists():
            continue
        logger.info(f"Parsing CSV: {csv_path.name}")
        try:
            df = pd.read_csv(str(csv_path))
            subtype = _detect_csv_subtype(df)
            logger.info(f"  CSV subtype: {subtype}")

            if subtype == "resistance":
                resistance, ev = _parse_resistance_csv(df)
                program.resistance = resistance
                _merge_evidence(combined_ev, ev)
                program.add_trace(
                    len(program.agent_trace)+1, "CSV Parser (Resistance)",
                    "Resistance MIC parsing",
                    f"{len(resistance.resistant_strains)} resistant strains, "
                    f"{len(resistance.resistance_mutations)} mutations, "
                    f"fold-shift: {resistance.fold_shift}", "csv")

            elif subtype == "admet":
                ev = _parse_admet_csv(df)
                _merge_evidence(combined_ev, ev)
                program.add_trace(
                    len(program.agent_trace)+1, "CSV Parser (ADMET)",
                    "ADMET data detected",
                    f"solubility={ev.has_solubility}, stability={ev.has_metabolic_stability}", "csv")

            else:  # compound_screen
                compounds, compound_profile, ev = _parse_compound_screen(df)
                program.all_compounds = compounds
                if compound_profile.name:
                    program.compound = compound_profile
                top_hits = [c for c in compounds if c["flag"] == "TOP_HIT"]
                _merge_evidence(combined_ev, ev)
                program.add_trace(
                    len(program.agent_trace)+1, "CSV Parser (Screen)",
                    "Compound screen parsing",
                    f"{len(compounds)} compounds, {len(top_hits)} top hits, "
                    f"lead: {program.compound.name}", "csv")

        except Exception as e:
            logger.error(f"CSV parse failed for {csv_path.name}: {e}")
        program.uploaded_files.append(csv_path.name)

    # ── PDF ──────────────────────────────────────────────────────────────
    if pdf_path and pdf_path.exists():
        logger.info(f"Parsing PDF: {pdf_path.name}")
        try:
            from api.ingestion.parsers.pdf_parser import parse_pdf
            pdf_ctx = parse_pdf(pdf_path)
            if pdf_ctx.cell_line:
                program.efficacy.cell_line = pdf_ctx.cell_line
            if pdf_ctx.key_genes and not program.target.gene:
                program.target.gene = pdf_ctx.key_genes[0]
            if pdf_ctx.key_pathways:
                program.target.pathway = pdf_ctx.key_pathways[0]
            program.add_trace(
                len(program.agent_trace)+1, "PDF Parser",
                "PDF text extraction",
                f"genes: {pdf_ctx.key_genes[:3]}, cell line: {pdf_ctx.cell_line}", "pdf")
        except Exception as e:
            logger.error(f"PDF parse failed: {e}")
        program.uploaded_files.append(pdf_path.name)

    # ── Text notes ────────────────────────────────────────────────────────
    for txt_path in (text_paths or []):
        if not txt_path.exists():
            continue
        target_from_txt, lines = _parse_text_notes(txt_path)
        if target_from_txt.gene and not program.target.gene:
            program.target = target_from_txt
        elif target_from_txt.mechanism_of_action:
            program.target.mechanism_of_action = target_from_txt.mechanism_of_action
        program.uploaded_files.append(txt_path.name)

    # ── Merge evidence ────────────────────────────────────────────────────
    program.evidence = combined_ev

    # ── Infer target organism from resistance data ─────────────────────
    if program.resistance.resistance_mutations:
        gyra = any("gyr" in m.lower() for m in program.resistance.resistance_mutations)
        if gyra and not program.target.organism:
            program.target.organism = "E. coli (presumed)"
            program.target.gene = program.target.gene or "GyrA"
            program.target.target_class = "Type II topoisomerase"
            program.target.mechanism_of_action = "DNA gyrase inhibition"

    program.add_trace(
        len(program.agent_trace)+1, "Universal Parser",
        "DrugProgram assembled",
        f"compound: {program.compound.name or 'unknown'}, "
        f"target: {program.target.gene or 'unknown'}, "
        f"evidence completeness: {program.completeness_pct}%", "schema")

    return program


def _merge_evidence(base: EvidencePackage, additional: EvidencePackage) -> None:
    """Merge evidence fields — True wins (non-destructive)."""
    for field in base.model_fields:
        if field in ("blocking_gaps", "critical_gaps"):
            continue
        additional_val = getattr(additional, field, False)
        if additional_val and not getattr(base, field, False):
            setattr(base, field, additional_val)
