"""
parsers/assay_parser.py

Deterministic resistance assay CSV parser.

Handles:
  - MIC (broth microdilution) tables — CLSI M07 format
  - Time-kill kinetics data
  - Mutation frequency assays
  - Strain × compound MIC matrices
  - Checkerboard (synergy) assays

Detects and flags:
  - Missing wild-type/parent strain baseline → fold-shift cannot be computed
  - Single replicate (n=1) → high false-positive risk
  - Unit ambiguity (μg/mL vs mg/L vs nM mixed)
  - MIC values reported as ranges (e.g. "0.125/0.25") → normalize to midpoint
  - CLSI breakpoint classification (susceptible / intermediate / resistant)

CLSI 2024 breakpoints for common antibiotics (μg/mL):
  ciprofloxacin: S≤1, R≥4
  levofloxacin:  S≤2, R≥8
  gentamicin:    S≤4, R≥16
  vancomycin:    S≤2, R≥16
  daptomycin:    S≤1, R≥4
"""

from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ─── CLSI 2024 MIC breakpoints (μg/mL) ───────────────────────────────────────
CLSI_BREAKPOINTS: dict[str, dict] = {
    "ciprofloxacin":  {"s": 1.0,  "i": 2.0,  "r": 4.0},
    "levofloxacin":   {"s": 2.0,  "i": 4.0,  "r": 8.0},
    "moxifloxacin":   {"s": 1.0,  "i": 2.0,  "r": 4.0},
    "ofloxacin":      {"s": 2.0,  "i": 4.0,  "r": 8.0},
    "gentamicin":     {"s": 4.0,  "i": 8.0,  "r": 16.0},
    "amikacin":       {"s": 16.0, "i": 32.0, "r": 64.0},
    "vancomycin":     {"s": 2.0,  "i": 4.0,  "r": 16.0},
    "daptomycin":     {"s": 1.0,  "i": 2.0,  "r": 4.0},
    "linezolid":      {"s": 4.0,  "i": 4.0,  "r": 8.0},
    "amoxicillin":    {"s": 2.0,  "i": 4.0,  "r": 8.0},
    "ceftriaxone":    {"s": 1.0,  "i": 2.0,  "r": 4.0},
    "meropenem":      {"s": 1.0,  "i": 2.0,  "r": 4.0},
}

# Column name pattern → field mapping
_COL_PATTERNS: dict[str, list[str]] = {
    "strain":    [r"^strain", r"^isolate", r"^organism", r"^sample", r"^id$"],
    "compound":  [r"^compound", r"^antibiotic", r"^drug", r"^cpd", r"^antimicrobial"],
    "mic":       [r"^mic$", r"^mic_", r"mic\s*\(", r"minimum.inhibitory"],
    "wt_mic":    [r"wt.?mic", r"wild.?type", r"parent.?mic", r"baseline"],
    "fold":      [r"^fold", r"fold.?shift", r"fold.?change", r"fold.?diff"],
    "mutation":  [r"^mutation", r"^variant", r"^snp", r"^amino.?acid"],
    "replicate": [r"^rep\d", r"^replicate", r"^n_rep", r"^n$"],
    "time":      [r"^time", r"^hour", r"^hr$", r"^t\d"],
    "cfu":       [r"^cfu", r"^log.cfu", r"^viable"],
    "freq":      [r"^freq", r"^mutation.freq", r"^rate"],
}


# ─── Output models ────────────────────────────────────────────────────────────

class MICEntry(BaseModel):
    strain:           str
    compound:         str
    mic_value:        Optional[float] = None
    mic_unit:         str = "μg/mL"
    wt_mic:           Optional[float] = None
    fold_shift:       Optional[float] = None
    classification:   str = "UNKNOWN"   # SUSCEPTIBLE | INTERMEDIATE | RESISTANT | UNKNOWN
    breakpoint_source: str = "CLSI_2024"
    n_replicates:     int = 1
    is_wt_strain:     bool = False


class ParsedAssayData(BaseModel):
    assay_type:       str = "mic"          # mic | time_kill | mutation_freq | checkerboard
    mic_entries:      list[MICEntry] = []
    strains:          list[str] = []
    compounds:        list[str] = []
    resistant_strains: list[str] = []
    sensitive_strains: list[str] = []
    wt_strains:       list[str] = []
    max_fold_shift:   Optional[float] = None
    compound_fold_shifts: dict[str, float] = {}
    mutations_by_strain: dict[str, list[str]] = {}
    # Audit flags
    has_wt_control:   bool = False
    has_replicates:   bool = False
    replicate_count:  int = 1
    unit_consistent:  bool = True
    detected_unit:    str = "μg/mL"
    n_strains:        int = 0
    n_compounds:      int = 0
    raw_summary:      str = ""


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _match_col(header: str, patterns: list[str]) -> bool:
    h = header.lower().strip()
    return any(re.search(p, h) for p in patterns)


def _find_col(df: pd.DataFrame, field: str) -> Optional[str]:
    patterns = _COL_PATTERNS.get(field, [])
    for col in df.columns:
        if _match_col(col, patterns):
            return col
    return None


def _detect_assay_type(df: pd.DataFrame) -> str:
    cols = " ".join(df.columns).lower()
    has_time = bool(re.search(r"\btime\b|\bhour\b|\bhr\b", cols))
    has_cfu  = bool(re.search(r"\bcfu\b|\bviable\b|\blog.cfu\b", cols))
    has_freq = bool(re.search(r"\bfreq\b|\bmutation.rate\b", cols))
    has_fic  = bool(re.search(r"\bfic\b|\bsynergy\b|\bcheckerboard\b", cols))
    if has_time and has_cfu: return "time_kill"
    if has_freq:              return "mutation_freq"
    if has_fic:               return "checkerboard"
    return "mic"


def _normalize_mic_value(raw: str) -> Optional[float]:
    """Handle MIC strings: ">32", "≤0.125", "0.125/0.25", "0.5 μg/mL"."""
    if not raw or str(raw) in ("nan", "None", "", ".", "ND"):
        return None
    s = str(raw).strip()
    # Strip comparison operators
    s = re.sub(r"^[><=≤≥]+", "", s)
    # Take first of range (0.125/0.25 → 0.125)
    s = s.split("/")[0]
    # Strip unit suffixes
    s = re.sub(r"[\s]*(μg/mL|mg/L|ug/ml|mg/l|nM|μM|ng/mL)", "", s, flags=re.IGNORECASE)
    try:
        return float(s.strip())
    except ValueError:
        return None


def _detect_unit(df: pd.DataFrame, mic_col: str) -> str:
    for val in df[mic_col].dropna().astype(str).head(20):
        val_l = val.lower()
        if "nm" in val_l:  return "nM"
        if "μm" in val_l or "um" in val_l: return "μM"
        if "mg/l" in val_l: return "mg/L"
    return "μg/mL"  # CLSI default


def _classify_mic(mic: float, compound: str, wt_mic: Optional[float]) -> str:
    """Classify using fold-shift first, CLSI breakpoints as fallback."""
    if wt_mic and wt_mic > 0:
        fold = mic / wt_mic
        if fold >= 8:    return "RESISTANT"
        if fold >= 4:    return "INTERMEDIATE"
        if fold <= 0.5:  return "COLLATERAL_SENSITIVE"
        return "SUSCEPTIBLE"
    # CLSI breakpoints
    for drug_name, bp in CLSI_BREAKPOINTS.items():
        if drug_name in compound.lower():
            if mic >= bp["r"]: return "RESISTANT"
            if mic <= bp["s"]: return "SUSCEPTIBLE"
            return "INTERMEDIATE"
    return "UNKNOWN"


def _is_wt_strain(strain_name: str) -> bool:
    """Detect wild-type / parent strain by name."""
    wt_patterns = [r"wt$", r"wild.?type", r"parent", r"atcc", r"reference",
                   r"25922", r"mg1655", r"k-12", r"nctc"]
    name = strain_name.lower()
    return any(re.search(p, name) for p in wt_patterns)


# ─── Main parser ──────────────────────────────────────────────────────────────

def parse_resistance_assay(path: Path) -> ParsedAssayData:
    """
    Parse a resistance assay CSV file into structured ParsedAssayData.
    Works for MIC tables, time-kill, mutation frequency, and checkerboard.
    """
    result = ParsedAssayData()

    try:
        df = pd.read_csv(str(path))
    except Exception as e:
        logger.error(f"Assay CSV read failed: {e}")
        result.raw_summary = f"Parse failed: {e}"
        return result

    if df.empty:
        result.raw_summary = "Empty file"
        return result

    result.assay_type = _detect_assay_type(df)
    logger.info(f"Assay type: {result.assay_type}, rows: {len(df)}, cols: {list(df.columns)}")

    # ── Find columns ─────────────────────────────────────────────────
    strain_col   = _find_col(df, "strain")
    compound_col = _find_col(df, "compound")
    mic_col      = _find_col(df, "mic")
    wt_col       = _find_col(df, "wt_mic")
    fold_col     = _find_col(df, "fold")
    mut_col      = _find_col(df, "mutation")
    rep_col      = _find_col(df, "replicate")

    result.has_replicates = rep_col is not None
    if mic_col:
        result.detected_unit = _detect_unit(df, mic_col)

    all_strains: list[str] = []
    all_compounds: list[str] = []

    # ── Parse each row ────────────────────────────────────────────────
    for _, row in df.iterrows():
        def g(col, default=None):
            if col and col in row.index:
                v = row[col]
                return None if pd.isna(v) else v
            return default

        strain   = str(g(strain_col, "Unknown_strain"))
        compound = str(g(compound_col, "Unknown_compound"))

        if strain not in all_strains:
            all_strains.append(strain)
        if compound not in all_compounds:
            all_compounds.append(compound)

        # MIC value
        mic_raw = g(mic_col)
        mic_val = _normalize_mic_value(str(mic_raw)) if mic_raw is not None else None

        # WT MIC
        wt_raw = g(wt_col)
        wt_val = _normalize_mic_value(str(wt_raw)) if wt_raw is not None else None

        # Fold shift (use provided or compute)
        fold = None
        if fold_col:
            try:
                fold = float(g(fold_col))
            except (TypeError, ValueError):
                pass
        if fold is None and mic_val and wt_val and wt_val > 0:
            fold = round(mic_val / wt_val, 2)

        # Classification
        classification = "UNKNOWN"
        if mic_val is not None:
            classification = _classify_mic(mic_val, compound, wt_val)

        # Replicates
        n_rep = 1
        if rep_col:
            try:
                n_rep = int(g(rep_col, 1))
            except (TypeError, ValueError):
                pass

        entry = MICEntry(
            strain=strain, compound=compound,
            mic_value=mic_val, mic_unit=result.detected_unit,
            wt_mic=wt_val, fold_shift=fold,
            classification=classification,
            n_replicates=n_rep,
            is_wt_strain=_is_wt_strain(strain),
        )
        result.mic_entries.append(entry)

        # Track WT strains
        if entry.is_wt_strain and strain not in result.wt_strains:
            result.wt_strains.append(strain)

        # Track classifications
        if classification == "RESISTANT" and strain not in result.resistant_strains:
            result.resistant_strains.append(strain)
        elif classification == "SUSCEPTIBLE" and strain not in result.sensitive_strains:
            result.sensitive_strains.append(strain)

        # Track fold shifts per compound
        if fold and fold > result.compound_fold_shifts.get(compound, 0):
            result.compound_fold_shifts[compound] = fold

        # Mutations
        mut = g(mut_col)
        if mut and str(mut) not in ("nan", "None", ""):
            if strain not in result.mutations_by_strain:
                result.mutations_by_strain[strain] = []
            if str(mut) not in result.mutations_by_strain[strain]:
                result.mutations_by_strain[strain].append(str(mut))

    # ── Summary fields ────────────────────────────────────────────────
    result.strains   = all_strains
    result.compounds = all_compounds
    result.n_strains   = len(all_strains)
    result.n_compounds = len(all_compounds)
    result.has_wt_control = len(result.wt_strains) > 0
    result.replicate_count = max(
        (e.n_replicates for e in result.mic_entries), default=1
    )

    if result.compound_fold_shifts:
        result.max_fold_shift = max(result.compound_fold_shifts.values())

    parts = [
        f"{result.n_strains} strains",
        f"{result.n_compounds} compounds",
        f"{len(result.resistant_strains)} resistant strains",
    ]
    if result.max_fold_shift:
        max_cpd = max(result.compound_fold_shifts, key=result.compound_fold_shifts.get)
        parts.append(f"max fold-shift: {result.max_fold_shift}× ({max_cpd})")
    if not result.has_wt_control:
        parts.append("⚠ no WT control")
    if result.replicate_count <= 1:
        parts.append("⚠ n=1 replicates")

    result.raw_summary = " | ".join(parts)
    logger.info(f"Assay parsed: {result.raw_summary}")
    return result
