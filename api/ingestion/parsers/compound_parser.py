"""
parsers/compound_parser.py

Deterministic compound screen CSV parser.

Handles:
  - IC50 dose-response tables
  - HTS primary screens (z-score, log2FC, percent inhibition)
  - Plate-format data (well + compound + readout)
  - MIC-based compound screens (antibacterial)
  - Multi-point dose-response curves

Classifies compounds:
  TOP_HIT:     zscore ≥ 3.5 OR IC50 < 100 nM
  FOLLOW_UP:   zscore ≥ 2.5 OR IC50 < 500 nM
  LOW:         zscore 0–2.5
  DEPRIORITIZE: zscore < 0 OR inactive
  CONTRADICTION: flagged via contradiction detector (post-hoc)

Detects and flags:
  - No vehicle control well
  - Single concentration (not dose-response)
  - DMSO-sensitive scaffold patterns (from SMILES)
  - IC50 unit mixing (nM vs μM vs mg/L)
  - Z' factor if plate data present (< 0.5 = poor assay quality)
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

# ─── DMSO-sensitive scaffold patterns (SMARTS-lite, no rdkit needed) ─────────
# These structural features have known DMSO aggregation risk
_DMSO_RISK_SMILES_PATTERNS = [
    r"C\(=O\)Nc1",             # acyl amine
    r"S\(=O\)\(=O\)N",        # sulfonamide
    r"c1nc2ccccc2n1",          # benzimidazole
    r"C#N",                    # nitrile
    r"N=C\(N\)N",              # guanidine
    r"\[N\+\]",                # charged nitrogen (PAINS)
    r"c1ccc\(O\)cc1",          # para-phenol
    r"O=C1CC=CO1",             # butenolide
]

_COL_PATTERNS: dict[str, list[str]] = {
    "id":        [r"^id$", r"^cpd.?id", r"^compound.?id", r"^drug.?id", r"^chembl"],
    "name":      [r"^name$", r"^compound$", r"^drug$", r"^cpd.?name", r"^compound.?name"],
    "target":    [r"^target", r"^protein", r"^gene"],
    "ic50":      [r"^ic50", r"^ic_50", r"^ec50", r"^ec_50"],
    "mic":       [r"^mic$", r"^mic_", r"^mbc$"],
    "zscore":    [r"^zscore", r"^z.score", r"^z_score"],
    "log2fc":    [r"^log2.?fc", r"^log2.?fold", r"^lfc"],
    "neglogp":   [r"^.?log10.?p", r"^neg.?log", r"^neglog", r"^p.?value"],
    "pct_inh":   [r"^pct.?inh", r"^percent.?inh", r"^%inh", r"^inhibition"],
    "smiles":    [r"^smiles", r"^structure", r"^canonical"],
    "vehicle":   [r"^vehicle", r"^dmso.?ctrl", r"^solvent.?ctrl"],
    "replicate": [r"^rep\d", r"^replicate", r"^n_rep"],
    "well":      [r"^well", r"^position", r"^row", r"^col"],
    "plate":     [r"^plate", r"^plate.?id"],
    "zprime":    [r"^z.?prime", r"^z'"],
}


# ─── Output models ────────────────────────────────────────────────────────────

class ParsedCompound(BaseModel):
    id:            str
    name:          str
    target:        Optional[str] = None
    ic50_nm:       Optional[float] = None
    ic50_unit:     str = "nM"
    mic_ugml:      Optional[float] = None
    zscore:        Optional[float] = None
    log2fc:        Optional[float] = None
    neg_log10p:    Optional[float] = None
    pct_inhibition: Optional[float] = None
    smiles:        Optional[str] = None
    flag:          str = "UNCLASSIFIED"
    dmso_risk:     bool = False
    n_replicates:  int = 1
    is_vehicle_ctrl: bool = False
    raw_row:       dict = {}


class ParsedCompoundScreen(BaseModel):
    screen_type:   str = "ic50"    # ic50 | hts_zscore | hts_logfc | mic_screen
    compounds:     list[ParsedCompound] = []
    top_hits:      list[ParsedCompound] = []
    lead_compound: Optional[ParsedCompound] = None
    # Quality metrics
    has_vehicle_ctrl:  bool = False
    has_dose_response: bool = False
    has_replicates:    bool = False
    replicate_count:   int = 1
    zprime:            Optional[float] = None
    ic50_unit:         str = "nM"
    # Stats
    n_compounds:       int = 0
    n_top_hits:        int = 0
    hit_rate_pct:      float = 0.0
    best_ic50_nm:      Optional[float] = None
    # Audit flags
    unit_ambiguity:    bool = False
    raw_summary:       str = ""


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, field: str) -> Optional[str]:
    patterns = _COL_PATTERNS.get(field, [])
    for col in df.columns:
        h = col.lower().strip()
        if any(re.search(p, h) for p in patterns):
            return col
    return None


def _detect_screen_type(df: pd.DataFrame, ic50_col, zscore_col, logfc_col, mic_col) -> str:
    if mic_col:          return "mic_screen"
    if ic50_col:         return "ic50"
    if zscore_col:       return "hts_zscore"
    if logfc_col:        return "hts_logfc"
    return "ic50"


def _detect_ic50_unit(df: pd.DataFrame, ic50_col: str) -> tuple[str, bool]:
    """Returns (unit, has_ambiguity)."""
    vals = df[ic50_col].dropna().astype(str).tolist()
    nm_count = uM_count = 0
    for v in vals[:50]:
        v_l = v.lower()
        if "nm" in v_l:          nm_count += 1
        elif "μm" in v_l or "um" in v_l: uM_count += 1
    if nm_count > 0 and uM_count > 0:
        return "nM", True     # ambiguous
    if uM_count > nm_count:   return "μM", False
    return "nM", False        # default: nM


def _normalize_ic50_to_nm(value: Optional[float], unit: str) -> Optional[float]:
    if value is None: return None
    if unit == "μM":  return value * 1000
    if unit == "mM":  return value * 1_000_000
    if unit == "mg/L": return value * 1000  # rough — MW unknown
    return value  # already nM


def _has_dmso_risk(smiles: Optional[str]) -> bool:
    if not smiles: return False
    return any(re.search(p, smiles) for p in _DMSO_RISK_SMILES_PATTERNS)


def _compute_zscore_from_ic50(ic50_nm: Optional[float]) -> Optional[float]:
    """Heuristic zscore from IC50 when no zscore column present."""
    if ic50_nm is None: return None
    if ic50_nm < 10:    return 4.8
    if ic50_nm < 50:    return 4.2
    if ic50_nm < 100:   return 3.8
    if ic50_nm < 250:   return 3.2
    if ic50_nm < 500:   return 2.8
    if ic50_nm < 1000:  return 2.2
    if ic50_nm < 5000:  return 1.5
    return 0.5


def _classify_compound(
    ic50_nm: Optional[float],
    zscore: Optional[float],
    pct_inh: Optional[float],
    mic_ugml: Optional[float],
) -> str:
    # Use whatever signal is available, highest confidence first
    if zscore is not None:
        if zscore >= 3.5: return "TOP_HIT"
        if zscore >= 2.5: return "FOLLOW_UP"
        if zscore >= 0:   return "LOW"
        return "DEPRIORITIZE"
    if ic50_nm is not None:
        if ic50_nm < 100:   return "TOP_HIT"
        if ic50_nm < 500:   return "FOLLOW_UP"
        if ic50_nm < 2000:  return "LOW"
        return "DEPRIORITIZE"
    if mic_ugml is not None:
        if mic_ugml <= 0.5:  return "TOP_HIT"
        if mic_ugml <= 2.0:  return "FOLLOW_UP"
        if mic_ugml <= 8.0:  return "LOW"
        return "DEPRIORITIZE"
    if pct_inh is not None:
        if pct_inh >= 80: return "TOP_HIT"
        if pct_inh >= 50: return "FOLLOW_UP"
        if pct_inh >= 20: return "LOW"
        return "DEPRIORITIZE"
    return "UNCLASSIFIED"


def _is_vehicle_row(row_values: dict) -> bool:
    for k, v in row_values.items():
        kl = str(k).lower()
        vl = str(v).lower()
        if any(t in kl or t in vl for t in ["vehicle", "dmso_ctrl", "solvent", "negative_ctrl"]):
            return True
    return False


# ─── Main parser ──────────────────────────────────────────────────────────────

def parse_compound_screen(path: Path) -> ParsedCompoundScreen:
    """
    Parse a compound screen CSV into ParsedCompoundScreen.
    Handles IC50, HTS z-score, HTS log2FC, and MIC-based screens.
    """
    result = ParsedCompoundScreen()

    try:
        df = pd.read_csv(str(path))
    except Exception as e:
        logger.error(f"Compound CSV read failed: {e}")
        result.raw_summary = f"Parse failed: {e}"
        return result

    if df.empty:
        result.raw_summary = "Empty file"
        return result

    # ── Find columns ─────────────────────────────────────────────────
    id_col      = _find_col(df, "id")
    name_col    = _find_col(df, "name")
    target_col  = _find_col(df, "target")
    ic50_col    = _find_col(df, "ic50")
    mic_col     = _find_col(df, "mic")
    zscore_col  = _find_col(df, "zscore")
    logfc_col   = _find_col(df, "log2fc")
    negp_col    = _find_col(df, "neglogp")
    pct_col     = _find_col(df, "pct_inh")
    smiles_col  = _find_col(df, "smiles")
    veh_col     = _find_col(df, "vehicle")
    rep_col     = _find_col(df, "replicate")
    zprime_col  = _find_col(df, "zprime")

    result.screen_type      = _detect_screen_type(df, ic50_col, zscore_col, logfc_col, mic_col)
    result.has_vehicle_ctrl = veh_col is not None
    result.has_replicates   = rep_col is not None
    result.has_dose_response = ic50_col is not None

    # Detect IC50 unit
    ic50_unit = "nM"
    if ic50_col:
        ic50_unit, ambiguous = _detect_ic50_unit(df, ic50_col)
        result.ic50_unit       = ic50_unit
        result.unit_ambiguity  = ambiguous

    # Z' factor
    if zprime_col:
        try:
            zp_vals = pd.to_numeric(df[zprime_col], errors="coerce").dropna()
            if len(zp_vals) > 0:
                result.zprime = float(zp_vals.median())
        except Exception:
            pass

    # ── Parse rows ────────────────────────────────────────────────────
    for idx, row in df.iterrows():
        def g(col, default=None):
            if col and col in row.index:
                v = row[col]
                return None if pd.isna(v) else v
            return default

        row_dict = {str(c): str(row[c]) for c in df.columns if not pd.isna(row[c])}

        # Skip vehicle control rows
        if _is_vehicle_row(row_dict):
            result.has_vehicle_ctrl = True
            continue

        cpd_id   = str(g(id_col, f"CPD_{str(idx+1).zfill(4)}"))
        cpd_name = str(g(name_col, f"Compound {idx+1}"))
        target   = str(g(target_col)) if g(target_col) else None

        # IC50
        ic50_raw = g(ic50_col)
        ic50_val = None
        if ic50_raw is not None:
            try:
                ic50_str = re.sub(r"[><=≤≥\s]", "", str(ic50_raw))
                ic50_str = re.sub(r"(nM|μM|uM|mg/L|mM)", "", ic50_str, flags=re.IGNORECASE)
                ic50_val = float(ic50_str.split(",")[0])
                ic50_val = _normalize_ic50_to_nm(ic50_val, ic50_unit)
            except (ValueError, TypeError):
                pass

        # MIC
        mic_val = None
        if mic_col:
            try:
                mic_val = float(re.sub(r"[><=≤≥\s\D]", "", str(g(mic_col, "")).split(",")[0]))
            except (ValueError, TypeError):
                pass

        # Z-score
        zscore = None
        if zscore_col:
            try:
                zscore = float(g(zscore_col))
            except (TypeError, ValueError):
                pass
        if zscore is None and ic50_val is not None:
            zscore = _compute_zscore_from_ic50(ic50_val)

        # Log2FC
        log2fc = None
        if logfc_col:
            try:
                log2fc = float(g(logfc_col))
            except (TypeError, ValueError):
                pass

        # -log10(p)
        neg_log10p = None
        if negp_col:
            try:
                neg_log10p = float(g(negp_col))
            except (TypeError, ValueError):
                pass

        # % inhibition
        pct_inh = None
        if pct_col:
            try:
                pct_inh = float(g(pct_col))
            except (TypeError, ValueError):
                pass

        smiles = str(g(smiles_col)) if g(smiles_col) else None
        dmso_risk = _has_dmso_risk(smiles)

        n_rep = 1
        if rep_col:
            try:
                n_rep = int(g(rep_col, 1))
            except (TypeError, ValueError):
                pass

        flag = _classify_compound(ic50_val, zscore, pct_inh, mic_val)

        compound = ParsedCompound(
            id=cpd_id, name=cpd_name, target=target,
            ic50_nm=round(ic50_val, 3) if ic50_val else None,
            ic50_unit=ic50_unit, mic_ugml=mic_val,
            zscore=round(zscore, 3) if zscore else None,
            log2fc=round(log2fc, 3) if log2fc else None,
            neg_log10p=round(neg_log10p, 3) if neg_log10p else None,
            pct_inhibition=round(pct_inh, 1) if pct_inh else None,
            smiles=smiles, flag=flag, dmso_risk=dmso_risk,
            n_replicates=n_rep, raw_row=row_dict,
        )
        result.compounds.append(compound)

    # ── Summary fields ────────────────────────────────────────────────
    result.n_compounds = len(result.compounds)
    result.top_hits    = [c for c in result.compounds if c.flag == "TOP_HIT"]
    result.n_top_hits  = len(result.top_hits)
    result.hit_rate_pct = round(result.n_top_hits / max(result.n_compounds, 1) * 100, 1)
    result.replicate_count = max((c.n_replicates for c in result.compounds), default=1)

    ic50s = [c.ic50_nm for c in result.compounds if c.ic50_nm and c.ic50_nm > 0]
    if ic50s:
        result.best_ic50_nm = min(ic50s)

    # Lead compound: lowest IC50 or highest zscore among TOP_HIT
    if result.top_hits:
        hits_with_ic50 = [c for c in result.top_hits if c.ic50_nm]
        if hits_with_ic50:
            result.lead_compound = min(hits_with_ic50, key=lambda c: c.ic50_nm)
        else:
            hits_with_z = [c for c in result.top_hits if c.zscore]
            if hits_with_z:
                result.lead_compound = max(hits_with_z, key=lambda c: c.zscore)

    parts = [
        f"{result.n_compounds} compounds",
        f"{result.n_top_hits} top hits ({result.hit_rate_pct}%)",
    ]
    if result.lead_compound:
        parts.append(
            f"lead: {result.lead_compound.name} "
            f"(IC50 {result.lead_compound.ic50_nm:.1f} nM)" if result.lead_compound.ic50_nm
            else f"lead: {result.lead_compound.name}"
        )
    if not result.has_vehicle_ctrl:
        parts.append("⚠ no vehicle control")
    if result.replicate_count <= 1:
        parts.append("⚠ n=1 replicates")
    if result.unit_ambiguity:
        parts.append("⚠ IC50 unit ambiguity")

    result.raw_summary = " | ".join(parts)
    logger.info(f"Compound screen: {result.raw_summary}")
    return result
