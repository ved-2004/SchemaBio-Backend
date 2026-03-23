"""
parsers/vcf_parser.py

Deterministic VCF/genomics parser for antibiotic resistance programs.

Handles:
  - Standard VCF 4.x (with cyvcf2)
  - Plain-text VCF fallback (no cyvcf2 dependency)
  - Resistance-relevant gene filtering (QRDR, efflux pump genes)
  - CADD score extraction
  - ClinVar significance parsing

Key resistance genes tracked:
  Fluoroquinolone targets: gyrA, gyrB, parC, parE
  Beta-lactam targets:     pbpA, pbpB, pbpC, blaTEM, blaSHV, blaOXA
  Aminoglycoside:          aac, ant, aph
  Efflux pumps:            acrA, acrB, tolC, mexA, mexB, oprM
  MRSA:                    mecA, mecC
  General AMR:             marA, marR, ramA, soxS

Output: list[ParsedVariant] — each variant fully typed and classified
"""

from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ─── Known resistance genes and their roles ──────────────────────────────────

RESISTANCE_GENE_META: dict[str, dict] = {
    # Fluoroquinolone (gyrase/topoisomerase)
    "gyra":  {"mechanism": "fluoroquinolone_target", "pathway": "DNA_replication",
               "qrdr_residues": [83, 87], "protein": "DNA gyrase subunit A"},
    "gyrb":  {"mechanism": "fluoroquinolone_target", "pathway": "DNA_replication",
               "protein": "DNA gyrase subunit B"},
    "parc":  {"mechanism": "fluoroquinolone_target", "pathway": "DNA_replication",
               "qrdr_residues": [80, 84], "protein": "Topoisomerase IV subunit C"},
    "pare":  {"mechanism": "fluoroquinolone_target", "pathway": "DNA_replication",
               "protein": "Topoisomerase IV subunit E"},
    # Efflux pumps
    "acra":  {"mechanism": "efflux_pump", "pathway": "active_efflux",
               "protein": "AcrAB-TolC pump component A"},
    "acrb":  {"mechanism": "efflux_pump", "pathway": "active_efflux",
               "protein": "AcrAB-TolC pump component B"},
    "tolc":  {"mechanism": "efflux_pump", "pathway": "active_efflux",
               "protein": "TolC outer membrane channel"},
    "mexa":  {"mechanism": "efflux_pump", "pathway": "active_efflux",
               "protein": "MexAB-OprM pump (P. aeruginosa)"},
    # MRSA
    "meca":  {"mechanism": "beta_lactam_resistance", "pathway": "cell_wall_synthesis",
               "protein": "Penicillin-binding protein 2a"},
    "mecc":  {"mechanism": "beta_lactam_resistance", "pathway": "cell_wall_synthesis",
               "protein": "Penicillin-binding protein 2c"},
    # Broad AMR regulators
    "mara":  {"mechanism": "regulatory", "pathway": "mar_operon",
               "protein": "Multiple antibiotic resistance protein A"},
    "marr":  {"mechanism": "regulatory", "pathway": "mar_operon",
               "protein": "Mar operon repressor"},
}

QRDR_HOTSPOTS = {
    "gyra": {83: "Ser83", 87: "Asp87"},
    "parc": {80: "Ser80", 84: "Glu84"},
}

CONSEQUENCE_SEVERITY = {
    "frameshift_variant":       "HIGH",
    "stop_gained":              "HIGH",
    "splice_donor_variant":     "HIGH",
    "splice_acceptor_variant":  "HIGH",
    "missense_variant":         "MODERATE",
    "inframe_insertion":        "MODERATE",
    "inframe_deletion":         "MODERATE",
    "synonymous_variant":       "LOW",
    "5_prime_utr_variant":      "MODIFIER",
    "3_prime_utr_variant":      "MODIFIER",
    "intron_variant":           "MODIFIER",
}


# ─── Output model ─────────────────────────────────────────────────────────────

class ParsedVariant(BaseModel):
    chrom:          str
    pos:            int
    ref:            str
    alt:            str
    gene:           str = "UNKNOWN"
    gene_lower:     str = ""          # for lookup
    consequence:    Optional[str] = None
    impact:         str = "UNKNOWN"
    aa_change:      Optional[str] = None   # e.g. "D87N"
    hgvs_p:         Optional[str] = None   # e.g. "p.Asp87Asn"
    cadd_score:     Optional[float] = None
    clinvar_sig:    Optional[str] = None
    af:             Optional[float] = None  # allele frequency
    is_resistance_gene: bool = False
    resistance_mechanism: Optional[str] = None
    is_qrdr_hotspot:    bool = False
    qrdr_residue:       Optional[str] = None
    resistance_note:    Optional[str] = None
    raw_info:           dict = {}


# ─── Plain-text VCF parser (no cyvcf2 needed) ─────────────────────────────────

def _parse_vcf_plaintext(path: Path) -> list[ParsedVariant]:
    """
    Parse VCF without cyvcf2 — handles standard VCF 4.x plain text.
    Extracts INFO field annotations (ANN, CSQ, CADD, CLNSIG).
    """
    variants: list[ParsedVariant] = []
    header_cols: list[str] = []

    try:
        text = path.read_text(errors="ignore")
    except Exception as e:
        logger.error(f"VCF read failed: {e}")
        return variants

    for line in text.splitlines():
        if line.startswith("##"):
            continue
        if line.startswith("#CHROM"):
            header_cols = line.lstrip("#").split("\t")
            continue
        if not line.strip():
            continue

        parts = line.split("\t")
        if len(parts) < 5:
            continue

        chrom = parts[0]
        pos_str = parts[1]
        ref = parts[3]
        alt_field = parts[4]
        info_str = parts[7] if len(parts) > 7 else ""

        # Handle multi-allelic — take first alt
        alt = alt_field.split(",")[0]

        try:
            pos = int(pos_str)
        except ValueError:
            continue

        info = _parse_info(info_str)
        variant = _build_variant(chrom, pos, ref, alt, info)
        variants.append(variant)

    logger.info(f"VCF parsed: {len(variants)} variants")
    return variants


def _parse_info(info_str: str) -> dict:
    """Parse VCF INFO field into a dict."""
    info: dict = {}
    if not info_str or info_str == ".":
        return info
    for part in info_str.split(";"):
        if "=" in part:
            k, _, v = part.partition("=")
            info[k] = v
        else:
            info[part] = True
    return info


def _build_variant(chrom: str, pos: int, ref: str, alt: str, info: dict) -> ParsedVariant:
    """Construct a ParsedVariant from parsed fields."""
    variant = ParsedVariant(chrom=chrom, pos=pos, ref=ref, alt=alt, raw_info=info)

    # ── Gene from ANN/CSQ (SnpEff / VEP) ─────────────────────────────
    ann = info.get("ANN", info.get("CSQ", ""))
    if ann:
        _parse_annotation(variant, str(ann))

    # ── CADD score ────────────────────────────────────────────────────
    for cadd_key in ("CADD_PHRED", "CADD", "CADD_phred"):
        if cadd_key in info:
            try:
                variant.cadd_score = float(str(info[cadd_key]).split(",")[0])
            except (ValueError, TypeError):
                pass
            break

    # ── ClinVar ──────────────────────────────────────────────────────
    for clnkey in ("CLNSIG", "CLINVAR_SIG"):
        if clnkey in info:
            variant.clinvar_sig = str(info[clnkey]).replace("_", " ")
            break

    # ── Allele frequency ──────────────────────────────────────────────
    for afkey in ("AF", "AF_popmax", "gnomAD_AF"):
        if afkey in info:
            try:
                variant.af = float(str(info[afkey]).split(",")[0])
            except (ValueError, TypeError):
                pass
            break

    # ── Resistance gene metadata ──────────────────────────────────────
    gene_l = variant.gene.lower()
    variant.gene_lower = gene_l
    if gene_l in RESISTANCE_GENE_META:
        meta = RESISTANCE_GENE_META[gene_l]
        variant.is_resistance_gene = True
        variant.resistance_mechanism = meta.get("mechanism")

        # QRDR hotspot detection
        if gene_l in QRDR_HOTSPOTS and variant.aa_change:
            for res_num, res_label in QRDR_HOTSPOTS[gene_l].items():
                # Match residue number in aa_change (e.g. D87N → 87)
                if str(res_num) in variant.aa_change:
                    variant.is_qrdr_hotspot = True
                    variant.qrdr_residue = res_label
                    variant.resistance_note = (
                        f"QRDR hotspot {res_label} in {variant.gene} — "
                        f"associated with fluoroquinolone resistance"
                    )
                    break

    # ── Impact from consequence ───────────────────────────────────────
    if variant.consequence:
        cons_l = variant.consequence.lower().replace(" ", "_")
        for key, sev in CONSEQUENCE_SEVERITY.items():
            if key in cons_l:
                variant.impact = sev
                break

    return variant


def _parse_annotation(variant: ParsedVariant, ann: str) -> None:
    """
    Parse SnpEff ANN or VEP CSQ annotation string.
    ANN format: Allele|Annotation|Impact|Gene_Name|...
    CSQ format: Allele|Consequence|IMPACT|SYMBOL|...
    """
    first_entry = ann.split(",")[0]
    fields = first_entry.split("|")
    if len(fields) < 4:
        return

    # Try SnpEff ANN
    # ANN: [0]=allele [1]=annotation [2]=impact [3]=gene_name [4]=gene_id [9]=hgvs_p ...
    if len(fields) >= 4:
        variant.consequence = fields[1].replace("&", ", ")
        variant.impact      = fields[2] if fields[2] else variant.impact
        variant.gene        = fields[3] if fields[3] else variant.gene

    if len(fields) >= 10:
        hgvs_p = fields[9]
        if hgvs_p and hgvs_p != ".":
            variant.hgvs_p = hgvs_p
            # Extract short aa change (D87N from p.Asp87Asn)
            variant.aa_change = _shorten_hgvs(hgvs_p)

    # VEP CSQ override if gene still unknown
    if variant.gene in ("UNKNOWN", "") and len(fields) >= 4:
        variant.gene = fields[3]


def _shorten_hgvs(hgvs_p: str) -> Optional[str]:
    """
    Convert p.Asp87Asn → D87N using 3-to-1 amino acid codes.
    """
    AA3 = {
        "Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E",
        "Gly":"G","His":"H","Ile":"I","Leu":"L","Lys":"K","Met":"M","Phe":"F",
        "Pro":"P","Ser":"S","Thr":"T","Trp":"W","Tyr":"Y","Val":"V",
        "Ter":"*","Stop":"*",
    }
    # p.Asp87Asn pattern
    m = re.search(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|Ter|del|dup|ins|\*)", hgvs_p)
    if m:
        ref_aa = AA3.get(m.group(1), m.group(1))
        pos    = m.group(2)
        alt_aa = AA3.get(m.group(3), m.group(3))
        return f"{ref_aa}{pos}{alt_aa}"
    return None


# ─── cyvcf2 parser (preferred when available) ─────────────────────────────────

def _parse_vcf_cyvcf2(path: Path) -> list[ParsedVariant]:
    """Parse VCF with cyvcf2 (preferred — handles bgzipped, indexed files)."""
    try:
        import cyvcf2
    except ImportError:
        logger.info("cyvcf2 not available — falling back to plain-text parser")
        return _parse_vcf_plaintext(path)

    variants: list[ParsedVariant] = []
    try:
        vcf = cyvcf2.VCF(str(path))
        for record in vcf:
            for alt in record.ALT:
                info = dict(record.INFO)
                v = _build_variant(
                    chrom=record.CHROM,
                    pos=int(record.POS),
                    ref=record.REF,
                    alt=str(alt),
                    info={k: str(v) for k, v in info.items()},
                )
                variants.append(v)
    except Exception as e:
        logger.error(f"cyvcf2 parse failed: {e} — falling back")
        return _parse_vcf_plaintext(path)

    return variants


# ─── Public entry point ───────────────────────────────────────────────────────

def parse_vcf(path: Path) -> list[ParsedVariant]:
    """
    Parse a VCF file. Tries cyvcf2 first, falls back to plain-text.
    Returns list of ParsedVariant, sorted by resistance relevance.
    """
    try:
        import cyvcf2  # noqa
        variants = _parse_vcf_cyvcf2(path)
    except ImportError:
        variants = _parse_vcf_plaintext(path)

    # Sort: resistance gene hotspots first
    variants.sort(key=lambda v: (
        0 if v.is_qrdr_hotspot else
        1 if v.is_resistance_gene else
        2 if v.impact == "HIGH" else
        3 if v.impact == "MODERATE" else 4
    ))

    high_impact = [v for v in variants if v.impact in ("HIGH", "MODERATE")]
    resistance  = [v for v in variants if v.is_resistance_gene]
    logger.info(
        f"VCF: {len(variants)} total, {len(high_impact)} high/moderate impact, "
        f"{len(resistance)} resistance genes"
    )
    return variants
