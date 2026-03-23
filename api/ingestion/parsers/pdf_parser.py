"""
parsers/pdf_parser.py

Deterministic PDF text extractor for scientific papers and target rationale docs.

Extracts (no LLM):
  - Gene names (from curated AMR gene list + regex)
  - Organism names (Gram-negative, Gram-positive, species names)
  - Mechanism keywords (gyrase, efflux, beta-lactamase, etc.)
  - Quantitative claims (IC50, MIC, MBC values with units)
  - Cell lines mentioned
  - Key pathways
  - Document type (methods paper / review / rationale notes)

Uses PyMuPDF (fitz) with plain-text fallback via pdfminer.
"""

from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ─── Known AMR genes (curated) ────────────────────────────────────────────────
AMR_GENES = {
    # Fluoroquinolone
    "gyrA","gyrB","parC","parE",
    # Beta-lactamases
    "blaTEM","blaSHV","blaOXA","blaCTX","blaNDM","blaKPC","blaVIM","blaIMP",
    # MRSA
    "mecA","mecC","PBP2a",
    # Aminoglycoside modifying enzymes
    "aac","ant","aph",
    # Efflux pumps
    "acrA","acrB","tolC","mexA","mexB","mexC","mexD","oprM","oprN",
    "mdrA","smeDEF",
    # Regulators
    "marA","marR","ramA","soxS","robA",
    # MRSA virulence
    "spa","sbi","coa",
    # Other
    "rpoB","rpoC","katG","inhA","embB",  # TB resistance genes
}

# Gram-negative organisms
GRAM_NEGATIVE = [
    "E. coli", "Escherichia coli", "E.coli",
    "K. pneumoniae", "Klebsiella pneumoniae",
    "A. baumannii", "Acinetobacter baumannii",
    "P. aeruginosa", "Pseudomonas aeruginosa",
    "Enterobacteriaceae", "Salmonella", "Shigella",
    "H. pylori", "Helicobacter pylori",
    "N. gonorrhoeae", "Neisseria",
]

# Gram-positive organisms
GRAM_POSITIVE = [
    "S. aureus", "Staphylococcus aureus", "MRSA", "MSSA",
    "E. faecalis", "Enterococcus faecalis",
    "E. faecium", "Enterococcus faecium", "VRE",
    "S. pneumoniae", "Streptococcus pneumoniae",
    "M. tuberculosis", "Mycobacterium tuberculosis",
]

# Resistance mechanisms
MECHANISM_KEYWORDS = {
    "gyrase inhibition":      ["gyrase", "topoisomerase", "gyrA", "DNA supercoiling", "QRDR"],
    "efflux pump":            ["efflux", "AcrAB", "TolC", "MexAB", "pump inhibitor", "CCCP"],
    "beta-lactamase":         ["beta-lactamase", "β-lactamase", "ESBL", "carbapenemase", "NDM", "KPC"],
    "target modification":    ["PBP", "penicillin binding", "transpeptidase", "murein"],
    "ribosomal":              ["ribosom", "30S", "50S", "rRNA", "aminoglycoside"],
    "membrane permeability":  ["porin", "outer membrane", "ompF", "ompC", "lipopolysaccharide"],
    "enzymatic inactivation": ["acetyltransferase", "nucleotidyltransferase", "phosphotransferase"],
    "dna_repair":             ["SOS response", "RecA", "lexA", "DNA damage"],
}

# Quantitative patterns
_IC50_PATTERN  = re.compile(
    r"IC50\s*(?:of|=|:)?\s*([\d.]+)\s*([nNμuU]?[Mm](?:ol)?(?:/[lL])?|mg/[lL])",
    re.IGNORECASE
)
_MIC_PATTERN   = re.compile(
    r"MIC(?:50|90)?\s*(?:of|=|:)?\s*([\d.]+)\s*(μg/mL|mg/L|ug/mL|ng/mL|μM|nM)",
    re.IGNORECASE
)
_MBC_PATTERN   = re.compile(
    r"MBC\s*(?:of|=|:)?\s*([\d.]+)\s*(μg/mL|mg/L|ug/mL)",
    re.IGNORECASE
)
_FOLD_PATTERN  = re.compile(
    r"(\d+(?:\.\d+)?)[- ]?fold\s+(?:increase|decrease|resistance|reduction|change)",
    re.IGNORECASE
)
_CADD_PATTERN  = re.compile(r"CADD\s+(?:score|phred)?\s*(?:of|=|:)?\s*([\d.]+)", re.IGNORECASE)

# Cell lines
_CELL_LINE_PATTERNS = [
    r"\bHeLa\b", r"\bHEK293\b", r"\bA549\b", r"\bMCF-7\b", r"\bCaco-2\b",
    r"\bHepG2\b", r"\bRAW264\b", r"\bTHP-1\b", r"\bU87\b", r"\bPC-3\b",
    r"(?:ATCC\s+\d+)", r"ESKAPE",
]


# ─── Output model ─────────────────────────────────────────────────────────────

class QuantitativeClaim(BaseModel):
    claim_type:  str         # ic50 | mic | mbc | fold_change
    value:       float
    unit:        str
    context:     str = ""    # surrounding sentence for verification
    normalized_nm: Optional[float] = None


class ParsedPDF(BaseModel):
    filename:      str = ""
    doc_type:      str = "unknown"   # paper | review | methods | rationale_notes
    page_count:    int = 0
    key_genes:     list[str] = []
    organisms:     list[str] = []
    gram_class:    str = "unknown"   # gram_negative | gram_positive | both | unknown
    mechanisms:    list[str] = []
    mechanism_keywords: list[str] = []
    cell_line:     Optional[str] = None
    key_pathways:  list[str] = []
    quantitative_claims: list[QuantitativeClaim] = []
    fold_changes:  list[float] = []
    abstract:      str = ""
    full_text_snippet: str = ""
    raw_summary:   str = ""


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _normalize_to_nm(value: float, unit: str) -> Optional[float]:
    unit_l = unit.lower()
    if "nm" in unit_l:   return value
    if "μm" in unit_l or "um" in unit_l: return value * 1000
    if "mm" in unit_l:   return value * 1_000_000
    if "μg/ml" in unit_l or "ug/ml" in unit_l: return value * 1000  # rough
    if "mg/l" in unit_l: return value * 1000
    return None


def _detect_doc_type(text: str) -> str:
    text_l = text[:3000].lower()
    if any(t in text_l for t in ["abstract", "introduction", "methods", "results", "discussion"]):
        return "paper"
    if any(t in text_l for t in ["review", "current opinion", "perspectives"]):
        return "review"
    if any(t in text_l for t in ["target rationale", "hypothesis", "proposal", "program rationale"]):
        return "rationale_notes"
    if any(t in text_l for t in ["protocol", "procedure", "step 1", "materials and methods"]):
        return "methods"
    return "paper"


def _extract_abstract(text: str) -> str:
    """Extract abstract from paper text."""
    patterns = [
        r"(?:Abstract|ABSTRACT)\s*\n(.+?)(?:\n\n|\nIntroduction|\nINTRODUCTION)",
        r"(?:Abstract|ABSTRACT)[:\s]*(.{200,1500})(?:Introduction|Keywords|Background)",
    ]
    for p in patterns:
        m = re.search(p, text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).replace("\n", " ").strip()[:1500]
    # Fallback: first 800 chars after stripping headers
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 50]
    return " ".join(lines[:6])[:800]


def _extract_genes(text: str) -> list[str]:
    found = []
    # Exact gene name matching (case-sensitive for AMR genes)
    for gene in AMR_GENES:
        # Match gene name surrounded by word boundaries
        pattern = r'\b' + re.escape(gene) + r'\b'
        if re.search(pattern, text):
            if gene not in found:
                found.append(gene)
    # Also match common patterns like "gyrA D87N" style
    mutation_gene_pattern = re.compile(r'\b(gyr[AB]|parC|parE|mecA)\b', re.IGNORECASE)
    for m in mutation_gene_pattern.finditer(text):
        gene = m.group(1)
        if gene not in found:
            found.append(gene)
    return found[:20]


def _extract_organisms(text: str) -> tuple[list[str], str]:
    """Returns (organisms_found, gram_class)."""
    found = []
    has_neg = has_pos = False
    for org in GRAM_NEGATIVE:
        if org.lower() in text.lower() and org not in found:
            found.append(org)
            has_neg = True
    for org in GRAM_POSITIVE:
        if org.lower() in text.lower() and org not in found:
            found.append(org)
            has_pos = True
    gram = "both" if (has_neg and has_pos) else "gram_negative" if has_neg else "gram_positive" if has_pos else "unknown"
    return found[:10], gram


def _extract_mechanisms(text: str) -> tuple[list[str], list[str]]:
    """Returns (mechanism_names, raw_keywords)."""
    mechs = []
    keywords = []
    text_l = text.lower()
    for mech_name, kws in MECHANISM_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in text_l:
                if mech_name not in mechs:
                    mechs.append(mech_name)
                if kw not in keywords:
                    keywords.append(kw)
                break
    return mechs, keywords


def _extract_quantitative(text: str) -> list[QuantitativeClaim]:
    claims = []
    sentences = re.split(r'[.!?]\s+', text)

    for pattern, claim_type in [
        (_IC50_PATTERN, "ic50"),
        (_MIC_PATTERN, "mic"),
        (_MBC_PATTERN, "mbc"),
    ]:
        for m in pattern.finditer(text):
            try:
                value = float(m.group(1))
                unit  = m.group(2)
                # Find surrounding sentence for context
                start = max(0, m.start() - 100)
                end   = min(len(text), m.end() + 100)
                context = text[start:end].replace("\n", " ").strip()

                nm_val = _normalize_to_nm(value, unit)
                claims.append(QuantitativeClaim(
                    claim_type=claim_type, value=value, unit=unit,
                    context=context[:200], normalized_nm=nm_val,
                ))
            except (ValueError, IndexError):
                pass

    # Deduplicate by value + type
    seen = set()
    unique = []
    for c in claims:
        key = (c.claim_type, round(c.value, 3))
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique[:30]


def _extract_cell_line(text: str) -> Optional[str]:
    for pattern in _CELL_LINE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(0)
    return None


def _extract_fold_changes(text: str) -> list[float]:
    folds = []
    for m in _FOLD_PATTERN.finditer(text):
        try:
            folds.append(float(m.group(1)))
        except ValueError:
            pass
    return sorted(set(folds), reverse=True)[:10]


# ─── PDF text extraction ──────────────────────────────────────────────────────

def _extract_text_pymupdf(path: Path) -> tuple[str, int]:
    """Extract text using PyMuPDF (fitz)."""
    try:
        import fitz
        doc = fitz.open(str(path))
        pages = []
        for page in doc:
            pages.append(page.get_text())
        text = "\n".join(pages)
        return text, len(doc)
    except ImportError:
        logger.info("PyMuPDF not available")
        return "", 0
    except Exception as e:
        logger.error(f"PyMuPDF extraction failed: {e}")
        return "", 0


def _extract_text_pdfminer(path: Path) -> tuple[str, int]:
    """Fallback text extraction using pdfminer.six."""
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(str(path))
        # Rough page count from form feeds
        page_count = text.count("\x0c") + 1
        return text, page_count
    except ImportError:
        logger.info("pdfminer not available")
        return "", 0
    except Exception as e:
        logger.error(f"pdfminer extraction failed: {e}")
        return "", 0


# ─── Main entry ───────────────────────────────────────────────────────────────

def parse_pdf(path: Path) -> ParsedPDF:
    """
    Parse a scientific paper or notes PDF.
    Tries PyMuPDF first, then pdfminer, then reports empty.
    """
    result = ParsedPDF(filename=path.name)

    # Extract text
    text, page_count = _extract_text_pymupdf(path)
    if not text:
        text, page_count = _extract_text_pdfminer(path)
    if not text:
        logger.warning(f"PDF text extraction failed for {path.name}")
        result.raw_summary = "PDF extraction failed — no text extracted"
        return result

    result.page_count = page_count

    # Run all extractors
    result.doc_type             = _detect_doc_type(text)
    result.abstract             = _extract_abstract(text)
    result.full_text_snippet    = text[:2000].replace("\n", " ")
    result.key_genes            = _extract_genes(text)
    result.organisms, result.gram_class = _extract_organisms(text)
    result.mechanisms, result.mechanism_keywords = _extract_mechanisms(text)
    result.cell_line            = _extract_cell_line(text)
    result.quantitative_claims  = _extract_quantitative(text)
    result.fold_changes         = _extract_fold_changes(text)

    # Pathways from mechanism keywords
    result.key_pathways = list({
        kw for kw in result.mechanism_keywords
        if any(pw in kw.lower() for pw in ["pathway", "operon", "circuit", "regulon"])
    })[:5]

    parts = [
        f"{result.doc_type}",
        f"{page_count}pp",
        f"genes: {', '.join(result.key_genes[:4]) or 'none'}",
        f"organisms: {len(result.organisms)}",
        f"mechanisms: {len(result.mechanisms)}",
        f"quantitative claims: {len(result.quantitative_claims)}",
    ]
    result.raw_summary = " | ".join(parts)
    logger.info(f"PDF parsed: {result.raw_summary}")
    return result
