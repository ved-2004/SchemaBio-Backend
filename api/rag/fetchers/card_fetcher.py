"""
CARD Fetcher — Comprehensive Antibiotic Resistance Database
https://card.mcmaster.ca/

Downloads the live CARD canonical data package from:
  https://card.mcmaster.ca/latest/data   (tar.bz2, ~4.6 MB)

Memory-efficient implementation:
  - Streams the tar.bz2 and processes one entry at a time
  - Only extracts card.json and aro_index.tsv (skips all other archive members)
  - Filters strictly: documents must match BOTH a target gene AND a drug class
    (when both are provided) rather than the old OR-based match
  - Hard cap at MAX_DOCUMENTS to prevent memory spikes on broad queries

card.json entry keys used:
  model_id, model_name, ARO_accession, ARO_name, ARO_description,
  ARO_category, model_type, model_description, CARD_short_name
"""
from __future__ import annotations

import io
import json
import logging
import tarfile
from typing import Any

import httpx

logger = logging.getLogger(__name__)

CARD_DATA_URL = "https://card.mcmaster.ca/latest/data"
CARD_BASE = "https://card.mcmaster.ca"

MAX_DOCUMENTS = 500  # Hard cap — prevents memory spikes on broad queries

# Gene / keyword filters for each supported target class
GENE_KEYWORDS: dict[str, list[str]] = {
    "gyrA":  ["gyra", "gyrase subunit a", "dna gyrase a", "fluoroquinolone resistant gyra"],
    "gyrB":  ["gyrb", "gyrase subunit b", "dna gyrase b"],
    "parC":  ["parc", "topoisomerase iv subunit a", "topo iv a"],
    "parE":  ["pare", "topoisomerase iv subunit b", "topo iv b"],
    "acrB":  ["acrb", "acrab", "acr efflux", "acrabtolc", "acrb efflux"],
    "tolC":  ["tolc", "outer membrane channel tolc"],
    "marA":  ["mara", "multiple antibiotic resistance"],
    "mecA":  ["meca", "pbp2a", "penicillin-binding protein 2a", "methicillin resistant"],
    "blaTEM": ["blatem", "tem beta-lactamase", "tem-1", "tem-2"],
    "blaSHV": ["blashv", "shv beta-lactamase"],
    "blaKPC": ["blakpc", "kpc", "klebsiella pneumoniae carbapenemase"],
    "qnrA":  ["qnra", "quinolone resistance protein"],
    "qnrB":  ["qnrb"],
    "qnrS":  ["qnrs"],
}

DRUG_CLASS_KEYWORDS = {
    "fluoroquinolone": ["fluoroquinolone", "quinolone", "ciprofloxacin", "levofloxacin"],
    "beta-lactam":     ["beta-lactam", "penicillin", "cephalosporin", "carbapenem"],
    "aminoglycoside":  ["aminoglycoside", "gentamicin", "tobramycin"],
}


# ── Download and parse CARD archive ──────────────────────────────────────────

async def _download_card_archive() -> bytes:
    """Download the CARD latest data tar.bz2 (~4.6 MB)."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(CARD_DATA_URL, follow_redirects=True)
        resp.raise_for_status()
        return resp.content


def _parse_aro_index(archive: tarfile.TarFile) -> dict[str, dict]:
    """Parse aro_index.tsv into {accession: row_dict} lookup.

    Returns a dict keyed by both "ARO:XXXXXXX" and bare "XXXXXXX" forms.
    """
    f = archive.extractfile("./aro_index.tsv")
    if not f:
        return {}
    lines = f.read().decode("utf-8", errors="replace").splitlines()
    f.close()
    if len(lines) < 2:
        return {}
    header = [h.strip() for h in lines[0].split("\t")]
    lookup: dict[str, dict] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        row = {header[i]: parts[i].strip() if i < len(parts) else "" for i in range(len(header))}
        acc = row.get("ARO Accession", "")
        if acc:
            lookup[acc] = row                         # "ARO:3003294"
            lookup[acc.replace("ARO:", "")] = row     # "3003294"
    return lookup


def _matches_genes(text: str, genes_lower: list[str]) -> bool:
    t = text.lower()
    return any(kw in t for gene in genes_lower for kw in GENE_KEYWORDS.get(gene, [gene]))


def _matches_drug_class(text: str, classes_lower: list[str]) -> bool:
    t = text.lower()
    return any(kw in t for dc in classes_lower for kw in DRUG_CLASS_KEYWORDS.get(dc, [dc]))


def _format_card_entry(model: dict, aro_row: dict | None = None) -> str:
    """Format a card.json model entry as a plain-text chunk."""
    aro_acc = model.get("ARO_accession", "")
    aro_name = model.get("ARO_name", model.get("model_name", ""))
    aro_desc = model.get("ARO_description", model.get("model_description", ""))
    short_name = model.get("CARD_short_name", "")
    model_type = model.get("model_type", "")

    # Drug class and mechanism from ARO_category (dict keyed by cvterm_id strings)
    aro_cat = model.get("ARO_category", {})
    drug_classes: list[str] = []
    mechanisms: list[str] = []
    gene_families: list[str] = []
    antibiotics: list[str] = []
    if isinstance(aro_cat, dict):
        for cat in aro_cat.values():
            if not isinstance(cat, dict):
                continue
            cat_name = cat.get("category_aro_name", "")
            cat_class = cat.get("category_aro_class_name", "")
            if cat_class == "AMR Gene Family":
                gene_families.append(cat_name)
            elif cat_class == "Drug Class":
                drug_classes.append(cat_name)
            elif cat_class == "Resistance Mechanism":
                mechanisms.append(cat_name)
            elif cat_class == "Antibiotic":
                antibiotics.append(cat_name)

    # Supplement with aro_index row if available (most reliable flat metadata)
    if aro_row:
        dc_from_index = aro_row.get("Drug Class", "")
        mech_from_index = aro_row.get("Resistance Mechanism", "")
        gf_from_index = aro_row.get("AMR Gene Family", "")
        if dc_from_index and not drug_classes:
            drug_classes = [dc_from_index]
        if mech_from_index and not mechanisms:
            mechanisms = [mech_from_index]
        if gf_from_index and not gene_families:
            gene_families = [gf_from_index]

    aro_display = f"ARO:{aro_acc}" if not aro_acc.startswith("ARO:") else aro_acc
    parts = [f"CARD {aro_display} — {aro_name}."]
    if short_name:
        parts.append(f"Short name: {short_name}.")
    if model_type:
        parts.append(f"Model type: {model_type}.")
    if gene_families:
        parts.append(f"AMR gene family: {'; '.join(gene_families)}.")
    if drug_classes:
        parts.append(f"Drug class: {'; '.join(drug_classes)}.")
    elif antibiotics:
        parts.append(f"Active against: {'; '.join(antibiotics[:5])}.")
    if mechanisms:
        parts.append(f"Resistance mechanism: {'; '.join(mechanisms)}.")
    if aro_desc:
        parts.append(aro_desc)
    parts.append(f"Source: CARD ({CARD_BASE}/aro/{aro_acc.replace('ARO:', '')}).")
    return " ".join(parts)


# ── Public entry point ────────────────────────────────────────────────────────

async def fetch_card_documents(
    target_genes: list[str] | None = None,
    drug_classes: list[str] | None = None,
) -> list[dict]:
    """
    Download the live CARD data package and extract resistance gene documents
    matching the given target_genes / drug_classes.

    Filtering logic:
      - If both target_genes and drug_classes are provided, a document must match
        at least one gene AND at least one drug class (AND filter, strict).
      - If only one is provided, matches on that dimension alone.
      - If neither is provided, returns an empty list (no broad-scan).

    Capped at MAX_DOCUMENTS to prevent memory spikes.

    Returns list of {"id", "text", "metadata"} ready for VectorStore.add_documents().
    """
    genes_lower = [g.lower() for g in target_genes] if target_genes else []
    classes_lower = [d.lower() for d in drug_classes] if drug_classes else []

    if not genes_lower and not classes_lower:
        logger.info("CARD: no target genes or drug classes specified — skipping fetch.")
        return []

    # Download
    logger.info("CARD: downloading latest data from %s ...", CARD_DATA_URL)
    raw = await _download_card_archive()
    logger.info("CARD: downloaded %d bytes.", len(raw))

    # Open archive and parse aro_index first (small, needed for enrichment)
    archive = tarfile.open(fileobj=io.BytesIO(raw), mode="r:bz2")
    aro_index_lookup = _parse_aro_index(archive)

    # Stream-process card.json: parse incrementally, filter, cap
    card_file = archive.extractfile("./card.json")
    if not card_file:
        archive.close()
        logger.error("CARD: card.json not found in archive.")
        return []

    card_models: dict = json.load(card_file)
    card_file.close()

    # Free the archive and raw bytes as soon as possible
    archive.close()
    del raw

    documents: list[dict] = []
    seen_ids: set[str] = set()

    for model_key, model in card_models.items():
        if not isinstance(model, dict):
            continue

        # Stop once we've hit the cap
        if len(documents) >= MAX_DOCUMENTS:
            break

        aro_acc_raw = str(model.get("ARO_accession", ""))
        aro_name = model.get("ARO_name", "")
        aro_desc = model.get("ARO_description", "")
        search_text = f"{aro_name} {aro_desc}"

        aro_row = aro_index_lookup.get(aro_acc_raw)
        if aro_row:
            search_text += (
                f" {aro_row.get('Drug Class', '')}"
                f" {aro_row.get('AMR Gene Family', '')}"
                f" {aro_row.get('Resistance Mechanism', '')}"
            )

        # Strict AND filter when both dimensions provided
        if genes_lower and classes_lower:
            if not (_matches_genes(search_text, genes_lower) and _matches_drug_class(search_text, classes_lower)):
                continue
        elif genes_lower:
            if not _matches_genes(search_text, genes_lower):
                continue
        elif classes_lower:
            if not _matches_drug_class(search_text, classes_lower):
                continue

        doc_id = f"card_{aro_acc_raw}"
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)

        aro_full = f"ARO:{aro_acc_raw}" if aro_acc_raw else ""
        text = _format_card_entry(model, aro_row)

        dc_meta = aro_row.get("Drug Class", "") if aro_row else ""
        mech_meta = aro_row.get("Resistance Mechanism", "") if aro_row else ""
        gf_meta = aro_row.get("AMR Gene Family", "") if aro_row else ""

        documents.append(
            {
                "id": doc_id,
                "text": text,
                "metadata": {
                    "source": "CARD",
                    "aro_accession": aro_full,
                    "source_url": f"{CARD_BASE}/aro/{aro_acc_raw}",
                    "gene_family": gf_meta,
                    "drug_class": dc_meta,
                    "mechanism": mech_meta,
                    "doc_type": "card_model",
                },
            }
        )

    # Free the large dict now that we're done iterating
    del card_models

    logger.info(
        "CARD: extracted %d relevant documents (genes=%s, classes=%s, cap=%d).",
        len(documents), target_genes, drug_classes, MAX_DOCUMENTS,
    )
    return documents
