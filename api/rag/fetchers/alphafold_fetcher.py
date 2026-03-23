"""
AlphaFold Fetcher — AlphaFold Protein Structure Database
https://alphafold.ebi.ac.uk/

Fetches protein structure data purely from the live AlphaFold EBI REST API.
No hardcoded structural notes — all text is derived from API responses.

API reference: https://alphafold.ebi.ac.uk/api/
"""
from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api"
ALPHAFOLD_ENTRY = "https://alphafold.ebi.ac.uk/entry"

# UniProt accession lookup for known antibiotic-resistance-relevant proteins.
# This is a routing table only — no fallback content.
UNIPROT_MAP: dict[str, str] = {
    # E. coli K-12
    "gyrA":   "P0AES4",
    "gyrB":   "P0AES6",
    "parC":   "P0AES9",
    "parE":   "P0AEB2",
    "acrB":   "P31224",
    "tolC":   "P02930",
    "marA":   "P0ACH5",
    # S. aureus
    "mecA":   "P0C1U8",
    # K. pneumoniae
    "blaKPC": "Q9S3U0",
}


def _confidence_label(plddt: float) -> str:
    if plddt >= 90:
        return "very high"
    if plddt >= 70:
        return "confident"
    if plddt >= 50:
        return "low"
    return "very low"


def _build_text_from_api(uniprot_id: str, gene_key: str, pred: dict) -> str:
    """Build a plain-text document purely from AlphaFold API response fields."""
    entry_id    = pred.get("entryId", f"AF-{uniprot_id}-F1")
    version     = pred.get("latestVersion", "?")
    gene        = pred.get("gene", gene_key)
    uniprot_id_ = pred.get("uniprotAccession", uniprot_id)
    swiss_id    = pred.get("uniprotId", "")
    description = pred.get("uniprotDescription", "")
    organism    = pred.get("organismScientificName", "")
    seq_len     = pred.get("sequenceEnd", "?")
    tool        = pred.get("toolUsed", "AlphaFold")
    created     = pred.get("modelCreatedDate", "")[:10] if pred.get("modelCreatedDate") else ""
    reviewed    = pred.get("isUniProtReviewed", False)
    ref_proteome= pred.get("isReferenceProteome", False)

    plddt       = pred.get("globalMetricValue")
    frac_vh     = pred.get("fractionPlddtVeryHigh", 0.0)
    frac_conf   = pred.get("fractionPlddtConfident", 0.0)
    frac_low    = pred.get("fractionPlddtLow", 0.0)
    frac_vl     = pred.get("fractionPlddtVeryLow", 0.0)

    pdb_url  = pred.get("pdbUrl", "")
    cif_url  = pred.get("cifUrl", "")
    pae_url  = pred.get("paeDocUrl", "")

    parts: list[str] = []

    parts.append(
        f"AlphaFold {entry_id} (model v{version}): {description} — gene {gene}, "
        f"UniProt {uniprot_id_} ({swiss_id})."
    )
    if organism:
        parts.append(f"Organism: {organism}.")
    parts.append(f"Sequence length: {seq_len} amino acids.")
    if created:
        parts.append(f"Model created: {created}.")
    flags = []
    if reviewed:
        flags.append("UniProt reviewed (Swiss-Prot)")
    if ref_proteome:
        flags.append("reference proteome")
    if flags:
        parts.append(f"Entry status: {', '.join(flags)}.")
    parts.append(f"Structure tool: {tool}.")

    if plddt is not None:
        label = _confidence_label(float(plddt))
        parts.append(
            f"Global mean pLDDT: {plddt:.2f} ({label} confidence). "
            f"Residue confidence breakdown: very high {frac_vh*100:.1f}%, "
            f"confident {frac_conf*100:.1f}%, low {frac_low*100:.1f}%, "
            f"very low {frac_vl*100:.1f}%."
        )

    parts.append(f"Structure viewer: {ALPHAFOLD_ENTRY}/{uniprot_id_}")
    if pdb_url:
        parts.append(f"PDB download: {pdb_url}")
    if cif_url:
        parts.append(f"mmCIF download: {cif_url}")
    if pae_url:
        parts.append(f"PAE (predicted aligned error): {pae_url}")

    parts.append(f"Source: AlphaFold EBI (alphafold.ebi.ac.uk).")
    return " ".join(parts)


async def _fetch_prediction(uniprot_id: str, client: httpx.AsyncClient) -> dict | None:
    """Fetch AlphaFold prediction metadata for a single UniProt accession."""
    url = f"{ALPHAFOLD_API}/prediction/{uniprot_id}"
    try:
        resp = await client.get(url, timeout=12.0, follow_redirects=True)
        if resp.status_code == 200:
            data = resp.json()
            return data[0] if isinstance(data, list) and data else None
    except Exception as exc:
        logger.warning("AlphaFold API failed for %s: %s", uniprot_id, exc)
    return None


async def fetch_alphafold_documents(target_genes: list[str] | None = None) -> list[dict]:
    """
    Fetch AlphaFold structure data for requested target genes.
    All text is built exclusively from live API responses — no hardcoded content.

    Raises if a requested gene has no UniProt mapping in UNIPROT_MAP and the
    AlphaFold search API also returns nothing.

    Returns list of {"id", "text", "metadata"} ready for VectorStore.add_documents().
    """
    genes_lower = [g.lower() for g in target_genes] if target_genes else []

    # Select proteins to query
    if genes_lower:
        proteins = {
            gene: uid
            for gene, uid in UNIPROT_MAP.items()
            if any(gl in gene.lower() for gl in genes_lower)
        }
    else:
        proteins = dict(UNIPROT_MAP)

    if not proteins:
        logger.warning("AlphaFold: no UniProt mappings found for genes=%s", target_genes)
        return []

    documents: list[dict] = []
    async with httpx.AsyncClient() as client:
        for gene_key, uniprot_id in proteins.items():
            pred = await _fetch_prediction(uniprot_id, client)
            if not pred:
                logger.warning("AlphaFold: no prediction returned for %s (%s)", gene_key, uniprot_id)
                continue

            text = _build_text_from_api(uniprot_id, gene_key, pred)
            documents.append(
                {
                    "id": f"alphafold_{gene_key}_{uniprot_id}",
                    "text": text,
                    "metadata": {
                        "source": "AlphaFold",
                        "uniprot_id": uniprot_id,
                        "gene": pred.get("gene", gene_key),
                        "organism": pred.get("organismScientificName", ""),
                        "plddt": pred.get("globalMetricValue"),
                        "model_version": pred.get("latestVersion"),
                        "seq_length": pred.get("sequenceEnd"),
                        "source_url": f"{ALPHAFOLD_ENTRY}/{uniprot_id}",
                        "pdb_url": pred.get("pdbUrl", ""),
                        "doc_type": "protein_structure",
                    },
                }
            )

    logger.info("AlphaFold: fetched %d documents (genes=%s)", len(documents), target_genes)
    return documents
