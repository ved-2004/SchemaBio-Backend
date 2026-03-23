"""
IMGT Fetcher — International Immunogenetics Information System
https://www.imgt.org/

Fetches live IG/TR gene data from IMGT/GENE-DB.
No hardcoded content — all documents come from live API responses.

IMGT/GENE-DB query endpoint:
  GET https://www.imgt.org/genedb/GENElect?query={type}+{gene}&species={species}

Query types used:
  7.2  — Gene segment by gene name (returns HTML with gene data)
  1.1  — All gene segments for a species/locus

SSL note: IMGT's certificate is not in the Windows default trust store.
          verify=False is required on Windows.
"""
from __future__ import annotations

import logging
import re

import httpx

logger = logging.getLogger(__name__)

IMGT_BASE = "https://www.imgt.org"
IMGT_GENE_DB = f"{IMGT_BASE}/genedb/GENElect"

# Therapeutically relevant human IG gene segments to query.
# These are routinely used in antibody engineering and are present in IMGT/GENE-DB.
# Queried regardless of target gene — provides antibody engineering context for
# biologics development, target validation assays, and PK reagent design.
CORE_IG_QUERIES: list[tuple[str, str]] = [
    # (gene_name, species)
    ("IGHV3-23",  "Homo sapiens"),   # Most common VH in therapeutic antibodies
    ("IGHV1-69",  "Homo sapiens"),   # Common anti-bacterial antibodies
    ("IGHV3-30",  "Homo sapiens"),   # Broad germline template
    ("IGHJ6",     "Homo sapiens"),   # Longest CDR3-contributing J segment
    ("IGKV1-39",  "Homo sapiens"),   # Most common kappa light chain
    ("IGLV2-14",  "Homo sapiens"),   # Common lambda light chain
    ("IGHV3-23",  "Mus musculus"),   # Mouse VH3-23 — murine mAb engineering
    ("TRAV12-2",  "Homo sapiens"),   # TCR alpha variable — bacterial antigen response
    ("TRBV20-1",  "Homo sapiens"),   # TCR beta variable — infection immune monitoring
]


def _strip_html(html: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    clean = re.sub(r"<[^>]+>", " ", html)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


async def _query_gene_db(gene_name: str, species: str) -> str | None:
    """
    Query IMGT/GENE-DB for a specific gene + species combination.
    Returns stripped plain-text content or None if unavailable / empty.
    verify=False required for Windows SSL compatibility with IMGT.
    """
    params = {"query": f"7.2+{gene_name}", "species": species}
    try:
        async with httpx.AsyncClient(verify=False, timeout=14.0) as client:
            resp = await client.get(IMGT_GENE_DB, params=params, follow_redirects=True)
        if resp.status_code == 200:
            text = _strip_html(resp.text)
            # IMGT returns ~200 chars even for "no results" pages; require meaningful content
            if len(text) > 400:
                return text[:1200]
    except Exception as exc:
        logger.warning("IMGT GENE-DB failed for %s (%s): %s", gene_name, species, exc)
    return None


async def fetch_imgt_documents(target_genes: list[str] | None = None) -> list[dict]:
    """
    Fetch live IMGT/GENE-DB entries for therapeutically relevant IG/TR gene segments
    and any target-gene-specific entries from the program state.

    All documents come exclusively from live IMGT API responses.
    Returns list of {"id", "text", "metadata"} ready for VectorStore.add_documents().
    """
    documents: list[dict] = []
    seen_ids: set[str] = set()

    # ── 1. Core IG/TR gene queries (always run) ───────────────────────────────
    for gene_name, species in CORE_IG_QUERIES:
        doc_id = f"imgt_{gene_name.lower().replace('-', '_')}_{species.split()[0].lower()}"
        if doc_id in seen_ids:
            continue
        raw = await _query_gene_db(gene_name, species)
        if raw:
            seen_ids.add(doc_id)
            documents.append(
                {
                    "id": doc_id,
                    "text": f"IMGT/GENE-DB {gene_name} ({species}): {raw} Source: IMGT ({IMGT_BASE}).",
                    "metadata": {
                        "source": "IMGT",
                        "gene": gene_name,
                        "species": species,
                        "source_url": IMGT_GENE_DB,
                        "doc_type": "gene_db_entry",
                    },
                }
            )
        else:
            logger.debug("IMGT: no content for %s / %s", gene_name, species)

    # ── 2. Target-gene-specific queries ──────────────────────────────────────
    if target_genes:
        for gene in target_genes:
            for species in ("Homo sapiens", "Mus musculus"):
                doc_id = f"imgt_target_{gene.lower()}_{species.split()[0].lower()}"
                if doc_id in seen_ids:
                    continue
                raw = await _query_gene_db(gene, species)
                if raw:
                    seen_ids.add(doc_id)
                    documents.append(
                        {
                            "id": doc_id,
                            "text": f"IMGT/GENE-DB {gene} ({species}): {raw} Source: IMGT ({IMGT_BASE}).",
                            "metadata": {
                                "source": "IMGT",
                                "gene": gene,
                                "species": species,
                                "source_url": IMGT_GENE_DB,
                                "doc_type": "target_gene_db_entry",
                            },
                        }
                    )

    logger.info("IMGT: fetched %d live documents (target_genes=%s)", len(documents), target_genes)
    return documents
