"""
agents/literature_agent.py

Literature retrieval and quantitative claim extraction.

Queries PubMed E-utilities for relevant papers based on:
  - Gene name (e.g. "gyrA")
  - Resistance mechanism (e.g. "fluoroquinolone resistance")
  - Compound name (e.g. "Compound-14")
  - Organism (e.g. "Escherichia coli")

Extracts quantitative claims from abstracts via Claude.
Falls back to pre-cached demo literature on API failure.
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
from typing import Optional

import anthropic
import httpx

from api.legacy.models.drug_program import DrugProgram, LiteratureResult

logger = logging.getLogger(__name__)

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEMO_CACHE_PATH = os.path.join(os.path.dirname(__file__), "../../data/demo/cached_literature.json")

_CLAIM_EXTRACT_PROMPT = """Extract all quantitative claims from this abstract.
Return ONLY valid JSON array, no markdown:
[
  {"type": "ic50|mic|mbc|fold_change|frequency", "value": number, "unit": "string", "target": "string", "context": "≤15 word quote"}
]
If no quantitative claims, return [].
"""


async def _pubmed_search(query: str, max_results: int = 5) -> list[str]:
    """Search PubMed and return list of PMIDs."""
    api_key = os.environ.get("PUBMED_API_KEY", "")
    params = {
        "db": "pubmed", "term": query, "retmax": max_results,
        "retmode": "json", "sort": "relevance",
    }
    if api_key:
        params["api_key"] = api_key

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(f"{PUBMED_BASE}/esearch.fcgi", params=params)
            data = resp.json()
            return data.get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        logger.warning(f"PubMed search failed: {e}")
        return []


async def _pubmed_fetch(pmids: list[str]) -> list[dict]:
    """Fetch PubMed records for a list of PMIDs."""
    if not pmids:
        return []
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{PUBMED_BASE}/efetch.fcgi",
                params={"db":"pubmed","id":",".join(pmids),"retmode":"xml","rettype":"abstract"},
            )
            return _parse_pubmed_xml(resp.text, pmids)
    except Exception as e:
        logger.warning(f"PubMed fetch failed: {e}")
        return []


def _parse_pubmed_xml(xml: str, pmids: list[str]) -> list[dict]:
    """Simple XML parser for PubMed records (no lxml dependency)."""
    import re
    results = []

    def extract(text, tag):
        m = re.search(f"<{tag}[^>]*>(.*?)</{tag}>", text, re.DOTALL)
        return m.group(1).strip() if m else ""

    # Split by PubmedArticle
    articles = re.split(r"<PubmedArticle>", xml)[1:]
    for i, article in enumerate(articles):
        pmid = pmids[i] if i < len(pmids) else ""
        title    = re.sub(r"<[^>]+>", "", extract(article, "ArticleTitle"))
        abstract = re.sub(r"<[^>]+>", "", extract(article, "AbstractText"))
        journal  = re.sub(r"<[^>]+>", "", extract(article, "Title"))
        year_m   = re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", article, re.DOTALL)
        year     = int(year_m.group(1)) if year_m else 0

        # Authors: last name of first author
        author_m = re.search(r"<Author[^>]*>.*?<LastName>(.*?)</LastName>.*?<Initials>(.*?)</Initials>",
                              article, re.DOTALL)
        authors = f"{author_m.group(1)} {author_m.group(2)} et al." if author_m else "Author et al."

        if title:
            results.append({
                "pmid": pmid, "title": title[:200], "abstract": abstract[:1500],
                "journal": journal[:80], "year": year, "authors": authors,
            })
    return results


async def _extract_claims_llm(abstract: str) -> list[dict]:
    """Extract quantitative claims from abstract via Claude."""
    if not abstract.strip():
        return []
    try:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY",""))
        msg = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=300,
            messages=[{"role":"user","content": f"{_CLAIM_EXTRACT_PROMPT}\n\n{abstract[:800]}"}],
        )
        raw = msg.content[0].text.strip().replace("```json","").replace("```","").strip()
        return json.loads(raw) if raw.startswith("[") else []
    except Exception as e:
        logger.warning(f"Claim extraction failed: {e}")
        return []


def _load_demo_cache() -> list[dict]:
    """Load pre-cached demo literature."""
    try:
        with open(DEMO_CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return _HARDCODED_DEMO_LIT


_HARDCODED_DEMO_LIT = [
    {
        "pmid": "37104821",
        "title": "Quinolone scaffold activity against E.coli GyrA D87N mutants",
        "authors": "Chen X et al.", "journal": "J Med Chem", "year": 2023,
        "abstract": "Classic quinolone-based compounds showed IC50 > 890 nM against GyrA D87N in cell-free enzyme assay. The D87N mutation disrupts the Mg2+-water bridge critical for quinolone binding. Non-chelating scaffolds may retain activity.",
        "relevance_score": 0.95, "triggered_by": "GyrA",
        "quantitative_claims": [{"type":"ic50","value":890,"unit":"nM","target":"GyrA D87N"}],
    },
    {
        "pmid": "36892011",
        "title": "Structural basis of fluoroquinolone resistance in E. coli gyrase",
        "authors": "Blanco D et al.", "journal": "Nature Commun", "year": 2022,
        "abstract": "Crystal structure of GyrA D87N shows altered active site geometry. Compounds binding via Mg2+ chelation show 600–1200× reduced activity. Non-chelating scaffolds may retain full activity against QRDR mutants.",
        "relevance_score": 0.91, "triggered_by": "gyrA D87N",
        "quantitative_claims": [{"type":"fold_change","value":900,"unit":"fold","target":"GyrA D87N"}],
    },
    {
        "pmid": "32217743",
        "title": "Frequency of fluoroquinolone resistance in clinical E. coli isolates",
        "authors": "Piddock LJ et al.", "journal": "AAC", "year": 2020,
        "abstract": "gyrA D87N detected in 34% of fluoroquinolone-resistant E. coli clinical isolates in EU. AcrAB-TolC efflux co-selection observed in 67% of resistant isolates. Dual mechanism resistance requires dual targeting strategy.",
        "relevance_score": 0.82, "triggered_by": "resistance_mutations",
        "quantitative_claims": [],
    },
    {
        "pmid": "30110579",
        "title": "Novel non-chelating gyrase inhibitors bypass resistance mutations",
        "authors": "Mayer C et al.", "journal": "JACS", "year": 2023,
        "abstract": "Compounds targeting GyrB rather than GyrA QRDR retain activity against D87N/S83L mutants. Novel binding mode confirmed by SPR (Kd 2.3 nM). GyrB inhibitors show 4–32× lower MIC against resistant isolates vs. classic fluoroquinolones.",
        "relevance_score": 0.88, "triggered_by": "bypass resistance",
        "quantitative_claims": [{"type":"mic","value":2.3,"unit":"nM","target":"GyrB inhibitor"}],
    },
]


async def retrieve_literature(program: DrugProgram) -> DrugProgram:
    """
    Retrieve and process literature for the drug program.
    Updates program.literature in place.
    """
    gene    = program.target.gene or ""
    compound = program.compound.name or ""
    organism = program.target.organism or "E. coli"
    mutations = " ".join(program.resistance.resistance_mutations[:2])

    queries = []
    if gene:
        queries.append(f"{gene} antibiotic resistance {organism}")
    if mutations:
        queries.append(f"{mutations} fluoroquinolone resistance mechanism")
    if compound and compound not in ("Unknown", ""):
        queries.append(f"{compound} antimicrobial activity {gene}")

    # Try live PubMed
    papers_raw = []
    if os.environ.get("ANTHROPIC_API_KEY") and queries:
        for query in queries[:2]:
            pmids = await _pubmed_search(query, max_results=3)
            fetched = await _pubmed_fetch(pmids)
            papers_raw.extend(fetched)

    # Fall back to demo cache
    if not papers_raw:
        logger.info("Using demo literature cache")
        papers_raw = _load_demo_cache()

    # Convert to LiteratureResult + extract claims
    results = []
    for p in papers_raw[:6]:
        claims = p.get("quantitative_claims", [])
        if not claims and p.get("abstract"):
            claims = await _extract_claims_llm(p["abstract"])

        relevance = p.get("relevance_score", 0.8)
        # Boost relevance if gene or mutation in title/abstract
        title_abs = (p.get("title","") + " " + p.get("abstract","")).lower()
        if gene.lower() in title_abs:   relevance = min(1.0, relevance + 0.05)
        if mutations.lower() in title_abs: relevance = min(1.0, relevance + 0.08)

        results.append(LiteratureResult(
            pmid=str(p.get("pmid","")),
            title=p.get("title","")[:200],
            authors=p.get("authors",""),
            journal=p.get("journal",""),
            year=p.get("year",0),
            abstract=p.get("abstract","")[:1500],
            relevance_score=round(relevance, 3),
            triggered_by=p.get("triggered_by", gene or "search"),
            quantitative_claims=claims,
        ))

    # Sort by relevance
    results.sort(key=lambda r: r.relevance_score, reverse=True)
    program.literature = results[:6]

    program.add_trace(
        len(program.agent_trace)+1, "LiteratureAgent",
        "Literature retrieval",
        f"{len(results)} papers, {sum(len(r.quantitative_claims) for r in results)} quantitative claims",
        "lit",
    )
    return program
