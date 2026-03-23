"""
FDA Agent — openFDA API Integration
=========================================================
Queries FDA's public openFDA API (no auth required) to retrieve:
  1. Approved drugs for the target organism/indication (drug labels)
  2. Drug approval history + exclusivity data (Drugs@FDA)
  3. QIDP-designated drugs list (GAIN Act — static reference, no API exists)

API base: https://api.fda.gov/
Docs:     https://open.fda.gov/apis/

All calls are async with graceful fallback — never blocks the pipeline.
"""

import httpx
import re
from typing import Optional

OPENFDA_BASE = "https://api.fda.gov"

# ─────────────────────────────────────────────
# QIDP (Qualified Infectious Disease Product) Designated Drugs
# Source: FDA GAIN Act QIDP Database (no API — maintained as static reference)
# https://www.fda.gov/industry/prescription-drug-user-fee-amendments/qidp-drug-development-database
# Last updated: 2024
# ─────────────────────────────────────────────

QIDP_DESIGNATED_DRUGS = {
    "ceftazidime-avibactam":     {"brand": "Avycaz",     "nda": "206494", "approved": 2015, "pathogens": ["klebsiella pneumoniae", "pseudomonas aeruginosa", "escherichia coli"]},
    "ceftolozane-tazobactam":    {"brand": "Zerbaxa",    "nda": "206829", "approved": 2014, "pathogens": ["pseudomonas aeruginosa", "escherichia coli", "klebsiella pneumoniae"]},
    "meropenem-vaborbactam":     {"brand": "Vabomere",   "nda": "209816", "approved": 2017, "pathogens": ["klebsiella pneumoniae", "escherichia coli", "enterobacter"]},
    "imipenem-cilastatin-relebactam": {"brand": "Recarbrio", "nda": "212819", "approved": 2019, "pathogens": ["pseudomonas aeruginosa", "klebsiella pneumoniae"]},
    "cefiderocol":               {"brand": "Fetroja",    "nda": "209445", "approved": 2019, "pathogens": ["klebsiella pneumoniae", "acinetobacter baumannii", "pseudomonas aeruginosa", "escherichia coli"]},
    "aztreonam-avibactam":       {"brand": "Emblaveo",   "nda": "218226", "approved": 2024, "pathogens": ["klebsiella pneumoniae", "escherichia coli", "enterobacter", "pseudomonas aeruginosa"]},
    "sulbactam-durlobactam":     {"brand": "Xacduro",    "nda": "216862", "approved": 2023, "pathogens": ["acinetobacter baumannii"]},
    "omadacycline":              {"brand": "Nuzyra",     "nda": "209816", "approved": 2018, "pathogens": ["staphylococcus aureus"]},
    "lefamulin":                 {"brand": "Xenleta",    "nda": "211672", "approved": 2019, "pathogens": ["staphylococcus aureus", "streptococcus pneumoniae"]},
    "dalbavancin":               {"brand": "Dalvance",   "nda": "021883", "approved": 2014, "pathogens": ["staphylococcus aureus", "enterococcus faecalis"]},
    "oritavancin":               {"brand": "Orbactiv",   "nda": "206334", "approved": 2014, "pathogens": ["staphylococcus aureus", "enterococcus faecium"]},
    "tedizolid":                 {"brand": "Sivextro",   "nda": "205435", "approved": 2014, "pathogens": ["staphylococcus aureus"]},
    "delafloxacin":              {"brand": "Baxdela",    "nda": "208610", "approved": 2017, "pathogens": ["staphylococcus aureus", "escherichia coli"]},
    "eravacycline":              {"brand": "Xerava",     "nda": "210234", "approved": 2018, "pathogens": ["klebsiella pneumoniae", "escherichia coli", "acinetobacter baumannii"]},
    "gepotidacin":               {"brand": "Blujepa",    "nda": "218670", "approved": 2024, "pathogens": ["escherichia coli"]},
}

# Organism -> exclusivity expiry estimates (QIDP gives 5-yr extension on top of standard exclusivity)
# These are approximate — actual expiry depends on NDA filing date
EXCLUSIVITY_MAP = {
    "ceftazidime-avibactam":  {"standard_expiry": "2031", "qidp_expiry": "2036", "notes": "Core avibactam patents expire ~2033"},
    "cefiderocol":            {"standard_expiry": "2031", "qidp_expiry": "2036", "notes": "Shionogi composition patents extend to 2033+"},
    "aztreonam-avibactam":    {"standard_expiry": "2038", "qidp_expiry": "2043", "notes": "Most recent approval — longest runway"},
    "sulbactam-durlobactam":  {"standard_expiry": "2038", "qidp_expiry": "2043", "notes": "OXA-targeted — limited to A. baumannii"},
}


# ─────────────────────────────────────────────
# openFDA: Approved drug labels for organism/indication
# ─────────────────────────────────────────────

async def fetch_approved_drugs_for_organism(organism: str) -> list[dict]:
    """
    Query openFDA drug label database for approved antibiotics
    with the target organism in their indications_and_usage section.

    Returns list of approved drugs relevant to the organism.
    Falls back to empty list on any failure.
    """
    if not organism:
        return []

    org_lower = organism.lower()
    search_terms = _build_fda_search_terms(org_lower)

    results = []
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            for term in search_terms[:2]:  # limit to 2 queries max
                r = await client.get(
                    f"{OPENFDA_BASE}/drug/label.json",
                    params={
                        "search": f'indications_and_usage:"{term}" AND openfda.pharm_class_cs:"Antibacterial"',
                        "limit": 8,
                        "fields": "openfda.brand_name,openfda.generic_name,openfda.application_number,indications_and_usage",
                    }
                )
                if r.status_code == 200:
                    data = r.json()
                    for hit in data.get("results", []):
                        openfda = hit.get("openfda", {})
                        brand = openfda.get("brand_name", ["Unknown"])[0]
                        generic = openfda.get("generic_name", ["Unknown"])[0]
                        app_num = openfda.get("application_number", [""])[0]
                        indication_raw = hit.get("indications_and_usage", [""])[0]
                        indication = indication_raw[:300].strip() + "..." if len(indication_raw) > 300 else indication_raw

                        results.append({
                            "brand_name": brand,
                            "generic_name": generic.lower(),
                            "application_number": app_num,
                            "indication_snippet": indication,
                            "source": "openFDA drug label",
                        })

    except Exception:
        pass  # Graceful fallback

    # Deduplicate by generic name
    seen = set()
    unique = []
    for r in results:
        key = r["generic_name"]
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique[:6]  # cap at 6 results


def _build_fda_search_terms(org_lower: str) -> list[str]:
    """Map organism name to terms used in FDA label indications."""
    term_map = {
        "klebsiella": ["Klebsiella pneumoniae", "Klebsiella"],
        "acinetobacter": ["Acinetobacter baumannii", "Acinetobacter"],
        "pseudomonas": ["Pseudomonas aeruginosa", "Pseudomonas"],
        "escherichia": ["Escherichia coli", "E. coli"],
        "staphylococcus": ["Staphylococcus aureus", "MRSA"],
        "enterococcus": ["Enterococcus faecium", "VRE", "vancomycin-resistant Enterococcus"],
        "enterobacter": ["Enterobacter", "Enterobacteriaceae"],
        "mycobacterium": ["Mycobacterium tuberculosis", "tuberculosis"],
        "streptococcus": ["Streptococcus pneumoniae", "Streptococcus"],
        "neisseria": ["Neisseria gonorrhoeae", "gonorrhea"],
    }
    for key, terms in term_map.items():
        if key in org_lower:
            return terms
    return [org_lower]


# ─────────────────────────────────────────────
# openFDA: Drug approval history (Drugs@FDA)
# ─────────────────────────────────────────────

async def fetch_drug_approval_history(compound_name: str) -> Optional[dict]:
    """
    Query Drugs@FDA for approval history of a specific compound.
    Returns NDA/BLA number, approval date, applicant, and active ingredients.
    """
    if not compound_name:
        return None
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(
                f"{OPENFDA_BASE}/drug/drugsfda.json",
                params={
                    "search": f'openfda.generic_name:"{compound_name}"',
                    "limit": 1,
                }
            )
            if r.status_code == 200:
                results = r.json().get("results", [])
                if results:
                    hit = results[0]
                    products = hit.get("products", [{}])[0]
                    submissions = hit.get("submissions", [])
                    approval_date = None
                    for sub in submissions:
                        if sub.get("submission_type") == "ORIG" and sub.get("submission_status") == "AP":
                            approval_date = sub.get("submission_status_date", "Unknown")
                            break
                    return {
                        "application_number": hit.get("application_number"),
                        "applicant": hit.get("applicant"),
                        "brand_name": products.get("brand_name"),
                        "dosage_form": products.get("dosage_form"),
                        "approval_date": approval_date,
                        "source": "Drugs@FDA",
                    }
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────
# QIDP Lookup (static reference — no FDA API exists for this)
# ─────────────────────────────────────────────

def get_qidp_drugs_for_organism(organism: str) -> list[dict]:
    """
    Return list of QIDP-designated approved drugs for the target organism.
    Data sourced from FDA GAIN Act QIDP database (static, updated 2024).
    """
    org_lower = organism.lower()
    matches = []

    for drug_name, info in QIDP_DESIGNATED_DRUGS.items():
        if any(p in org_lower or org_lower in p for p in info["pathogens"]):
            entry = {
                "drug": drug_name,
                "brand": info["brand"],
                "nda": info["nda"],
                "fda_approved": info["approved"],
                "qidp_designated": True,
                "exclusivity": EXCLUSIVITY_MAP.get(drug_name, {}).get("qidp_expiry", "Est. 2029-2033"),
                "exclusivity_notes": EXCLUSIVITY_MAP.get(drug_name, {}).get("notes", ""),
            }
            matches.append(entry)

    return matches


def get_market_exclusivity_gap(organism: str) -> dict:
    """
    Assess the patent/exclusivity cliff for the target organism's treatment space.
    Returns drugs with expiring exclusivity (near-term genericization opportunity or cliff risk).
    """
    org_lower = organism.lower()
    expiring_soon = []
    protected = []

    for drug_name, info in QIDP_DESIGNATED_DRUGS.items():
        if any(p in org_lower or org_lower in p for p in info["pathogens"]):
            excl = EXCLUSIVITY_MAP.get(drug_name, {})
            expiry_str = excl.get("qidp_expiry", "2033")
            try:
                expiry_year = int(expiry_str)
                if expiry_year <= 2030:
                    expiring_soon.append({"drug": drug_name, "brand": info["brand"], "qidp_expiry": expiry_str})
                else:
                    protected.append({"drug": drug_name, "brand": info["brand"], "qidp_expiry": expiry_str})
            except ValueError:
                protected.append({"drug": drug_name, "brand": info["brand"], "qidp_expiry": expiry_str})

    return {
        "expiring_by_2030": expiring_soon,
        "still_protected": protected,
        "opportunity_note": (
            "Exclusivity cliffs for some treatments create genericization risk — new entrants should target differentiated mechanisms"
            if expiring_soon else
            "Most approved treatments have >5 years of remaining QIDP exclusivity — competitive window narrow unless differentiated"
        ),
    }


# ─────────────────────────────────────────────
# Main entry point for Layer 3
# ─────────────────────────────────────────────

async def enrich_fda_intelligence(
    organism: str,
    compound_names: list[str],
) -> dict:
    """
    Main FDA intelligence function called by Layer 3.
    Combines:
      - Live openFDA approved drug label lookup
      - QIDP-designated competitor list (static, no API)
      - Exclusivity gap analysis
      - Drug approval history for user's lead compounds

    Returns structured dict for ProgramState.fda_intelligence.
    """
    import asyncio

    approved_labels_task = fetch_approved_drugs_for_organism(organism)

    compound_tasks = [
        fetch_drug_approval_history(name)
        for name in (compound_names or [])[:3]  # max 3 compounds
    ]

    approved_labels, *compound_histories = await asyncio.gather(
        approved_labels_task,
        *compound_tasks,
    )

    # Static lookups (instant, no network)
    qidp_competitors = get_qidp_drugs_for_organism(organism)
    exclusivity_gap = get_market_exclusivity_gap(organism)

    return {
        "approved_drugs_on_label": approved_labels,        # live from openFDA
        "qidp_competitors": qidp_competitors,              # QIDP-designated drugs for organism
        "exclusivity_gap": exclusivity_gap,                # patent cliff analysis
        "compound_approval_history": [h for h in compound_histories if h],  # user's compounds in FDA
        "source_note": "openFDA API (live) + FDA QIDP Database (static ref, updated 2024)",
    }
