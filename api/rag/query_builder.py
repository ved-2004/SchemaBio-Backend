"""
Query Builder — converts Layer 1 ingestion output (program_state) into
semantic query strings for the RAG vector store.
"""
from __future__ import annotations

from typing import Any


def build_queries(program_state: dict[str, Any]) -> list[str]:
    """
    Derive up to 12 ranked query strings from the ingestion program_state.
    Queries are ordered from most-specific (entity+variant) to most-general (stage).
    """
    queries: list[str] = []
    entities: list[dict] = program_state.get("entities", [])
    signals: list[dict] = program_state.get("signals", [])
    stage: str = program_state.get("stage_estimate", {}).get("name", "")

    # ── Entity-based queries ──────────────────────────────────────────────────
    for ent in entities:
        etype = ent.get("type", "")
        value = ent.get("value", "")
        if not value:
            continue

        if etype == "variant":
            gene = value.split()[0] if " " in value else value
            mut = value.split()[1] if " " in value else ""
            queries.append(
                f"{value} resistance mutation {gene} QRDR fluoroquinolone mechanism"
            )
            if mut:
                queries.append(
                    f"{gene} {mut} antibiotic resistance clinical MIC fold-shift"
                )
        elif etype == "target":
            queries.append(
                f"{value} protein structure inhibition antibiotic resistance mechanism"
            )
            queries.append(
                f"{value} QRDR drug binding site fluoroquinolone gyrase topoisomerase"
            )
        elif etype == "organism":
            queries.append(
                f"antibiotic resistance {value} resistance genes efflux mutation mechanism"
            )
        elif etype == "compound":
            queries.append(
                f"gyrase inhibitor {value} mechanism IC50 resistance overcome lead optimisation"
            )
        elif etype == "drug_class":
            queries.append(
                f"{value} antibiotic resistance mechanisms target mutations efflux pump"
            )
        elif etype == "assay_type":
            queries.append(
                f"minimum inhibitory concentration {value} resistance breakpoints CLSI EUCAST interpretation"
            )

    # ── Signal-based queries ──────────────────────────────────────────────────
    for sig in signals:
        kind = sig.get("kind", "")
        value = sig.get("value")

        if kind == "resistance_fold_shift" and value is not None:
            queries.append(
                f"high-level antibiotic resistance {value}x fold shift clinical significance MIC"
            )
        elif kind == "resistance_associated_variant":
            queries.append(
                f"{value} resistance variant molecular mechanism fluoroquinolone QRDR"
            )
        elif kind == "compound_hit":
            queries.append(
                f"gyrase inhibitor compound lead optimisation overcoming resistance GyrA D87N"
            )
        elif kind == "lead_ic50_nm":
            queries.append(
                f"gyrase inhibitor potency selectivity IC50 {value} nM target engagement"
            )
        elif kind == "mechanism_hint":
            queries.append(f"{value} antibacterial resistance mechanism efflux target mutation")

    # ── Stage-based contextual queries ───────────────────────────────────────
    stage_queries: dict[str, list[str]] = {
        "resistance_mechanism_characterization": [
            "efflux vs target mutation fluoroquinolone resistance E. coli discrimination assay",
            "DNA gyrase GyrA QRDR mutation enzyme inhibition wild-type mutant comparison",
            "efflux pump inhibitor carbonyl cyanide CCCP PA-βN ciprofloxacin MIC",
            "time-kill kinetics bactericidal bacteriostatic resistance mechanism antibacterial",
        ],
        "hit_discovery": [
            "antibiotic hit discovery compound screen gyrase topoisomerase inhibitor scaffold",
            "novel antibacterial mechanism of action gyrase inhibitor fluoroquinolone class",
        ],
        "experimental_validation_planning": [
            "antibacterial compound validation ADMET pharmacokinetics in vitro in vivo",
            "resistance mechanism validation enzyme inhibition selectivity gyrase ParC",
        ],
        "preclinical_package_gap_analysis": [
            "preclinical antibacterial ADMET toxicity cytotoxicity hERG selectivity",
            "in vivo efficacy murine thigh infection model neutropenic pharmacodynamics",
        ],
        "manufacturing_feasibility_review": [
            "antibacterial compound synthesis manufacturability GMP CDMO scale-up",
            "process chemistry API synthesis steps yield cost-of-goods",
        ],
    }
    queries.extend(stage_queries.get(stage, [
        "antibiotic resistance mechanism drug discovery AMR",
    ]))

    # ── Deduplicate preserving order ──────────────────────────────────────────
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)

    return unique[:12]


def extract_genes(program_state: dict[str, Any]) -> list[str]:
    """Return unique gene names extracted from target/variant entities."""
    genes: list[str] = []
    for ent in program_state.get("entities", []):
        if ent.get("type") in ("target", "variant"):
            gene = ent.get("value", "").split()[0].lower()
            if gene and gene not in genes:
                genes.append(gene)
    return genes


def extract_drug_classes(program_state: dict[str, Any]) -> list[str]:
    """Return unique drug class strings from entities."""
    classes: list[str] = []
    for ent in program_state.get("entities", []):
        if ent.get("type") == "drug_class":
            dc = ent.get("value", "").lower()
            if dc and dc not in classes:
                classes.append(dc)
    return classes
