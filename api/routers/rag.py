"""
FastAPI router — RAG endpoints

POST /api/rag/query   — Query CARD/AlphaFold/IMGT for a program state (Layers 2 & 3)
POST /api/rag/index   — Trigger (re-)indexing for a program state
GET  /api/rag/status  — Collection document counts
DELETE /api/rag/index — Clear all collections

Consumed by:
  • experiment_design layer (Layer 2) to enrich experiment rationale
  • execution_planning layer (Layer 3) to ground translational context
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.rag.service import (
    ensure_indexed_and_query,
    index_for_program_state,
    query_rag,
    get_vector_store,
)
from api.rag.vector_store import COLLECTIONS

router = APIRouter(prefix="/api/rag", tags=["rag"])


# ── Request / response models ─────────────────────────────────────────────────

class RAGQueryRequest(BaseModel):
    """
    Input to /api/rag/query.
    program_state must match the IngestionResponse.program_state schema
    produced by Layer 1 (ingestion).
    """
    program_state: dict[str, Any]
    top_k: int = Field(default=5, ge=1, le=20, description="Max docs to return per source.")
    auto_index: bool = Field(
        default=True,
        description="Auto-populate vector store if empty. Set False to query only.",
    )


class RAGIndexRequest(BaseModel):
    """Input to POST /api/rag/index."""
    program_state: dict[str, Any]
    force_refresh: bool = Field(
        default=False,
        description="If True, clears all collections before re-indexing.",
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/query", summary="Query RAG for experiment design / execution planning context")
async def rag_query(request: RAGQueryRequest) -> dict[str, Any]:
    """
    Query the RAG vector store with a Layer 1 program_state.

    Returns a RAGContextBundle with relevant documents from:
      • CARD  — resistance genes, mechanisms, MIC breakpoints
      • AlphaFold — protein structure metadata, pLDDT confidence, structural notes
      • IMGT  — immunogenetics, antibody engineering, immunogenicity context

    This endpoint is the primary hook for Layers 2 and 3 to retrieve grounding
    evidence before LLM-driven experiment design or translational planning.
    """
    try:
        if request.auto_index:
            result = await ensure_indexed_and_query(
                request.program_state, top_k=request.top_k
            )
        else:
            result = await query_rag(request.program_state, top_k=request.top_k)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {exc}") from exc


@router.post("/index", summary="Index CARD / AlphaFold / IMGT for a program state")
async def rag_index(request: RAGIndexRequest) -> dict[str, Any]:
    """
    Fetch and embed documents from CARD, AlphaFold, and IMGT for the given
    program_state. Returns document counts per source.

    Call this explicitly to pre-populate the vector store before running Layers 2/3,
    or to refresh stale data (force_refresh=true).
    """
    try:
        counts = await index_for_program_state(
            request.program_state, force_refresh=request.force_refresh
        )
        store = get_vector_store()
        return {
            "status": "ok",
            "documents_indexed": counts,
            "collection_totals": {
                "CARD": store.collection_count(COLLECTIONS["card"]),
                "AlphaFold": store.collection_count(COLLECTIONS["alphafold"]),
                "IMGT": store.collection_count(COLLECTIONS["imgt"]),
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc


@router.get("/status", summary="RAG index health and document counts")
async def rag_status() -> dict[str, Any]:
    """Returns current document counts for each RAG collection."""
    store = get_vector_store()
    counts = {
        "CARD": store.collection_count(COLLECTIONS["card"]),
        "AlphaFold": store.collection_count(COLLECTIONS["alphafold"]),
        "IMGT": store.collection_count(COLLECTIONS["imgt"]),
    }
    total = sum(counts.values())
    return {
        "status": "ok",
        "total_documents": total,
        "collections": counts,
        "sources": {
            "CARD": "https://card.mcmaster.ca",
            "AlphaFold": "https://alphafold.ebi.ac.uk",
            "IMGT": "https://www.imgt.org",
        },
    }


@router.delete("/index", summary="Clear all RAG collections")
async def rag_clear() -> dict[str, Any]:
    """Wipes all three vector store collections. Re-index with POST /api/rag/index."""
    store = get_vector_store()
    for coll in COLLECTIONS.values():
        store.clear_collection(coll)
    return {"status": "ok", "message": "All RAG collections cleared."}
