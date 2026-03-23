"""
SchemaBio RAG Layer
Retrieval-augmented generation over CARD, AlphaFold, and IMGT databases.
"""
from .service import ensure_indexed_and_query, index_for_program_state, query_rag

__all__ = ["ensure_indexed_and_query", "index_for_program_state", "query_rag"]
