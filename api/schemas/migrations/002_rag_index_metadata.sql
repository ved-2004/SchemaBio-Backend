-- 002_rag_index_metadata.sql
-- Tracks when each RAG collection (CARD, AlphaFold, IMGT) was last indexed.
-- Used to skip re-fetching on cold starts when data is still fresh.

CREATE TABLE IF NOT EXISTS rag_index_metadata (
    collection_name  text PRIMARY KEY,
    last_indexed_at  timestamptz NOT NULL DEFAULT now(),
    document_count   int NOT NULL DEFAULT 0
);
