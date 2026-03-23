"""
ChromaDB-based persistent vector store for RAG documents.
Three collections: card_resistance, imgt_sequences, alphafold_structures.

ChromaDB is imported lazily so the app can start without it; RAG/Layer 2 will
fail at first use with a clear error if chromadb is not installed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

CHROMA_PATH = Path(__file__).parent.parent / "data" / "chromadb"

COLLECTIONS = {
    "card": "card_resistance",
    "imgt": "imgt_sequences",
    "alphafold": "alphafold_structures",
}


def _import_chromadb() -> Any:
    """Deferred import so backend can start without chromadb installed."""
    try:
        import chromadb
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        return chromadb, DefaultEmbeddingFunction
    except ImportError as e:
        raise ImportError(
            "chromadb is required for RAG (Layer 2). Install with: pip install chromadb"
        ) from e


CHUNK_SIZE = 512       # target characters per chunk
CHUNK_OVERLAP = 64     # overlap between adjacent chunks


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split long text into overlapping chunks for better retrieval granularity."""
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind(". ")
            if last_period > chunk_size // 2:
                end = start + last_period + 2
                chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if c]


class VectorStore:
    """ChromaDB persistent vector store with cosine similarity, chunking, and re-ranking."""

    def __init__(self, persist_path: Optional[Path] = None):
        chromadb, DefaultEmbeddingFunction = _import_chromadb()
        self._chromadb = chromadb
        self.persist_path = persist_path or CHROMA_PATH
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_path))
        self._embedding_fn = DefaultEmbeddingFunction()
        self._collections: dict[str, Any] = {}

    def _get_collection(self, name: str) -> Any:
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    # ChromaDB hard limit per batch call
    _CHROMA_BATCH_LIMIT = 5000

    def add_documents(self, collection_name: str, documents: list[dict]) -> int:
        """
        Add documents to a collection. Skips already-indexed IDs.
        Long documents are split into overlapping chunks for better retrieval.
        documents: list of {"id": str, "text": str, "metadata": dict}
        Returns number of new document chunks added.
        """
        if not documents:
            return 0

        collection = self._get_collection(collection_name)

        # Expand long documents into chunks
        expanded: list[tuple[str, str, dict]] = []
        for doc in documents:
            text = doc["text"]
            meta = doc.get("metadata", {})
            chunks = _chunk_text(text)
            if len(chunks) == 1:
                expanded.append((doc["id"], text, meta))
            else:
                for ci, chunk in enumerate(chunks):
                    chunk_meta = {**meta, "chunk_index": ci, "total_chunks": len(chunks), "parent_id": doc["id"]}
                    expanded.append((f"{doc['id']}_chunk{ci}", chunk, chunk_meta))

        # Dedup check
        all_ids = [e[0] for e in expanded]
        existing: set[str] = set()
        for i in range(0, len(all_ids), self._CHROMA_BATCH_LIMIT):
            batch_ids = all_ids[i : i + self._CHROMA_BATCH_LIMIT]
            existing.update(collection.get(ids=batch_ids)["ids"])

        new = [e for e in expanded if e[0] not in existing]
        if not new:
            return 0

        # Insert in batches
        added = 0
        for i in range(0, len(new), self._CHROMA_BATCH_LIMIT):
            batch = new[i : i + self._CHROMA_BATCH_LIMIT]
            batch_ids, batch_texts, batch_metas = zip(*batch)
            collection.add(
                ids=list(batch_ids),
                documents=list(batch_texts),
                metadatas=list(batch_metas),
            )
            added += len(batch)

        return added

    def query(
        self,
        collection_name: str,
        query_texts: list[str],
        n_results: int = 5,
        keyword_boost: float = 0.15,
    ) -> list[dict]:
        """
        Hybrid search: semantic (embedding cosine) + keyword overlap re-ranking.

        1. Semantic search via ChromaDB embeddings (cosine distance)
        2. Keyword overlap score: fraction of query words found in the document
        3. Combined score = semantic_score + keyword_boost * keyword_score
        4. Re-rank by combined score, return top n_results

        Returns list of {"text", "metadata", "distance", "relevance_score"}.
        """
        collection = self._get_collection(collection_name)
        count = collection.count()
        if count == 0:
            return []

        # Fetch more candidates for re-ranking (2x requested)
        n_candidates = min(n_results * 2, count)
        results = collection.query(query_texts=query_texts, n_results=n_candidates)

        # Merge results from all queries; keep lowest distance per unique text
        seen: dict[str, dict] = {}
        for qi in range(len(query_texts)):
            docs = results["documents"][qi]
            metas = results["metadatas"][qi]
            dists = results["distances"][qi]
            for doc, meta, dist in zip(docs, metas, dists):
                if doc not in seen or dist < seen[doc]["distance"]:
                    seen[doc] = {"text": doc, "metadata": meta, "distance": dist}

        # Keyword overlap re-ranking
        query_words = set()
        for qt in query_texts:
            query_words.update(w.lower() for w in qt.split() if len(w) > 2)

        ranked: list[dict] = []
        for r in seen.values():
            semantic_score = max(0.0, 1.0 - r["distance"])

            # Keyword overlap: fraction of query words found in document
            doc_words = set(w.lower() for w in r["text"].split() if len(w) > 2)
            keyword_overlap = len(query_words & doc_words) / max(len(query_words), 1)

            combined = semantic_score + keyword_boost * keyword_overlap
            ranked.append({
                "text": r["text"],
                "metadata": r["metadata"],
                "distance": r["distance"],
                "relevance_score": round(combined, 4),
                "semantic_score": round(semantic_score, 4),
                "keyword_score": round(keyword_overlap, 4),
            })

        ranked.sort(key=lambda x: x["relevance_score"], reverse=True)
        return ranked[:n_results]

    def collection_count(self, collection_name: str) -> int:
        try:
            return self._get_collection(collection_name).count()
        except Exception:
            return 0

    def clear_collection(self, collection_name: str) -> None:
        try:
            self._client.delete_collection(collection_name)
            self._collections.pop(collection_name, None)
        except Exception:
            pass
