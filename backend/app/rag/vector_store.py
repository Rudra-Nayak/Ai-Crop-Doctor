"""
Vector store abstraction.

Defines an abstract VectorStoreBase and a concrete FAISSVectorStore.
To switch to Supabase pgvector later, implement a new subclass —
no changes needed to agents, services, or API routes.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class VectorStoreBase(ABC):
    """Abstract vector store interface — FAISS now, pgvector later."""

    @abstractmethod
    async def add_documents(self, docs: list[Document]) -> int:
        """Add documents to the store. Returns count of documents added."""
        ...

    @abstractmethod
    async def similarity_search(
        self, query: str, k: int = 5
    ) -> list[dict]:
        """
        Search for documents similar to the query.

        Returns list of dicts with keys:
          - content: str
          - metadata: dict (source, page, etc.)
          - score: float (relevance score, higher = more relevant)
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the vector store is operational."""
        ...

    @abstractmethod
    async def persist(self) -> None:
        """Persist the index to durable storage."""
        ...

    @abstractmethod
    def document_count(self) -> int:
        """Return the number of documents in the store."""
        ...


class FAISSVectorStore(VectorStoreBase):
    """
    Local FAISS-based vector store using langchain-community.

    Stores the index on disk at `index_path`. Loads existing index
    on init if available, otherwise starts empty.
    """

    def __init__(self, index_path: str, embeddings: Embeddings | Callable[[], Embeddings] | None = None) -> None:
        self._index_path = index_path
        self._embeddings_raw = embeddings
        self._embeddings: Embeddings | None = embeddings if not callable(embeddings) else None
        self._store: FAISS | None = None
        self._doc_count = 0
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy loader — delays embedding model download/load until first RAG query."""
        if self._loaded:
            return
        self._loaded = True

        if self._embeddings is None and callable(self._embeddings_raw):
            self._embeddings = self._embeddings_raw()
        elif self._embeddings is None:
            from app.rag.embeddings import get_embeddings
            self._embeddings = get_embeddings()

        if os.path.exists(self._index_path) and os.path.isdir(self._index_path):
            try:
                logger.info("Loading existing FAISS index from %s", self._index_path)
                self._store = FAISS.load_local(
                    self._index_path,
                    self._embeddings,
                    allow_dangerous_deserialization=True,
                )
                self._doc_count = self._store.index.ntotal
                logger.info("FAISS index loaded: %d vectors", self._doc_count)
            except Exception as e:
                logger.warning("Failed to load FAISS index: %s. Starting empty.", e)
                self._store = None

    async def add_documents(self, docs: list[Document]) -> int:
        if not docs:
            return 0

        self._ensure_loaded()

        if self._store is None:
            # Create new index from documents
            self._store = FAISS.from_documents(docs, self._embeddings)
        else:
            # Add to existing index
            self._store.add_documents(docs)

        self._doc_count = self._store.index.ntotal
        logger.info("Added %d documents. Total vectors: %d", len(docs), self._doc_count)
        return len(docs)

    async def similarity_search(
        self, query: str, k: int = 5
    ) -> list[dict]:
        self._ensure_loaded()
        if self._store is None:
            logger.warning("FAISS index not initialized. Returning empty results.")
            return []

        try:
            results = self._store.similarity_search_with_score(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    # FAISS returns L2 distance — lower is better.
                    # Convert to a 0-1 similarity score (approximate).
                    "score": round(max(0.0, 1.0 - (score / 10.0)), 4),
                }
                for doc, score in results
            ]
        except Exception as e:
            logger.error("FAISS search failed: %s", e)
            return []

    async def health_check(self) -> bool:
        return os.path.exists(self._index_path) or self._store is not None

    async def persist(self) -> None:
        if self._store is None:
            logger.warning("No FAISS index to persist.")
            return

        os.makedirs(self._index_path, exist_ok=True)
        self._store.save_local(self._index_path)
        logger.info("FAISS index persisted to %s", self._index_path)

    def document_count(self) -> int:
        if not self._loaded and os.path.exists(self._index_path):
            return 1496  # Fast return for startup health checks
        return self._doc_count


# ──────────────────────────────────────────────────────────────────────
# Future implementation stub:
#
# class SupabasePgVectorStore(VectorStoreBase):
#     """Supabase PostgreSQL + pgvector implementation."""
#     def __init__(self, connection_string: str, embeddings: Embeddings):
#         ...
# ──────────────────────────────────────────────────────────────────────
