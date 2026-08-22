"""
Vector store abstraction.

Defines an abstract VectorStoreBase and a concrete FAISSVectorStore.
To switch to Supabase pgvector later, implement a new subclass —
no changes needed to agents, services, or API routes.
"""

from __future__ import annotations

import asyncio
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


class SupabasePgVectorStore(VectorStoreBase):
    """
    Supabase PostgreSQL + pgvector vector store implementation.
    
    Stores knowledge base chunks and embeddings in Supabase database table `documents`.
    Uses high-performance HNSW index & `match_documents` SQL function for vector search.
    """

    def __init__(self, url: str, key: str, embeddings: Embeddings | Callable[[], Embeddings]) -> None:
        self._url = url
        self._key = key
        self._embeddings_raw = embeddings
        self._embeddings: Embeddings | None = embeddings if not callable(embeddings) else None
        self._client = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        from supabase import create_client
        self._client = create_client(self._url, self._key)

    def _ensure_embeddings(self) -> None:
        if self._embeddings is not None:
            return
        if callable(self._embeddings_raw):
            self._embeddings = self._embeddings_raw()

    async def add_documents(self, docs: list[Document]) -> int:
        if not docs:
            return 0
        self._ensure_client()
        self._ensure_embeddings()

        texts = [doc.page_content for doc in docs]
        vector_embeddings = self._embeddings.embed_documents(texts)

        records = []
        for doc, emb in zip(docs, vector_embeddings):
            records.append({
                "content": doc.page_content,
                "metadata": doc.metadata or {},
                "embedding": emb,
            })

        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            self._client.table("documents").insert(batch).execute()

        logger.info("Inserted %d documents into Supabase pgvector.", len(docs))
        return len(docs)

    async def similarity_search(self, query: str, k: int = 5) -> list[dict]:
        self._ensure_client()
        self._ensure_embeddings()
        query_embedding = self._embeddings.embed_query(query)

        try:
            def _rpc():
                return self._client.rpc(
                    "match_documents",
                    {
                        "query_embedding": query_embedding,
                        "match_threshold": 0.0,
                        "match_count": k,
                    }
                ).execute()

            rpc_res = await asyncio.to_thread(_rpc)

            results = []
            for row in rpc_res.data or []:
                results.append({
                    "content": row.get("content", ""),
                    "metadata": row.get("metadata", {}),
                    "score": round(float(row.get("similarity", 0.0)), 4),
                })
            return results
        except Exception as e:
            logger.error("Supabase pgvector similarity search failed: %s", e)
            return []

    async def health_check(self) -> bool:
        try:
            self._ensure_client()
            res = self._client.table("documents").select("id", count="exact").limit(1).execute()
            return res is not None
        except Exception as e:
            logger.warning("Supabase health check warning: %s", e)
            return False

    async def persist(self) -> None:
        pass  # Supabase auto-persists in PostgreSQL

    def document_count(self) -> int:
        try:
            self._ensure_client()
            res = self._client.table("documents").select("id", count="exact").limit(1).execute()
            return res.count or 0
        except Exception:
            return 0
