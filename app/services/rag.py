"""
RAG service — high-level retrieval orchestration.

Provides a simple interface for the agent to query the knowledge base.
Handles the full pipeline: query → vector search → rerank → format results.
Degrades gracefully: returns empty results on any failure.
"""

from __future__ import annotations

import logging

from app.rag.reranker import rerank_results
from app.rag.vector_store import VectorStoreBase

logger = logging.getLogger(__name__)


class RAGService:
    """Orchestrates knowledge base retrieval."""

    def __init__(self, vector_store: VectorStoreBase, top_k: int = 5, rerank_top_n: int = 3) -> None:
        self._store = vector_store
        self._top_k = top_k
        self._rerank_top_n = rerank_top_n

    async def query(self, question: str, k: int | None = None) -> list[dict]:
        """
        Search the knowledge base for information relevant to the question.

        Returns a list of dicts with 'content', 'metadata', and 'score'.
        Returns an empty list on any failure — the agent's prompt instructs
        it to NOT hallucinate when RAG returns nothing.
        """
        if not question or not question.strip():
            return []

        k = k or self._top_k

        try:
            # Step 1: Vector similarity search
            results = await self._store.similarity_search(question, k=k)

            if not results:
                logger.info("RAG returned 0 results for: '%s'", question[:100])
                return []

            logger.info(
                "RAG found %d results for: '%s'",
                len(results),
                question[:100],
            )

            # Step 2: Rerank for relevance
            reranked = await rerank_results(
                query=question,
                results=results,
                top_n=self._rerank_top_n,
            )

            return reranked

        except Exception as e:
            logger.error("RAG query failed: %s", e, exc_info=True)
            return []

    async def health_check(self) -> dict:
        """Check RAG subsystem health."""
        store_ok = await self._store.health_check()
        return {
            "vector_store_loaded": store_ok,
            "document_count": self._store.document_count(),
        }

    def format_results_for_agent(self, results: list[dict]) -> str:
        """
        Format RAG results into a readable string for the agent.
        Called by the agent's SearchKnowledgeBaseTool.
        """
        if not results:
            return "No relevant information found in the knowledge base."

        formatted_parts = []
        for i, result in enumerate(results, 1):
            source = result.get("metadata", {}).get("source", "unknown")
            score = result.get("score", 0)
            rerank_score = result.get("rerank_score", "N/A")
            content = result.get("content", "")

            formatted_parts.append(
                f"[Source {i}: {source} | Relevance: {score:.2f} | Rerank: {rerank_score}]\n"
                f"{content}\n"
            )

        return "\n---\n".join(formatted_parts)
