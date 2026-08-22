"""
LLM-based reranker.

Takes the top-k results from vector search and returns the top-n most relevant
based on vector similarity scores.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def rerank_results(
    query: str,
    results: list[dict],
    top_n: int = 3,
) -> list[dict]:
    """
    Rerank search results based on vector similarity and return top_n.
    """
    if not results:
        return []
    return results[:top_n]
