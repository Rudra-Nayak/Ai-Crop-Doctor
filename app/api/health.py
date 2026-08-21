"""
Health check endpoint.

Reports status of all subsystems so the frontend can show
system readiness and operators can monitor.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from app.schemas.responses import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """
    System health check.

    Returns status of:
    - FAISS vector store
    - Embedding model
    - Groq API configuration
    - Active session count
    - Knowledge base document count
    """
    config = request.app.state.config
    rag_service = request.app.state.rag_service
    session_manager = request.app.state.session_manager

    try:
        rag_health = await rag_service.health_check()

        from app.rag.embeddings import is_loaded as embeddings_loaded

        active_cases = await session_manager._store.list_active_cases()

        return HealthResponse(
            status="ok",
            faiss_loaded=rag_health.get("vector_store_loaded", False),
            embedding_model_loaded=embeddings_loaded(),
            groq_api_configured=bool(config.groq_api_key),
            active_sessions=len(active_cases),
            knowledge_base_chunks=rag_health.get("document_count", 0),
        )

    except Exception as e:
        logger.error("Health check error: %s", e)
        return HealthResponse(
            status="degraded",
            faiss_loaded=False,
            embedding_model_loaded=False,
            groq_api_configured=bool(config.groq_api_key),
            active_sessions=0,
            knowledge_base_chunks=0,
        )
