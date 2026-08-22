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


@router.get("/debug_step")
async def debug_step(request: Request, step: int = 1):
    import time
    t0 = time.time()
    if step == 1:
        sm = request.app.state.session_manager
        case = await sm.get_or_create_case(None)
        return {"step": 1, "status": "ok", "case_id": case.case_id, "elapsed_ms": round((time.time() - t0) * 1000, 1)}
    elif step == 2:
        rag = request.app.state.rag_service
        docs = await rag.query("Tomato early blight")
        return {"step": 2, "status": "ok", "docs_count": len(docs), "elapsed_ms": round((time.time() - t0) * 1000, 1)}
    elif step == 3:
        from groq import AsyncGroq
        cfg = request.app.state.config
        client = AsyncGroq(api_key=cfg.groq_api_key)
        res = await client.chat.completions.create(
            model=cfg.groq_text_model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
        )
        return {"step": 3, "status": "ok", "response": res.choices[0].message.content, "elapsed_ms": round((time.time() - t0) * 1000, 1)}
    return {"error": "invalid step"}
