"""
AI Crop Doctor — FastAPI Application.

Main entry point. Sets up the application with:
- CORS middleware for React frontend
- Lifespan events for startup/shutdown (load embeddings, FAISS index)
- All API route registration
- Dependency injection via app.state
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.agent.diagnostic_agent import DiagnosticAgent
from app.agent.flow import DiagnosticFlow
from app.api.diagnosis import router as diagnosis_router, legacy_dignosis
from app.api.health import router as health_router
from app.api.voice import router as voice_router
from app.config import get_settings
from app.rag.embeddings import get_embeddings
from app.services.rag import RAGService
from app.services.speech import SpeechService
from app.services.vision import VisionService
from app.session.manager import SessionManager
from app.session.store import InMemorySessionStore

# Apply compatibility patches for CrewAI / LiteLLM on non-Anthropic providers
try:
    import litellm
    litellm.drop_params = True
    litellm.num_retries = 5

    _orig_litellm_comp = litellm.completion

    def _sanitized_completion(*args, **kwargs):
        if "messages" in kwargs and isinstance(kwargs["messages"], list):
            for msg in kwargs["messages"]:
                if isinstance(msg, dict):
                    msg.pop("cache_breakpoint", None)
                    msg.pop("cache_control", None)
        return _orig_litellm_comp(*args, **kwargs)

    litellm.completion = _sanitized_completion
except Exception:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events.

    Startup:
    - Load configuration
    - Initialize embedding model
    - Load FAISS vector store
    - Create all services and the diagnostic agent
    - Store everything in app.state for route access

    Shutdown:
    - Persist FAISS index if modified
    - Clean up resources
    """
    logger.info("=" * 60)
    logger.info("AI Crop Doctor — Starting up")
    logger.info("=" * 60)

    config = get_settings()

    # Create upload and log directories
    os.makedirs(config.upload_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # ── Pre-load embedding model singleton ONCE at startup ────────
    logger.info("Pre-loading embedding model singleton once at startup...")
    embeddings_instance = get_embeddings(config.embedding_model)
    embeddings_instance.embed_query("startup warm-up")
    logger.info("Embedding model is 100%% ready and cached in RAM.")

    # ── Initialize RAG pipeline (Supabase pgvector) ───────────────
    from app.rag.vector_store import SupabasePgVectorStore
    logger.info("Using Supabase pgvector RAG store (%s)", config.supabase_url or "cloud")
    vector_store = SupabasePgVectorStore(config.supabase_url, config.supabase_key, embeddings_instance)

    rag_service = RAGService(
        vector_store=vector_store,
        top_k=config.rag_top_k,
        rerank_top_n=config.rag_rerank_top_n,
    )

    # ── Initialize services ───────────────────────────────────────
    vision_service = VisionService()
    speech_service = SpeechService()
    from app.services.storage import StorageService
    storage_service = StorageService(config)

    # ── Initialize session management ─────────────────────────────
    if config.use_supabase_session and config.supabase_url and config.supabase_key:
        logger.info("Using Supabase PostgreSQL session store")
        from app.session.store import SupabaseSessionStore
        session_store = SupabaseSessionStore(config.supabase_url, config.supabase_key)
    else:
        session_store = InMemorySessionStore()

    session_manager = SessionManager(session_store)

    # ── Initialize the diagnostic agent & flow ────────────────────
    diagnostic_agent = DiagnosticAgent(
        config=config,
        vision_service=vision_service,
        rag_service=rag_service,
    )
    diagnostic_flow = DiagnosticFlow(
        agent=diagnostic_agent,
        session_manager=session_manager,
    )

    # ── Store everything in app.state ─────────────────────────────
    app.state.config = config
    app.state.vector_store = vector_store
    app.state.rag_service = rag_service
    app.state.vision_service = vision_service
    app.state.speech_service = speech_service
    app.state.storage_service = storage_service
    app.state.session_manager = session_manager
    app.state.diagnostic_agent = diagnostic_agent
    app.state.diagnostic_flow = diagnostic_flow

    rag_health = await rag_service.health_check()
    logger.info("RAG status: %s", rag_health)
    logger.info("Groq API configured: %s", bool(config.groq_api_key))
    logger.info("=" * 60)
    logger.info("AI Crop Doctor — Ready!")
    logger.info("=" * 60)

    yield  # ── Application running ──

    # ── Shutdown ──────────────────────────────────────────────────
    logger.info("Shutting down...")
    try:
        await vector_store.persist()
    except Exception as e:
        logger.warning("Failed to persist FAISS index on shutdown: %s", e)
    logger.info("Goodbye!")


# ── Create the FastAPI app ────────────────────────────────────────────

app = FastAPI(
    title="AI Crop Doctor",
    description=(
        "Multimodal agentic AI assistant for crop disease diagnosis. "
        "Accepts voice, text, and images. Uses CrewAI for reasoning, "
        "LangChain for RAG, and Groq for fast inference."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS middleware (for React frontend) ──────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai-crop-doctor-six.vercel.app",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ─────────────────────────────────────────────────

app.include_router(diagnosis_router)
app.include_router(voice_router)
app.include_router(health_router)

# Register legacy direct endpoint at root level
app.post("/dignosis")(legacy_dignosis)

# ── Mount static frontend (HTML/CSS/JS) ──────────────────────────────
from fastapi.staticfiles import StaticFiles

frontend_candidates = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend")),
    os.path.abspath("frontend"),
    os.path.abspath("../frontend"),
]
for fc in frontend_candidates:
    if os.path.exists(fc):
        app.mount("/", StaticFiles(directory=fc, html=True), name="frontend")
        break

