"""
API response models.

Defines the shape of all JSON responses returned by the API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.session.models import DiagnosisResult


class DiagnosisResponse(BaseModel):
    """Response from the /api/diagnosis endpoint."""

    case_id: str = ""
    response_text: str = ""
    diagnosis: DiagnosisResult | None = None
    needs_followup: bool = False
    followup_question: str | None = None
    confidence: float = 0.0


class TranscribeResponse(BaseModel):
    """Response from the /api/voice/transcribe endpoint."""

    text: str = ""
    language: str = ""
    duration_s: float = 0.0


class HealthResponse(BaseModel):
    """Response from the /api/health endpoint."""

    status: str = "ok"
    faiss_loaded: bool = False
    embedding_model_loaded: bool = False
    groq_api_configured: bool = False
    active_sessions: int = 0
    knowledge_base_chunks: int = 0


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = ""
    detail: str = ""
    case_id: str | None = None
