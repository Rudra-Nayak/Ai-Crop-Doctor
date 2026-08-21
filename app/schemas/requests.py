"""
API request models.

These are separate from the session/domain models — they describe
what the API endpoints accept from the client.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DiagnosisTextRequest(BaseModel):
    """Text-only follow-up request (JSON body)."""

    text: str = ""
    case_id: str | None = None


class SynthesizeRequest(BaseModel):
    """Request body for text-to-speech synthesis."""

    text: str = Field(..., min_length=1, max_length=5000)
