"""
Voice API endpoints.

Separate STT and TTS endpoints for the frontend to use independently
of the diagnosis flow (e.g., transcribe-then-display before submitting).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import Response

from app.schemas.responses import TranscribeResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/voice", tags=["voice"])


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    request: Request,
    audio: UploadFile = File(...),
):
    """
    Speech-to-text using Groq Whisper.

    Accepts an audio file and returns the transcription.
    Frontend can use this to show the user what was heard
    before submitting to the diagnosis endpoint.
    """
    speech_service = request.app.state.speech_service

    try:
        audio_bytes = await audio.read()

        if not audio_bytes:
            return TranscribeResponse(text="", language="", duration_s=0.0)

        result = await speech_service.transcribe(
            audio_bytes=audio_bytes,
            filename=audio.filename or "audio.wav",
        )

        return TranscribeResponse(
            text=result.get("text", ""),
            language=result.get("language", ""),
            duration_s=result.get("duration_s", 0.0),
        )

    except Exception as e:
        logger.error("Transcribe endpoint error: %s", e)
        return TranscribeResponse(text="", language="", duration_s=0.0)


@router.post("/synthesize")
async def synthesize(
    request: Request,
    text: str = Form(...),
):
    """
    Text-to-speech using Groq Orpheus.

    Returns WAV audio bytes. On failure, returns a 200 with
    an empty body and X-TTS-Failed header so the frontend
    can fall back to displaying text.
    """
    speech_service = request.app.state.speech_service

    try:
        audio_bytes = await speech_service.synthesize(text)

        if not audio_bytes:
            return Response(
                content=b"",
                media_type="audio/wav",
                headers={"X-TTS-Failed": "true"},
            )

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "X-TTS-Failed": "false",
                "Content-Disposition": "inline; filename=response.wav",
            },
        )

    except Exception as e:
        logger.error("Synthesize endpoint error: %s", e)
        return Response(
            content=b"",
            media_type="audio/wav",
            headers={"X-TTS-Failed": "true"},
        )
