"""
Diagnosis API endpoints.

Unified endpoint accepting any combination of image, text, and audio.
Handles new cases and follow-up conversations.
"""

from __future__ import annotations

import logging
import os
import uuid

from fastapi import APIRouter, File, Form, Request, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.schemas.responses import DiagnosisResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["diagnosis"])


@router.post("/diagnosis", response_model=DiagnosisResponse)
async def diagnose(
    request: Request,
    image: UploadFile | None = File(None),
    text: str = Form(""),
    audio: UploadFile | None = File(None),
    case_id: str | None = Form(None),
):
    """
    Unified diagnosis endpoint.

    Accepts any combination of:
    - image: crop photo (UploadFile)
    - text: symptom description (string)
    - audio: voice recording → auto-transcribed (UploadFile)
    - case_id: continue existing conversation (string)

    Returns diagnosis, follow-up questions, or conversational response.
    """
    # Access app-level dependencies
    session_manager = request.app.state.session_manager
    diagnostic_flow = request.app.state.diagnostic_flow
    speech_service = request.app.state.speech_service
    config = request.app.state.config

    # Step 1: Get or create case
    case = await session_manager.get_or_create_case(case_id)
    current_case_id = case.case_id

    # Step 2: Handle audio → transcribe to text
    user_text = text
    if audio and audio.filename:
        try:
            audio_bytes = await audio.read()
            if audio_bytes:
                transcript = await speech_service.transcribe(
                    audio_bytes=audio_bytes,
                    filename=audio.filename,
                )
                if transcript.get("text"):
                    user_text = transcript["text"]
                    logger.info("Audio transcribed: '%s'", user_text[:100])
                elif transcript.get("error"):
                    logger.warning("Audio transcription failed: %s", transcript["error"])
                    # Fall through — use any text that was also provided
        except Exception as e:
            logger.error("Audio processing error: %s", e)

    # Step 3: Handle image upload
    image_path = None
    if image and image.filename:
        try:
            # Validate file type
            content_type = image.content_type or ""
            if not content_type.startswith("image/"):
                return DiagnosisResponse(
                    case_id=current_case_id,
                    response_text="Please upload an image file (JPEG, PNG, etc.).",
                    needs_followup=False,
                    confidence=0.0,
                )

            # Save the image
            upload_dir = config.upload_dir
            os.makedirs(upload_dir, exist_ok=True)
            ext = os.path.splitext(image.filename)[1] or ".jpg"
            filename = f"{uuid.uuid4()}{ext}"
            image_path = os.path.join(upload_dir, filename)

            with open(image_path, "wb") as f:
                content = await image.read()
                f.write(content)

            logger.info("Image saved: %s", image_path)

        except Exception as e:
            logger.error("Image save failed: %s", e)
            image_path = None

    # Step 4: Validate we have some input
    if not user_text and not image_path:
        return DiagnosisResponse(
            case_id=current_case_id,
            response_text=(
                "Please provide a description of your crop problem, "
                "upload a photo of the affected plant, or record a voice message."
            ),
            needs_followup=True,
            followup_question="What symptoms are you seeing on your crops?",
            confidence=0.0,
        )

    # Step 5: Run the diagnostic flow
    state = await diagnostic_flow.run(
        case_id=current_case_id,
        user_message=user_text,
        image_path=image_path,
    )

    # Step 6: Build response
    return DiagnosisResponse(
        case_id=current_case_id,
        response_text=state.response_text,
        diagnosis=state.diagnosis,
        needs_followup=state.needs_followup,
        followup_question=state.followup_question,
        confidence=state.confidence,
    )


@router.post("/dignosis")
async def legacy_dignosis(
    request: Request,
    image: UploadFile = File(...),
):
    """
    Legacy direct one-shot leaf diagnosis endpoint.
    """
    vision_service = request.app.state.vision_service
    config = request.app.state.config

    # Validate file type
    content_type = image.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    # Save the image
    upload_dir = config.upload_dir
    os.makedirs(upload_dir, exist_ok=True)
    ext = os.path.splitext(image.filename)[1] or ".jpg"
    filename = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join(upload_dir, filename)

    try:
        with open(file_path, "wb") as f:
            content = await image.read()
            f.write(content)
        logger.info("Saved legacy upload: %s", file_path)
    except Exception as e:
        logger.error("Failed to save legacy upload: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to save image: {str(e)}")

    # Call the legacy analysis
    result = await vision_service.analyze_image_legacy(file_path)

    return JSONResponse(content={"result": result})
