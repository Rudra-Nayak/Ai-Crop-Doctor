"""
Diagnostic flow — CrewAI Flow with structured Pydantic state.

Wraps the DiagnosticAgent execution with state management,
session tracking, and result routing.
"""

from __future__ import annotations

import logging
import time

from pydantic import BaseModel, Field

from app.agent.diagnostic_agent import DiagnosticAgent
from app.config import get_settings
from app.session.manager import SessionManager
from app.session.models import DiagnosisResult

logger = logging.getLogger(__name__)


class DiagnosisState(BaseModel):
    """Structured state for a diagnostic flow run."""

    case_id: str = ""
    image_path: str | None = None
    user_message: str = ""
    conversation_history: str = ""

    # Results
    response_text: str = ""
    diagnosis: DiagnosisResult | None = None
    needs_followup: bool = False
    followup_question: str | None = None
    confidence: float = 0.0

    # Metadata
    tools_called: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None
    raw_output: str = ""


class DiagnosticFlow:
    """
    Orchestrates a single diagnostic interaction.

    Flow:
    1. Prepare context from the session/case
    2. Run the diagnostic agent
    3. Route result: diagnosis or follow-up question
    4. Update the session with the result
    """

    def __init__(
        self,
        agent: DiagnosticAgent,
        session_manager: SessionManager,
    ) -> None:
        self._agent = agent
        self._session_manager = session_manager

    async def run(
        self,
        case_id: str,
        user_message: str = "",
        image_path: str | None = None,
    ) -> DiagnosisState:
        """
        Execute the diagnostic flow for a case.

        Returns a DiagnosisState with the result — either a diagnosis
        or a follow-up question for the farmer.
        """
        state = DiagnosisState(
            case_id=case_id,
            user_message=user_message,
            image_path=image_path,
        )

        start_time = time.time()

        try:
            # Step 1: Load conversation history
            case = await self._session_manager.get_case(case_id)
            if case:
                state.conversation_history = case.get_conversation_text()

            # Step 2: Record the farmer's message
            if user_message:
                await self._session_manager.add_farmer_message(
                    case_id=case_id,
                    content=user_message,
                    modality="text",
                )

            if image_path:
                await self._session_manager.record_image(case_id, image_path)

            # Step 3: Run the diagnostic agent
            result = await self._agent.run(
                user_message=user_message,
                image_path=image_path,
                conversation_history=state.conversation_history,
            )

            # Step 4: Populate state from result
            state.response_text = result.get("response_text", "")
            state.diagnosis = result.get("diagnosis")
            state.needs_followup = result.get("needs_followup", False)
            state.followup_question = result.get("followup_question")
            state.confidence = result.get("confidence", 0.0)
            state.tools_called = result.get("tools_called", [])
            state.raw_output = result.get("raw_output", "")

            if result.get("error"):
                state.error = result["error"]

            # Step 5: Record the agent's response
            await self._session_manager.add_agent_message(
                case_id=case_id,
                content=state.response_text,
                metadata={
                    "confidence": state.confidence,
                    "needs_followup": state.needs_followup,
                    "has_diagnosis": state.diagnosis is not None,
                },
            )

            # Step 6: Finalize diagnosis if we have one
            if state.diagnosis and not state.needs_followup:
                await self._session_manager.finalize_diagnosis(
                    case_id=case_id,
                    diagnosis=state.diagnosis,
                )

        except Exception as e:
            logger.error("Diagnostic flow failed: %s", e, exc_info=True)
            state.error = str(e)
            state.response_text = (
                "I apologize, but I encountered an issue processing your request. "
                "Please try again or describe your crop problem in more detail."
            )

        state.latency_ms = (time.time() - start_time) * 1000
        logger.info(
            "Flow complete: case=%s, followup=%s, confidence=%.2f, latency=%.0fms",
            case_id,
            state.needs_followup,
            state.confidence,
            state.latency_ms,
        )

        return state
