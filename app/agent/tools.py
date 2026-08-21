"""
Agent tools — LangChain BaseTool implementations for the Diagnostic Agent.

Each tool has a clear description so the agent knows WHEN to call it.
The agent autonomously decides which tool to use based on the current
state of evidence and its reasoning.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict

from app.services.rag import RAGService
from app.services.vision import VisionService

logger = logging.getLogger(__name__)


# ── Tool Input Schemas ────────────────────────────────────────────────

class ImageAnalysisInput(BaseModel):
    image_path: str = Field(description="Absolute path to the crop image file to analyze")


class KnowledgeBaseInput(BaseModel):
    query: str = Field(description="Search query describing symptoms, disease name, or crop issue to look up")


class ConfidenceCheckInput(BaseModel):
    evidence_summary: str = Field(
        description="Summary of all evidence gathered so far: image findings, "
        "knowledge base results, and farmer statements"
    )


class FollowupQuestionInput(BaseModel):
    question: str = Field(
        description="A specific follow-up question to ask the farmer for more information"
    )


# ── Tools ─────────────────────────────────────────────────────────────

class AnalyzeCropImageTool(BaseTool):
    """Analyze a crop/plant image to identify visible symptoms and disease indicators."""

    name: str = "analyze_crop_image"
    description: str = (
        "Analyze a crop/plant image to identify the plant species, visible symptoms, "
        "disease indicators, pest damage, and severity. Use this when the farmer has "
        "provided a crop image that hasn't been analyzed yet. Returns structured "
        "observations about what's visible in the image."
    )
    args_schema: Type[BaseModel] = ImageAnalysisInput

    # Service injected at construction
    vision_service: Any = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, image_path: str) -> str:
        """Synchronous wrapper — runs the async vision service."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run, self.vision_service.analyze_image(image_path)
                    ).result()
            else:
                result = asyncio.run(self.vision_service.analyze_image(image_path))
        except RuntimeError:
            result = asyncio.run(self.vision_service.analyze_image(image_path))

        if "error" in result:
            return (
                f"Image analysis encountered an issue: {result['error']}. "
                "Please ask the farmer to describe the symptoms in text instead."
            )

        # Format for the agent
        parts = []
        if result.get("is_plant") is False:
            return (
                f"The uploaded image does not appear to be a plant or crop. "
                f"Reason: {result.get('not_plant_reason', 'unrecognized content')}. "
                f"Please ask the farmer to upload a clear photo of the affected crop."
            )

        if result.get("plant_name"):
            parts.append(f"Plant identified: {result['plant_name']}")
        parts.append(f"Health status: {result.get('health_status', 'uncertain')}")
        if result.get("symptoms"):
            parts.append(f"Visible symptoms: {', '.join(result['symptoms'])}")
        if result.get("disease_indicators"):
            parts.append(f"Disease indicators: {', '.join(result['disease_indicators'])}")
        if result.get("severity"):
            parts.append(f"Severity: {result['severity']}")
        if result.get("affected_parts"):
            parts.append(f"Affected parts: {', '.join(result['affected_parts'])}")
        if result.get("observations"):
            parts.append(f"Observations: {result['observations']}")
        parts.append(f"Vision confidence: {result.get('confidence', 0)}")

        return "\n".join(parts)


class SearchKnowledgeBaseTool(BaseTool):
    """Search the agricultural knowledge base for disease and treatment information."""

    name: str = "search_knowledge_base"
    description: str = (
        "Search the agricultural knowledge base for diseases matching observed symptoms, "
        "treatment recommendations, fertilizer advice, or pest management information. "
        "Use this after identifying symptoms (from image or farmer description) to find "
        "confirmed disease information and evidence-based treatments. "
        "Provide specific symptom descriptions for best results."
    )
    args_schema: Type[BaseModel] = KnowledgeBaseInput

    # Service injected at construction
    rag_service: Any = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, query: str) -> str:
        """Search the knowledge base."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    results = pool.submit(
                        asyncio.run, self.rag_service.query(query)
                    ).result()
            else:
                results = asyncio.run(self.rag_service.query(query))
        except RuntimeError:
            results = asyncio.run(self.rag_service.query(query))

        if not results:
            return (
                "No relevant information found in the knowledge base for this query. "
                "Do NOT guess treatments. Recommend the farmer consult a local agricultural "
                "extension officer or plant pathologist for this specific issue."
            )

        return self.rag_service.format_results_for_agent(results)


class CheckConfidenceTool(BaseTool):
    """Evaluate whether accumulated evidence is sufficient for a confident diagnosis."""

    name: str = "check_confidence"
    description: str = (
        "Evaluate whether the accumulated evidence is sufficient for a diagnosis. "
        "Pass a summary of ALL evidence gathered so far (image analysis results, "
        "knowledge base matches, farmer's description, conversation history). "
        "Returns a confidence score (0-100) and whether to proceed with diagnosis "
        "or gather more evidence."
    )
    args_schema: Type[BaseModel] = ConfidenceCheckInput

    # LLM client for confidence evaluation
    groq_client: Any = None
    groq_model: str = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, evidence_summary: str) -> str:
        """Evaluate evidence quality using LLM."""
        try:
            from groq import Groq

            client = self.groq_client or Groq()

            response = client.chat.completions.create(
                model=self.groq_model or "groq/compound",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an agricultural evidence quality assessor. "
                            "Evaluate the evidence summary and return ONLY a JSON object: "
                            '{"confidence": <0-100>, "sufficient": <true/false>, '
                            '"gaps": ["list of missing evidence"], '
                            '"recommendation": "proceed_with_diagnosis|gather_more_evidence|escalate"}'
                        ),
                    },
                    {"role": "user", "content": evidence_summary},
                ],
                temperature=0.0,
                max_completion_tokens=300,
            )

            raw = response.choices[0].message.content.strip()
            # Clean markdown
            raw = raw.replace("```json", "").replace("```", "").strip()

            try:
                result = json.loads(raw)
                confidence = result.get("confidence", 50)
                sufficient = result.get("sufficient", False)
                gaps = result.get("gaps", [])
                recommendation = result.get("recommendation", "gather_more_evidence")

                return (
                    f"Confidence: {confidence}%\n"
                    f"Evidence sufficient: {sufficient}\n"
                    f"Recommendation: {recommendation}\n"
                    f"Evidence gaps: {', '.join(gaps) if gaps else 'None identified'}"
                )
            except json.JSONDecodeError:
                return f"Confidence assessment: {raw}"

        except Exception as e:
            logger.error("Confidence check failed: %s", e)
            return (
                "Confidence check unavailable. "
                "Proceed with caution and state your confidence level honestly."
            )


class AskFollowupQuestionTool(BaseTool):
    """Ask the farmer a follow-up question to gather more evidence."""

    name: str = "ask_followup_question"
    description: str = (
        "Ask the farmer a specific follow-up question to gather more diagnostic evidence. "
        "Use this when you need information that is NOT available from image analysis or "
        "the knowledge base, such as: how long symptoms have been present, geographic region, "
        "recent weather conditions, previous treatments applied, which parts of the field "
        "are affected, or recent farming activities. "
        "The question will be presented to the farmer and their response will be provided "
        "in the next interaction."
    )
    args_schema: Type[BaseModel] = FollowupQuestionInput

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, question: str) -> str:
        """
        Format the follow-up question.

        This tool is special: when the agent calls it, the diagnosis flow
        should pause and return the question to the farmer. The flow.py
        handles this by checking for the FOLLOWUP marker in the output.
        """
        return f"FOLLOWUP_QUESTION: {question}"
