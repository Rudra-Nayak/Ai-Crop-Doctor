"""
Diagnostic agent — single CrewAI agent with tool-calling evidence loop.

This is the core intelligence of the system. One agent, four tools,
autonomous reasoning about what evidence to gather next.
"""

from __future__ import annotations

import json
import os
import logging
from typing import Any

from groq import Groq

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

from app.agent.prompts import DIAGNOSTIC_AGENT_BACKSTORY, build_diagnostic_task_description
from app.agent.tools import (
    AnalyzeCropImageTool,
    AskFollowupQuestionTool,
    CheckConfidenceTool,
    SearchKnowledgeBaseTool,
)
from app.config import Settings
from app.services.rag import RAGService
from app.services.vision import VisionService
from app.session.models import DiagnosisResult

logger = logging.getLogger(__name__)


class DiagnosticAgent:
    """
    Factory and runner for the single diagnostic agent.

    Creates a CrewAI Agent equipped with 4 tools and runs it
    against a diagnostic task built from the current case state.
    """

    def __init__(
        self,
        config: Settings,
        vision_service: VisionService,
        rag_service: RAGService,
    ) -> None:
        self._config = config
        self._vision_service = vision_service
        self._rag_service = rag_service
        self._groq_client = Groq(api_key=config.groq_api_key)

    def _create_agent(self) -> Any:
        """Create the diagnostic agent instance lazily."""
        from crewai import Agent, LLM
        llm = LLM(
            model=f"groq/{self._config.groq_text_model}",
            api_key=self._config.groq_api_key,
            temperature=0.2,
            max_tokens=2048,
        )
        return Agent(
            role="Senior Agricultural Diagnostician",
            goal=(
                "Accurately diagnose crop diseases by gathering evidence through "
                "image analysis, knowledge base research, and farmer communication. "
                "Only diagnose when evidence is sufficient. Ask follow-up questions "
                "or escalate when uncertain."
            ),
            backstory=DIAGNOSTIC_AGENT_BACKSTORY,
            tools=self._tools,
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iter=self._config.max_agent_iterations,
        )

    async def run(
        self,
        user_message: str = "",
        image_path: str | None = None,
        conversation_history: str = "",
    ) -> dict:
        """
        Run the diagnostic agent and return results.

        Returns a dict with:
            - diagnosis: DiagnosisResult or None
            - response_text: str (conversational response for the farmer)
            - needs_followup: bool
            - followup_question: str or None
            - tools_called: list[str]
            - confidence: float
            - raw_output: str
        """
        logger.info(
            "Running diagnostic agent (message: '%s', image: %s, history_len: %d)",
            user_message[:100] if user_message else "(none)",
            bool(image_path),
            len(conversation_history),
        )

        return await self._run_direct_pipeline(user_message, image_path, conversation_history)

    async def _run_direct_pipeline(
        self,
        user_message: str = "",
        image_path: str | None = None,
        conversation_history: str = "",
    ) -> dict:
        """Resilient Direct Diagnostic Pipeline using Groq + Vision + RAG."""
        try:
            # Gather Vision findings if image is provided
            vision_context = ""
            if image_path and os.path.exists(image_path):
                v_res = await self._vision_service.analyze_image(image_path)
                vision_context = json.dumps(v_res, indent=2)

            # Gather RAG knowledge safely
            rag_context = "No RAG matches."
            try:
                search_query = f"{user_message} {vision_context[:200]}".strip()
                rag_docs = await self._rag_service.query(search_query)
                if rag_docs:
                    rag_context = self._rag_service.format_results_for_agent(rag_docs)
            except Exception as rag_err:
                logger.warning("RAG retrieval skipped due to error: %s", rag_err)

            system_prompt = (
                "You are an expert AI Crop Doctor and plant pathologist with 20 years of field experience. "
                "Analyze the farmer's crop condition using the provided image analysis, symptoms, "
                "conversation history, and agricultural knowledge base.\n\n"
                "Language Guidelines:\n"
                "- If the farmer communicates or writes in Hindi (हिंदी) or Hinglish, provide 'response_text', "
                "'organic_treatment', 'chemical_treatment', 'prevention', and 'symptoms' in clear, empathetic Hindi (Devanagari script).\n"
                "- If the farmer communicates or writes in Punjabi (ਪੰਜਾਬੀ), provide 'response_text', "
                "'organic_treatment', 'chemical_treatment', 'prevention', and 'symptoms' in clear, empathetic Punjabi (Gurmukhi script).\n"
                "- For 'plant_name' and 'disease', include bilingual names (e.g. 'आलू (Potato)' / 'ਆਲੂ (Potato)' and 'पछेती झुलसा (Late Blight)' / 'ਪਿਛੇਤੀ ਝੁਲਸਾ (Late Blight)').\n"
                "- If the farmer communicates in English, provide all fields in English.\n\n"
                "Instructions:\n"
                "1. If evidence is sufficient (confidence >= 70%), return a full diagnosis JSON.\n"
                "2. If more info is needed, ask a specific follow-up question.\n\n"
                "Return ONLY valid JSON with this exact schema:\n"
                "{\n"
                '  "plant_name": "identified plant species",\n'
                '  "disease": "identified disease or condition (or Healthy)",\n'
                '  "confidence": 0.88,\n'
                '  "severity": "mild|moderate|severe|critical",\n'
                '  "symptoms": ["symptom 1", "symptom 2"],\n'
                '  "cause": "pathogen/environmental etiology",\n'
                '  "organic_treatment": ["organic remedy 1", "organic remedy 2"],\n'
                '  "chemical_treatment": ["chemical intervention 1"],\n'
                '  "prevention": ["prevention practice 1", "prevention practice 2"],\n'
                '  "evidence_sources": ["University Pathology Guide / Extension"],\n'
                '  "response_text": "Farmer-friendly clear explanation of the diagnosis and immediate action steps.",\n'
                '  "needs_followup": false,\n'
                '  "followup_question": null\n'
                "}"
            )

            user_prompt = (
                f"Farmer Message: {user_message or 'Please examine this crop.'}\n\n"
                f"Vision Analysis:\n{vision_context or 'No image provided.'}\n\n"
                f"Agronomic Reference Knowledge:\n{rag_context or 'No RAG matches.'}\n\n"
                f"Conversation History:\n{conversation_history or 'New consultation.'}"
            )

            resp = self._groq_client.chat.completions.create(
                model=self._config.groq_text_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_completion_tokens=2048,
            )

            fallback_raw = resp.choices[0].message.content
            return self._parse_output(fallback_raw)

        except Exception as fb_err:
            logger.error("Direct diagnostic pipeline failed: %s", fb_err, exc_info=True)
            return {
                "diagnosis": None,
                "response_text": (
                    "I have received your crop inquiry, but encountered an issue retrieving the diagnosis. "
                    "Please provide more symptom details or attach a clear photo of the leaf."
                ),
                "needs_followup": False,
                "followup_question": None,
                "tools_called": [],
                "confidence": 0.0,
                "raw_output": str(fb_err),
                "error": str(fb_err),
            }

    def _parse_output(self, raw_output: str) -> dict:
        """Parse the agent's output into a structured result."""
        # Strip <think>...</think> if present
        clean_text = raw_output.strip()
        if "<think>" in clean_text:
            parts = clean_text.split("</think>")
            if len(parts) > 1:
                clean_text = parts[-1].strip()

        # Check if the agent is asking a follow-up question
        if "FOLLOWUP_QUESTION:" in clean_text:
            question = clean_text.split("FOLLOWUP_QUESTION:")[-1].strip()
            # Clean up: the question might have extra text after it
            question = question.split("\n")[0].strip()
            return {
                "diagnosis": None,
                "response_text": question,
                "needs_followup": True,
                "followup_question": question,
                "tools_called": [],
                "confidence": 0.0,
                "raw_output": raw_output,
            }

        # Try to extract JSON diagnosis from the output
        diagnosis_dict = self._extract_json(clean_text)

        if diagnosis_dict:
            # Extract conversational response
            response_text = diagnosis_dict.pop("response_text", "")

            # Build DiagnosisResult
            try:
                diagnosis = DiagnosisResult(
                    plant_name=diagnosis_dict.get("plant_name", ""),
                    disease=diagnosis_dict.get("disease", ""),
                    confidence=float(diagnosis_dict.get("confidence", 0)),
                    severity=diagnosis_dict.get("severity", ""),
                    symptoms=diagnosis_dict.get("symptoms", []),
                    cause=diagnosis_dict.get("cause", ""),
                    organic_treatment=diagnosis_dict.get("organic_treatment", []),
                    chemical_treatment=diagnosis_dict.get("chemical_treatment", []),
                    prevention=diagnosis_dict.get("prevention", []),
                    evidence_sources=diagnosis_dict.get("evidence_sources", []),
                    is_escalated=diagnosis_dict.get("is_escalated", False),
                    escalation_reason=diagnosis_dict.get("escalation_reason", ""),
                    additional_notes=diagnosis_dict.get("additional_notes", ""),
                )
            except Exception as e:
                logger.warning("Failed to parse DiagnosisResult: %s", e)
                diagnosis = None
                response_text = clean_text[:500]

            if not response_text and diagnosis:
                response_text = self._generate_response_text(diagnosis)

            return {
                "diagnosis": diagnosis,
                "response_text": response_text,
                "needs_followup": False,
                "followup_question": None,
                "tools_called": [],
                "confidence": diagnosis.confidence if diagnosis else 0.0,
                "raw_output": raw_output,
            }

        # Couldn't parse JSON — treat the whole output as conversational
        return {
            "diagnosis": None,
            "response_text": clean_text[:500],
            "needs_followup": "?" in clean_text,
            "followup_question": clean_text[:500] if "?" in clean_text else None,
            "tools_called": [],
            "confidence": 0.0,
            "raw_output": raw_output,
        }

    def _extract_json(self, text: str) -> dict | None:
        """Extract a JSON object from text that may contain other content."""
        cleaned = text.strip()
        if "<think>" in cleaned:
            parts = cleaned.split("</think>")
            if len(parts) > 1:
                cleaned = parts[-1].strip()

        # Try the whole text first
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Find the outermost JSON object
        brace_depth = 0
        start = -1
        for i, char in enumerate(cleaned):
            if char == "{":
                if brace_depth == 0:
                    start = i
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
                if brace_depth == 0 and start >= 0:
                    try:
                        return json.loads(cleaned[start : i + 1])
                    except json.JSONDecodeError:
                        start = -1
                        continue

        # Use json_repair as a resilient fallback
        try:
            import json_repair
            repaired = json_repair.loads(cleaned)
            if isinstance(repaired, dict):
                return repaired
        except Exception:
            pass

        return None

    def _generate_response_text(self, diagnosis: DiagnosisResult) -> str:
        """Generate a farmer-friendly response from a diagnosis."""
        if diagnosis.is_escalated:
            return (
                f"Based on my analysis, I suspect this might be related to {diagnosis.disease}, "
                f"but I'm only {int(diagnosis.confidence * 100)}% confident. "
                f"I recommend consulting a local agricultural extension officer for confirmation. "
                f"{diagnosis.escalation_reason}"
            )

        if diagnosis.disease.lower() == "healthy":
            return (
                f"Great news! Your {diagnosis.plant_name} looks healthy. "
                f"Keep up the good farming practices!"
            )

        return (
            f"Based on my analysis, your {diagnosis.plant_name} appears to have "
            f"{diagnosis.disease} (confidence: {int(diagnosis.confidence * 100)}%). "
            f"The severity is {diagnosis.severity}. "
            f"I've prepared treatment recommendations for you."
        )
