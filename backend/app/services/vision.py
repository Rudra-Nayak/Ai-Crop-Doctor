"""
Vision service — crop image analysis via Groq.

Uses the Groq SDK directly (not LangChain) because LangChain's ChatGroq
wrapper has limited vision support. The Groq SDK fully supports
multimodal messages with image_url content blocks.
"""

from __future__ import annotations

from groq import AsyncGroq
import base64
import json
import logging
import os

from app.config import get_settings

logger = logging.getLogger(__name__)

VISION_SYSTEM_PROMPT = """You are an expert agricultural plant pathologist analyzing a crop image.

Analyze the uploaded plant image carefully and provide a structured assessment.

Instructions:
- Identify the plant species if possible.
- Identify any disease, pest damage, or nutrient deficiency visible.
- If the plant appears healthy, clearly state "Healthy".
- Describe visible symptoms in detail (color changes, spots, patterns, wilting, etc.).
- Assess severity: None, Mild, Moderate, Severe, Critical.
- If uncertain about identification, state your uncertainty honestly.
- If the image is not a plant/crop, say so clearly.

Return ONLY valid JSON with this schema:
{
    "is_plant": true,
    "plant_name": "",
    "health_status": "healthy|diseased|pest_damage|nutrient_deficiency|uncertain",
    "disease_indicators": [],
    "symptoms": [],
    "severity": "none|mild|moderate|severe|critical",
    "affected_parts": [],
    "confidence": 0.0,
    "observations": "",
    "not_plant_reason": ""
}

Do NOT wrap the JSON in markdown code blocks. Return raw JSON only.
"""

LEGACY_SYSTEM_PROMPT = """
You are an expert agricultural plant pathologist.

Analyze the uploaded plant image.

Instructions:
- Identify the plant if possible.
- Identify the disease or pest if present.
- If healthy, clearly state Healthy.
- If uncertain, mention the uncertainty instead of guessing.
- Never reveal your reasoning.
- Never output <think>.
- Never output markdown.
- Return ONLY valid JSON.

Return this JSON schema:

{
  "plant_name": "",
  "disease": "",
  "confidence": 0,
  "severity": "",
  "symptoms": [],
  "cause": "",
  "organic_treatment": [],
  "chemical_treatment": [],
  "fertilizer": [],
  "prevention": [],
  "harvest_safe": "",
  "additional_notes": ""
}
"""


def _encode_image(image_path: str) -> str:
    """Base64-encode an image file."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _detect_mime_type(image_path: str) -> str:
    """Detect MIME type from file extension."""
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mime_map.get(ext, "image/jpeg")


class VisionService:
    """Groq-powered crop image analysis (Async)."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = AsyncGroq(api_key=self._settings.groq_api_key)

    async def analyze_image(self, image_path: str) -> dict:
        """
        Analyze a crop image using Groq's vision model asynchronously.
        """
        try:
            if not os.path.exists(image_path):
                return self._error_result(f"Image file not found: {image_path}")

            base64_image = _encode_image(image_path)
            mime_type = _detect_mime_type(image_path)
            image_url = f"data:{mime_type};base64,{base64_image}"

            logger.info("Analyzing image: %s (model: %s)", image_path, self._settings.groq_vision_model)

            candidate_vision_models = [
                self._settings.groq_vision_model,
                "meta-llama/llama-3.2-11b-vision-instruct",
                "llama-3.2-11b-vision-preview",
                "llama-3.2-90b-vision-preview",
            ]
            seen_vision = set()
            models_to_try = [m for m in candidate_vision_models if m and not (m in seen_vision or seen_vision.add(m))]

            response = None
            for v_model in models_to_try:
                try:
                    response = await self._client.chat.completions.create(
                        model=v_model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": VISION_SYSTEM_PROMPT},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": image_url},
                                    },
                                ],
                            }
                        ],
                        temperature=0.1,
                        max_completion_tokens=4096,
                    )
                    break
                except Exception as v_err:
                    logger.warning("Vision model '%s' failed: %s. Trying next candidate...", v_model, v_err)

            if response is None:
                raise RuntimeError("All candidate Groq vision models failed or reached rate limits.")

            raw = response.choices[0].message.content
            logger.debug("Vision raw response: %s", raw[:500])

            # Clean and parse JSON
            result = self._parse_json(raw)
            result["_raw_response"] = raw[:1000]

            logger.info(
                "Vision result: plant=%s, status=%s, confidence=%.2f",
                result.get("plant_name", "unknown"),
                result.get("health_status", "unknown"),
                result.get("confidence", 0),
            )
            return result

        except Exception as e:
            logger.error("Vision analysis failed: %s", e, exc_info=True)
            return self._error_result(str(e))

    async def analyze_image_legacy(self, image_path: str) -> dict:
        """
        Legacy direct one-shot leaf analysis using the old prompt.
        """
        try:
            if not os.path.exists(image_path):
                return {"error": f"Image file not found: {image_path}"}

            base64_image = _encode_image(image_path)
            mime_type = _detect_mime_type(image_path)
            image_url = f"data:{mime_type};base64,{base64_image}"

            response = await self._client.chat.completions.create(
                model=self._settings.groq_vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": LEGACY_SYSTEM_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                        ],
                    }
                ],
                temperature=0.1,
                max_completion_tokens=4096,
            )

            raw = response.choices[0].message.content.strip()
            logger.debug("Legacy vision raw response: %s", raw[:500])

            # Clean and parse JSON
            result = self._parse_json(raw)
            return result

        except Exception as e:
            logger.error("Legacy vision analysis failed: %s", e, exc_info=True)
            return {"error": str(e)}

    def _parse_json(self, text: str) -> dict:
        """Parse JSON from the LLM response, handling common formatting issues."""
        cleaned = text.strip()

        # Strip <think>...</think> block if present
        if "<think>" in cleaned:
            parts = cleaned.split("</think>")
            if len(parts) > 1:
                cleaned = parts[-1].strip()
            else:
                cleaned = ""

        # Strip markdown code blocks if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(cleaned[start:end])
                except json.JSONDecodeError:
                    pass

            return {
                "is_plant": True,
                "health_status": "uncertain",
                "observations": cleaned[:500],
                "confidence": 0.0,
                "error": "Failed to parse structured response",
            }

    def _error_result(self, error_msg: str) -> dict:
        """Return a safe error result that the agent can handle."""
        return {
            "is_plant": True,
            "plant_name": "",
            "health_status": "error",
            "disease_indicators": [],
            "symptoms": [],
            "severity": "unknown",
            "affected_parts": [],
            "confidence": 0.0,
            "observations": "",
            "error": f"Unable to analyze image: {error_msg}",
        }
