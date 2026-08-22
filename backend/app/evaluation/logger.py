"""
Structured event logger.

Logs diagnostic events as JSON lines to logs/events.jsonl.
Each event captures metrics for diagnosis quality, retrieval,
confidence, latency, and cost tracking.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from app.config import get_settings

logger = logging.getLogger(__name__)


class DiagnosticEventLogger:
    """Writes structured events to a JSONL log file."""

    def __init__(self, log_dir: str | None = None) -> None:
        settings = get_settings()
        self._log_dir = log_dir or settings.log_dir
        os.makedirs(self._log_dir, exist_ok=True)
        self._log_path = os.path.join(self._log_dir, "events.jsonl")

    def log_diagnosis_event(
        self,
        case_id: str,
        event_type: str,
        *,
        confidence: float = 0.0,
        tools_called: list[str] | None = None,
        tool_call_count: int = 0,
        rag_chunks_retrieved: int = 0,
        rag_chunks_used: int = 0,
        agent_iterations: int = 0,
        latency_ms: float = 0.0,
        was_escalated: bool = False,
        followup_questions_asked: int = 0,
        vision_used: bool = False,
        voice_used: bool = False,
        tts_generated: bool = False,
        predicted_disease: str = "",
        predicted_plant: str = "",
        error: str | None = None,
        extra: dict | None = None,
    ) -> None:
        """
        Log a diagnostic event with structured metrics.

        Args:
            case_id: The diagnostic case ID.
            event_type: Type of event (e.g., "diagnosis_complete", "followup_asked",
                        "escalated", "error").
            **kwargs: All metric fields.
        """
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "case_id": case_id,
            "event": event_type,
            "metrics": {
                "confidence": round(confidence, 4),
                "tools_called": tools_called or [],
                "tool_call_count": tool_call_count,
                "rag_chunks_retrieved": rag_chunks_retrieved,
                "rag_chunks_used": rag_chunks_used,
                "agent_iterations": agent_iterations,
                "latency_ms": round(latency_ms, 1),
                "was_escalated": was_escalated,
                "followup_questions_asked": followup_questions_asked,
                "vision_used": vision_used,
                "voice_used": voice_used,
                "tts_generated": tts_generated,
                "predicted_disease": predicted_disease,
                "predicted_plant": predicted_plant,
            },
        }

        if error:
            event["error"] = error

        if extra:
            event["extra"] = extra

        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error("Failed to write event log: %s", e)

    def read_events(self, limit: int = 100) -> list[dict]:
        """Read the most recent events from the log file."""
        if not os.path.exists(self._log_path):
            return []

        events = []
        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error("Failed to read event log: %s", e)

        return events[-limit:]
