"""
Session domain models.

These Pydantic models define the core data structures for tracking
a farmer's diagnostic case through the conversation lifecycle.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single turn in the conversation."""

    role: Literal["farmer", "agent", "system"] = "farmer"
    content: str = ""
    modality: Literal["text", "voice", "image"] = "text"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = Field(default_factory=dict)
    # metadata examples:
    #   {"image_path": "uploads/abc.jpg"}
    #   {"audio_duration_s": 2.3, "original_audio": "uploads/abc.wav"}


class DiagnosisResult(BaseModel):
    """Structured output from the diagnostic agent."""

    plant_name: str = ""
    disease: str = ""
    confidence: float = 0.0
    severity: str = ""
    symptoms: list[str] = Field(default_factory=list)
    cause: str = ""
    organic_treatment: list[str] = Field(default_factory=list)
    chemical_treatment: list[str] = Field(default_factory=list)
    prevention: list[str] = Field(default_factory=list)
    evidence_sources: list[str] = Field(default_factory=list)
    is_escalated: bool = False
    escalation_reason: str = ""
    additional_notes: str = ""


class Case(BaseModel):
    """
    A diagnostic case — one farmer interaction that may span multiple
    conversation turns, images, and follow-up questions.
    """

    case_id: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    messages: list[Message] = Field(default_factory=list)
    current_diagnosis: DiagnosisResult | None = None
    image_paths: list[str] = Field(default_factory=list)
    status: Literal["active", "diagnosed", "escalated", "closed"] = "active"
    agent_state: dict = Field(default_factory=dict)

    def add_message(self, message: Message) -> None:
        """Append a message and update the timestamp."""
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)

    def get_conversation_text(self) -> str:
        """Format conversation history as a readable string for the agent."""
        lines = []
        for msg in self.messages:
            role_label = "Farmer" if msg.role == "farmer" else "Agent"
            if msg.modality == "image":
                lines.append(f"[{role_label} uploaded an image: {msg.metadata.get('image_path', 'unknown')}]")
            else:
                lines.append(f"{role_label}: {msg.content}")
        return "\n".join(lines)
