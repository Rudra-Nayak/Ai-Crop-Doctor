"""
Session manager.

High-level API for session lifecycle management.
API routes call this — never the store directly.
"""

from __future__ import annotations

from app.session.models import Case, DiagnosisResult, Message
from app.session.store import SessionStore


class SessionManager:
    """Manages diagnostic case lifecycle."""

    def __init__(self, store: SessionStore) -> None:
        self._store = store

    async def start_case(self) -> Case:
        """Create a new diagnostic case."""
        return await self._store.create_case()

    async def get_case(self, case_id: str) -> Case | None:
        """Retrieve an existing case."""
        return await self._store.get_case(case_id)

    async def get_or_create_case(self, case_id: str | None) -> Case:
        """Get existing case by ID, or create a new one if ID is None / not found."""
        if case_id:
            case = await self._store.get_case(case_id)
            if case is not None:
                return case
        return await self._store.create_case()

    async def add_farmer_message(
        self,
        case_id: str,
        content: str,
        modality: str = "text",
        metadata: dict | None = None,
    ) -> Case | None:
        """Record a farmer's message in the case."""
        msg = Message(
            role="farmer",
            content=content,
            modality=modality,
            metadata=metadata or {},
        )
        return await self._store.add_message(case_id, msg)

    async def add_agent_message(
        self,
        case_id: str,
        content: str,
        metadata: dict | None = None,
    ) -> Case | None:
        """Record the agent's response in the case."""
        msg = Message(
            role="agent",
            content=content,
            modality="text",
            metadata=metadata or {},
        )
        return await self._store.add_message(case_id, msg)

    async def record_image(self, case_id: str, image_path: str) -> Case | None:
        """Record that the farmer uploaded an image."""
        case = await self._store.get_case(case_id)
        if case is None:
            return None
        case.image_paths.append(image_path)
        # Also add as a message for conversation history
        msg = Message(
            role="farmer",
            content="[Uploaded a crop image]",
            modality="image",
            metadata={"image_path": image_path},
        )
        case.add_message(msg)
        await self._store.update_case(case)
        return case

    async def finalize_diagnosis(
        self,
        case_id: str,
        diagnosis: DiagnosisResult,
    ) -> Case | None:
        """Set the final diagnosis and close the case."""
        case = await self._store.get_case(case_id)
        if case is None:
            return None
        case.current_diagnosis = diagnosis
        case.status = "escalated" if diagnosis.is_escalated else "diagnosed"
        await self._store.update_case(case)
        return case

    async def close_case(self, case_id: str) -> Case | None:
        """Mark a case as closed."""
        case = await self._store.get_case(case_id)
        if case is None:
            return None
        case.status = "closed"
        await self._store.update_case(case)
        return case
