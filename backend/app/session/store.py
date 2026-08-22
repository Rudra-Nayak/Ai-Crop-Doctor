"""
Session storage abstraction.

Defines an abstract SessionStore interface and an in-memory implementation.
To add persistent storage (e.g. Supabase PostgreSQL), create a new subclass
of SessionStore — no changes needed to agents, services, or API routes.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from app.session.models import Case, Message


class SessionStore(ABC):
    """Abstract session store — swap implementations without changing callers."""

    @abstractmethod
    async def create_case(self) -> Case:
        """Create a new diagnostic case and return it."""
        ...

    @abstractmethod
    async def get_case(self, case_id: str) -> Case | None:
        """Retrieve a case by ID, or None if not found."""
        ...

    @abstractmethod
    async def update_case(self, case: Case) -> None:
        """Persist updates to an existing case."""
        ...

    @abstractmethod
    async def add_message(self, case_id: str, message: Message) -> Case | None:
        """Add a message to a case. Returns updated case or None if not found."""
        ...

    @abstractmethod
    async def list_active_cases(self) -> list[Case]:
        """List all cases with status 'active'."""
        ...


class InMemorySessionStore(SessionStore):
    """
    Dictionary-backed session store.

    Good for hackathon MVP and development. Not persistent across restarts.
    Replace with SupabaseSessionStore for production.
    """

    def __init__(self) -> None:
        self._cases: dict[str, Case] = {}

    async def create_case(self) -> Case:
        case_id = str(uuid.uuid4())
        case = Case(
            case_id=case_id,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            status="active",
        )
        self._cases[case_id] = case
        return case

    async def get_case(self, case_id: str) -> Case | None:
        return self._cases.get(case_id)

    async def update_case(self, case: Case) -> None:
        if case.case_id in self._cases:
            case.updated_at = datetime.now(timezone.utc)
            self._cases[case.case_id] = case

    async def add_message(self, case_id: str, message: Message) -> Case | None:
        case = self._cases.get(case_id)
        if case is None:
            return None
        case.add_message(message)
        return case

    async def list_active_cases(self) -> list[Case]:
        return [c for c in self._cases.values() if c.status == "active"]


class SupabaseSessionStore(SessionStore):
    """
    Supabase PostgreSQL-backed session store.

    Persists diagnostic consultation history, messages, and structured prescriptions
    to table `diagnostic_cases` in Supabase PostgreSQL.
    """

    def __init__(self, url: str, key: str) -> None:
        from supabase import create_client
        self._client = create_client(url, key)

    async def create_case(self) -> Case:
        case_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        case = Case(
            case_id=case_id,
            created_at=now,
            updated_at=now,
            status="active",
        )
        data = {
            "case_id": case_id,
            "status": "active",
            "messages": [],
            "diagnosis": None,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }
        self._client.table("diagnostic_cases").insert(data).execute()
        return case

    async def get_case(self, case_id: str) -> Case | None:
        try:
            res = self._client.table("diagnostic_cases").select("*").eq("case_id", case_id).execute()
            if not res.data:
                return None
            row = res.data[0]
            # Deserialize row to Case object
            case = Case.model_validate(row)
            return case
        except Exception:
            return None

    async def update_case(self, case: Case) -> None:
        try:
            case.updated_at = datetime.now(timezone.utc)
            data = case.model_dump(mode="json")
            self._client.table("diagnostic_cases").update(data).eq("case_id", case.case_id).execute()
        except Exception:
            pass

    async def add_message(self, case_id: str, message: Message) -> Case | None:
        case = await self.get_case(case_id)
        if case is None:
            return None
        case.add_message(message)
        await self.update_case(case)
        return case

    async def list_active_cases(self) -> list[Case]:
        try:
            res = self._client.table("diagnostic_cases").select("*").eq("status", "active").execute()
            return [Case.model_validate(row) for row in res.data or []]
        except Exception:
            return []
