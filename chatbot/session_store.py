"""
chatbot/session_store.py
─────────────────────────
A minimal, thread-safe, in-memory store for `ChatSession` objects.

Flask request handlers are stateless by design, but `chatbot.bot.ChatSession`
is a stateful, multi-turn conversation object (it tracks collected slots,
clarification turns, the last assessment, etc.). This module is the seam
that bridges the two: the frontend holds a `session_id` (a UUID4 string)
and sends it with every `/api/chat/message` call; this store maps that id
back to the in-process `ChatSession` instance.

Scope and limitations (by design, documented rather than hidden):
  - Single-process only. If you run multiple Gunicorn/uWSGI workers, each
    worker has its own session dict, and a user's follow-up message could
    land on a different worker with no memory of their session. Fine for a
    local demo / single-process deployment; for production, swap the `dict`
    below for Redis (or any shared key-value store) keyed the same way.
  - No persistence across restarts — sessions are purely in memory.
  - Idle sessions are swept on access based on SESSION_TTL_MINUTES so the
    process does not grow unbounded over a long-running server.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict

from chatbot.bot import ChatSession
from config import SESSION_TTL_MINUTES


@dataclass
class _Entry:
    session: ChatSession
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SessionStore:
    """Thread-safe in-memory map of session_id -> ChatSession."""

    def __init__(self, ttl_minutes: int = SESSION_TTL_MINUTES):
        self._sessions: Dict[str, _Entry] = {}
        self._lock = threading.Lock()
        self._ttl = timedelta(minutes=max(1, ttl_minutes))

    def _sweep_expired(self) -> None:
        cutoff = datetime.now(timezone.utc) - self._ttl
        expired = [sid for sid, entry in self._sessions.items() if entry.last_seen < cutoff]
        for sid in expired:
            self._sessions.pop(sid, None)

    def create(self) -> str:
        """Start a brand new session and return its id."""
        with self._lock:
            self._sweep_expired()
            session_id = uuid.uuid4().hex
            self._sessions[session_id] = _Entry(session=ChatSession())
            return session_id

    def get(self, session_id: str | None) -> tuple[str, ChatSession]:
        """
        Return (session_id, ChatSession) for an existing id, or transparently
        start a new session if the id is missing/unknown/expired. The caller
        should always echo the returned session_id back to the client.
        """
        with self._lock:
            self._sweep_expired()
            if session_id and session_id in self._sessions:
                entry = self._sessions[session_id]
                entry.last_seen = datetime.now(timezone.utc)
                return session_id, entry.session

            new_id = uuid.uuid4().hex
            self._sessions[new_id] = _Entry(session=ChatSession())
            return new_id, self._sessions[new_id].session

    def reset(self, session_id: str) -> None:
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is not None:
                entry.session.reset()
                entry.last_seen = datetime.now(timezone.utc)

    def drop(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def active_count(self) -> int:
        with self._lock:
            self._sweep_expired()
            return len(self._sessions)


# A single process-wide store, imported by api/routes.py.
session_store = SessionStore()
