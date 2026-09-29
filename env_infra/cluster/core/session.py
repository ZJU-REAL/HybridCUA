"""SessionRecord + SessionStore: the protocol-layer session bookkeeping.

A *session* is the protocol-layer, world-polymorphic handle a runner holds onto
(``world_id`` + capabilities + unified obs/action). It is intentionally separate
from the OSWorld-era *lease* (resource-layer: ``user_id``/``job_id``/``env_id``).
On an OSWorld node a session still "rides" on a pool slot; today the convention
is ``session_id == slot_id``, recorded via ``SessionRecord.slot_id``.

The store mirrors the proven lease-lifecycle mechanisms (TTL expiry, dead-node
purge) but keeps session data structures free of any OSWorld-specific field.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SessionStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    CLOSED = "closed"
    ERROR = "error"
    EXPIRED = "expired"


def _now() -> float:
    return time.time()


def new_session_id() -> str:
    return f"sess-{uuid.uuid4().hex[:16]}"


@dataclass
class SessionRecord:
    session_id: str
    world_id: str
    node_id: str = ""
    node_url: str = ""
    slot_id: Optional[str] = None          # underlying pool slot / lease id on the node
    env_id: Optional[str] = None           # node-reported env/slot handle (eval façade)
    status: str = SessionStatus.PENDING.value
    mode: str = "eval"
    capabilities: Dict[str, Any] = field(default_factory=dict)
    task_payload: Dict[str, Any] = field(default_factory=dict)
    ttl_seconds: int = 7200
    user_id: str = "anonymous"             # kept for quota counting (not a lease coupling)
    # Eval-bookkeeping fields (the unified model absorbed the old LeaseRecord;
    # these drive quota/job/release/reconcile). Non-OSWorld worlds simply leave
    # the defaults — nothing in the protocol layer reads them.
    task_type: str = "evaluation"
    episode_id: Optional[str] = None
    job_id: Optional[str] = None
    job_script: Optional[str] = None
    source_addr: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_ts: float = field(default_factory=_now)
    last_activity_ts: float = field(default_factory=_now)

    def is_expired(self, now: Optional[float] = None) -> bool:
        if self.ttl_seconds <= 0:
            return False
        now = now if now is not None else _now()
        return (now - self.last_activity_ts) > self.ttl_seconds

    def touch(self) -> None:
        self.last_activity_ts = _now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "world_id": self.world_id,
            "node_id": self.node_id,
            "node_url": self.node_url,
            "slot_id": self.slot_id,
            "env_id": self.env_id,
            "status": self.status,
            "mode": self.mode,
            "capabilities": dict(self.capabilities),
            "ttl_seconds": self.ttl_seconds,
            "user_id": self.user_id,
            "task_type": self.task_type,
            "episode_id": self.episode_id,
            "job_id": self.job_id,
            "job_script": self.job_script,
            "source_addr": self.source_addr,
            "age_seconds": round(_now() - self.created_ts, 1),
            "idle_seconds": round(_now() - self.last_activity_ts, 1),
            "metadata": dict(self.metadata),
        }


class SessionStore:
    """Thread-safe in-memory session table with TTL and node-scoped queries."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: Dict[str, SessionRecord] = {}

    def create(self, record: SessionRecord) -> SessionRecord:
        with self._lock:
            self._sessions[record.session_id] = record
            return record

    def get(self, session_id: str) -> Optional[SessionRecord]:
        with self._lock:
            return self._sessions.get(session_id)

    def delete(self, session_id: str) -> Optional[SessionRecord]:
        with self._lock:
            return self._sessions.pop(session_id, None)

    def touch(self, session_id: str) -> None:
        with self._lock:
            rec = self._sessions.get(session_id)
            if rec:
                rec.touch()

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def list_all(
        self,
        *,
        world_id: Optional[str] = None,
        node_id: Optional[str] = None,
        status: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[SessionRecord]:
        with self._lock:
            out = list(self._sessions.values())
        if world_id is not None:
            out = [s for s in out if s.world_id == world_id]
        if node_id is not None:
            out = [s for s in out if s.node_id == node_id]
        if status is not None:
            out = [s for s in out if s.status == status]
        if user_id is not None:
            out = [s for s in out if s.user_id == user_id]
        return out

    def get_by_node(self, node_id: str) -> List[SessionRecord]:
        return self.list_all(node_id=node_id)

    def count_by_user(self, user_id: str) -> tuple[int, int]:
        """Return (total, training) session counts for a user — quota accounting.

        Replaces the old ``_count_user_leases`` now that sessions are the single
        bookkeeping model.
        """
        total = 0
        training = 0
        with self._lock:
            for s in self._sessions.values():
                if s.user_id == user_id:
                    total += 1
                    if s.task_type == "training":
                        training += 1
        return total, training

    def expire_stale(self, now: Optional[float] = None) -> List[SessionRecord]:
        """Mark expired sessions and return them (caller tears down resources).

        Mirrors the master's lease reaper: scan ``last_activity_ts`` against TTL.
        """
        now = now if now is not None else _now()
        expired: List[SessionRecord] = []
        with self._lock:
            for rec in self._sessions.values():
                if rec.status in (SessionStatus.CLOSED.value, SessionStatus.EXPIRED.value):
                    continue
                if rec.is_expired(now):
                    rec.status = SessionStatus.EXPIRED.value
                    expired.append(rec)
        return expired

    def purge_for_node(self, node_id: str) -> List[SessionRecord]:
        """Remove all sessions on a dead node; return what was removed."""
        with self._lock:
            victims = [s for s in self._sessions.values() if s.node_id == node_id]
            for s in victims:
                self._sessions.pop(s.session_id, None)
        return victims
