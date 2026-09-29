from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NodeInfo:
    node_id: str
    node_url: str
    max_envs: int
    total_envs: int = 0
    busy_envs: int = 0
    reserved_envs: int = 0
    idle_envs: int = 0
    resources: dict[str, Any] = field(default_factory=dict)
    # Read-only snapshot of the node's slots, as last reported by its heartbeat
    # (slot_id / world_id / busy / session_id / live_view). The node stays the
    # source of truth; the master only caches its latest self-report so it can
    # locate a slot (live view) and reconcile without re-querying /slots.
    slots: list[dict] = field(default_factory=list)
    slots_ts: float = 0.0
    prewarm_done: bool = False
    provider_name: str = ""
    os_type: str = "Ubuntu"
    action_space: str = "pyautogui"
    # Multi-world fields (additive; legacy OSWorld-only nodes leave these empty,
    # which the capability scheduler treats as "serves anything" for back-compat).
    runtimes: list[str] = field(default_factory=list)
    capabilities: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    draining: bool = False
    status: str = "healthy"
    registered_ts: float = field(default_factory=time.time)
    last_heartbeat_ts: float = field(default_factory=time.time)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @property
    def free_slots(self) -> int:
        return max(0, self.max_envs - self.busy_envs - self.reserved_envs)

    def to_dict(self) -> dict[str, Any]:
        resources = dict(self.resources or {})
        return {
            "node_id": self.node_id,
            "node_url": self.node_url,
            "max_envs": self.max_envs,
            "total_envs": self.total_envs,
            "busy_envs": self.busy_envs,
            "reserved_envs": self.reserved_envs,
            "free_slots": self.free_slots,
            "idle_envs": self.idle_envs,
            "slots": list(self.slots),
            "resources": resources,
            "qemu_count": int(resources.get("qemu_count", 0) or 0),
            "qemu_d_state_count": int(resources.get("qemu_d_state_count", 0) or 0),
            "prewarm_done": self.prewarm_done,
            "provider_name": self.provider_name,
            "os_type": self.os_type,
            "action_space": self.action_space,
            "runtimes": list(self.runtimes),
            "capabilities": dict(self.capabilities),
            "labels": dict(self.labels),
            "draining": self.draining,
            "status": self.status,
            "age_seconds": round(time.time() - self.registered_ts, 1),
            "heartbeat_age_seconds": round(time.time() - self.last_heartbeat_ts, 1),
        }


# NOTE: the former ``LeaseRecord`` was removed when the master unified onto a
# single session model. Session bookkeeping now lives in
# :class:`cluster.core.session.SessionRecord` / ``SessionStore``; the wire uses
# ``session_id`` throughout.


@dataclass
class UserQuota:
    user_id: str
    max_envs: int = 0
    max_training_envs: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "max_envs": self.max_envs,
            "max_training_envs": self.max_training_envs,
        }


@dataclass
class MasterJobRecord:
    job_id: str
    node_id: str
    node_url: str
    script_name: str
    user_id: str = "anonymous"
    status: str = "running"
    started_ts: float = field(default_factory=time.time)
    env_vars: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "node_id": self.node_id,
            "script_name": self.script_name,
            "user_id": self.user_id,
            "status": self.status,
            "age_seconds": round(time.time() - self.started_ts, 1),
            "env_vars": self.env_vars,
        }
