"""core: benchmark-neutral platform models, registry, session store, scheduler.

This layer holds the control-plane data structures and logic. It MUST NOT
import any benchmark SDK or any world adapter — only the protocol envelopes and
the world manifest model.
"""

from __future__ import annotations

from cluster.core.registry import WorldRegistry
from cluster.core.scheduler import (
    LeastLoadedScheduler,
    RoundRobinScheduler,
    Scheduler,
    create_scheduler,
    node_can_serve,
)
from cluster.core.session import SessionRecord, SessionStatus, SessionStore

__all__ = [
    "WorldRegistry",
    "SessionRecord",
    "SessionStatus",
    "SessionStore",
    "Scheduler",
    "LeastLoadedScheduler",
    "RoundRobinScheduler",
    "create_scheduler",
    "node_can_serve",
]
