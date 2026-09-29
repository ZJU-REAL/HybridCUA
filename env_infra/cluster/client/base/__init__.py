"""client.base: the platform-level, benchmark-neutral client + eval contract.

The "base" of the client package — the framework every concrete benchmark
client (``cluster/client/<id>/``) builds on. It contains the generic
:class:`ClusterSessionClient` (acquire/reset/step/observe/evaluate/release over
the unified /v1/sessions protocol) and the benchmark-neutral eval orchestrator
(:class:`EvalRunner`) plus its task-source ABC (:class:`EvalTaskSource`).

Nothing in this package may import a benchmark SDK or a benchmark-specific
client. Mirrors ``cluster/worlds/base/``.
"""
from __future__ import annotations

from cluster.client.base.eval_worker import (
    Agent,
    AgentFactory,
    EnvFactory,
    EvalRunner,
    EvalTaskSource,
    add_common_args,
    setup_logging,
)
from cluster.client.base.session_client import ClusterSessionClient

__all__ = [
    "ClusterSessionClient",
    "EvalRunner",
    "EvalTaskSource",
    "Agent",
    "AgentFactory",
    "EnvFactory",
    "add_common_args",
    "setup_logging",
]
