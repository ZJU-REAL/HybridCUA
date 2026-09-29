"""cluster.client — remote clients + eval runner for the multi-world platform.

Layered ``base/ + osworld/ + mobileworld/`` (mirrors ``cluster/worlds/``):

- :mod:`cluster.client.base` — benchmark-neutral :class:`ClusterSessionClient`
  (the unified /v1/sessions protocol) and :class:`EvalRunner` +
  :class:`EvalTaskSource` (the eval orchestration + task-source ABC).
- :mod:`cluster.client.osworld` — OSWorld session client + eval task source.
- :mod:`cluster.client.mobileworld` — MobileWorld session client + eval task source.

Top-level names below are backwards-compatible aliases so existing imports
(``from cluster.client import EvalRunner, OSWorldRemoteClient, ...``) keep
working; new code should import from the layered subpackages.
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
from cluster.client.mobileworld.session_client import (
    MobileWorldRemoteClient,
    MobileWorldSessionClient,
)
from cluster.client.osworld.session_client import (
    OSWorldRemoteClient,
    OSWorldSessionClient,
)

__all__ = [
    # base
    "ClusterSessionClient",
    "EvalRunner",
    "EvalTaskSource",
    "Agent",
    "AgentFactory",
    "EnvFactory",
    "add_common_args",
    "setup_logging",
    # osworld
    "OSWorldSessionClient",
    "OSWorldRemoteClient",
    # mobileworld
    "MobileWorldSessionClient",
    "MobileWorldRemoteClient",
]
