"""client.osworld: OSWorld benchmark client + eval task source."""
from __future__ import annotations

from cluster.client.osworld.eval import (
    KimiOSWorldEvalSource,
    OSWorldEvalSource,
)
from cluster.client.osworld.session_client import (
    OSWorldRemoteClient,
    OSWorldSessionClient,
    to_cluster_action,
    to_osworld_obs,
)

__all__ = [
    "OSWorldSessionClient",
    "OSWorldRemoteClient",
    "OSWorldEvalSource",
    "KimiOSWorldEvalSource",
    "to_osworld_obs",
    "to_cluster_action",
]
