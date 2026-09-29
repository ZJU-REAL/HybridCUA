"""worlds.base: the platform-level, benchmark-neutral world contract.

This is the "base" of the worlds package — the framework every concrete world
(``cluster/worlds/<id>/``) builds on. It contains the single interface every
world must implement (:class:`WorldAdapter`), the surface base classes that fix
each surface's observation/action dialect, the world manifest model, and the
HTTP sidecar server that exposes an adapter over the standard EnvSession REST
endpoints.

Nothing in this package may import a benchmark SDK.
"""

from __future__ import annotations

from cluster.worlds.base.adapter import WorldAdapter
from cluster.worlds.base.manifest import (
    DriverConfig,
    DriverKind,
    ResourceRequest,
    WorldCapability,
    WorldManifest,
)

__all__ = [
    "WorldAdapter",
    "WorldManifest",
    "WorldCapability",
    "DriverConfig",
    "DriverKind",
    "ResourceRequest",
]
