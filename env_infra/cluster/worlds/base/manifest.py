"""World manifest model: the benchmark-neutral declaration loaded from world.yaml.

A ``world.yaml`` declares how a world is launched (driver), what it needs
(resources) and what it can do (capabilities). The platform reads only this
file to decide scheduling — it never imports the world's code to learn its
abilities. The two orthogonal axes:

- ``capabilities.environment`` -> the *surface* (desktop/mobile/web/others)
- ``driver.kind``              -> how the adapter reaches the runtime
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class DriverKind(str, Enum):
    IN_PROCESS = "in_process"  # adapter constructs the env object in-process
    DOCKER = "docker"          # sidecar runs inside a docker image
    SUBPROCESS = "subprocess"  # sidecar runs as a local subprocess
    REMOTE = "remote"          # adapter talks to a remote/hosted service


@dataclass
class DriverConfig:
    kind: str = DriverKind.IN_PROCESS.value
    image: str | None = None
    command: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    #: {container_port: field_name} for docker-ps scanning. Part of the driver
    #: contract (which ports the container publishes and what they mean), so the
    #: world-neutral node parser never has to know any world's port layout.
    port_fields: Dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind, "image": self.image, "command": list(self.command),
            "env": dict(self.env), "port_fields": dict(self.port_fields),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "DriverConfig":
        data = data or {}
        return cls(
            kind=str(data.get("kind", DriverKind.IN_PROCESS.value)),
            image=data.get("image"),
            command=list(data.get("command", []) or []),
            env=dict(data.get("env", {}) or {}),
            port_fields={int(k): str(v) for k, v in (data.get("port_fields") or {}).items()},
        )


@dataclass
class ResourceRequest:
    cpu_cores: float = 0.0
    memory_gb: float = 0.0
    gpu: bool = False
    kvm: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"cpu_cores": self.cpu_cores, "memory_gb": self.memory_gb, "gpu": self.gpu, "kvm": self.kvm}

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "ResourceRequest":
        data = data or {}
        return cls(
            cpu_cores=float(data.get("cpu_cores", 0) or 0),
            memory_gb=float(data.get("memory_gb", 0) or 0),
            gpu=bool(data.get("gpu", False)),
            kvm=bool(data.get("kvm", False)),
        )


@dataclass
class WorldCapability:
    """What a world can do. ``environment`` is the surface."""

    environment: str = "unknown"             # desktop / mobile / web / others
    os: str = ""                             # ubuntu / android / ...
    observation: List[str] = field(default_factory=list)
    action: List[str] = field(default_factory=list)
    evaluation: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)  # surface-specific extras (proxy, account_pool, ...)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "environment": self.environment,
            "os": self.os,
            "observation": list(self.observation),
            "action": list(self.action),
            "evaluation": list(self.evaluation),
            "artifacts": list(self.artifacts),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "WorldCapability":
        data = dict(data or {})
        known = {"environment", "os", "observation", "action", "evaluation", "artifacts", "extra"}
        # Merge unknown top-level keys with an explicit nested "extra" so a
        # to_dict()->from_dict() round-trip is stable (to_dict emits a top-level
        # "extra"; omitting it from `known` would re-nest it as extra["extra"]).
        extra = {k: v for k, v in data.items() if k not in known}
        extra.update(dict(data.get("extra", {}) or {}))
        return cls(
            environment=str(data.get("environment", "unknown")),
            os=str(data.get("os", "") or ""),
            observation=list(data.get("observation", []) or []),
            action=list(data.get("action", []) or []),
            evaluation=list(data.get("evaluation", []) or []),
            artifacts=list(data.get("artifacts", []) or []),
            extra=extra,
        )

    def satisfies(self, requirement: Dict[str, Any] | None) -> bool:
        """True if this capability set covers every field in ``requirement``.

        ``requirement`` is the ``capability_requirements`` of a session request.
        - ``environment``/``os`` (scalars) must match exactly when requested.
        - ``observation``/``action``/``evaluation`` (lists) must be subsets of
          what this world provides.
        An empty/absent requirement field imposes no constraint.
        """
        if not requirement:
            return True

        env_req = requirement.get("environment")
        if env_req and env_req != self.environment:
            return False
        os_req = requirement.get("os")
        if os_req and os_req != self.os:
            return False

        for key, provided in (
            ("observation", self.observation),
            ("action", self.action),
            ("evaluation", self.evaluation),
        ):
            needed = requirement.get(key)
            if needed and not set(needed).issubset(set(provided)):
                return False
        return True


@dataclass
class WorldManifest:
    """Parsed ``world.yaml``."""

    world_id: str
    version: int = 1
    display_name: str = ""
    driver: DriverConfig = field(default_factory=DriverConfig)
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    capabilities: WorldCapability = field(default_factory=WorldCapability)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    #: dotted path "module:ClassName" of the WorldAdapter subclass for this world
    adapter: str | None = None

    @property
    def image(self) -> str | None:
        """The authoritative container image for this world.

        A world's image is part of its *driver contract* (``driver.image``), not
        a tunable in ``config_schema``. This is the single source every consumer
        (docker-ps scanning, container launch) reads, so the image can never
        drift between two declarations.
        """
        return self.driver.image

    def to_dict(self) -> Dict[str, Any]:
        return {
            "world_id": self.world_id,
            "version": self.version,
            "display_name": self.display_name or self.world_id,
            "driver": self.driver.to_dict(),
            "resources": self.resources.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "config_schema": dict(self.config_schema),
            "adapter": self.adapter,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldManifest":
        if not data or "world_id" not in data:
            # tolerate the envsession-era key name for forward/back compat
            if data and "runtime_id" in data:
                data = {**data, "world_id": data["runtime_id"]}
            else:
                raise ValueError("world manifest requires 'world_id'")
        return cls(
            world_id=str(data["world_id"]),
            version=int(data.get("version", 1) or 1),
            display_name=str(data.get("display_name", "") or ""),
            driver=DriverConfig.from_dict(data.get("driver")),
            resources=ResourceRequest.from_dict(data.get("resources")),
            capabilities=WorldCapability.from_dict(data.get("capabilities")),
            config_schema=dict(data.get("config_schema", {}) or {}),
            adapter=data.get("adapter"),
        )
