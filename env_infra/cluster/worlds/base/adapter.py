"""The single lifecycle interface every world must implement.

A :class:`WorldAdapter` wraps exactly one live environment session of one
benchmark ("world"). The platform (node pool, sidecar server) drives an adapter
purely through this contract and never touches the underlying benchmark.

World authors do NOT subclass this directly — they subclass one of the surface
base classes in :mod:`cluster.worlds.base.surfaces` (DesktopWorld / MobileWorld /
WebWorld / OthersWorld), which fix the observation/action dialect for that
surface and leave only thin "call the official API" translations to implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from cluster.schemas import Action, EvaluationResult, Observation, StepResponse


class WorldAdapter(ABC):
    """Standard lifecycle interface for a single environment session."""

    #: Surface name this adapter belongs to. Surface base classes set it; it is
    #: matched against ``capabilities.environment`` in the world manifest.
    surface: str = "unknown"

    @abstractmethod
    def reset(self, task_payload: Dict[str, Any]) -> Observation:
        """Reset the environment with the given task and return the first obs."""
        ...

    @abstractmethod
    def step(self, action: Action, pause: float | None = None) -> StepResponse:
        """Execute one action and return the resulting observation/done/info.

        ``pause`` is the post-action settle (seconds) requested by the caller;
        ``None`` means use the adapter's own default. Worlds with no settable
        settle may ignore it.
        """
        ...

    @abstractmethod
    def observe(self) -> Observation:
        """Return the current observation without taking an action."""
        ...

    @abstractmethod
    def evaluate(self) -> EvaluationResult:
        """Evaluate task completion for the current episode."""
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the adapter and underlying environment are healthy."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release all resources held by this adapter."""
        ...

    def get_info(self) -> Dict[str, Any]:
        """Optional: return world-specific metadata (surface, versions, ...)."""
        return {"surface": self.surface}

    def liveness(self) -> bool:
        """Optional deep pre-reuse probe, asked before handing an idle slot to a
        new session. Defaults to :meth:`health_check`; a world whose underlying
        resource can die without the health endpoint noticing (e.g. a Docker
        container that exited) overrides this to verify the resource is truly
        alive. Must be cheap/non-blocking — heavy checks belong behind a cache."""
        return self.health_check()

    def resource_id(self) -> Optional[str]:
        """Optional stable id of the underlying resource this adapter holds
        (e.g. a docker container id). Used by the pool's orphan reclaim to spare
        live resources without the pool knowing what kind of id it is. Defaults
        to None (no reclaimable external resource)."""
        return None
