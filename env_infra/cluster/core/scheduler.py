"""Capability-based node selection for the EnvSession control plane.

The scheduler picks a node for a session request by filtering on health, the
worlds a node hosts, and capability/resource fit — never by hardcoding a
benchmark name. ``node_can_serve`` is the world-neutral predicate; the legacy
free-slots-only behavior is preserved as a fallback when a request carries no
runtime/capability constraints (back-compat with the OSWorld ``/allocate`` path).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from cluster.worlds.base.manifest import ResourceRequest, WorldCapability


class _NodeLike(Protocol):
    node_id: str
    status: str
    draining: bool
    free_slots: int
    total_envs: int
    runtimes: List[str]
    capabilities: Dict[str, Any]
    resources: Dict[str, Any]


def _node_resources_ok(node: _NodeLike, req: Dict[str, Any] | None) -> bool:
    """Best-effort resource fit. Missing node metrics are treated as 'ok'."""
    if not req:
        return True
    want = ResourceRequest.from_dict(req)
    res = node.resources or {}
    if want.kvm and res.get("kvm") is False:
        return False
    if want.gpu and res.get("gpu") is False:
        return False
    # cpu/memory are advisory; only reject when the node reports a hard ceiling
    free_cpu = res.get("free_cpu_cores")
    if free_cpu is not None and want.cpu_cores and float(free_cpu) < want.cpu_cores:
        return False
    free_mem = res.get("free_memory_gb")
    if free_mem is not None and want.memory_gb and float(free_mem) < want.memory_gb:
        return False
    return True


def node_can_serve(node: _NodeLike, request: Dict[str, Any] | None) -> bool:
    """True if ``node`` can host the requested world + capabilities + resources.

    ``request`` keys (all optional):
      - ``runtime``: the world_id; node must host it (empty node.runtimes = legacy
        node that hosts anything).
      - ``capability_requirements``: matched via ``WorldCapability.satisfies`` against
        the node's advertised capabilities for that world.
      - ``resources``: matched against node.resources.
    """
    if node.status != "healthy" or node.draining or node.free_slots <= 0:
        return False
    if not request:
        return True

    runtime = request.get("runtime")
    node_runtimes = list(getattr(node, "runtimes", []) or [])
    if runtime and node_runtimes and runtime not in node_runtimes:
        return False

    cap_req = request.get("capability_requirements")
    if cap_req:
        node_caps = getattr(node, "capabilities", {}) or {}
        # node.capabilities maps world_id -> capability dict; if absent, no info => allow
        cap_dict = node_caps.get(runtime) if runtime else None
        if cap_dict is None and len(node_caps) == 1:
            cap_dict = next(iter(node_caps.values()))
        if cap_dict is not None:
            if not WorldCapability.from_dict(cap_dict).satisfies(cap_req):
                return False

    if not _node_resources_ok(node, request.get("resources")):
        return False
    return True


class Scheduler:
    def select_node(self, nodes: List[_NodeLike], request: Optional[Dict[str, Any]] = None) -> Optional[_NodeLike]:
        raise NotImplementedError


class LeastLoadedScheduler(Scheduler):
    def select_node(self, nodes: List[_NodeLike], request: Optional[Dict[str, Any]] = None) -> Optional[_NodeLike]:
        candidates = [n for n in nodes if node_can_serve(n, request)]
        if not candidates:
            return None
        return max(candidates, key=lambda n: (n.free_slots, -n.total_envs))


class RoundRobinScheduler(Scheduler):
    def __init__(self) -> None:
        self._cursor = 0

    def select_node(self, nodes: List[_NodeLike], request: Optional[Dict[str, Any]] = None) -> Optional[_NodeLike]:
        candidates = sorted(
            [n for n in nodes if node_can_serve(n, request)], key=lambda n: n.node_id
        )
        if not candidates:
            return None
        node = candidates[self._cursor % len(candidates)]
        self._cursor += 1
        return node


SCHEDULERS = {
    "least-loaded": LeastLoadedScheduler,
    "round-robin": RoundRobinScheduler,
}


def create_scheduler(name: str = "least-loaded") -> Scheduler:
    cls = SCHEDULERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown scheduler: {name!r}. Available: {list(SCHEDULERS)}")
    return cls()
