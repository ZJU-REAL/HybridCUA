"""Node-side EnvSession API: serve /v1/sessions/* backed by RuntimePool(s).

This is the world-neutral execution surface on a node. It holds one
:class:`~cluster.node.pool.RuntimePool` per world the node hosts, each
built from the world's manifest via the adapter's ``make_driver``. A session is
a slot reservation in the matching pool.

It is mounted as a Flask blueprint on the world-neutral node server
(:mod:`cluster.node.world_server`).
"""

from __future__ import annotations

import importlib
import logging
import threading
import uuid
from typing import Any, Callable, Dict, Optional

from flask import Blueprint, jsonify, request

from cluster.core.registry import WorldRegistry
from cluster.node.pool import NodeCapacity, RuntimePool, RuntimePoolExhausted
from cluster.schemas import Action
from cluster.worlds.base.manifest import DriverKind, WorldManifest

logger = logging.getLogger("cluster.node.sessions")


def _load_adapter_driver(manifest: WorldManifest, config: Dict[str, Any]) -> "RuntimeDriver":
    """Resolve ``manifest.adapter`` ("module:Class") and return a RuntimeDriver.

    The adapter class exposes ``make_driver(config) -> RuntimeDriver`` which
    encapsulates all lifecycle hooks (create, create_batch, adopt).

    Importing the adapter module happens here (node side), never on the master.
    """
    from cluster.node.pool import RuntimeDriver

    if not manifest.adapter:
        raise ValueError(f"world {manifest.world_id!r} manifest has no 'adapter'")
    mod_path, _, cls_name = manifest.adapter.partition(":")
    module = importlib.import_module(mod_path)
    cls = getattr(module, cls_name)
    if not hasattr(cls, "make_driver"):
        raise ValueError(f"adapter {manifest.adapter!r} has no make_driver()")
    merged = {
        k: v["default"]
        for k, v in manifest.config_schema.items()
        if isinstance(v, dict) and "default" in v
    }
    merged.update(config)
    # Inject the manifest image ONLY for docker-driver worlds, which need it to
    # launch their container. in_process worlds (e.g. OSWorld) pass this config
    # straight to their env constructor, where an unknown 'image' key would raise
    # — so the image stays a docker-driver concern, not a global config key.
    if manifest.image and manifest.driver.kind == DriverKind.DOCKER.value:
        merged["image"] = manifest.image
    result = cls.make_driver(merged)
    if isinstance(result, RuntimeDriver):
        return result
    # Backward compat: if make_driver returns a bare callable, wrap it
    return RuntimeDriver(create=result)


class NodeSessionManager:
    """Owns one RuntimePool per hosted world and maps session_id -> world pool."""

    def __init__(
        self,
        registry: WorldRegistry,
        *,
        world_ids: list[str],
        max_slots_per_world: int = 1,
        world_config: Optional[Dict[str, Dict[str, Any]]] = None,
        idle_ttl_seconds: int = 0,
        prewarm: int = 0,
        prewarm_concurrency: int = 2,
        scale_buffer: int = 0,
        scale_interval: int = 5,
    ) -> None:
        self.registry = registry
        self.world_ids = world_ids
        self.world_config = world_config or {}
        self._lock = threading.RLock()
        self._pools: Dict[str, RuntimePool] = {}
        self._session_world: Dict[str, str] = {}
        # One node-wide slot budget shared across every world's pool, so a busy
        # world can use the whole node instead of a static per-world slice.
        self._capacity = NodeCapacity(max_slots_per_world * max(1, len(world_ids)))
        for wid in world_ids:
            manifest = registry.get(wid)
            driver = _load_adapter_driver(manifest, self.world_config.get(wid, {}))
            pool = RuntimePool(
                driver,
                world_id=wid,
                max_slots=max_slots_per_world,
                capacity=self._capacity,
                idle_ttl_seconds=idle_ttl_seconds,
                prewarm=prewarm,
                prewarm_concurrency=prewarm_concurrency,
                scale_buffer=scale_buffer,
                scale_interval=scale_interval,
            )
            self._pools[wid] = pool

    def create_session(self, world_id: str, session_id: Optional[str] = None) -> str:
        if world_id not in self._pools:
            raise KeyError(f"node does not host world {world_id!r}")
        session_id = session_id or f"sess-{uuid.uuid4().hex[:16]}"
        self._pools[world_id].acquire(session_id)
        with self._lock:
            self._session_world[session_id] = world_id
        return session_id

    def _pool_for(self, session_id: str) -> RuntimePool:
        with self._lock:
            wid = self._session_world.get(session_id)
        if wid is None:
            raise KeyError(f"unknown session {session_id!r}")
        return self._pools[wid]

    def reset(self, session_id: str, task_payload: Dict[str, Any]):
        return self._pool_for(session_id).reset(session_id, task_payload)

    def step(self, session_id: str, action: Action, pause: float | None = None):
        return self._pool_for(session_id).step(session_id, action, pause)

    def observe(self, session_id: str):
        return self._pool_for(session_id).observe(session_id)

    def evaluate(self, session_id: str):
        return self._pool_for(session_id).evaluate(session_id)

    def close(self, session_id: str) -> None:
        pool = self._pool_for(session_id)
        pool.release(session_id, close=False)
        with self._lock:
            self._session_world.pop(session_id, None)

    def status(self) -> Dict[str, Any]:
        return {wid: pool.status() for wid, pool in self._pools.items()}

    def set_max_envs(self, max_envs: int) -> int:
        """Set the node's TOTAL slot cap, shared across all hosted worlds.

        The master speaks in node-wide ``max_envs``; the node's per-world pools
        share one :class:`NodeCapacity` budget, so any single world can use the
        whole node instead of a static per-world slice. Returns the cap applied.
        """
        self._capacity.set_max(max_envs)
        return max_envs

    def reconcile_sessions(self, valid_ids: set[str], *, min_age_seconds: float = 0.0) -> int:
        """Release (keep-warm) sessions the master no longer recognizes.

        Fans the master's authoritative ``valid_ids`` set across every pool and
        prunes the manager's ``_session_world`` map for any session that is no
        longer bound in any pool, so a later session-id reuse can't collide.
        Returns the number of sessions released. Node-push counterpart of the
        master's pull reconcile — see the platform recovery design.
        """
        with self._lock:
            pools = list(self._pools.values())
        released = sum(
            p.release_orphan_sessions(valid_ids, min_age_seconds=min_age_seconds) for p in pools
        )
        with self._lock:
            for sid in list(self._session_world):
                if sid not in valid_ids and not any(p.has_session(sid) for p in pools):
                    self._session_world.pop(sid, None)
        return released

    def reap_idle(self) -> int:
        """Close idle slots past their TTL across all pools. Returns count reaped.

        This is how a session orphaned by a master restart (unbound keep-warm by
        the node-push reclaim, never re-acquired) is eventually reclaimed: the
        master deliberately does NOT tear orphans down, so idle TTL on the node is
        the safety net that frees a truly-abandoned environment.
        """
        with self._lock:
            pools = list(self._pools.values())
        return sum(p.reap_idle() for p in pools)

    def reap_unhealthy_idle(self) -> int:
        """Probe idle slots across all pools and drop ones whose adapter is dead.

        World-neutral self-healing: reclaims phantom slots (e.g. a docker
        container that vanished) so the slot accounting re-converges on the real
        resources. Returns the total number of slots reaped.
        """
        with self._lock:
            pools = list(self._pools.values())
        return sum(p.reap_unhealthy_idle() for p in pools)

    def release_stale_busy(self) -> int:
        """Free slots stuck busy with no session across all pools. Normally this
        runs lazily inside acquire(); calling it from the maintenance loop ensures
        a fully-loaded node (no new acquires) still reclaims leaked busy slots
        instead of letting them accumulate. Returns the count released.
        """
        with self._lock:
            pools = list(self._pools.values())
        return sum(p._release_stale_busy() for p in pools)

    def reclaim_orphans(self, *, grace: float) -> int:
        """Ask each world's driver to free leaked resources not owned by any
        slot (e.g. docker containers left after a crash). World-neutral: the
        node only fans the call out; whether a world has orphans and how to
        reclaim them lives entirely in its driver hook. Returns count freed.
        """
        with self._lock:
            pools = list(self._pools.values())
        return sum(p.reclaim_orphans(grace) for p in pools)

    def create_slots(self, world_id: str, count: int, concurrency: int = 8) -> int:
        """Create idle slots (no session bound) in parallel. Returns count created."""
        if world_id not in self._pools:
            raise KeyError(f"node does not host world {world_id!r}")
        return self._pools[world_id].create_slots(count, concurrency=concurrency)

    def _world_for_slot(self, slot_id: str) -> str:
        """Resolve which hosted world owns ``slot_id`` (slot ids are unique across
        pools). Lets admin ops target a slot without the caller knowing its world."""
        for wid, pool in self._pools.items():
            if slot_id in pool._slots:
                return wid
        raise KeyError(f"unknown slot {slot_id!r}")

    def delete_slot(self, slot_id: str, world_id: Optional[str] = None) -> None:
        """Admin: destroy a specific slot. ``world_id`` is optional — when omitted
        it is resolved from the slot id."""
        world_id = world_id or self._world_for_slot(slot_id)
        if world_id not in self._pools:
            raise KeyError(f"node does not host world {world_id!r}")
        self._pools[world_id].delete_slot(slot_id)

    def delete_all_slots(self, world_id: Optional[str] = None) -> int:
        """Admin: destroy all slots. If world_id given, only that world; else all."""
        if world_id:
            if world_id not in self._pools:
                raise KeyError(f"node does not host world {world_id!r}")
            return self._pools[world_id].delete_all()
        return sum(pool.delete_all() for pool in self._pools.values())

    def restart_slot(self, slot_id: str, world_id: Optional[str] = None) -> Dict[str, Any]:
        """Admin: restart a slot's adapter. ``world_id`` optional — resolved from
        the slot id when omitted."""
        world_id = world_id or self._world_for_slot(slot_id)
        if world_id not in self._pools:
            raise KeyError(f"node does not host world {world_id!r}")
        return self._pools[world_id].restart_slot(slot_id)

    def list_slots(self) -> Dict[str, list]:
        """Return all slot statuses grouped by world."""
        return {wid: pool.list_slots() for wid, pool in self._pools.items()}

    def health_check(self) -> Dict[str, Dict[str, bool]]:
        """Run health_check on all adapters across all pools."""
        return {wid: pool.health_check() for wid, pool in self._pools.items()}

    def get_slot_live_view(self, slot_id: str) -> Optional[Dict[str, Any]]:
        """Return live_view info for a slot, or None if not available."""
        # Resolve the slot under the lock, but call get_info() OUTSIDE it — an
        # adapter's get_info() may touch its underlying resource and block
        # (e.g. during a reset rebuild); holding pool._lock across it would
        # stall the pool.
        for pool in self._pools.values():
            with pool._lock:
                slot = pool._slots.get(slot_id)
            if slot:
                try:
                    info = slot.adapter.get_info()
                    return info.get("live_view")
                except Exception:
                    return None
        return None

    def capabilities(self) -> Dict[str, Any]:
        return {wid: self.registry.get(wid).capabilities.to_dict() for wid in self.world_ids}


def create_node_sessions_blueprint(manager: NodeSessionManager) -> Blueprint:
    """Build a Flask blueprint exposing /v1/sessions/* on the node."""
    bp = Blueprint("node_sessions", __name__)

    def _body() -> Dict[str, Any]:
        return request.get_json(force=True, silent=True) or {}

    @bp.post("/v1/sessions")
    def create():  # type: ignore[unused-ignore]
        body = _body()
        world_id = body.get("runtime") or body.get("world_id")
        session_id = body.get("session_id")
        if not world_id:
            return jsonify({"ok": False, "error": "runtime (world_id) required"}), 400
        try:
            sid = manager.create_session(world_id, session_id)
            return jsonify({"ok": True, "session_id": sid, "world_id": world_id})
        except RuntimePoolExhausted as exc:
            return jsonify({"ok": False, "error": str(exc)}), 503
        except KeyError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404

    @bp.post("/v1/sessions/<session_id>/reset")
    def reset(session_id):  # type: ignore[unused-ignore]
        try:
            obs = manager.reset(session_id, _body().get("task_payload", {}) or {})
            return jsonify({"ok": True, "observation": obs.to_dict()})
        except KeyError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404

    @bp.post("/v1/sessions/<session_id>/step")
    def step(session_id):  # type: ignore[unused-ignore]
        try:
            body = _body()
            resp = manager.step(session_id, Action.from_dict(body.get("action")), body.get("pause"))
            return jsonify({"ok": True, **resp.to_dict()})
        except KeyError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        except ValueError as exc:
            # malformed/missing action body — reject cleanly, not as a 500
            return jsonify({"ok": False, "error": str(exc)}), 400

    @bp.post("/v1/sessions/<session_id>/observe")
    def observe(session_id):  # type: ignore[unused-ignore]
        try:
            obs = manager.observe(session_id)
            return jsonify({"ok": True, "observation": obs.to_dict()})
        except KeyError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404

    @bp.post("/v1/sessions/<session_id>/evaluate")
    def evaluate(session_id):  # type: ignore[unused-ignore]
        try:
            result = manager.evaluate(session_id)
            return jsonify({"ok": True, **result.to_dict()})
        except KeyError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404

    @bp.delete("/v1/sessions/<session_id>")
    def close(session_id):  # type: ignore[unused-ignore]
        try:
            manager.close(session_id)
            return jsonify({"ok": True})
        except KeyError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404

    def _world_id_from(body: dict) -> str | None:
        """Resolve the target world: explicit runtime/world_id, else the node's
        sole hosted world (unambiguous), else None. No hardcoded benchmark name."""
        wid = body.get("runtime") or body.get("world_id")
        if wid:
            return wid
        return manager.world_ids[0] if len(manager.world_ids) == 1 else None

    @bp.post("/env/create")
    def env_create():  # type: ignore[unused-ignore]
        body = _body()
        world_id = _world_id_from(body)
        if not world_id:
            return jsonify({"ok": False, "error": "world_id (runtime) required"}), 400
        count = max(1, min(int(body.get("count", 1)), 50))
        concurrency = max(1, min(int(body.get("concurrency", 8)), 16))
        try:
            pool = manager._pools.get(world_id)
            if pool is None:
                return jsonify({"ok": False, "error": f"node does not host world {world_id!r}"}), 404

            import threading as _threading
            def _bg_create():
                try:
                    manager.create_slots(world_id, count, concurrency=concurrency)
                except Exception as exc:
                    logger.warning("Background env_create for %s failed: %s", world_id, exc)
            _threading.Thread(target=_bg_create, daemon=True).start()
            return jsonify({"ok": True, "created": 0, "world_id": world_id, "async": True})
        except KeyError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404

    @bp.post("/env/delete")
    def env_delete():  # type: ignore[unused-ignore]
        body = _body()
        world_id = _world_id_from(body)
        slot_id = body.get("slot_id")
        delete_all = body.get("all", False)
        try:
            if delete_all:
                count = manager.delete_all_slots(body.get("runtime") or body.get("world_id"))
                return jsonify({"ok": True, "deleted": count})
            if not slot_id:
                return jsonify({"ok": False, "error": "slot_id required (or pass all: true)"}), 400
            # world_id is optional: slot ids are unique, so the node resolves the
            # owning world from the slot id when the caller doesn't supply one.
            manager.delete_slot(slot_id, world_id)
            return jsonify({"ok": True, "slot_id": slot_id})
        except KeyError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404

    @bp.post("/env/restart")
    def env_restart():  # type: ignore[unused-ignore]
        body = _body()
        world_id = _world_id_from(body)
        slot_id = body.get("slot_id")
        if not slot_id:
            return jsonify({"ok": False, "error": "slot_id required"}), 400
        # world_id optional — resolved from the slot id when absent.
        try:
            status = manager.restart_slot(slot_id, world_id)
            return jsonify({"ok": True, "slot": status})
        except KeyError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404

    # -- async TTL caches (never block /slots or the heartbeat) --------------
    # Host metrics (psutil) and per-slot health probes are cached + refreshed in
    # the background so /slots never blocks on a slow probe (principle #5). The
    # node does NOT scan docker here — docker is an adapter-internal concern; any
    # per-slot container facts arrive via the adapter's get_info() (passed
    # through RuntimeSlot.to_status()), not via a docker-ps scan.
    from cluster.utils.async_cache import AsyncTTLCache
    from cluster.node.diagnostics import sample_resources

    _resources_cache = AsyncTTLCache(sample_resources, ttl=15.0, default={})
    _resources_cache.prime()

    # manager.health_check() serially probes every slot; a slow adapter probe
    # (e.g. MobileWorld's blocking HTTP health on a wedged container) would stall
    # the whole /slots response and blank out the dashboard. Per principle #5,
    # keep it off the hot path: serve a cached snapshot, refresh in the background.
    _health_cache = AsyncTTLCache(manager.health_check, ttl=15.0, default={})
    _health_cache.prime()

    @bp.get("/slots")
    @bp.get("/emulators")
    def list_slots():  # type: ignore[unused-ignore]
        slots_by_world = manager.list_slots()
        # The slots already carry whatever the adapter chose to self-report
        # (incl. container_id/container_name/docker_state for docker worlds) via
        # to_status(); the neutral node neither scans nor interprets it. Only
        # backfill env_id, which the frontend expects on every slot.
        for world_slots in slots_by_world.values():
            for slot in world_slots:
                if not slot.get("env_id"):
                    slot["env_id"] = slot.get("slot_id")
        return jsonify({
            "ok": True,
            "health": _health_cache.get(),
            "resources": _resources_cache.get(),
            "slots": slots_by_world,
        })

    @bp.get("/v1/sessions")
    def status():  # type: ignore[unused-ignore]
        return jsonify({"ok": True, "pools": manager.status()})

    return bp
