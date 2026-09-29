"""RuntimePool: a benchmark-neutral pool of WorldAdapter slots.

This is the world-agnostic generalization of the OSWorld ``EnvPool``. It manages
slots that each hold one :class:`~cluster.worlds.base.adapter.WorldAdapter`, created
by a *driver* — a zero-arg callable returning a fresh adapter. The pool knows
nothing about DesktopEnv, Docker or QEMU; OSWorld-specific lifecycle (container
adoption, snapshot revert) lives in the OSWorld driver/adapter, not here.

It mirrors the proven EnvPool mechanics — capacity cap, idle reuse, crash
replacement, idle reaping, prewarm, background scaling, per-slot locking,
stale-busy release, admin ops (delete/restart) — without OSWorld coupling.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from cluster.schemas import Action, EvaluationResult, Observation, StepResponse
from cluster.utils.debug_events import record_exception
from cluster.worlds.base.adapter import WorldAdapter

logger = logging.getLogger("cluster.node.pool")

class RuntimeDriver:
    """Unified driver protocol for world adapter lifecycle.

    Encapsulates all hooks a world provides to the pool:
    - create(): build a single adapter on demand (required)
    - create_batch(n): build N adapters at once with world-controlled serialization (optional)
    - adopt(): discover and reclaim existing running environments (optional)

    When create_batch is not provided, batch operations fall back to repeated create() calls.
    """

    def __init__(
        self,
        *,
        create: Callable[[], WorldAdapter],
        create_batch: Optional[Callable[[int], List[WorldAdapter]]] = None,
        adopt: Optional[Callable[[], List[WorldAdapter]]] = None,
        reclaim_orphans: Optional[Callable[[float, set], int]] = None,
    ) -> None:
        self._create = create
        self._create_batch = create_batch
        self._adopt = adopt
        self._reclaim_orphans = reclaim_orphans

    def __call__(self) -> WorldAdapter:
        """Create a single adapter (backward-compatible callable interface)."""
        return self._create()

    def create_batch(self, count: int) -> List[WorldAdapter]:
        """Create N adapters. Uses world's batch implementation if available,
        otherwise falls back to sequential single-create calls."""
        if self._create_batch:
            return self._create_batch(count)
        return [self._create() for _ in range(count)]

    @property
    def supports_batch(self) -> bool:
        """True if the world provides a dedicated batch creation strategy."""
        return self._create_batch is not None

    def adopt(self) -> List[WorldAdapter]:
        """Discover and adopt existing running environments."""
        if self._adopt:
            return self._adopt()
        return []

    def reclaim_orphans(self, grace_seconds: float, live_ids: set) -> int:
        """Reclaim leaked resources not mapped to any live slot (the inverse of
        adopt). Default: no-op. ``live_ids`` are the resource ids the pool still
        owns (so the hook spares them). A world whose resources can be orphaned
        (e.g. a Docker container left after a crash) provides this; all
        world-specific teardown stays inside the hook, never in the pool."""
        if self._reclaim_orphans:
            return self._reclaim_orphans(grace_seconds, live_ids)
        return 0


class RuntimePoolExhausted(RuntimeError):
    """Raised when the pool is full and has no idle slot to reuse."""


class NodeCapacity:
    """A node-wide slot budget shared across all per-world :class:`RuntimePool`s.

    The node runs one pool per world, but the host can only run so many
    environments in total. Rather than statically splitting ``max_envs`` across
    worlds (which strands capacity on idle worlds), every pool consults this
    shared budget before building a slot. Occupancy is the live sum of each
    pool's slot count — there is no separate counter to drift out of sync.
    """

    def __init__(self, max_envs: int) -> None:
        self._max = max(1, int(max_envs))
        self._lock = threading.RLock()
        self._pools: List["RuntimePool"] = []

    def register(self, pool: "RuntimePool") -> None:
        with self._lock:
            self._pools.append(pool)

    def set_max(self, max_envs: int) -> None:
        with self._lock:
            self._max = max(1, int(max_envs))

    @property
    def max(self) -> int:
        return self._max

    def used(self) -> int:
        """Node-wide occupancy = sum of live slot counts across all pools."""
        return sum(p._slot_count() for p in self._pools)

    def has_room(self) -> int:
        """Remaining node-wide headroom (may be negative if over-built)."""
        with self._lock:
            return self._max - self.used()


def _new_slot_id() -> str:
    return f"slot-{uuid.uuid4().hex[:12]}"


@dataclass
class RuntimeSlot:
    slot_id: str
    adapter: WorldAdapter
    busy: bool = False
    session_id: Optional[str] = None
    created_ts: float = field(default_factory=time.time)
    acquired_ts: float = field(default_factory=time.time)
    last_activity_ts: float = field(default_factory=time.time)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def touch(self) -> None:
        self.last_activity_ts = time.time()

    def to_status(self) -> Dict[str, Any]:
        now = time.time()
        info: Dict[str, Any] = {
            "slot_id": self.slot_id,
            "busy": self.busy,
            "session_id": self.session_id,
            "created_seconds": round(now - self.created_ts, 1),
            "age_seconds": round(now - self.acquired_ts, 1),
            "idle_seconds": round(now - self.last_activity_ts, 1),
        }
        try:
            extra = self.adapter.get_info()
            if extra:
                info.update(extra)
        except Exception:  # noqa: BLE001
            pass
        return info


class RuntimePool:
    """Manages up to ``max_slots`` adapter slots built by ``driver``."""

    def __init__(
        self,
        driver: RuntimeDriver,
        *,
        world_id: str = "unknown",
        max_slots: int = 1,
        capacity: Optional[NodeCapacity] = None,
        idle_ttl_seconds: int = 0,
        prewarm: int = 0,
        prewarm_concurrency: int = 2,
        scale_buffer: int = 0,
        scale_interval: int = 5,
        reset_on_release: bool = True,
    ) -> None:
        self.driver = driver
        self.world_id = world_id
        # When ``capacity`` is set, the build gate is node-wide (shared across
        # worlds); ``max_slots`` is the single-world fallback for back-compat.
        self._capacity = capacity
        self.max_slots = max(1, int(max_slots))
        self.idle_ttl_seconds = int(idle_ttl_seconds)
        self.reset_on_release = reset_on_release
        self._prewarm_count = max(0, min(int(prewarm), self.max_slots))
        self._prewarm_concurrency = max(1, int(prewarm_concurrency))
        self._scale_buffer = max(0, int(scale_buffer))
        self._scale_interval = max(1, int(scale_interval))
        self._lock = threading.RLock()
        self._slots: Dict[str, RuntimeSlot] = {}
        self._session_to_slot: Dict[str, str] = {}
        self._prewarm_done = False
        self._prewarm_errors: List[str] = []
        self._scaling = False
        self._scale_paused_until: float = 0.0

        # Register with the shared budget now that _lock/_slots exist and before
        # adopt/prewarm/scaler run, so node-wide occupancy counts this pool.
        if self._capacity is not None:
            self._capacity.register(self)

        adopted = self._run_adopt()

        remaining_prewarm = max(0, self._prewarm_count - adopted)
        if remaining_prewarm > 0:
            threading.Thread(
                target=self._background_prewarm,
                args=(remaining_prewarm,),
                daemon=True,
            ).start()
        else:
            self._prewarm_done = True

        if self._scale_buffer > 0:
            threading.Thread(target=self._background_scaler, daemon=True).start()

    def _run_adopt(self) -> int:
        """Run the driver's adopt hook to reclaim existing adapters (e.g. Docker
        containers from a previous node instance). Returns number of slots adopted."""
        try:
            adapters = self.driver.adopt()
        except Exception:
            logger.exception("adopt failed for world '%s'", self.world_id)
            return 0
        if not adapters:
            return 0
        adopted = 0
        for adapter in adapters:
            if len(self._slots) >= self.max_slots:
                try:
                    adapter.close()
                except Exception:  # noqa: BLE001
                    pass
                break
            slot = RuntimeSlot(slot_id=_new_slot_id(), adapter=adapter)
            self._slots[slot.slot_id] = slot
            adopted += 1
        if adopted:
            logger.info("Adopted %d existing adapter(s) for world '%s'", adopted, self.world_id)
        return adopted

    # -- capacity ----------------------------------------------------------
    def _slot_count(self) -> int:
        """Live slot count (busy or idle). Used by :class:`NodeCapacity`."""
        with self._lock:
            return len(self._slots)

    def _at_capacity_locked(self) -> bool:
        """Whether a new slot can NOT be built. Caller must hold ``self._lock``.

        Node-wide when a shared :class:`NodeCapacity` is wired, else the legacy
        per-pool ``max_slots`` cap.
        """
        if self._capacity is not None:
            return self._capacity.has_room() <= 0
        return len(self._slots) >= self.max_slots

    # -- slot lifecycle ----------------------------------------------------
    def _build_slot(self, *, claim: bool = False) -> RuntimeSlot:
        adapter = self.driver()
        slot = RuntimeSlot(slot_id=_new_slot_id(), adapter=adapter)
        with self._lock:
            # Re-check capacity under the lock: driver() ran unlocked, so other
            # threads/pools may have filled the node-wide budget meanwhile.
            # Discard the surplus adapter rather than exceed capacity.
            if self._at_capacity_locked():
                try:
                    adapter.close()
                finally:
                    raise RuntimePoolExhausted(
                        f"pool '{self.world_id}' exhausted after build: "
                        f"{len(self._slots)}/{self.max_slots} slots in use"
                    )
            # When the new slot is being acquired for a session, mark it busy
            # before releasing the lock so a concurrent acquire() can't grab it.
            slot.busy = claim
            self._slots[slot.slot_id] = slot
        return slot

    def _create_and_store_slot(self, idx: int, count: int) -> None:
        with self._lock:
            if self._at_capacity_locked():
                return
        try:
            self._build_slot()
        except Exception as exc:
            logger.exception("Failed to create slot (%d/%d) for world '%s'", idx + 1, count, self.world_id)
            with self._lock:
                self._prewarm_errors.append(f"slot {idx + 1}/{count}: {exc}")
            return
        logger.info("Created slot (%d/%d) for world '%s'", idx + 1, count, self.world_id)

    # -- prewarm -----------------------------------------------------------
    def _prewarm(self, count: int) -> None:
        if self._prewarm_concurrency <= 1:
            for idx in range(count):
                self._create_and_store_slot(idx, count)
            return
        logger.info("Prewarming %d slots with concurrency=%d for world '%s'",
                    count, self._prewarm_concurrency, self.world_id)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._prewarm_concurrency) as executor:
            futures = [
                executor.submit(self._create_and_store_slot, idx, count)
                for idx in range(count)
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception:
                    logger.exception("Prewarm task raised an unexpected error")

    def _background_prewarm(self, count: int) -> None:
        try:
            self._prewarm(count)
        except Exception:
            logger.exception("Background prewarm crashed for world '%s'", self.world_id)
            with self._lock:
                self._prewarm_errors.append("prewarm crashed unexpectedly")
        finally:
            self._prewarm_done = True
            with self._lock:
                ok_count = len(self._slots)
                err_count = len(self._prewarm_errors)
            logger.info("Prewarm finished for world '%s': %d/%d slots ready, %d errors",
                        self.world_id, ok_count, count, err_count)

    # -- background scaler -------------------------------------------------
    def _background_scaler(self) -> None:
        while True:
            time.sleep(self._scale_interval)
            try:
                self._maybe_scale_up()
            except Exception:
                logger.exception("Scaler iteration failed for world '%s'", self.world_id)

    def _pause_scaler(self, seconds: float = 60.0) -> None:
        self._scale_paused_until = time.time() + seconds

    def _maybe_scale_up(self) -> None:
        with self._lock:
            if self._scaling or not self._prewarm_done:
                return
            if time.time() < self._scale_paused_until:
                return
            total = len(self._slots)
            idle = sum(1 for s in self._slots.values() if not s.busy)
            headroom = self.max_slots - total
            needed = self._scale_buffer - idle
            if needed <= 0 or headroom <= 0:
                return
            batch = min(needed, headroom, self._prewarm_concurrency)
            self._scaling = True

        try:
            self._prewarm(batch)
        finally:
            with self._lock:
                self._scaling = False

    # -- stale busy release ------------------------------------------------
    def _release_stale_busy(self, timeout: float = 120.0) -> int:
        """Release slots stuck in busy state without a session (e.g. after a
        failed release). Mirrors EnvPool._release_stale_busy_locked."""
        now = time.time()
        stale_timeout = max(timeout, float(self.idle_ttl_seconds or 120))
        released = 0
        with self._lock:
            for slot in self._slots.values():
                if slot.busy and slot.session_id is None:
                    if (now - slot.last_activity_ts) > stale_timeout:
                        logger.warning("Force-releasing stale busy slot %s (no session, idle %.0fs)",
                                       slot.slot_id, now - slot.last_activity_ts)
                        slot.busy = False
                        slot.touch()
                        released += 1
        return released

    # -- acquire / release -------------------------------------------------
    def acquire(self, session_id: Optional[str] = None) -> RuntimeSlot:
        """Reserve a slot for a session: reuse a healthy idle one or create a new one."""
        session_id = session_id or f"sess-{uuid.uuid4().hex[:12]}"
        with self._lock:
            if session_id in self._session_to_slot:
                return self._slots[self._session_to_slot[session_id]]

        # Find a healthy idle slot WITHOUT holding self._lock across any docker
        # call. liveness() and close() (docker exec/stop, seconds-long) used to
        # run inside the lock here, starving the heartbeat's status() of the lock
        # during prewarm storms and getting the node marked dead. Now the lock is
        # only ever held for in-memory bookkeeping (microseconds).
        idle = self._claim_healthy_idle()

        if idle is None:
            with self._lock:
                if self._at_capacity_locked():
                    raise RuntimePoolExhausted(
                        f"pool '{self.world_id}' exhausted: {len(self._slots)}/{self.max_slots} slots in use"
                    )
            idle = self._build_slot(claim=True)

        with self._lock:
            # idle.busy is already True (claimed under the lock by
            # _claim_healthy_idle or _build_slot); just bind the session.
            idle.session_id = session_id
            idle.acquired_ts = time.time()
            idle.touch()
            self._session_to_slot[session_id] = idle.slot_id
            return idle

    def _claim_healthy_idle(self) -> Optional[RuntimeSlot]:
        """Return a claimed (busy=True), liveness-verified idle slot, or None.

        Concurrency-safe by reusing reap_unhealthy_idle's proven pattern:

        1. Under the lock, optimistically claim ONE idle candidate (busy=True) so
           no concurrent acquire() can grab the same slot — find-and-claim stays
           atomic, the gui-env EnvPool invariant.
        2. OUTSIDE the lock, probe the claimed candidate's liveness() (a docker
           call). Healthy -> return it (still busy, ready to bind). Unhealthy ->
           tear it down off-lock and loop to the next candidate.
        3. Tear-down re-checks under the lock that it's still the same, still-busy,
           session-less slot before detaching, then closes the adapter off-lock.

        Because each candidate is claimed and probed one at a time, the lock is
        never held across a docker call and at most one extra slot is briefly
        marked busy — no fleet-wide optimistic reservation.
        """
        while True:
            with self._lock:
                candidate = None
                for slot in self._slots.values():
                    if not slot.busy:
                        slot.busy = True  # claim atomically under the lock
                        candidate = slot
                        break
                if candidate is None:
                    return None

            try:
                healthy = bool(candidate.adapter.liveness())
            except Exception:  # noqa: BLE001 - an unanswerable probe means dead
                healthy = False

            if healthy:
                return candidate

            # Unhealthy: detach under the lock (re-check identity/state first),
            # then close the adapter (docker stop) OUTSIDE the lock.
            logger.warning("Slot %s failed health check on acquire, removing", candidate.slot_id)
            detached = False
            with self._lock:
                if (
                    self._slots.get(candidate.slot_id) is candidate
                    and candidate.busy
                    and candidate.session_id is None
                ):
                    self._slots.pop(candidate.slot_id, None)
                    detached = True
            if detached:
                self._close_adapter_unlocked(candidate)
            # loop to claim the next idle candidate

    def _slot_for_session(self, session_id: str) -> RuntimeSlot:
        slot_id = self._session_to_slot.get(session_id)
        if slot_id is None or slot_id not in self._slots:
            raise KeyError(f"no slot for session {session_id!r}")
        return self._slots[slot_id]

    # -- proxied lifecycle ops (per-slot lock) -----------------------------
    def reset(self, session_id: str, task_payload: Dict[str, Any]) -> Observation:
        with self._lock:
            slot = self._slot_for_session(session_id)
        with slot.lock:
            obs = self._guard(slot, "reset", lambda: slot.adapter.reset(task_payload))
            slot.touch()
        return obs

    def step(self, session_id: str, action: Action, pause: float | None = None) -> StepResponse:
        with self._lock:
            slot = self._slot_for_session(session_id)
        with slot.lock:
            resp = self._guard(slot, "step", lambda: slot.adapter.step(action, pause))
            slot.touch()
        return resp

    def observe(self, session_id: str) -> Observation:
        with self._lock:
            slot = self._slot_for_session(session_id)
        with slot.lock:
            obs = self._guard(slot, "observe", lambda: slot.adapter.observe())
            slot.touch()
        return obs

    def evaluate(self, session_id: str) -> EvaluationResult:
        with self._lock:
            slot = self._slot_for_session(session_id)
        with slot.lock:
            result = self._guard(slot, "evaluate", lambda: slot.adapter.evaluate())
            slot.touch()
        return result

    def release(self, session_id: str, *, close: bool = True) -> None:
        """Free the slot for a session.

        If ``close=True`` (default): destroy the adapter and remove the slot.
        If ``close=False``: keep the adapter warm, return slot to idle for reuse.
        When ``reset_on_release`` is set and close=False, reset the adapter first.
        """
        with self._lock:
            slot_id = self._session_to_slot.pop(session_id, None)
            if slot_id is None:
                return
            slot = self._slots.get(slot_id)
            if slot is None:
                return
            slot.session_id = None
            if close:
                slot.busy = False
                self._destroy_slot_locked(slot)
                return
            slot.busy = True  # keep busy while resetting

        if self.reset_on_release:
            try:
                with slot.lock:
                    slot.adapter.reset({})
            except Exception:
                logger.exception("Failed to reset adapter on release for slot %s; replacing", slot.slot_id)
                self._replace_slot(slot)
                return

        with self._lock:
            if self._slots.get(slot.slot_id) is slot:
                slot.busy = False
                slot.touch()

    def has_session(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._session_to_slot

    def release_orphan_sessions(self, valid_ids: set[str], *, min_age_seconds: float = 0.0) -> int:
        """Release busy slots whose session is no longer recognized by the master."""
        now = time.time()
        with self._lock:
            victims = [
                s
                for s in self._slots.values()
                if s.busy
                and s.session_id
                and s.session_id not in valid_ids
                and (now - s.acquired_ts) >= min_age_seconds
            ]
        released = 0
        for slot in victims:
            self.release(slot.session_id, close=False)
            released += 1
        if released:
            logger.info("pool '%s' released %d orphan session(s)", self.world_id, released)
        return released

    # -- slot destruction / replacement ------------------------------------
    def _destroy_slot_locked(self, slot: RuntimeSlot) -> None:
        try:
            slot.adapter.close()
        except Exception:  # noqa: BLE001
            logger.exception("error closing adapter for slot %s", slot.slot_id)
        self._slots.pop(slot.slot_id, None)

    def _close_adapter_unlocked(self, slot: RuntimeSlot) -> None:
        """Close a slot's adapter (docker stop) WITHOUT holding self._lock.

        The caller must already have detached the slot from ``_slots`` (and any
        session binding) under the lock, so no other thread can hand this slot
        out while we block on the docker stop. adapter.close() is idempotent
        (guarded by the adapter's _closed flag), so a double close is harmless."""
        try:
            slot.adapter.close()
        except Exception:  # noqa: BLE001
            logger.exception("error closing adapter for slot %s", slot.slot_id)

    def _replace_slot(self, slot: RuntimeSlot) -> Optional[RuntimeSlot]:
        """Replace a broken slot with a fresh one. Returns the new slot or None."""
        old_adapter = slot.adapter
        try:
            new_adapter = self.driver()
        except Exception:
            logger.exception("Failed to build replacement for slot %s; removing", slot.slot_id)
            with self._lock:
                self._slots.pop(slot.slot_id, None)
                if slot.session_id:
                    self._session_to_slot.pop(slot.session_id, None)
            try:
                old_adapter.close()
            except Exception:  # noqa: BLE001
                pass
            return None

        with self._lock:
            if self._slots.get(slot.slot_id) is slot:
                slot.adapter = new_adapter
                slot.busy = False
                slot.session_id = None
                slot.created_ts = time.time()
                slot.touch()
        try:
            old_adapter.close()
        except Exception:  # noqa: BLE001
            pass
        return slot

    def _guard(self, slot: RuntimeSlot, op: str, fn: Callable[[], Any]) -> Any:
        """Run an adapter op; record a structured envpool_error on failure, then
        let the exception propagate to the client.

        The slot is NOT destroyed on failure — a single operation error does not
        mean the environment is dead. The client decides how to handle it (skip
        task, retry, or close the session). This matches gui-env's behavior; the
        structured record restores gui-env's debug-events observability that the
        multi-world refactor dropped (sidecar otherwise loses the slot/session
        context by the time it catches the exception).
        """
        try:
            return fn()
        except Exception as exc:
            record_exception(
                exc,
                type="envpool_error",
                service="node",
                component=f"runtime_pool.{op}",
                session_id=slot.session_id,
                env_id=slot.slot_id,
                world_id=self.world_id,
                stacklevel=2,
            )
            logger.exception(
                "pool '%s' %s failed slot=%s session=%s",
                self.world_id, op, slot.slot_id, slot.session_id,
            )
            raise

    # -- admin ops ---------------------------------------------------------
    def delete_slot(self, slot_id: str) -> None:
        """Admin: destroy a specific slot and its adapter.

        Fire-and-forget: the slot is detached from ``_slots`` (and any session
        binding) under the lock, then the adapter's docker stop/remove (seconds
        long) is handed to a background daemon thread so the caller returns
        immediately instead of blocking on docker. Removing the slot from
        ``_slots`` first means the scaler's target-size accounting (which keys off
        ``len(self._slots)``) already sees it gone; ``_pause_scaler()`` gives a
        further cushion. Teardown errors only reach the log — ``close()`` is
        best-effort by design."""
        self._pause_scaler()
        with self._lock:
            slot = self._slots.get(slot_id)
            if slot is None:
                raise KeyError(f"Unknown slot_id: {slot_id}")
            if slot.session_id:
                self._session_to_slot.pop(slot.session_id, None)
            self._slots.pop(slot_id, None)
        threading.Thread(
            target=self._close_adapter_unlocked,
            args=(slot,),
            name=f"delete-slot-{slot_id}",
            daemon=True,
        ).start()

    def delete_all(self) -> int:
        """Admin: destroy all slots in parallel. Returns count destroyed."""
        self._pause_scaler()
        with self._lock:
            slots = list(self._slots.values())
            self._slots.clear()
            self._session_to_slot.clear()
        if not slots:
            return 0

        def _close_one(slot: RuntimeSlot) -> bool:
            try:
                slot.adapter.close()
                return True
            except Exception:  # noqa: BLE001
                logger.exception("Failed to close slot %s during delete_all", slot.slot_id)
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(slots), 16)) as executor:
            results = list(executor.map(_close_one, slots))
        return sum(1 for r in results if r)

    def restart_slot(self, slot_id: str) -> Dict[str, Any]:
        """Admin: close and rebuild a slot's adapter in place."""
        with self._lock:
            slot = self._slots.get(slot_id)
            if slot is None:
                raise KeyError(f"Unknown slot_id: {slot_id}")
            if slot.session_id:
                self._session_to_slot.pop(slot.session_id, None)
                slot.session_id = None
                slot.busy = False

        try:
            slot.adapter.close()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to close adapter during restart of %s", slot_id)

        new_adapter = self.driver()
        with self._lock:
            slot.adapter = new_adapter
            slot.created_ts = time.time()
            slot.busy = False
            slot.touch()
        return slot.to_status()

    # -- capacity ----------------------------------------------------------
    def set_max_slots(self, max_slots: int) -> None:
        with self._lock:
            self.max_slots = max(1, int(max_slots))

    # -- batch create ------------------------------------------------------
    def create_slots(self, count: int, concurrency: int = 8) -> int:
        """Create up to ``count`` idle slots. Uses the driver's batch strategy
        if available, otherwise falls back to parallel per-slot creation."""
        with self._lock:
            headroom = self._capacity.has_room() if self._capacity else self.max_slots - len(self._slots)
        count = min(count, headroom)
        if count <= 0:
            return 0

        if self.driver.supports_batch:
            return self._create_slots_batch(count)

        return self._create_slots_parallel(count, concurrency)

    def _create_slots_batch(self, count: int) -> int:
        """Batch path: delegate to the driver's create_batch for world-controlled creation."""
        try:
            adapters = self.driver.create_batch(count)
        except Exception as exc:
            raise RuntimeError(f"create_batch failed for world '{self.world_id}': {exc}") from exc
        created = 0
        with self._lock:
            for adapter in adapters:
                if self._at_capacity_locked():
                    break
                slot = RuntimeSlot(slot_id=_new_slot_id(), adapter=adapter)
                self._slots[slot.slot_id] = slot
                created += 1
        if not created and count > 0:
            raise RuntimeError(f"create_batch returned no usable adapters for world '{self.world_id}'")
        logger.info("create_batch created %d/%d slots for world '%s'", created, count, self.world_id)
        return created

    def _create_slots_parallel(self, count: int, concurrency: int = 8) -> int:
        """Parallel path: create slots via ThreadPoolExecutor (original behavior)."""
        errors: list[str] = []

        def _create_one(idx: int) -> bool:
            with self._lock:
                if self._at_capacity_locked():
                    return False
            try:
                self._build_slot()
                return True
            except Exception as exc:
                logger.warning("create_slots failed (%d/%d) for world '%s': %s",
                               idx + 1, count, self.world_id, exc)
                errors.append(str(exc))
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(count, concurrency)) as executor:
            results = list(executor.map(_create_one, range(count)))
        created = sum(1 for r in results if r)
        if errors and not created:
            raise RuntimeError(f"Failed to create any slots: {'; '.join(errors[:3])}")
        return created

    # -- maintenance -------------------------------------------------------
    def reap_idle(self) -> int:
        """Close idle slots older than idle_ttl_seconds. Returns count reaped."""
        if self.idle_ttl_seconds <= 0:
            return 0
        now = time.time()
        reaped = 0
        with self._lock:
            for slot in list(self._slots.values()):
                if not slot.busy and (now - slot.last_activity_ts) > self.idle_ttl_seconds:
                    self._destroy_slot_locked(slot)
                    reaped += 1
        return reaped

    def reap_unhealthy_idle(self) -> int:
        """Probe idle slots via the adapter's health_check(); drop the dead ones.

        World-agnostic: the pool never decides what "alive" means — it asks the
        adapter. For a docker world an adapter whose container has vanished
        returns False, so its phantom slot is reclaimed and the slot accounting
        re-converges on the real resources. Busy slots are left untouched (never
        kill a slot mid-session). Returns the number of slots reaped.

        Mirrors reap_idle's reclaim pattern; like list_slots/health_check it
        snapshots under self._lock but calls the adapter outside it (per-slot
        lock), so a slow health probe never stalls the pool.
        """
        with self._lock:
            idle = [s for s in self._slots.values() if not s.busy]
        victims: List[RuntimeSlot] = []
        for slot in idle:
            try:
                with slot.lock:
                    healthy = bool(slot.adapter.liveness())
            except Exception:  # noqa: BLE001 - an unanswerable probe means dead
                healthy = False
            if not healthy:
                victims.append(slot)
        reaped = 0
        with self._lock:
            for slot in victims:
                # Re-check under the lock: it must still be the same, still-idle slot
                # (a concurrent acquire() may have claimed it after our probe).
                if self._slots.get(slot.slot_id) is slot and not slot.busy:
                    logger.warning("Reaping unhealthy idle slot %s (world '%s')", slot.slot_id, self.world_id)
                    self._destroy_slot_locked(slot)
                    reaped += 1
        return reaped

    def reclaim_orphans(self, grace_seconds: float) -> int:
        """Ask the driver to free leaked resources, sparing those this pool still
        owns. The set of live resource ids is collected from each slot's
        ``get_info()`` (world-neutral: the pool doesn't know what an id is, only
        that the driver may want to spare it). All teardown lives in the driver."""
        with self._lock:
            slots = list(self._slots.values())
        live_ids: set = set()
        for slot in slots:
            try:
                rid = slot.adapter.resource_id()
            except Exception:  # noqa: BLE001
                rid = None
            if rid:
                live_ids.add(rid)
        return self.driver.reclaim_orphans(grace_seconds, live_ids)

    def health_check(self) -> Dict[str, bool]:
        with self._lock:
            slots = list(self._slots.values())
        out: Dict[str, bool] = {}
        for slot in slots:
            try:
                with slot.lock:
                    out[slot.slot_id] = bool(slot.adapter.health_check())
            except Exception:  # noqa: BLE001
                out[slot.slot_id] = False
        return out

    def list_slots(self) -> List[Dict[str, Any]]:
        # Snapshot slot references under the lock, but build statuses OUTSIDE it.
        # to_status() calls adapter.get_info(), which for OSWorld touches the
        # docker container (a potentially slow/blocking call during a reset
        # container rebuild). Holding self._lock across it would stall every
        # other pool op (acquire/release/reset) and make /slots time out.
        with self._lock:
            slots = list(self._slots.values())
        return [s.to_status() for s in slots]

    def status(self) -> Dict[str, Any]:
        # Snapshot under the lock, build statuses outside it (see list_slots).
        with self._lock:
            slots = list(self._slots.values())
            meta = {
                "world_id": self.world_id,
                # Under shared node-wide capacity the cap is the node budget,
                # not this pool's legacy per-world max_slots.
                "max_slots": self._capacity.max if self._capacity else self.max_slots,
                "total_slots": len(slots),
                "busy_slots": sum(1 for s in slots if s.busy),
                "idle_slots": sum(1 for s in slots if not s.busy),
                "prewarm_done": self._prewarm_done,
                "prewarm_errors": list(self._prewarm_errors),
                "scaling_in_progress": self._scaling,
                "reset_on_release": self.reset_on_release,
            }
        meta["slots"] = [s.to_status() for s in slots]
        return meta

    def close_all(self) -> None:
        with self._lock:
            for slot in list(self._slots.values()):
                self._destroy_slot_locked(slot)
            self._session_to_slot.clear()
