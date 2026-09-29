"""Async cluster session client — benchmark-agnostic (asyncio + aiohttp).

The asyncio counterpart of :class:`ClusterSessionClient`. Same unified
/v1/sessions protocol (shared via ``_protocol``), same data-plane/control-plane
split, same retry policy — only the transport differs.

One instance binds to one session. To drive many sessions concurrently, the
caller runs ONE event loop and fans out with ``asyncio.gather`` /
``asyncio.Semaphore`` over multiple clients — this is what lets a single process
hold N concurrent rollouts (concurrency decoupled from OS processes), instead of
the eval driver's process-per-session model. Example: ``scripts/python/
async_session_demo.py``.

Usage::

    async with AsyncClusterSessionClient(runtime="osworld") as c:
        obs = await c.reset({"id": task_id, "instruction": "..."})
        data = await c.step({"kind": "gui", "type": "pyautogui", "payload": {...}})
        score = (await c.evaluate()).get("score")
"""
from __future__ import annotations

import asyncio
import atexit
import logging
import os
from typing import Any, Dict, Optional

import aiohttp

from . import _protocol as proto

logger = logging.getLogger("cluster.client.async_session")


# Process-level shared aiohttp connector, keyed by event loop. Sharing one
# connector across all session clients on a loop lets keep-alive connections to
# a node be reused across trajectories (vs a fresh connector per trajectory,
# which paid a cold TCP/TLS handshake on every step — ~+2s/step in training).
# Keyed by id(loop) because an aiohttp connector is bound to the loop that
# created it; multi-process rollout workers each have their own loop and thus
# their own pool, matching the per-worker single-process world.
_SHARED_CONNECTORS: dict[int, aiohttp.TCPConnector] = {}


def _shared_connector_limit() -> int:
    """Pool size >= concurrent in-flight requests on this loop.

    Each trajectory may have its main op + heartbeat in flight, so size to ~2x
    the per-process trajectory-concurrency budget (already sliced per worker via
    GUI_TRAJECTORY_CONCURRENCY) with a generous floor.
    """
    conc = os.getenv("GUI_TRAJECTORY_CONCURRENCY") or os.getenv("GUI_POOL_MAX_ENVS") or "64"
    try:
        return max(proto.POOL_SIZE, int(conc) * 2)
    except ValueError:
        return max(proto.POOL_SIZE, 128)


def _shared_connector() -> aiohttp.TCPConnector:
    loop = asyncio.get_event_loop()
    key = id(loop)
    conn = _SHARED_CONNECTORS.get(key)
    if conn is None or conn.closed:
        limit = _shared_connector_limit()
        conn = aiohttp.TCPConnector(limit=limit, limit_per_host=limit)
        _SHARED_CONNECTORS[key] = conn
        logger.info("Created shared aiohttp connector (limit=%d) for loop %d", limit, key)
    return conn


@atexit.register
def _close_shared_connectors() -> None:
    """Close pooled connectors on interpreter shutdown.

    The shared connectors are intentionally process-lived (not owned by any
    ClientSession), so close them here to avoid leaking sockets and the noisy
    'Event loop is closed' warning aiohttp's __del__ emits at GC time.
    """
    for conn in list(_SHARED_CONNECTORS.values()):
        try:
            if conn.closed:
                continue
            # Prefer the synchronous internal close: the async close() is a
            # coroutine and the loop is already gone at interpreter exit.
            sync_close = getattr(conn, "_close", None)
            if callable(sync_close):
                sync_close()
        except Exception:
            pass
    _SHARED_CONNECTORS.clear()


class AsyncClusterSessionClient:
    """Async stateful client bound to one cluster session.

    Construct, then ``await acquire()`` (or use ``async with``, which acquires on
    enter and releases on exit). Not safe to share one instance across tasks
    that call it concurrently — give each concurrent rollout its own client.
    """

    def __init__(
        self,
        cluster_url: str | None = None,
        runtime: str = "osworld",
        *,
        timeout: float = 300,
        user_id: str | None = None,
        task_type: str | None = None,
        job_id: str | None = None,
    ):
        self.cluster_url = proto.normalize_url(cluster_url or proto.DEFAULT_CLUSTER_URL)
        self.runtime = runtime
        self.timeout = timeout
        # Session-attribution fields, forwarded on the wire by acquire() so the
        # master can bucket this session by user / training-vs-eval. None => not
        # sent (legacy body). Set per-allocate by the caller, never shared.
        self.user_id = user_id
        self.task_type = task_type
        self.job_id = job_id
        self.session_id: Optional[str] = None
        self._node_url: Optional[str] = None
        # The aiohttp ClientSession is created lazily in acquire(), which always
        # runs inside the event loop (creating it in __init__ would bind to the
        # wrong/absent loop and trigger aiohttp warnings).
        self._http: Optional[aiohttp.ClientSession] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

    def _ensure_http(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            # Share ONE process-level TCPConnector across all sessions on this
            # event loop so keep-alive connections to a node are reused across
            # trajectories (a fresh connector per trajectory paid a cold
            # TCP/TLS handshake on every step — measured ~+2s/step vs the legacy
            # global-pool client). Each client keeps its own lightweight
            # ClientSession but does NOT own the connector (connector_owner=False),
            # so release()/close() never tears down the shared pool.
            self._http = aiohttp.ClientSession(
                connector=_shared_connector(),
                connector_owner=False,
            )
        return self._http

    async def _request(self, method: str, url: str, json: Dict[str, Any] | None, timeout: float) -> Dict[str, Any]:
        """HTTP with bounded backoff on transient failures; returns parsed JSON.

        Mirrors the sync client's policy via ``_protocol``: retry 502/503 and
        transport errors within a bounded budget; surface everything else.
        """
        attempt = 0
        ct = aiohttp.ClientTimeout(total=timeout)
        http = self._ensure_http()
        while True:
            try:
                async with http.request(method, url, json=json or {}, timeout=ct) as resp:
                    if proto.should_retry_status(resp.status, attempt):
                        await asyncio.sleep(proto.backoff_delay(attempt))
                        attempt += 1
                        continue
                    resp.raise_for_status()
                    return await resp.json()
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
                if not proto.can_retry_after_error(attempt):
                    raise
                await asyncio.sleep(proto.backoff_delay(attempt))
                attempt += 1

    async def acquire(self) -> "AsyncClusterSessionClient":
        url, body = proto.acquire_request(
            self.cluster_url,
            self.runtime,
            user_id=self.user_id,
            task_type=self.task_type,
            job_id=self.job_id,
        )
        data = proto.check_envelope("acquire", await self._request("POST", url, body, self.timeout))
        self.session_id, self._node_url = proto.parse_acquire(data)
        logger.info(
            "Acquired session %s (runtime=%s, node_url=%s)",
            self.session_id, self.runtime, self._node_url,
        )
        self._start_heartbeat()
        return self

    async def _call(self, op: str, body: Dict[str, Any] | None = None, timeout: float | None = None) -> Dict[str, Any]:
        method, url = proto.op_request(op, self.session_id, self.cluster_url, self._node_url)
        eff_timeout = timeout if timeout is not None else (proto.OP_TIMEOUTS.get(op) or self.timeout)
        return proto.check_envelope(op, await self._request(method, url, body, eff_timeout))

    # -- heartbeat (asyncio.Task, not a thread) --------------------------------

    def _start_heartbeat(self) -> None:
        if proto.HEARTBEAT_INTERVAL <= 0 or self.session_id is None:
            return
        if self._heartbeat_task and not self._heartbeat_task.done():
            return
        self._heartbeat_task = asyncio.ensure_future(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        try:
            while self.session_id is not None:
                await asyncio.sleep(proto.HEARTBEAT_INTERVAL)
                session_id = self.session_id
                if session_id is None:
                    return
                try:
                    _, url = proto.op_request("heartbeat", session_id, self.cluster_url, self._node_url)
                    await self._request("POST", url, {}, 10)
                except Exception as exc:
                    logger.debug("Session heartbeat failed for %s: %s", session_id, exc)
        except asyncio.CancelledError:
            pass

    async def _stop_heartbeat(self) -> None:
        task = self._heartbeat_task
        self._heartbeat_task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    # -- session operations ----------------------------------------------------

    async def reset(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        data = await self._call("reset", {"task_payload": task_payload})
        return data.get("observation", {})

    async def step(self, action: Dict[str, Any], pause: float | None = None) -> Dict[str, Any]:
        body = {"action": action} if pause is None else {"action": action, "pause": pause}
        return await self._call("step", body)

    async def observe(self) -> Dict[str, Any]:
        data = await self._call("observe")
        return data.get("observation", {})

    async def evaluate(self) -> Dict[str, Any]:
        return await self._call("evaluate")

    async def release(self) -> None:
        await self._stop_heartbeat()
        if self.session_id is not None and self._http is not None:
            try:
                _, url = proto.op_request("release", self.session_id, self.cluster_url, self._node_url)
                await self._http.delete(url, timeout=aiohttp.ClientTimeout(total=30))
                logger.info("Released session %s", self.session_id)
            except Exception as exc:
                logger.warning("Failed to release session %s: %s", self.session_id, exc)
            finally:
                self.session_id = None
                self._node_url = None
        if self._http is not None and not self._http.closed:
            await self._http.close()

    async def close(self) -> None:
        await self.release()

    async def __aenter__(self) -> "AsyncClusterSessionClient":
        return await self.acquire()

    async def __aexit__(self, *exc) -> None:
        await self.release()
