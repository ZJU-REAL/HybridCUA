"""Generic cluster session client — benchmark-agnostic (synchronous).

Provides acquire/reset/step/observe/evaluate/release via the unified
/v1/sessions protocol. Benchmark-specific clients (OSWorldSessionClient,
MobileWorldSessionClient) inherit and adapt to their own env interface.

Data-path calls (reset/step/observe/evaluate) go directly to the Node for
lower latency and reduced Master bandwidth. Control-plane calls (acquire/
release/heartbeat) stay on the Master.

The wire protocol (URLs, envelope, routing, retry policy) is inlined here as
private helpers/class attributes — this base client is self-contained. (An
earlier ``_protocol`` module factored these out to share between sync and async
transports; the async transport has been retired, so the indirection is gone.)
"""
from __future__ import annotations

import logging
import os
import random
import threading
import time
from typing import Any, Dict, Optional, Tuple

import requests

from cluster.utils.http_client import make_session

logger = logging.getLogger("cluster.client.session")


# -- protocol constants ------------------------------------------------------

# Data-plane ops bypass the master and go straight to the owning node when a
# node_url is known. Everything else (acquire/heartbeat/release) is control-plane.
_DATA_PLANE_OPS = frozenset({"reset", "step", "observe", "evaluate"})

# Per-op default timeouts (seconds); None => use the client's default timeout.
# reset carries snapshot revert + setup downloads + the adapter's post-reset settle;
# step must outlast run_python's 200s HTTP timeout so a slow bash command surfaces as
# TimeoutExpired rather than killing the request.
_OP_TIMEOUTS: Dict[str, Optional[float]] = {
    "reset": 420.0,
    "step": 240.0,
    "observe": 60.0,
    "evaluate": 120.0,
}

# Transient HTTP statuses worth retrying: 503 (no free slot yet — the scheduler
# may free one soon, so a worker queues instead of crashing) and 502 (a sidecar
# blipped). Permanent 4xx are never retried — they won't fix themselves.
_RETRY_STATUSES = (502, 503)
_RETRY_MAX = int(os.environ.get("CLUSTER_CLIENT_RETRY_MAX", "6"))
_RETRY_BASE = float(os.environ.get("CLUSTER_CLIENT_RETRY_BASE", "0.5"))
_RETRY_CAP = float(os.environ.get("CLUSTER_CLIENT_RETRY_MAX_DELAY", "20"))

# Connection-pool size for a client's HTTP session. Must be >= the number of
# requests one client has in flight at once: with process-per-env sampling that
# is ~2 (main + heartbeat), so 8 is ample.
_POOL_SIZE = int(os.environ.get("CLUSTER_CLIENT_POOL_SIZE", "8"))

_HEARTBEAT_INTERVAL = float(os.environ.get("CLUSTER_CLIENT_HEARTBEAT_INTERVAL", "30"))

DEFAULT_CLUSTER_URL = os.environ.get("GUI_ENV_SERVER_URL", "http://127.0.0.1:19000")


# -- protocol helpers (inlined, IO-free) -------------------------------------

def _normalize_url(url: str) -> str:
    return url.rstrip("/")


def _acquire_request(cluster_url: str, runtime: str) -> Tuple[str, Dict[str, Any]]:
    """(url, body) for acquiring a session on the master."""
    return f"{cluster_url}/v1/sessions", {"runtime": runtime}


def _parse_acquire(data: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Extract (session_id, node_url) from an acquire response.

    Master returns ``{sessions: [{session_id, node_url}]}``; a node returns
    ``{session_id, node_url}`` directly.
    """
    if "sessions" in data:
        session = data["sessions"][0]
        node_url = session.get("node_url")
        return session["session_id"], (_normalize_url(node_url) if node_url else None)
    node_url = data.get("node_url")
    return data["session_id"], (_normalize_url(node_url) if node_url else None)


def _op_base(op: str, cluster_url: str, node_url: Optional[str]) -> str:
    """Pick the base URL for ``op``: node for data-plane (when known), else master."""
    if node_url and op in _DATA_PLANE_OPS:
        return node_url
    return cluster_url


def _op_request(
    op: str, session_id: str, cluster_url: str, node_url: Optional[str]
) -> Tuple[str, str]:
    """(method, url) for a session op (reset/step/observe/evaluate/heartbeat/release)."""
    base = _op_base(op, cluster_url, node_url)
    if op == "release":
        return "DELETE", f"{base}/v1/sessions/{session_id}"
    return "POST", f"{base}/v1/sessions/{session_id}/{op}"


def _check_envelope(path: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Raise on ``ok=False`` envelopes; return data otherwise."""
    if not data.get("ok"):
        raise RuntimeError(f"Session API error on {path}: {data.get('error')}")
    return data


def _should_retry_status(status: int, attempt: int) -> bool:
    """True if an HTTP status warrants another attempt within the bounded budget."""
    return status in _RETRY_STATUSES and attempt < _RETRY_MAX


def _can_retry_after_error(attempt: int) -> bool:
    """True if a transport error (connection/timeout) may be retried."""
    return attempt < _RETRY_MAX


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter for retry attempt ``attempt`` (0-based)."""
    return min(_RETRY_BASE * 2 ** attempt, _RETRY_CAP) + random.uniform(0, _RETRY_BASE)


# -- the client --------------------------------------------------------------

class ClusterSessionClient:
    """Stateful client bound to one cluster session.

    Benchmark-specific clients subclass this and adapt the reset/step/observe/
    evaluate primitives to their benchmark's native env interface
    (DesktopEnv / AndroidEnvClient). One instance binds to one session.
    """

    def __init__(
        self,
        cluster_url: str | None = None,
        runtime: str = "osworld",
        *,
        timeout: float = 300,
    ):
        self.cluster_url = _normalize_url(cluster_url or DEFAULT_CLUSTER_URL)
        self.runtime = runtime
        self.timeout = timeout
        self.session_id: Optional[str] = None
        self._node_url: Optional[str] = None
        # Pooled session reused across this client's lifetime (data path to one
        # node + heartbeat to the master); size via CLUSTER_CLIENT_POOL_SIZE.
        self._http = make_session(pool_connections=_POOL_SIZE, pool_maxsize=_POOL_SIZE)
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._acquire()

    def _request(self, method: str, url: str, json: Dict[str, Any] | None, timeout: float) -> requests.Response:
        """HTTP with bounded exponential backoff + jitter on transient failures.

        Retries on 502/503 and on transport errors (connection/timeout): a 503
        means "no free slot yet", so a worker queues instead of crashing. Permanent
        4xx are raised immediately.
        """
        attempt = 0
        while True:
            try:
                resp = self._http.request(method, url, json=json or {}, timeout=timeout)
            except (requests.ConnectionError, requests.Timeout):
                if not _can_retry_after_error(attempt):
                    raise
            else:
                if not _should_retry_status(resp.status_code, attempt):
                    resp.raise_for_status()
                    return resp
            time.sleep(_backoff_delay(attempt))
            attempt += 1

    def _acquire(self) -> None:
        url, body = _acquire_request(self.cluster_url, self.runtime)
        resp = self._request("POST", url, body, self.timeout)
        data = _check_envelope("acquire", resp.json())
        self.session_id, self._node_url = _parse_acquire(data)
        logger.info(
            "Acquired session %s (runtime=%s, node_url=%s)",
            self.session_id, self.runtime, self._node_url,
        )
        self._start_heartbeat()

    def _call(self, op: str, body: Dict[str, Any] | None = None, timeout: float | None = None) -> Dict[str, Any]:
        """Issue a session op (reset/step/observe/evaluate) and unwrap the envelope."""
        method, url = _op_request(op, self.session_id, self.cluster_url, self._node_url)
        eff_timeout = timeout if timeout is not None else (_OP_TIMEOUTS.get(op) or self.timeout)
        resp = self._request(method, url, body, eff_timeout)
        return _check_envelope(op, resp.json())

    # -- heartbeat (keeps Master TTL alive while data-path bypasses it) --------

    def _start_heartbeat(self) -> None:
        if _HEARTBEAT_INTERVAL <= 0 or self.session_id is None:
            return
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"ClusterSessionHeartbeat-{self.session_id}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        thread = self._heartbeat_thread
        if thread and thread.is_alive():
            thread.join(timeout=2)
        self._heartbeat_thread = None

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(_HEARTBEAT_INTERVAL):
            session_id = self.session_id
            if session_id is None:
                return
            try:
                _, url = _op_request("heartbeat", session_id, self.cluster_url, self._node_url)
                self._http.post(url, json={}, timeout=10)
            except Exception as exc:
                logger.debug("Session heartbeat failed for %s: %s", session_id, exc)

    # -- session operations ----------------------------------------------------

    def reset(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        data = self._call("reset", {"task_payload": task_payload})
        return data.get("observation", {})

    def step(self, action: Dict[str, Any], pause: float | None = None) -> Dict[str, Any]:
        body = {"action": action} if pause is None else {"action": action, "pause": pause}
        return self._call("step", body)

    def observe(self) -> Dict[str, Any]:
        data = self._call("observe")
        return data.get("observation", {})

    def evaluate(self) -> Dict[str, Any]:
        return self._call("evaluate")

    def release(self) -> None:
        if self.session_id is None:
            return
        self._stop_heartbeat()
        try:
            _, url = _op_request("release", self.session_id, self.cluster_url, self._node_url)
            self._http.delete(url, timeout=30)
            logger.info("Released session %s", self.session_id)
        except Exception as exc:
            logger.warning("Failed to release session %s: %s", self.session_id, exc)
        finally:
            self.session_id = None
            self._node_url = None
            self._http.close()

    def close(self) -> None:
        self.release()

    def __del__(self):
        self.release()
