"""EnvSession wire protocol — IO-free.

The single source of truth for *what* an EnvSession call looks like on the wire
(which URL, which body, how to read the ``{ok, ...}`` envelope, when to retry),
shared by the sync (`requests`) and async (`aiohttp`) clients so neither has to
re-spell the protocol.

Design rules:
- No ``requests`` / ``aiohttp`` import here — pure URL/dict/decision helpers.
- Data-plane ops (reset/step/observe/evaluate) route to the node directly when a
  ``node_url`` is known; control-plane ops (acquire/heartbeat/release) go to the
  master. This preserves the data-plane/control-plane split.
- Retry policy is shared and deliberately conservative: a transient transport
  error or a 502/503 is retried; everything else surfaces immediately. The
  caller never blindly replays a non-idempotent POST beyond this bounded policy.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional, Tuple

# Data-plane ops bypass the master and go straight to the owning node when a
# node_url is known. Everything else (acquire/heartbeat/release) is control-plane.
DATA_PLANE_OPS = frozenset({"reset", "step", "observe", "evaluate"})

# Per-op default timeouts (seconds); None => use the client's default timeout.
OP_TIMEOUTS: Dict[str, Optional[float]] = {
    "reset": None,
    "step": 120.0,
    "observe": 60.0,
    "evaluate": 120.0,
}

# Transient HTTP statuses worth retrying: 503 (no free slot yet — the scheduler
# may free one soon, so a worker queues instead of crashing) and 502 (a sidecar
# blipped). Permanent 4xx are never retried — they won't fix themselves.
RETRY_STATUSES = (502, 503)
RETRY_MAX = int(os.environ.get("CLUSTER_CLIENT_RETRY_MAX", "6"))
RETRY_BASE = float(os.environ.get("CLUSTER_CLIENT_RETRY_BASE", "0.5"))
RETRY_CAP = float(os.environ.get("CLUSTER_CLIENT_RETRY_MAX_DELAY", "20"))

# Connection-pool size for a client's HTTP session / connector. Must be >= the
# number of requests one client has in flight at once: with process-per-env
# sampling that is ~2 (main + heartbeat), so 8 is ample; an async client driving
# N sessions through one event loop should raise it to >= N.
POOL_SIZE = int(os.environ.get("CLUSTER_CLIENT_POOL_SIZE", "8"))

HEARTBEAT_INTERVAL = float(os.environ.get("CLUSTER_CLIENT_HEARTBEAT_INTERVAL", "30"))

DEFAULT_CLUSTER_URL = os.environ.get("GUI_ENV_SERVER_URL", "http://127.0.0.1:19000")


def normalize_url(url: str) -> str:
    return url.rstrip("/")


# -- acquire ----------------------------------------------------------------

def acquire_request(
    cluster_url: str,
    runtime: str,
    *,
    user_id: Optional[str] = None,
    task_type: Optional[str] = None,
    job_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """(url, body) for acquiring a session on the master.

    ``user_id`` / ``task_type`` / ``job_id`` are sent on the wire only when
    provided, so the master can attribute the session to a user and bucket it as
    training vs evaluation on its dashboard. ``task_type`` is written only for the
    values the server understands (``"training"`` / ``"evaluation"``); anything
    else is dropped so a stray label can't create a phantom dashboard bucket.
    Omitting all three keeps the legacy ``{"runtime": ...}`` body (backward
    compatible with servers that ignore these fields).
    """
    body: Dict[str, Any] = {"runtime": runtime}
    if user_id is not None:
        body["user_id"] = user_id
    if task_type in ("training", "evaluation"):
        body["task_type"] = task_type
    if job_id:
        body["job_id"] = job_id
    return f"{cluster_url}/v1/sessions", body


def parse_acquire(data: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Extract (session_id, node_url) from an acquire response.

    Master returns ``{sessions: [{session_id, node_url}]}``; a node returns
    ``{session_id, node_url}`` directly.
    """
    if "sessions" in data:
        session = data["sessions"][0]
        node_url = session.get("node_url")
        return session["session_id"], (normalize_url(node_url) if node_url else None)
    node_url = data.get("node_url")
    return data["session_id"], (normalize_url(node_url) if node_url else None)


# -- per-op routing ---------------------------------------------------------

def op_base(op: str, cluster_url: str, node_url: Optional[str]) -> str:
    """Pick the base URL for ``op``: node for data-plane (when known), else master."""
    if node_url and op in DATA_PLANE_OPS:
        return node_url
    return cluster_url


def op_request(
    op: str, session_id: str, cluster_url: str, node_url: Optional[str]
) -> Tuple[str, str]:
    """(method, url) for a session op (reset/step/observe/evaluate/heartbeat/release)."""
    base = op_base(op, cluster_url, node_url)
    if op == "release":
        return "DELETE", f"{base}/v1/sessions/{session_id}"
    return "POST", f"{base}/v1/sessions/{session_id}/{op}"


def check_envelope(path: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Raise on ``ok=False`` envelopes; return data otherwise."""
    if not data.get("ok"):
        raise RuntimeError(f"Session API error on {path}: {data.get('error')}")
    return data


# -- retry policy (shared by sync + async) ----------------------------------

def should_retry_status(status: int, attempt: int) -> bool:
    """True if an HTTP status warrants another attempt within the bounded budget."""
    return status in RETRY_STATUSES and attempt < RETRY_MAX


def can_retry_after_error(attempt: int) -> bool:
    """True if a transport error (connection/timeout) may be retried."""
    return attempt < RETRY_MAX


def backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter for retry attempt ``attempt`` (0-based)."""
    return min(RETRY_BASE * 2 ** attempt, RETRY_CAP) + random.uniform(0, RETRY_BASE)
