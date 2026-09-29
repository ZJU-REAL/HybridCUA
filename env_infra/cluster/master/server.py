from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import requests
from urllib.parse import quote

from flask import Flask, Response, jsonify, redirect, request as flask_request
from flask_sock import Sock

from cluster.master.models import MasterJobRecord, NodeInfo, UserQuota
from cluster.utils.common import json_body as _json_body, proxy_request
from cluster.utils.http_client import make_session
from cluster.utils.debug_events import tail_events
from cluster.utils.logging_helpers import configure_logging, short_excerpt
from cluster.utils.ws_proxy import close_quietly, http_to_ws_url, proxy_websocket
from cluster.core.scheduler import Scheduler, create_scheduler
from cluster.core.session import SessionRecord, SessionStatus, SessionStore, new_session_id

logger = logging.getLogger("cluster.master")
app = Flask(__name__)
sock = Sock(app)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_nodes: dict[str, NodeInfo] = {}
# slot_id -> node_id reverse index, rebuilt from each node's heartbeat slot list.
# Guarded by _nodes_lock (same lifecycle as _nodes). Lets the master locate the
# node hosting a slot in O(1) for live view, covering idle slots too (keyed by
# slot_id, which every slot has — unlike session_id, which only busy slots have).
_slot_index: dict[str, str] = {}
_nodes_lock = threading.RLock()

_quotas: dict[str, UserQuota] = {}
_quotas_lock = threading.RLock()

_master_jobs: dict[str, MasterJobRecord] = {}
_master_jobs_lock = threading.RLock()

_local_job_logs: dict[str, list[str]] = {}
MAX_LOG_LINES = 5000

_alloc_lock = threading.Lock()
_scheduler: Scheduler | None = None
_config: argparse.Namespace | None = None

# Process-wide pooled HTTP client for all master->node calls (keep-alive reuse).
# Live-view streaming (proxy_request/ws_proxy) is deliberately NOT routed here.
HTTP = make_session()


def _reindex_node_slots(node_id: str, slots: list[dict] | None) -> None:
    """Replace ``node_id``'s entries in the slot->node index with ``slots``.
    ``slots=None`` just drops the node's entries (deregister / dead). Caller need
    not hold _nodes_lock; this takes it."""
    with _nodes_lock:
        for sid in [s for s, n in _slot_index.items() if n == node_id]:
            del _slot_index[sid]
        for s in slots or []:
            sid = s.get("slot_id")
            if sid:
                _slot_index[sid] = node_id

# ---------------------------------------------------------------------------
# EnvSession is the SINGLE bookkeeping model (it absorbed the old LeaseRecord).
# Every client-facing /v1/sessions route and every dashboard env action is backed
# by this one store; data-plane ops run client -> node directly, while the master
# owns scheduling, bookkeeping, and liveness.
# ---------------------------------------------------------------------------
_sessions: SessionStore = SessionStore()
_sessions_lock = threading.RLock()  # serializes node selection during create
_FORWARD_TIMEOUT = 600


# ---------------------------------------------------------------------------
# EnvSession control plane — create / forward / translate
#
# _create_sessions() is the one place that places a session on a node (via the
# capability scheduler) and asks the node to build it through /v1/sessions.
# ---------------------------------------------------------------------------
def _create_sessions(
    *,
    world_id: str,
    count: int = 1,
    user_id: str = "anonymous",
    mode: str = "eval",
    task_type: str = "evaluation",
    episode_id: str | None = None,
    job_id: str | None = None,
    job_script: str | None = None,
    source_addr: str | None = None,
    task_payload: dict[str, Any] | None = None,
    capability_requirements: dict[str, Any] | None = None,
    resources: dict[str, Any] | None = None,
    ttl_seconds: int = 7200,
    only_node_id: str | None = None,
) -> tuple[list[SessionRecord], str | None, int]:
    """Place ``count`` sessions of ``world_id`` across nodes.

    Returns ``(created, error, status_code)``. ``created`` holds the
    SessionRecords already stored; on partial failure ``error`` is set and
    ``status_code`` reflects the first failure (409 no-node, 502 transport,
    503 node-refused). On full success: ``(records, None, 200)``.
    """
    request_for_sched = {
        "runtime": world_id,
        "capability_requirements": capability_requirements or {},
        "resources": resources or {},
    }
    created: list[SessionRecord] = []
    for _ in range(count):
        record, error, code = _create_one_session(
            world_id=world_id,
            request_for_sched=request_for_sched,
            only_node_id=only_node_id,
            mode=mode,
            task_type=task_type,
            episode_id=episode_id,
            job_id=job_id,
            job_script=job_script,
            source_addr=source_addr,
            task_payload=task_payload or {},
            ttl_seconds=ttl_seconds,
            user_id=user_id,
        )
        if record is None:
            return created, error, code
        created.append(record)
    return created, None, 200


def _create_one_session(
    *,
    world_id: str,
    request_for_sched: dict[str, Any],
    only_node_id: str | None,
    mode: str,
    task_type: str,
    episode_id: str | None,
    job_id: str | None,
    job_script: str | None,
    source_addr: str | None,
    task_payload: dict[str, Any],
    ttl_seconds: int,
    user_id: str,
) -> tuple[SessionRecord | None, str | None, int]:
    """Place ONE session, retrying across candidate nodes.

    This reuses the proven session-allocation lifecycle: reserve a slot up front
    (``reserved_envs``) so concurrent creates don't oversubscribe a node, ask
    the node to build it, and on any failure roll the reservation back and try
    the next candidate. Bookkeeping is a SessionRecord rather than a LeaseRecord
    — the lifecycle logic is identical.
    """
    tried: set[str] = set()
    last_error = ""
    while True:
        with _alloc_lock:
            with _nodes_lock:
                candidates = list(_nodes.values())
            if only_node_id is not None:
                candidates = [n for n in candidates if n.node_id == only_node_id]
            remaining = [n for n in candidates if n.node_id not in tried]
            node = _scheduler.select_node(remaining, request_for_sched) if _scheduler else None
            if node is None:
                if last_error:
                    return None, last_error, 503
                return None, (
                    f"no healthy node can serve world {world_id!r} with the requested capabilities"
                ), 409
            tried.add(node.node_id)
            # Claim the slot atomically: re-check capacity and reserve under the
            # SAME node.lock the scheduler's free_slots read needs. A heartbeat /
            # reconcile can mutate busy_envs between select_node and here, so the
            # slot the scheduler saw may already be gone — re-verify before taking
            # it, and skip this node (retry) rather than oversubscribe it.
            with node.lock:
                if node.free_slots <= 0:
                    continue
                node.reserved_envs += 1

        session_id = new_session_id()
        try:
            resp = HTTP.post(
                f"{node.node_url}/v1/sessions",
                json={"runtime": world_id, "session_id": session_id},
                timeout=_FORWARD_TIMEOUT,
            )
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            with node.lock:
                node.reserved_envs = max(0, node.reserved_envs - 1)
            logger.warning("create session on node %s failed: %s, trying next", node.node_id, exc)
            last_error = f"node {node.node_id}: {exc}"
            continue
        if not resp.ok or not data.get("ok"):
            with node.lock:
                node.reserved_envs = max(0, node.reserved_envs - 1)
            last_error = data.get("error", f"node refused (HTTP {resp.status_code})")
            logger.warning("create session on node %s rejected: %s, trying next", node.node_id, last_error)
            continue

        with node.lock:
            node.reserved_envs = max(0, node.reserved_envs - 1)
            node.busy_envs += 1

        node_env_id = data.get("env_id") or data.get("session_id") or session_id
        record = SessionRecord(
            session_id=session_id,
            world_id=world_id,
            node_id=node.node_id,
            node_url=node.node_url,
            slot_id=session_id,
            env_id=node_env_id,
            status=SessionStatus.ACTIVE.value,
            mode=mode,
            capabilities=request_for_sched.get("capability_requirements", {}),
            task_payload=task_payload,
            ttl_seconds=ttl_seconds,
            user_id=user_id,
            task_type=task_type,
            episode_id=episode_id,
            job_id=job_id,
            job_script=job_script,
            source_addr=source_addr,
        )
        _sessions.create(record)
        return record, None, 200


def _drop_session(record: SessionRecord, *, timeout: int = 30) -> None:
    """DELETE the session on its node and remove it from the store.

    The shared teardown action only. ``busy_envs`` accounting is intentionally
    left to the caller: each call site decrements under a different, correct
    condition (e.g. close() only when still ACTIVE; the reaper unconditionally,
    since expire_stale() already flipped status to EXPIRED). Do NOT fold the
    decrement in here.
    """
    try:
        HTTP.delete(
            f"{record.node_url}/v1/sessions/{record.session_id}", timeout=timeout
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("session %s close on node failed: %s", record.session_id, exc)
    _sessions.delete(record.session_id)


def _close_session(record: SessionRecord, *, timeout: int = 30) -> None:
    """Tear down one session: DELETE it on the node, drop it, decrement busy."""
    _drop_session(record, timeout=timeout)
    with _nodes_lock:
        node = _nodes.get(record.node_id)
    if node is not None and record.status == SessionStatus.ACTIVE.value:
        with node.lock:
            node.busy_envs = max(0, node.busy_envs - 1)


_CLUSTER_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _CLUSTER_DIR.parent
_FRONTEND_DIR = _CLUSTER_DIR / "frontend"
_DASHBOARD_PATH = _FRONTEND_DIR / "dashboard.html"
_DEBUG_EVENT_FILTER_KEYS = ("type", "service", "path", "session_id", "env_id", "node_id")


def _debug_event_filters() -> dict[str, str | None]:
    return {key: flask_request.args.get(key) for key in _DEBUG_EVENT_FILTER_KEYS}


def _collect_debug_events(
    limit: int = 100,
    *,
    filters: dict[str, str | None] | None = None,
    include_nodes: bool = True,
) -> list[dict[str, Any]]:
    try:
        limit = max(0, min(int(limit or 0), 2000))
    except (TypeError, ValueError):
        limit = 100
    if limit == 0:
        return []
    filters = filters or {}
    events = tail_events(limit, **filters)
    if include_nodes:
        with _nodes_lock:
            nodes = [n for n in _nodes.values() if n.status != "dead"]
        per_node_limit = max(20, min(limit, 200))
        for node in nodes:
            try:
                resp = HTTP.get(
                    f"{node.node_url}/debug/events",
                    params={"tail": per_node_limit, **{k: v for k, v in filters.items() if v}},
                    timeout=2,
                )
                data = resp.json()
            except Exception as exc:
                logger.debug(
                    "Debug events fetch failed node_id=%s node_url=%s: %s",
                    node.node_id,
                    node.node_url,
                    exc,
                )
                continue
            if not data.get("ok"):
                continue
            for event in data.get("events", []):
                if not isinstance(event, dict):
                    continue
                item = dict(event)
                item.setdefault("node_id", node.node_id)
                item.setdefault("node_url", node.node_url)
                events.append(item)
    events.sort(key=lambda item: str(item.get("ts") or ""))
    return events[-limit:]


def _build_user_rows(sessions: list[SessionRecord]) -> list[dict[str, Any]]:
    """Per-user session tallies joined with quotas, for the dashboard and /users.

    Shared so user accounting (total/training/evaluation per user) lives in one
    place. ``sessions`` is the session set to count over (a snapshot or live list).
    """
    user_stats: dict[str, dict[str, int]] = {}
    for r in sessions:
        s = user_stats.setdefault(r.user_id, {"total": 0, "training": 0, "evaluation": 0})
        s["total"] += 1
        s[r.task_type] = s.get(r.task_type, 0) + 1
    with _quotas_lock:
        all_user_ids = sorted(set(user_stats.keys()) | set(_quotas.keys()))
        quotas = {uid: _quotas.get(uid) for uid in all_user_ids}
    rows = []
    for uid in all_user_ids:
        s = user_stats.get(uid, {"total": 0, "training": 0, "evaluation": 0})
        q = quotas.get(uid)
        rows.append({
            "user_id": uid,
            "total_sessions": s["total"],
            "training_sessions": s["training"],
            "evaluation_sessions": s["evaluation"],
            "quota": q.to_dict() if q else None,
        })
    return rows


def _build_dashboard_data() -> dict[str, Any]:
    """Collect all dashboard data into a single dict."""
    with _nodes_lock:
        node_snapshot = list(_nodes.values())
        max_envs = sum(n.max_envs for n in node_snapshot)
        healthy = sum(1 for n in node_snapshot if n.status == "healthy")
    sessions_snapshot = _sessions.list_all()
    session_count = len(sessions_snapshot)
    training_sessions = sum(1 for r in sessions_snapshot if r.task_type == "training")
    all_sessions_wire = [r.to_dict() for r in sessions_snapshot]
    users = _build_user_rows(sessions_snapshot)
    with _quotas_lock:
        quotas_out = []
        for q in _quotas.values():
            d = q.to_dict()
            t, tr = _sessions.count_by_user(q.user_id)
            d["current_envs"] = t
            d["current_training_envs"] = tr
            quotas_out.append(d)

    _refresh_job_statuses()
    with _master_jobs_lock:
        jobs = [r.to_dict() for r in _master_jobs.values()]

    all_emus: list[dict[str, Any]] = []
    node_env_counts: dict[str, tuple[int, int]] = {}  # node_id -> (total, busy)
    node_resources: dict[str, dict[str, Any]] = {}
    healthy_nodes_list = [n for n in node_snapshot if n.status != "dead"]
    for nd in healthy_nodes_list:
        slots_data: dict[str, Any] = {}
        try:
            slots_resp = HTTP.get(f"{nd.node_url}/slots", timeout=10)
            slots_data = slots_resp.json()
        except Exception as exc:
            logger.debug("Dashboard /slots fetch failed node_id=%s: %s", nd.node_id, exc)
            slots_data = {"ok": False, "slots": {}, "resources": {}}

        resources = slots_data.get("resources") or {}
        node_resources[nd.node_id] = resources
        if resources:
            with nd.lock:
                nd.resources = resources

        # Each slot carries whatever its adapter self-reported (incl. per-slot
        # container_id/docker_state for docker worlds); the master neither scans
        # nor aggregates docker — it only flattens the slots for display.
        node_emus = _fetch_node_slots_from_data(slots_data)
        node_busy = sum(1 for e in node_emus if e.get("busy"))
        node_env_counts[nd.node_id] = (len(node_emus), node_busy)
        for emu in node_emus:
            emu["node_id"] = nd.node_id
            emu["node_url"] = nd.node_url
            all_emus.append(emu)

    with _nodes_lock:
        nodes = [n.to_dict() for n in _nodes.values()]

    # Override heartbeat-cached counts with live data from /slots
    for n in nodes:
        live = node_env_counts.get(n["node_id"])
        if live is not None:
            n["total_envs"] = live[0]
            n["busy_envs"] = live[1]
            n["idle_envs"] = live[0] - live[1]

    total_envs = sum(n.get("total_envs", 0) for n in nodes)
    busy_envs = sum(n.get("busy_envs", 0) for n in nodes)
    idle_envs = total_envs - busy_envs
    cluster_qemu_count = sum(int((n.get("resources") or {}).get("qemu_count", 0) or 0) for n in nodes)
    cluster_qemu_d_state_count = sum(
        int((n.get("resources") or {}).get("qemu_d_state_count", 0) or 0) for n in nodes
    )

    return {
        "cluster": {
            "scheduler": _config.scheduler if _config else None,
            "total_nodes": len(nodes), "healthy_nodes": healthy,
            "total_envs": total_envs, "busy_envs": busy_envs,
            "idle_envs": idle_envs, "max_envs": max_envs,
            "active_sessions": session_count,
            "training_sessions": training_sessions,
            "evaluation_sessions": session_count - training_sessions,
            "qemu_count": cluster_qemu_count,
            "qemu_d_state_count": cluster_qemu_d_state_count,
        },
        "nodes": nodes, "sessions": all_sessions_wire,
        "users": users, "quotas": quotas_out,
        "jobs": jobs, "available_scripts": _scan_available_scripts(),
        "script_params": _scan_all_script_params(),
        "emulators": all_emus,
        "node_resources": node_resources,
        "debug_events": _collect_debug_events(100, include_nodes=True),
    }


def _redirect_with_flash(msg: str = "", error: str = "") -> Response:
    parts = []
    if msg:
        parts.append(f"msg={quote(msg)}")
    if error:
        parts.append(f"error={quote(error)}")
    qs = f"?{'&'.join(parts)}" if parts else ""
    return redirect(f"/{qs}")


def _forward_admin_to_node(node_id: str, path: str, body: dict, *, timeout: int = 300, label: str = "") -> Response:
    node_url = _get_node_url(node_id)
    if not node_url:
        return _redirect_with_flash(error=f"Node {node_id!r} not found")
    try:
        resp = HTTP.post(f"{node_url}{path}", json=body, timeout=timeout)
        data = resp.json()
        if not data.get("ok"):
            err = data.get("error", "unknown error")
            logger.warning("%s on %s failed: %s", path, node_id, err)
            return _redirect_with_flash(error=f"{label or path} failed: {err}")
    except requests.Timeout:
        logger.warning("%s on %s timed out (%ds)", path, node_id, timeout)
        return _redirect_with_flash(error=f"{label or path} timed out after {timeout}s")
    except Exception as exc:
        logger.warning("%s on %s failed: %s", path, node_id, exc)
        return _redirect_with_flash(error=f"{label or path} failed: {exc}")
    return _redirect_with_flash(msg=f"{label or path} succeeded")


_ACTION_DISPATCH: dict[str, callable] = {}


def _dashboard_action(name: str):
    """Register a function as a dashboard action dispatched via ?action=name."""
    def decorator(fn):
        _ACTION_DISPATCH[name] = fn
        return fn
    return decorator


@app.route("/", methods=["GET", "POST"])
@app.route("/dashboard", methods=["GET", "POST"])
@app.route("/dashboard/", methods=["GET", "POST"])
def dashboard():
    action = flask_request.args.get("action") or flask_request.form.get("action") or ""
    if action and action in _ACTION_DISPATCH:
        return _ACTION_DISPATCH[action]()

    html = _DASHBOARD_PATH.read_text(encoding="utf-8")
    data = _build_dashboard_data()
    flash_msg = flask_request.args.get("msg", "")
    flash_err = flask_request.args.get("error", "")
    inject = f"window.__REPO_ROOT__ = {json.dumps(str(_REPO_ROOT))};\n"
    inject += f"window.__DATA__ = {json.dumps(data)};\n"
    if flash_msg:
        inject += f"window.__FLASH_MSG__ = {json.dumps(flash_msg)};\n"
    if flash_err:
        inject += f"window.__FLASH_ERR__ = {json.dumps(flash_err)};\n"
    html = html.replace("/*__INJECT__*/", inject, 1)
    resp = Response(html, mimetype="text/html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.get("/debug/events")
def debug_events_api():
    tail = flask_request.args.get("tail", default=200, type=int) or 200
    include_nodes = str(flask_request.args.get("include_nodes", "1")).lower() not in {"0", "false", "no"}
    return jsonify({
        "ok": True,
        "events": _collect_debug_events(tail, filters=_debug_event_filters(), include_nodes=include_nodes),
    })


# ---------------------------------------------------------------------------
# Node management
# ---------------------------------------------------------------------------
def _check_node_secret(body: dict[str, Any]) -> Response | None:
    """Reject (403) if a node secret is configured and the body's doesn't match.
    Returns an error Response to short-circuit, or None to proceed."""
    if _config and _config.node_secret and body.get("secret") != _config.node_secret:
        return jsonify({"ok": False, "error": "invalid secret"}), 403
    return None


@app.post("/node/register")
def node_register():
    body = _json_body()
    node_id = body.get("node_id")
    node_url = body.get("node_url")
    if not node_id or not node_url:
        return jsonify({"ok": False, "error": "node_id and node_url are required"}), 400

    rejected = _check_node_secret(body)
    if rejected is not None:
        return rejected

    node_url = node_url.rstrip("/")
    with _nodes_lock:
        existing = _nodes.get(node_id)
        if existing is not None:
            with existing.lock:
                existing.node_url = node_url
                existing.max_envs = int(body.get("max_envs", existing.max_envs))
                existing.labels = body.get("labels", existing.labels)
                existing.provider_name = body.get("provider_name", existing.provider_name)
                existing.os_type = body.get("os_type", existing.os_type)
                existing.action_space = body.get("action_space", existing.action_space)
                existing.runtimes = body.get("runtimes", existing.runtimes)
                existing.capabilities = body.get("capabilities", existing.capabilities)
                existing.status = "healthy"
                existing.last_heartbeat_ts = time.time()
            logger.info("Node %s re-registered at %s", node_id, node_url)
        else:
            node = NodeInfo(
                node_id=node_id,
                node_url=node_url,
                max_envs=int(body.get("max_envs", 1)),
                labels=body.get("labels", {}),
                provider_name=body.get("provider_name", ""),
                os_type=body.get("os_type", "Ubuntu"),
                action_space=body.get("action_space", "pyautogui"),
                runtimes=body.get("runtimes", []),
                capabilities=body.get("capabilities", {}),
            )
            _nodes[node_id] = node
            logger.info("Node %s registered at %s (max_envs=%d)", node_id, node_url, node.max_envs)

    return jsonify({"ok": True, "node_id": node_id})


@app.post("/node/heartbeat")
def node_heartbeat():
    body = _json_body()
    node_id = body.get("node_id")
    if not node_id:
        return jsonify({"ok": False, "error": "node_id is required"}), 400

    rejected = _check_node_secret(body)
    if rejected is not None:
        return rejected

    with _nodes_lock:
        node = _nodes.get(node_id)
    if node is None:
        return jsonify({"ok": False, "error": f"unknown node_id: {node_id}", "re_register": True}), 404

    now = time.time()
    with node.lock:
        node.last_heartbeat_ts = now
        node.total_envs = int(body.get("total_envs", node.total_envs))
        node.busy_envs = int(body.get("busy_envs", node.busy_envs))
        node.idle_envs = int(body.get("idle_envs", node.idle_envs))
        # reserved_envs is NOT reset here: it is in-flight allocation state owned
        # solely by _create_one_session (every +1 has a paired -1 on all exits).
        # Zeroing it from a heartbeat would erase reservations for creates still
        # waiting on a slow node build, letting a concurrent create oversubscribe.
        resources = body.get("resources")
        if isinstance(resources, dict):
            node.resources = resources
        node.prewarm_done = bool(body.get("prewarm_done", node.prewarm_done))
        slots = body.get("slots")
        if isinstance(slots, list):
            node.slots = slots
            node.slots_ts = now
        if node.status in ("unhealthy", "dead"):
            node.status = "healthy"
            logger.info("Node %s recovered to healthy", node_id)

    if isinstance(slots, list):
        _reindex_node_slots(node_id, slots)

    valid_sessions = [r.session_id for r in _sessions.get_by_node(node_id)]
    return jsonify({"ok": True, "valid_sessions": valid_sessions})


@app.post("/node/deregister")
def node_deregister():
    body = _json_body()
    node_id = body.get("node_id")
    if not node_id:
        return jsonify({"ok": False, "error": "node_id is required"}), 400

    with _nodes_lock:
        node = _nodes.pop(node_id, None)
    if node is None:
        return jsonify({"ok": False, "error": f"unknown node_id: {node_id}"}), 404
    _reindex_node_slots(node_id, None)

    orphaned = [r.session_id for r in _sessions.purge_for_node(node_id)]
    if orphaned:
        logger.info("Cleaned %d orphaned sessions from deregistered node %s", len(orphaned), node_id)

    logger.info("Node %s deregistered", node_id)
    return jsonify({"ok": True})


@app.get("/nodes")
def list_nodes():
    with _nodes_lock:
        nodes = [n.to_dict() for n in _nodes.values()]
    return jsonify({"ok": True, "nodes": nodes})


# ---------------------------------------------------------------------------
# Health reaper
# ---------------------------------------------------------------------------
_RECONCILE_GRACE_SECONDS = float(os.environ.get("RECONCILE_GRACE_SECONDS", "30"))


def _health_reaper_loop(unhealthy_timeout: float, dead_timeout: float) -> None:
    while True:
        time.sleep(10)
        now = time.time()
        with _nodes_lock:
            for node in _nodes.values():
                with node.lock:
                    age = now - node.last_heartbeat_ts
                    if age > dead_timeout and node.status != "dead":
                        logger.warning("Node %s marked dead (no heartbeat for %.0fs)", node.node_id, age)
                        node.status = "dead"
                    elif age > unhealthy_timeout and node.status == "healthy":
                        logger.warning("Node %s marked unhealthy (no heartbeat for %.0fs)", node.node_id, age)
                        node.status = "unhealthy"

        # Reap idle/expired sessions and purge sessions on dead nodes. Sessions
        # are the single bookkeeping model, so this is the only reaper.
        _reap_sessions(now)


def _reap_sessions(now: float) -> None:
    """Expire stale sessions and purge sessions belonging to dead nodes."""
    for rec in _sessions.expire_stale(now):
        logger.warning(
            "Session %s expired (world=%s node=%s idle %.0fs), closing",
            rec.session_id, rec.world_id, rec.node_id, now - rec.last_activity_ts,
        )
        _drop_session(rec, timeout=10)
        # Unconditional decrement is correct: expire_stale() already flipped
        # status to EXPIRED, and every busy session was ACTIVE when it counted.
        with _nodes_lock:
            node = _nodes.get(rec.node_id)
        if node is not None:
            with node.lock:
                node.busy_envs = max(0, node.busy_envs - 1)

    with _nodes_lock:
        dead_node_ids = [nid for nid, n in _nodes.items() if n.status == "dead"]
    for nid in dead_node_ids:
        _reindex_node_slots(nid, None)
        for rec in _sessions.purge_for_node(nid):
            logger.warning("Purged session %s on dead node %s", rec.session_id, nid)


# ---------------------------------------------------------------------------
# Reconciliation — prune master bookkeeping that has drifted from node reality.
#
# The node is the source of truth. Capacity counts AND the slot view are kept
# fresh by the heartbeat (every ~10s); reconcile (every ~60s) reuses that fed
# snapshot — no extra GET — to do the one thing the heartbeat can't: drop a
# PHANTOM session (master has a record the node's slots no longer back).
#
# It deliberately does NOT touch ORPHANS (a session the node runs but the master
# forgot, e.g. after a master restart): tearing one down would kill a live eval
# on master amnesia. The node's own heartbeat reclaim unbinds those keep-warm.
# ---------------------------------------------------------------------------
def _reconcile_node(node: NodeInfo, now: float) -> None:
    # session_id is present only on busy slots; idle slots carry None and simply
    # don't contribute (they back no session, so can't make one a phantom).
    node_session_ids = {
        s.get("session_id") for s in node.slots
        if s.get("busy") and s.get("session_id")
    }
    phantoms = [
        rec.session_id
        for rec in _sessions.get_by_node(node.node_id)
        if rec.session_id not in node_session_ids
        and now - rec.created_ts > _RECONCILE_GRACE_SECONDS
    ]
    for sid in phantoms:
        _sessions.delete(sid)
    if phantoms:
        logger.warning("Reconcile: removed %d phantom session(s) from node %s: %s",
                       len(phantoms), node.node_id, phantoms)


def _reconcile_loop(interval: float = 60.0) -> None:
    logger.info("Reconcile loop started (interval=%.0fs)", interval)
    while True:
        time.sleep(interval)
        now = time.time()
        with _nodes_lock:
            nodes_snapshot = list(_nodes.values())
        for node in nodes_snapshot:
            if node.status == "dead":
                dead = [r.session_id for r in _sessions.purge_for_node(node.node_id)]
                if dead:
                    logger.warning("Reconcile: purged %d sessions from dead node %s", len(dead), node.node_id)
                continue
            _reconcile_node(node, now)


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------
_JOB_USER_TOKEN = "@@osworld-job:"


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "__all__":
        return None
    return text


def _count_user_sessions(user_id: str) -> tuple[int, int]:
    """(total, training) session counts for a user — quota accounting."""
    return _sessions.count_by_user(user_id)




# ---------------------------------------------------------------------------
# VNC proxy — forward browser VNC requests through master to the correct node.
# HTTP static assets and the noVNC WebSocket stream both stay on master:18000.
# ---------------------------------------------------------------------------

#: A container-local origin (any port). Web viewers (Gradio) emit these both in
#: their HTML/JSON config and inline in SSE events; we rewrite them to the public
#: master path so the browser reaches assets/files through us, not 127.0.0.1.
_LOCAL_ORIGIN_RE = re.compile(rb"http://127\.0\.0\.1:\d{1,5}")


def _rewrite_stream(chunks, repl: bytes):
    """Rewrite origins in a byte stream, flushing on newlines. SSE is newline-
    framed and an origin token never spans a line, so buffering until the last
    '\\n' makes the rewrite immune to where 4096-byte chunk boundaries fall."""
    buf = b""
    for chunk in chunks:
        buf += chunk
        nl = buf.rfind(b"\n")
        if nl >= 0:
            yield _LOCAL_ORIGIN_RE.sub(repl, buf[:nl + 1])
            buf = buf[nl + 1:]
    if buf:
        yield _LOCAL_ORIGIN_RE.sub(repl, buf)


def _fetch_node_slots_from_data(data: dict) -> list[dict]:
    """Extract flat slot list from /slots response data."""
    if "slots" in data and isinstance(data["slots"], dict):
        return [s for slots in data["slots"].values() for s in slots]
    return data.get("slots", data.get("emulators", []))


def _fetch_node_slots(node_url: str, timeout: float = 3) -> list[dict]:
    """Fetch slot list from a node, handling both response formats."""
    resp = HTTP.get(f"{node_url}/slots", timeout=timeout)
    return _fetch_node_slots_from_data(resp.json())


def _find_node_for_slot(slot_id: str) -> str | None:
    """Resolve the node hosting ``slot_id`` from the heartbeat-fed slot index.

    O(1) and always fresh to within one heartbeat — no rescan, no TTL cache that
    could outlive the slot and point at a dead node. A dead node's url is never
    returned even if a stale index entry lingers until the next reindex."""
    with _nodes_lock:
        node = _nodes.get(_slot_index.get(slot_id, ""))
        if node and node.status != "dead":
            return node.node_url
    return None


@sock.route("/ws/view/<slot_id>")
def master_view_websockify(ws, slot_id: str):
    node_url = _find_node_for_slot(slot_id)
    if not node_url:
        logger.warning("Rejecting view websocket for %s: not found on any node", slot_id)
        close_quietly(ws)
        return
    upstream_url = http_to_ws_url(f"{node_url}/view/{quote(slot_id, safe='')}/websockify")
    qs = flask_request.query_string.decode()
    if qs:
        upstream_url += f"?{qs}"
    proxy_websocket(ws, upstream_url, logger, label=f"view ws {slot_id}")


@app.route("/view/<slot_id>/", defaults={"subpath": ""}, methods=["GET", "POST"])
@app.route("/view/<slot_id>/<path:subpath>", methods=["GET", "POST"])
def master_view_proxy(slot_id: str, subpath: str):
    node_url = _find_node_for_slot(slot_id)
    if not node_url:
        return jsonify({"ok": False, "error": f"slot {slot_id} not found"}), 404
    target_url = f"{node_url}/view/{slot_id}/{subpath}"
    qs = flask_request.query_string.decode()
    if qs:
        target_url += f"?{qs}"
    try:
        resp = proxy_request(target_url)
        # Rewrite the upstream's local origin (127.0.0.1:<container-port>) to the
        # public master path so the viewer's assets/config AND the absolute file
        # URLs Gradio pushes inside its SSE event stream resolve through us.
        ct = resp.content_type or ""
        public_root = f"http://{flask_request.host}/view/{slot_id}".encode()
        if "event-stream" in ct:
            # SSE is long-lived: rewrite chunk-by-chunk, never buffer the whole body.
            resp.response = _rewrite_stream(resp.response, public_root)
        elif "html" in ct or "json" in ct:
            resp.set_data(_LOCAL_ORIGIN_RE.sub(public_root, resp.get_data()))
        return resp
    except Exception as exc:
        return jsonify({"ok": False, "error": f"view proxy failed: {exc}"}), 502


# ---------------------------------------------------------------------------
# Session management plane (data-plane ops normally go client -> node directly).
# ---------------------------------------------------------------------------
# Clients talk the neutral /v1/sessions protocol straight to the owning node
# (node_url from the acquire response); the master owns scheduling, bookkeeping
# (/v1/sessions DELETE), and liveness (/v1/sessions/<id>/heartbeat). The master
# also exposes /v1/sessions/<id>/{reset,step,observe,evaluate} forwarders below,
# but they are only a fallback for callers that lack a node_url — the SDK
# clients, which always carry node_url, bypass them.
def _touch_session(session_id: str) -> bool:
    rec = _sessions.get(session_id)
    if rec is None:
        return False
    rec.touch()
    return True




# ---------------------------------------------------------------------------
# Programmatic EnvSession API (/v1/sessions) — the world-neutral control plane.
# The explicit, runtime-typed entry point used by the dashboard and SDK clients.
# Forwarding to the node passes the neutral protocol through unchanged (no
# OSWorld translation — callers here already speak the /v1 dialect).
# ---------------------------------------------------------------------------
def _forward_v1(record: SessionRecord, method: str, subpath: str, body: dict[str, Any] | None):
    url = f"{record.node_url}/v1/sessions/{record.session_id}{subpath}"
    try:
        resp = HTTP.request(method, url, json=body or {}, timeout=_FORWARD_TIMEOUT)
        return resp.json(), resp.status_code
    except Exception as exc:  # noqa: BLE001
        logger.warning("forward %s %s failed: %s", method, url, exc)
        return {"ok": False, "error": str(exc)}, 502


@app.post("/v1/sessions")
def v1_create_session():
    body = _json_body()
    world_id = body.get("runtime") or body.get("world_id")
    if not world_id:
        return jsonify({"ok": False, "error": "runtime (world_id) required"}), 400
    created, error, code = _create_sessions(
        world_id=world_id,
        count=int(body.get("count", 1) or 1),
        ttl_seconds=int(body.get("ttl_seconds", 7200) or 7200),
        user_id=body.get("user_id", "anonymous"),
        mode=body.get("mode", "eval"),
        task_type=body.get("task_type", "evaluation"),
        task_payload=body.get("task_payload", {}) or {},
        capability_requirements=body.get("capability_requirements", {}) or {},
        resources=body.get("resources", {}) or {},
    )
    sessions = [
        {"session_id": r.session_id, "runtime": r.world_id, "node_id": r.node_id, "node_url": r.node_url}
        for r in created
    ]
    if error:
        return jsonify({"ok": False, "error": error, "sessions": sessions}), code
    return jsonify({"ok": True, "sessions": sessions})


@app.post("/v1/sessions/<session_id>/reset")
def v1_reset(session_id):
    rec = _sessions.get(session_id)
    if rec is None:
        return jsonify({"ok": False, "error": "session not found"}), 404
    rec.touch()
    data, code = _forward_v1(rec, "POST", "/reset", {"task_payload": _json_body().get("task_payload", {})})
    return jsonify(data), code


@app.post("/v1/sessions/<session_id>/step")
def v1_step(session_id):
    rec = _sessions.get(session_id)
    if rec is None:
        return jsonify({"ok": False, "error": "session not found"}), 404
    rec.touch()
    b = _json_body()
    data, code = _forward_v1(rec, "POST", "/step", {"action": b.get("action"), "pause": b.get("pause")})
    return jsonify(data), code


@app.post("/v1/sessions/<session_id>/observe")
def v1_observe(session_id):
    rec = _sessions.get(session_id)
    if rec is None:
        return jsonify({"ok": False, "error": "session not found"}), 404
    rec.touch()
    data, code = _forward_v1(rec, "POST", "/observe", {})
    return jsonify(data), code


@app.post("/v1/sessions/<session_id>/evaluate")
def v1_evaluate(session_id):
    rec = _sessions.get(session_id)
    if rec is None:
        return jsonify({"ok": False, "error": "session not found"}), 404
    rec.touch()
    data, code = _forward_v1(rec, "POST", "/evaluate", {})
    return jsonify(data), code


@app.post("/v1/sessions/<session_id>/heartbeat")
def v1_heartbeat(session_id):
    if not _touch_session(session_id):
        return jsonify({"ok": False, "error": "session not found"}), 404
    return jsonify({"ok": True})


@app.delete("/v1/sessions/<session_id>")
def v1_close(session_id):
    rec = _sessions.get(session_id)
    if rec is None:
        return jsonify({"ok": False, "error": "session not found"}), 404
    _close_session(rec)
    return jsonify({"ok": True})


@app.get("/v1/sessions")
def v1_list_sessions():
    world_id = flask_request.args.get("runtime")
    node_id = flask_request.args.get("node_id")
    status_f = flask_request.args.get("status")
    sessions = _sessions.list_all(world_id=world_id, node_id=node_id, status=status_f)
    return jsonify({"ok": True, "sessions": [s.to_dict() for s in sessions]})


@app.get("/v1/sessions/<session_id>")
def v1_get_session(session_id):
    rec = _sessions.get(session_id)
    if rec is None:
        return jsonify({"ok": False, "error": "session not found"}), 404
    return jsonify({"ok": True, "session": rec.to_dict()})


# ---------------------------------------------------------------------------
# Quota management
# ---------------------------------------------------------------------------
@app.get("/quotas")
def list_quotas():
    with _quotas_lock:
        quotas = list(_quotas.values())
    result = []
    for q in quotas:
        total, training = _count_user_sessions(q.user_id)
        d = q.to_dict()
        d["current_envs"] = total
        d["current_training_envs"] = training
        result.append(d)
    return jsonify({"ok": True, "quotas": result})


@_dashboard_action("set_quota")
def set_quota():
    user_id = _form_val("user_id")
    if not user_id:
        return redirect("/")
    max_envs = int(_form_val("max_envs") or 0)
    max_training = int(_form_val("max_training_envs") or 0)
    with _quotas_lock:
        existing = _quotas.get(user_id)
        if existing is not None:
            existing.max_envs = max_envs
            existing.max_training_envs = max_training
        else:
            _quotas[user_id] = UserQuota(user_id=user_id, max_envs=max_envs, max_training_envs=max_training)
    return redirect("/")


@_dashboard_action("delete_quota")
def delete_quota():
    user_id = _form_val("user_id")
    if user_id:
        with _quotas_lock:
            _quotas.pop(user_id, None)
    return redirect("/")


# ---------------------------------------------------------------------------
# Admin management (all form-based, redirect to dashboard)
# ---------------------------------------------------------------------------
def _form_val(key: str) -> str:
    return (flask_request.form.get(key) or flask_request.args.get(key) or _json_body().get(key) or "").strip()


_RELEASE_FILTER_KEYS = ("session_id", "user_id", "task_type", "job_id", "job_script")


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "all"}


def _release_filters_from_values(values: dict[str, Any]) -> tuple[dict[str, str], bool, str | None]:
    filters: dict[str, str] = {}
    for key in _RELEASE_FILTER_KEYS:
        value = _optional_str(values.get(key))
        if value:
            filters[key] = value
    if "task_type" in filters and filters["task_type"] not in ("training", "evaluation"):
        return filters, False, "task_type must be training or evaluation"
    release_all = _bool_value(values.get("all"))
    if not filters and not release_all:
        return filters, release_all, "At least one release filter or all=true is required"
    return filters, release_all, None


def _session_matches_filters(record: SessionRecord, filters: dict[str, str]) -> bool:
    for key, expected in filters.items():
        if key == "session_id":
            actual = record.session_id
        else:
            actual = getattr(record, key, None)
        if actual != expected:
            return False
    return True


def _release_matching_sessions(
    filters: dict[str, str],
    *,
    reason: str,
    release_all: bool = False,
    close_timeout: int = 10,
) -> dict[str, Any]:
    matched = [
        record
        for record in _sessions.list_all()
        if release_all or _session_matches_filters(record, filters)
    ]
    for record in matched:
        _sessions.delete(record.session_id)

    node_counts: dict[str, int] = {}
    for record in matched:
        if record.status == SessionStatus.ACTIVE.value:
            node_counts[record.node_id] = node_counts.get(record.node_id, 0) + 1
    if node_counts:
        with _nodes_lock:
            for node_id, dec in node_counts.items():
                node = _nodes.get(node_id)
                if node is None:
                    continue
                with node.lock:
                    node.busy_envs = max(0, node.busy_envs - dec)

    failures: list[dict[str, str]] = []

    def _close_on_node(record: SessionRecord) -> dict[str, str] | None:
        try:
            resp = HTTP.delete(
                f"{record.node_url}/v1/sessions/{record.session_id}", timeout=close_timeout
            )
            if resp.status_code >= 400:
                return {
                    "session_id": record.session_id,
                    "node_id": record.node_id,
                    "error": f"HTTP {resp.status_code}: {short_excerpt(resp.text)}",
                }
        except Exception as exc:
            return {"session_id": record.session_id, "node_id": record.node_id, "error": str(exc)}
        return None

    if matched:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, len(matched))) as pool:
            futures = [pool.submit(_close_on_node, record) for record in matched]
            for future in concurrent.futures.as_completed(futures):
                failure = future.result()
                if failure is not None:
                    failures.append(failure)
                    logger.warning(
                        "Failed to close released session %s on node %s: %s",
                        failure["session_id"],
                        failure["node_id"],
                        failure["error"],
                    )

    session_ids = [record.session_id for record in matched]
    logger.info(
        "Released %d session(s) reason=%s all=%s filters=%s failures=%d",
        len(matched),
        reason,
        release_all,
        filters,
        len(failures),
    )
    return {
        "released": len(matched),
        "session_ids": session_ids,
        "filters": dict(filters),
        "all": bool(release_all),
        "failures": failures,
    }


@_dashboard_action("release_session")
def admin_release_session():
    session_id = _form_val("session_id")
    if not session_id:
        return _redirect_with_flash(error="session_id is required")
    result = _release_matching_sessions({"session_id": session_id}, reason="dashboard release_session", close_timeout=30)
    return _redirect_with_flash(msg=f"Released {result['released']} session(s)")


@_dashboard_action("release_all_sessions")
def admin_release_all_sessions():
    result = _release_matching_sessions({}, reason="dashboard release_all_sessions", release_all=True)
    return _redirect_with_flash(msg=f"Released {result['released']} session(s)")


@_dashboard_action("release_sessions")
def admin_release_sessions():
    values = {key: _form_val(key) for key in _RELEASE_FILTER_KEYS}
    filters, release_all, error = _release_filters_from_values(values)
    if error:
        return _redirect_with_flash(error=error)
    if not filters and not release_all:
        release_all = True
    result = _release_matching_sessions(filters, reason="dashboard release_session", release_all=release_all)
    return _redirect_with_flash(msg=f"Released {result['released']} session(s)")


@_dashboard_action("drain_node")
def admin_drain_node():
    node_id = _form_val("node_id")
    with _nodes_lock:
        node = _nodes.get(node_id)
    if node is not None:
        with node.lock:
            node.draining = True
        logger.info("Admin drained node %s", node_id)
    return redirect("/")


@_dashboard_action("undrain_node")
def admin_undrain_node():
    node_id = _form_val("node_id")
    with _nodes_lock:
        node = _nodes.get(node_id)
    if node is not None:
        with node.lock:
            node.draining = False
        logger.info("Admin undrained node %s", node_id)
    return redirect("/")


@_dashboard_action("set_max_envs")
def admin_set_max_envs():
    node_id = _form_val("node_id")
    max_envs = _form_val("max_envs")
    if not node_id or not max_envs:
        return redirect("/")
    max_envs_int = int(max_envs)
    with _nodes_lock:
        node = _nodes.get(node_id)
    if node is None:
        return _redirect_with_flash(error=f"Node {node_id!r} not found")
    # Update master state
    with node.lock:
        node.max_envs = max_envs_int
    # Sync to node
    try:
        HTTP.post(f"{node.node_url}/set_max_envs", json={"max_envs": max_envs_int}, timeout=5)
    except Exception as exc:
        logger.warning("Failed to sync max_envs to node %s: %s", node_id, exc)
    logger.info("Admin set node %s max_envs=%d", node_id, max_envs_int)
    return redirect("/")


# ---------------------------------------------------------------------------
# Environment management (forwarded to nodes)
# ---------------------------------------------------------------------------
def _get_node_url(node_id: str) -> str | None:
    with _nodes_lock:
        node = _nodes.get(node_id)
    return node.node_url if node else None


@_dashboard_action("env_create")
def admin_env_create():
    """Create idle env slots on a node (no session bound).

    Delegates to the node's POST /env/create which creates slots in parallel
    using a thread pool. The environments are idle and ready for allocation.
    """
    node_id = _form_val("node_id")
    runtime = _form_val("runtime").strip()  # empty -> node resolves its sole world
    try:
        count = max(1, min(int(_form_val("count") or 1), 50))
    except ValueError:
        return _redirect_with_flash(error="count must be an integer")

    if not node_id or node_id == "__all__":
        with _nodes_lock:
            candidates = list(_nodes.values())
        node = None
        for n in candidates:
            with n.lock:
                if n.status == "healthy" and not n.draining:
                    node = n
                    break
        if node is None:
            return _redirect_with_flash(error="No healthy node available")
        node_id = node.node_id
        node_url = node.node_url
    else:
        node_url = _get_node_url(node_id)
        if not node_url:
            return _redirect_with_flash(error=f"Node {node_id!r} not found")

    def _do_create() -> None:
        try:
            resp = HTTP.post(
                f"{node_url}/env/create",
                json={"runtime": runtime, "count": count},
                timeout=_FORWARD_TIMEOUT,
            )
            data = resp.json()
            if not resp.ok or not data.get("ok"):
                err = data.get("error", f"HTTP {resp.status_code}")
                logger.warning("Dashboard env_create on %s failed: %s", node_id, err)
                return
            created = int(data.get("created", 0) or 0)
            logger.info("Dashboard env_create on %s: created %d %s slot(s)",
                        node_id, created, runtime)
        except requests.Timeout:
            logger.warning("Dashboard env_create on %s timed out", node_id)
        except Exception as exc:
            logger.warning("Dashboard env_create on %s failed: %s", node_id, exc)

    threading.Thread(target=_do_create, daemon=True).start()
    return _redirect_with_flash(msg=f"Creating {count} {runtime} env(s) on {node_id}... refresh to see progress")


@_dashboard_action("env_delete")
def admin_env_delete():
    node_id = _form_val("node_id")
    env_id = _form_val("env_id")
    if not node_id or not env_id:
        return _redirect_with_flash(error="node_id and env_id are required")

    # If it's a session, close it on master side too
    record = _sessions.get(env_id)
    if record is not None:
        _close_session(record, timeout=120)
        return _redirect_with_flash(msg=f"Closed session {env_id}")

    # Otherwise it's an idle slot — ask the node to delete it directly
    node_url = _get_node_url(node_id)
    if not node_url:
        return _redirect_with_flash(error=f"Node {node_id!r} not found")
    try:
        resp = HTTP.post(
            f"{node_url}/env/delete",
            json={"slot_id": env_id},
            timeout=30,
        )
        data = resp.json()
        if not resp.ok or not data.get("ok"):
            return _redirect_with_flash(error=data.get("error", f"HTTP {resp.status_code}"))
        return _redirect_with_flash(msg=f"Deleted slot {env_id}")
    except Exception as exc:
        return _redirect_with_flash(error=f"Failed to delete on node: {exc}")


@_dashboard_action("env_delete_all")
def admin_env_delete_all():
    node_id = _form_val("node_id")

    # Close any active sessions
    if node_id and node_id != "__all__":
        victims = _sessions.get_by_node(node_id)
    else:
        victims = _sessions.list_all()
    for rec in victims:
        try:
            _close_session(rec, timeout=30)
        except Exception:
            pass

    # Delete all slots on target node(s)
    if node_id and node_id != "__all__":
        node_url = _get_node_url(node_id)
        if not node_url:
            return _redirect_with_flash(error=f"Node {node_id!r} not found")
        targets = [(node_id, node_url)]
    else:
        with _nodes_lock:
            targets = [(n.node_id, n.node_url) for n in _nodes.values() if n.status != "dead"]

    def _do_delete_all():
        def _delete_on_node(item):
            nid, url = item
            try:
                resp = HTTP.post(f"{url}/env/delete", json={"all": True}, timeout=300)
                deleted = resp.json().get("deleted", 0)
                logger.info("Dashboard env_delete_all on %s: deleted %d slot(s)", nid, deleted)
            except Exception as exc:
                logger.warning("Dashboard env_delete_all on %s failed: %s", nid, exc)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(targets))) as pool:
            pool.map(_delete_on_node, targets)

    threading.Thread(target=_do_delete_all, daemon=True).start()
    scope = node_id if (node_id and node_id != "__all__") else f"{len(targets)} nodes"
    return _redirect_with_flash(msg=f"Deleting all envs on {scope}... refresh to see progress")


@_dashboard_action("env_restart")
def admin_env_restart():
    node_id = _form_val("node_id")
    env_id = _form_val("env_id")
    if not node_id or not env_id:
        return _redirect_with_flash(error="node_id and env_id are required")

    # If it's an active session, close it first
    record = _sessions.get(env_id)
    if record is not None:
        _close_session(record, timeout=30)

    # Ask node to restart the slot in place
    node_url = _get_node_url(node_id)
    if not node_url:
        return _redirect_with_flash(error=f"Node {node_id!r} not found")
    try:
        resp = HTTP.post(f"{node_url}/env/restart", json={"slot_id": env_id}, timeout=120)
        data = resp.json()
        if not resp.ok or not data.get("ok"):
            return _redirect_with_flash(error=data.get("error", f"HTTP {resp.status_code}"))
        return _redirect_with_flash(msg=f"Restarted slot {env_id}")
    except Exception as exc:
        return _redirect_with_flash(error=f"Restart failed: {exc}")


# ---------------------------------------------------------------------------
# Job execution (local on master + remote on nodes)
# ---------------------------------------------------------------------------
_local_job_pids: dict[str, int] = {}
_ACTIVE_JOB_STATUSES = {"running", "pending"}


def _job_is_finished(status: str | None) -> bool:
    return str(status or "").lower() not in _ACTIVE_JOB_STATUSES


def _forget_master_job_locked(job_id: str) -> bool:
    removed = _master_jobs.pop(job_id, None) is not None
    _local_job_logs.pop(job_id, None)
    _local_job_pids.pop(job_id, None)
    return removed


def _try_clear_remote_job(record: MasterJobRecord) -> None:
    if record.node_id == "master" or not record.node_url:
        return
    try:
        resp = HTTP.post(f"{record.node_url}/jobs/{record.job_id}/clear", json={}, timeout=5)
        if resp.status_code >= 400 and resp.status_code != 404:
            logger.debug(
                "Remote job clear returned HTTP %s node_id=%s job_id=%s",
                resp.status_code,
                record.node_id,
                record.job_id,
            )
    except Exception as exc:
        logger.debug("Remote job clear failed node_id=%s job_id=%s: %s", record.node_id, record.job_id, exc)


def _run_local_job(job_id: str, script_path: str, env: dict[str, str] | None) -> None:
    logs = _local_job_logs.setdefault(job_id, [])
    try:
        proc = subprocess.Popen(
            ["bash", script_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, cwd=str(_REPO_ROOT),
            preexec_fn=os.setsid,
        )
        _local_job_pids[job_id] = proc.pid
        for raw_line in iter(proc.stdout.readline, b""):
            line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
            with _master_jobs_lock:
                logs.append(line)
                if len(logs) > MAX_LOG_LINES:
                    _local_job_logs[job_id] = logs[-MAX_LOG_LINES:]
        proc.wait()
        with _master_jobs_lock:
            record = _master_jobs.get(job_id)
            if record and record.status == "running":
                record.status = "done" if proc.returncode == 0 else "failed"
    except Exception as exc:
        with _master_jobs_lock:
            logs.append(f"[ERROR] {exc}")
            record = _master_jobs.get(job_id)
            if record:
                record.status = "failed"
    finally:
        _local_job_pids.pop(job_id, None)
        try:
            os.unlink(script_path)
        except OSError:
            pass


def _parse_script_and_env(form) -> tuple[str, str, dict[str, str]] | None:
    """Parse script content and env vars from form. Returns (script, script_name, env_vars) or None."""
    script_select = form.get("script_select") or ""
    script_content = form.get("script_content") or ""

    if script_select and script_select != "__custom__":
        sp = _REPO_ROOT / "scripts" / "bash" / script_select
        if not sp.is_file():
            return None
        script = sp.read_text(encoding="utf-8")
        script_name = script_select
    elif script_content.strip():
        script = script_content.strip()
        script_name = "custom"
    else:
        return None

    env_vars: dict[str, str] = {}
    for line in (form.get("env_vars") or "").strip().splitlines():
        line = line.strip()
        if "=" in line:
            k, v = line.split("=", 1)
            env_vars[k.strip()] = v.strip()
    return script, script_name, env_vars


@_dashboard_action("jobs_start")
def jobs_start():
    node_id = _form_val("node_id")
    user_id = _form_val("user_id") or "anonymous"

    parsed = _parse_script_and_env(flask_request.form)
    if parsed is None:
        return _redirect_with_flash(error="No script selected or script file not found")
    script, script_name, env_vars = parsed

    is_local = (node_id == "__master__")

    if is_local:
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        fd, sp = tempfile.mkstemp(prefix=f"osworld_{job_id}_", suffix=".sh")
        with os.fdopen(fd, "w") as f:
            f.write(script)
        run_env = os.environ.copy()
        run_env.update(env_vars)
        run_env["BACKGROUND"] = "0"
        run_env["OSWORLD_REAL_USER_ID"] = user_id
        run_env["OSWORLD_USER_ID"] = f"{user_id}{_JOB_USER_TOKEN}{job_id}"
        run_env["OSWORLD_TASK_TYPE"] = run_env.get("OSWORLD_TASK_TYPE") or "evaluation"
        run_env["OSWORLD_JOB_ID"] = job_id
        run_env["OSWORLD_JOB_SCRIPT"] = script_name

        record = MasterJobRecord(
            job_id=job_id, node_id="master", node_url="",
            script_name=script_name, user_id=user_id, env_vars=env_vars,
        )
        with _master_jobs_lock:
            _master_jobs[job_id] = record
            _local_job_logs[job_id] = []
        threading.Thread(target=_run_local_job, args=(job_id, sp, run_env), daemon=True).start()
        logger.info("Job %s started locally by %s (%s)", job_id, user_id, script_name)
    else:
        with _nodes_lock:
            node = _nodes.get(node_id)
        if node is None:
            return _redirect_with_flash(error=f"Unknown node: {node_id}")
        try:
            resp = HTTP.post(
                f"{node.node_url}/run_script",
                json={"script": script, "script_name": script_name, "env_vars": env_vars},
                timeout=15,
            )
            data = resp.json()
        except Exception as exc:
            return _redirect_with_flash(error=f"Failed to start job on node: {exc}")
        if not data.get("ok"):
            return _redirect_with_flash(error=f"Node error: {data.get('error')}")

        job_id = data["job_id"]
        record = MasterJobRecord(
            job_id=job_id, node_id=node_id, node_url=node.node_url,
            script_name=script_name, user_id=user_id, env_vars=env_vars,
        )
        with _master_jobs_lock:
            _master_jobs[job_id] = record
        logger.info("Job %s started on node %s by %s (%s)", job_id, node_id, user_id, script_name)

    return redirect("/")


def _get_job_logs_and_status(record: MasterJobRecord) -> tuple[list[str], str]:
    """Get log lines and current status for a job (local or remote)."""
    if record.node_id == "master":
        with _master_jobs_lock:
            lines = list(_local_job_logs.get(record.job_id, []))
        return lines, record.status
    else:
        try:
            resp = HTTP.get(f"{record.node_url}/jobs/{record.job_id}/logs", timeout=10)
            data = resp.json()
            status = data.get("status", record.status)
            if status != record.status:
                with _master_jobs_lock:
                    record.status = status
            return data.get("lines", []), status
        except Exception as exc:
            return [f"[ERROR] Failed to fetch logs from node: {exc}"], record.status


@app.get("/jobs/<job_id>/log")
def job_log_page(job_id: str):
    with _master_jobs_lock:
        record = _master_jobs.get(job_id)
    if record is None:
        return Response(f"Unknown job: {job_id}", status=404, mimetype="text/plain")

    lines, status = _get_job_logs_and_status(record)
    log_html = (_FRONTEND_DIR / "job_log.html").read_text(encoding="utf-8")
    log_data = json.dumps({"job_id": job_id, "status": status, "node_id": record.node_id,
                             "script_name": record.script_name, "user_id": record.user_id, "lines": lines})
    log_html = log_html.replace("/*__INJECT__*/", f"window.__LOG__ = {log_data};", 1)
    resp = Response(log_html, mimetype="text/html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.get("/node/<node_id>")
def node_detail_page(node_id: str):
    with _nodes_lock:
        node = _nodes.get(node_id)
    if node is None:
        return Response(f"Unknown node: {node_id}", status=404, mimetype="text/plain")

    # Fetch live data from the node (single /slots call returns everything)
    resources: dict[str, Any] = {}
    emulators: list[dict[str, Any]] = []
    try:
        slots_resp = HTTP.get(f"{node.node_url}/slots", timeout=5)
        slots_data = slots_resp.json()
        resources = slots_data.get("resources") or {}
        emulators = _fetch_node_slots_from_data(slots_data)
    except Exception:
        resources = dict(node.resources or {})

    # Update node info with live counts
    node_dict = node.to_dict()
    if emulators:
        node_dict["total_envs"] = len(emulators)
        node_dict["busy_envs"] = sum(1 for e in emulators if e.get("busy"))
        node_dict["idle_envs"] = node_dict["total_envs"] - node_dict["busy_envs"]

    page_data = json.dumps({
        "node": node_dict,
        "resources": resources,
        "emulators": emulators,
    })
    html = (_FRONTEND_DIR / "node_detail.html").read_text(encoding="utf-8")
    html = html.replace("/*__INJECT__*/", f"window.__NODE__ = {page_data};", 1)
    resp = Response(html, mimetype="text/html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@_dashboard_action("jobs_kill")
def jobs_kill():
    job_id = _form_val("job_id")
    if not job_id:
        return _redirect_with_flash(error="job_id is required")
    with _master_jobs_lock:
        record = _master_jobs.get(job_id)
    if record is None:
        return redirect("/")

    if record.node_id == "master":
        pid = _local_job_pids.get(job_id)
        if pid:
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        with _master_jobs_lock:
            record.status = "killed"
    else:
        try:
            resp = HTTP.post(f"{record.node_url}/jobs/{job_id}/kill", json={}, timeout=10)
            if resp.json().get("ok"):
                with _master_jobs_lock:
                    record.status = "killed"
        except Exception as exc:
            logger.warning("Failed to kill job %s: %s", job_id, exc)
    return redirect("/")


@_dashboard_action("jobs_clear")
def jobs_clear():
    job_id = _form_val("job_id")
    _refresh_job_statuses()

    if job_id:
        with _master_jobs_lock:
            record = _master_jobs.get(job_id)
        if record is None:
            return _redirect_with_flash(msg="Job already cleared")
        if not _job_is_finished(record.status):
            return _redirect_with_flash(error="Cannot clear a running job. Kill it first.")

        _try_clear_remote_job(record)
        with _master_jobs_lock:
            current = _master_jobs.get(job_id)
            if current is not None and _job_is_finished(current.status):
                removed = _forget_master_job_locked(job_id)
            else:
                removed = False
        if removed:
            return _redirect_with_flash(msg=f"Cleared job {job_id}")
        return _redirect_with_flash(error="Job became active again; not cleared")

    with _master_jobs_lock:
        records = [r for r in _master_jobs.values() if _job_is_finished(r.status)]

    for record in records:
        _try_clear_remote_job(record)

    removed = 0
    with _master_jobs_lock:
        for record in records:
            current = _master_jobs.get(record.job_id)
            if current is not None and _job_is_finished(current.status):
                if _forget_master_job_locked(record.job_id):
                    removed += 1

    return _redirect_with_flash(msg=f"Cleared {removed} finished job(s)")


def _refresh_job_statuses() -> None:
    with _master_jobs_lock:
        running = [r for r in _master_jobs.values() if r.status == "running" and r.node_id != "master"]
    for record in running:
        try:
            resp = HTTP.get(f"{record.node_url}/jobs/{record.job_id}/status", timeout=5)
            data = resp.json()
            if data.get("ok") and data.get("status") != "running":
                with _master_jobs_lock:
                    record.status = data["status"]
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Status & health
# ---------------------------------------------------------------------------
@app.get("/ping")
def ping():
    return jsonify({"ok": True, "message": "pong", "role": "master"})



@app.get("/healthz")
def healthz():
    with _nodes_lock:
        healthy = sum(1 for n in _nodes.values() if n.status == "healthy")
    return jsonify({"ok": healthy > 0 or len(_nodes) == 0, "healthy_nodes": healthy})


@app.get("/status")
def status():
    with _nodes_lock:
        total_envs = sum(n.total_envs for n in _nodes.values())
        busy_envs = sum(n.busy_envs for n in _nodes.values())
        idle_envs = sum(n.idle_envs for n in _nodes.values())
        max_envs = sum(n.max_envs for n in _nodes.values())
        healthy = sum(1 for n in _nodes.values() if n.status == "healthy")
        nodes = [n.to_dict() for n in _nodes.values()]
        qemu_count = sum(int((n.get("resources") or {}).get("qemu_count", 0) or 0) for n in nodes)
        qemu_d_state_count = sum(
            int((n.get("resources") or {}).get("qemu_d_state_count", 0) or 0) for n in nodes
        )
    sessions_snapshot = _sessions.list_all()
    session_count = len(sessions_snapshot)
    training_sessions = sum(1 for r in sessions_snapshot if r.task_type == "training")
    evaluation_sessions = session_count - training_sessions

    return jsonify({
        "ok": True,
        "cluster": {
            "scheduler": _config.scheduler if _config else None,
            "total_nodes": len(nodes),
            "healthy_nodes": healthy,
            "total_envs": total_envs,
            "busy_envs": busy_envs,
            "idle_envs": idle_envs,
            "max_envs": max_envs,
            "active_sessions": session_count,
            "training_sessions": training_sessions,
            "evaluation_sessions": evaluation_sessions,
            "qemu_count": qemu_count,
            "qemu_d_state_count": qemu_d_state_count,
        },
        "nodes": nodes,
    })


@app.get("/slots")
@app.get("/emulators")
def api_list_slots():
    all_slots: list[dict[str, Any]] = []
    with _nodes_lock:
        healthy_nodes = [n for n in _nodes.values() if n.status != "dead"]
    for node in healthy_nodes:
        try:
            for slot in _fetch_node_slots(node.node_url, timeout=10):
                slot["node_id"] = node.node_id
                slot["node_url"] = node.node_url
                all_slots.append(slot)
        except Exception as exc:
            logger.warning("Failed to fetch slots from node %s: %s", node.node_id, exc)
    return jsonify({"ok": True, "slots": all_slots})


@app.post("/v1/sessions/release")
def release_sessions_api():
    """Bulk release sessions by filter (user_id, task_type, job_id, etc.)."""
    body = _json_body()
    filters, release_all, error = _release_filters_from_values(body)
    if error:
        return jsonify({"ok": False, "error": error}), 400
    result = _release_matching_sessions(filters, reason="api /v1/sessions/release", release_all=release_all)
    return jsonify({"ok": True, **result})


@app.get("/users")
def list_users():
    return jsonify({"ok": True, "users": _build_user_rows(_sessions.list_all())})


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OSWorld Cluster Master Server")
    p.add_argument("--host", type=str, default=os.getenv("MASTER_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.getenv("MASTER_PORT", "18000")))
    p.add_argument(
        "--scheduler",
        type=str,
        default=os.getenv("MASTER_SCHEDULER", "least-loaded"),
        choices=["least-loaded", "round-robin"],
    )
    p.add_argument(
        "--node-secret",
        type=str,
        default=os.getenv("MASTER_NODE_SECRET", ""),
        help="Shared secret for node authentication. Empty means no auth.",
    )
    p.add_argument(
        "--unhealthy-timeout",
        type=float,
        default=float(os.getenv("MASTER_UNHEALTHY_TIMEOUT", "30")),
    )
    p.add_argument(
        "--dead-timeout",
        type=float,
        default=float(os.getenv("MASTER_DEAD_TIMEOUT", "60")),
    )
    p.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("MASTER_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def _scan_available_scripts() -> list[str]:
    scripts_dir = _REPO_ROOT / "scripts" / "bash"
    if not scripts_dir.is_dir():
        return []
    return sorted(str(f.relative_to(scripts_dir)) for f in scripts_dir.rglob("run_*.sh"))


_PARAM_RE = re.compile(r'\$\{(\w+):-((?:[^}$]|\$(?!\{)|\$\{[^}]*\})*)\}')
_INTERNAL_VARS = frozenset({
    "TIMESTAMP", "LOG_FILE", "PID_FILE", "PROXY_CONFIG_FILE",
    "HOST_PROXY_URL", "http_proxy", "https_proxy", "HTTP_PROXY",
    "HTTPS_PROXY", "all_proxy", "ALL_PROXY", "NO_PROXY", "no_proxy",
    "PYCMD", "BACKGROUND",
})
_NESTED_DEFAULT_RE = re.compile(r'^\$\{\w+:-(.+)\}$')


def _resolve_default(raw: str) -> str:
    """Resolve nested ${VAR:-val} to the innermost default."""
    m = _NESTED_DEFAULT_RE.match(raw)
    if m:
        return _resolve_default(m.group(1))
    return raw


def _parse_script_params(script_name: str) -> list[dict[str, str]]:
    scripts_dir = _REPO_ROOT / "scripts" / "bash"
    path = scripts_dir / script_name
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    seen: set[str] = set()
    params: list[dict[str, str]] = []
    for match in _PARAM_RE.finditer(text):
        name = match.group(1)
        default = _resolve_default(match.group(2))
        if name in seen or name.startswith("_") or name in _INTERNAL_VARS:
            continue
        if name.endswith(("_KEY", "_SECRET", "_PASSWORD", "_TOKEN")):
            continue
        seen.add(name)
        params.append({"name": name, "default": default})
    return params


def _scan_all_script_params() -> dict[str, list[dict[str, str]]]:
    scripts = _scan_available_scripts()
    return {s: _parse_script_params(s) for s in scripts}


def main() -> None:
    global _scheduler, _config

    args = parse_args()
    _config = args
    configure_logging(args.log_level)

    _scheduler = create_scheduler(args.scheduler)
    logger.info(
        "Starting Master on %s:%s  scheduler=%s  unhealthy_timeout=%.0fs  dead_timeout=%.0fs",
        args.host, args.port, args.scheduler, args.unhealthy_timeout, args.dead_timeout,
    )

    threading.Thread(
        target=_health_reaper_loop,
        args=(args.unhealthy_timeout, args.dead_timeout),
        daemon=True,
    ).start()

    threading.Thread(
        target=_reconcile_loop,
        args=(float(os.environ.get("RECONCILE_INTERVAL", "60")),),
        daemon=True,
    ).start()

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
