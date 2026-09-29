"""World-neutral cluster node server.

Hosts one or more worlds via the EnvSession ``/v1/sessions`` API, backed by a
:class:`~cluster.node.server.NodeSessionManager` (RuntimePool per world).
This server imports NO benchmark SDK: it discovers worlds from
``cluster/worlds/<id>/world.yaml`` and only the selected world's adapter module
is imported, lazily, when its pool is built.

Run::

    python -m cluster.node.world_server                                  # host all worlds
    python -m cluster.node.world_server --node-worlds osworld --master-url http://master:19000

It registers ``runtimes`` + ``capabilities`` with the master so the capability
scheduler can route sessions by world/surface without hardcoded names.
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import socket
import threading
import time
import uuid

import requests
from flask import Flask, jsonify, request

from cluster.core.registry import WorldRegistry
from cluster.node.diagnostics import sample_resources
from cluster.node.server import NodeSessionManager, create_node_sessions_blueprint
from cluster.utils.debug_events import tail_events

logger = logging.getLogger("cluster.node.world_server")

#: Min slot age before a heartbeat-driven (node-push) orphan release may fire,
#: guarding against reclaiming a freshly-acquired slot whose master record is
#: still in flight. Mirrors the master's reconcile grace window.
_NODE_ORPHAN_GRACE = float(os.getenv("NODE_ORPHAN_GRACE_SECONDS", "30"))

#: Orphan-resource reclaim (distinct from session-orphan grace above): the
#: world-neutral node periodically asks each world's driver to free resources
#: not mapped to any live slot AND older than this grace. What an "orphan" is and
#: how to free it lives entirely in the adapter's reclaim hook (for docker worlds:
#: an exited/leaked container). ENABLED by default so leaked resources (e.g. the
#: exited container a failed `stop().remove()` leaves behind) are cleaned up
#: instead of accumulating; set NODE_ORPHAN_RECLAIM_ENABLED=0 to keep debugging
#: resources around.
_ORPHAN_RECLAIM_ENABLED = os.getenv("NODE_ORPHAN_RECLAIM_ENABLED", "1") == "1"
_CONTAINER_ORPHAN_GRACE = float(os.getenv("NODE_CONTAINER_ORPHAN_GRACE_SECONDS", "120"))

_DEFAULT_WORLDS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "worlds")


def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _parse_labels(raw: str) -> dict:
    labels = {}
    for pair in (raw or "").split(","):
        pair = pair.strip()
        if "=" in pair:
            k, v = pair.split("=", 1)
            labels[k.strip()] = v.strip()
    return labels


def build_app(manager: NodeSessionManager) -> Flask:
    app = Flask("cluster-node-world-server")
    app.register_blueprint(create_node_sessions_blueprint(manager))

    # -- live view proxy (WebSocket must register before HTTP catch-all) -----
    try:
        from flask_sock import Sock
        from cluster.utils.ws_proxy import close_quietly, proxy_websocket

        sock = Sock(app)

        @sock.route("/view/<slot_id>/websockify")
        def view_websockify(ws, slot_id: str):  # type: ignore[unused-ignore]
            live_view = manager.get_slot_live_view(slot_id)
            if not live_view or live_view.get("protocol") != "vnc":
                close_quietly(ws)
                return
            port = live_view["port"]
            upstream_url = f"ws://127.0.0.1:{port}/websockify"
            proxy_websocket(ws, upstream_url, logger, label=f"view ws {slot_id}")
    except ImportError:
        logger.warning("flask-sock not installed; live view WebSocket proxy disabled")

    from cluster.utils.common import proxy_request

    import re as _re
    _EXTERNAL_CDN_RE = _re.compile(
        r'<(?:link|script)[^>]*(?:fonts\.googleapis\.com|fonts\.gstatic\.com|cdnjs\.cloudflare\.com)[^>]*/?>(?:</script>)?',
        _re.IGNORECASE,
    )

    @app.route("/view/<slot_id>/", defaults={"subpath": ""}, methods=["GET", "POST"])
    @app.route("/view/<slot_id>/<path:subpath>", methods=["GET", "POST"])
    def view_proxy(slot_id: str, subpath: str):  # type: ignore[unused-ignore]
        live_view = manager.get_slot_live_view(slot_id)
        if not live_view:
            return jsonify({"ok": False, "error": f"no live_view for slot {slot_id}"}), 404
        port = live_view["port"]
        target = f"http://127.0.0.1:{port}/{subpath}"
        qs = request.query_string.decode()
        if qs:
            target += f"?{qs}"
        try:
            resp = proxy_request(target)
            if not subpath and resp.content_type and "html" in resp.content_type:
                html = resp.get_data(as_text=True)
                html = _EXTERNAL_CDN_RE.sub("", html)
                resp.set_data(html)
            return resp
        except Exception:
            return jsonify({"ok": False, "error": f"live view not ready (port {port} not responding)"}), 503

    @app.get("/healthz")
    def healthz():  # type: ignore[unused-ignore]
        return jsonify({"ok": True, "worlds": manager.world_ids})

    @app.get("/v1/runtimes")
    def runtimes():  # type: ignore[unused-ignore]
        return jsonify({"ok": True, "runtimes": manager.capabilities()})

    @app.get("/debug/events")
    def debug_events():  # type: ignore[unused-ignore]
        # The master polls every node's /debug/events (it does NOT skip
        # world-neutral nodes); without this route the world node 404s and the
        # master silently drops it.
        tail = request.args.get("tail", default=200, type=int) or 200
        filters = {
            key: request.args.get(key)
            for key in ("type", "service", "path", "lease_id", "env_id", "node_id")
        }
        return jsonify({"ok": True, "events": tail_events(tail, **filters)})

    @app.post("/set_max_envs")
    def set_max_envs():  # type: ignore[unused-ignore]
        # Master speaks node-wide max_envs (legacy EnvPool endpoint name kept for
        # dashboard compatibility); the manager splits it across hosted worlds.
        body = request.get_json(force=True, silent=True) or {}
        try:
            max_envs = int(body.get("max_envs"))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "max_envs (int) required"}), 400
        per_world = manager.set_max_envs(max_envs)
        return jsonify({"ok": True, "max_envs": max_envs, "per_world": per_world})

    return app


def _register_with_master(master_url: str, node_id: str, node_url: str, secret: str,
                          manager: NodeSessionManager, max_envs: int, labels: dict,
                          max_retries: int = 30, retry_interval: float = 5.0) -> None:
    payload = {
        "node_id": node_id,
        "node_url": node_url,
        "secret": secret,
        "max_envs": max_envs,
        "runtimes": manager.world_ids,
        "capabilities": manager.capabilities(),
        "labels": labels,
    }
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(f"{master_url}/node/register", json=payload, timeout=10)
            if resp.json().get("ok"):
                logger.info("Registered with master %s (node_id=%s, worlds=%s)", master_url, node_id, manager.world_ids)
                return
        except Exception as exc:  # noqa: BLE001
            logger.warning("register attempt %d/%d failed: %s", attempt, max_retries, exc)
        time.sleep(retry_interval)
    raise RuntimeError(f"failed to register with master after {max_retries} attempts")


def _handle_heartbeat_response(resp, master_url: str, node_id: str, node_url: str,
                               secret: str, manager: NodeSessionManager,
                               max_envs: int, labels: dict) -> None:
    """Act on the master's heartbeat reply: re-register if forgotten, or run the
    node-push orphan reclaim from the returned ``valid_sessions``. Extracted from
    the loop so it can be unit-tested with a fabricated response."""
    # A non-JSON body (e.g. an HTML error page from an intermediary proxy when
    # the master is briefly down) must NOT abort re-registration: fall back to
    # an empty dict so the status_code == 404 branch is still reachable.
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        data = {}
    if not data.get("ok"):
        # After a master restart the node is unknown -> 404 + re_register.
        # Re-register on the spot so the master's reconcile can re-discover our
        # live sessions instead of dropping us until process restart.
        if data.get("re_register") or resp.status_code == 404:
            logger.info("Master requested re-registration; re-registering")
            _register_with_master(master_url, node_id, node_url, secret,
                                  manager, max_envs, labels, max_retries=3)
        else:
            logger.warning("Heartbeat rejected: %s", data.get("error"))
    elif "valid_sessions" in data:
        # Node-push reclaim: drop sessions the master no longer recognizes,
        # keeping the adapter warm (close=False). The master's pull reconcile is
        # authoritative; this is the fast (per-heartbeat) path.
        manager.reconcile_sessions(set(data["valid_sessions"]),
                                   min_age_seconds=_NODE_ORPHAN_GRACE)


def _slot_summaries(status: dict) -> list[dict]:
    """Lightweight per-slot summary for the heartbeat — id/world/busy/session/
    live_view only, never heavy payloads (screenshots, container attrs). Lets the
    master cache a slot->node view (locate a slot for live view, reconcile) from
    the data it already gets every heartbeat. ``slot_id`` is always present, so an
    idle slot (session_id=None) is included too; only ``session_id`` may be None."""
    out: list[dict] = []
    for world_id, pool in status.items():
        for s in pool.get("slots", []) or []:
            out.append({
                "slot_id": s.get("slot_id"),
                "world_id": world_id,
                "busy": s.get("busy"),
                "session_id": s.get("session_id"),
                "live_view": s.get("live_view"),
            })
    return out


def _heartbeat_loop(master_url: str, node_id: str, node_url: str, secret: str,
                    manager: NodeSessionManager, interval: float,
                    max_envs: int, labels: dict) -> None:
    while True:
        time.sleep(interval)
        try:
            status = manager.status()
            busy = sum(p.get("busy_slots", 0) for p in status.values())
            total = sum(p.get("total_slots", 0) for p in status.values())
            resp = requests.post(
                f"{master_url}/node/heartbeat",
                # Host-level resources (CPU/mem/disk/load/qemu) are psutil-only;
                # the master stores them on NodeInfo and the dashboard aggregates
                # them. Without this a node shows empty resource panels. The node
                # never reports docker — docker is an adapter-internal concern.
                # total_envs/idle_envs keep master capacity correct every
                # heartbeat (10s) instead of only every reconcile (60s).
                json={"node_id": node_id, "secret": secret,
                      "busy_envs": busy, "total_envs": total,
                      "idle_envs": max(0, total - busy),
                      "runtimes": manager.world_ids, "capabilities": manager.capabilities(),
                      "resources": sample_resources(),
                      "slots": _slot_summaries(status)},
                timeout=10,
            )
            _handle_heartbeat_response(resp, master_url, node_id, node_url,
                                       secret, manager, max_envs, labels)
        except Exception as exc:  # noqa: BLE001
            logger.warning("heartbeat failed: %s", exc)


def _maintenance_loop(manager: NodeSessionManager, interval: float) -> None:
    """Periodic pool upkeep, always on:

    - reclaim idle slots past their TTL (only when a TTL is configured), and
    - reap idle slots whose adapter is unhealthy (e.g. a docker container that
      vanished), so the slot accounting re-converges on the real resources.
    """
    while True:
        time.sleep(interval)
        try:
            reaped = manager.reap_idle()
            if reaped:
                logger.info("Idle reaper closed %d idle slot(s)", reaped)
        except Exception as exc:  # noqa: BLE001
            logger.warning("idle reaper failed: %s", exc)
        try:
            # Backstop: free slots stuck busy with no session. acquire() does this
            # lazily, but a fully-loaded node makes no new acquires, so leaked busy
            # slots would otherwise accumulate (drifting the slot count above NUM_ENVS).
            freed = manager.release_stale_busy()
            if freed:
                logger.info("Stale-busy reaper freed %d leaked busy slot(s)", freed)
        except Exception as exc:  # noqa: BLE001
            logger.warning("stale-busy reaper failed: %s", exc)
        try:
            unhealthy = manager.reap_unhealthy_idle()
            if unhealthy:
                logger.info("Health reaper reclaimed %d unhealthy idle slot(s)", unhealthy)
        except Exception as exc:  # noqa: BLE001
            logger.warning("health reaper failed: %s", exc)
        if _ORPHAN_RECLAIM_ENABLED:
            try:
                orphans = manager.reclaim_orphans(grace=_CONTAINER_ORPHAN_GRACE)
                if orphans:
                    logger.info("Orphan reclaimer removed %d orphan container(s)", orphans)
            except Exception as exc:  # noqa: BLE001
                logger.warning("orphan reclaimer failed: %s", exc)


def _deregister_from_master(master_url: str, node_id: str, secret: str) -> None:
    """Tell the master we are going away so it reclaims us immediately instead
    of waiting for the dead-timeout. Mirrors the legacy node (cluster/node/
    server.py); errors are swallowed so this is safe to run from atexit."""
    try:
        requests.post(
            f"{master_url}/node/deregister",
            json={"node_id": node_id, "secret": secret},
            timeout=5,
        )
        logger.info("Deregistered from master (node_id=%s)", node_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Deregister failed: %s", exc)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="World-neutral cluster node server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.getenv("NODE_PORT", "18080")))
    p.add_argument("--node-worlds", default=os.getenv("NODE_WORLDS", ""),
                   help="comma-separated world ids to host; empty or 'all' = host every world "
                        "declared under cluster/worlds/ (default)")
    p.add_argument("--worlds-dir", default=os.getenv("NODE_WORLDS_DIR", _DEFAULT_WORLDS_DIR))
    p.add_argument("--max-slots-per-world", type=int, default=int(os.getenv("NODE_MAX_SLOTS", "1")))
    p.add_argument("--idle-ttl-seconds", type=int, default=int(os.getenv("NODE_IDLE_TTL", "0")))
    p.add_argument("--prewarm", type=int, default=int(os.getenv("NODE_PREWARM", "0")),
                   help="Number of slots to pre-create at startup per world")
    p.add_argument("--prewarm-concurrency", type=int, default=int(os.getenv("NODE_PREWARM_CONCURRENCY", "2")),
                   help="Parallel workers used during startup prewarm")
    p.add_argument("--scale-buffer", type=int, default=int(os.getenv("NODE_SCALE_BUFFER", "0")),
                   help="Minimum idle slot buffer maintained by background scaler")
    p.add_argument("--scale-interval", type=int, default=int(os.getenv("NODE_SCALE_INTERVAL", "5")),
                   help="Seconds between scaler check iterations")
    p.add_argument("--world-config", default=os.getenv("NODE_WORLD_CONFIG", ""),
                   help="JSON mapping world_id -> config dict passed to the adapter")
    p.add_argument("--master-url", default=os.getenv("NODE_MASTER_URL", ""))
    p.add_argument("--node-url", default=os.getenv("NODE_URL", ""))
    p.add_argument("--node-id", default=os.getenv("NODE_ID", ""))
    p.add_argument("--node-secret", default=os.getenv("NODE_SECRET", ""))
    p.add_argument("--heartbeat-interval", type=float, default=float(os.getenv("NODE_HEARTBEAT_INTERVAL", "10")))
    p.add_argument("--node-labels", default=os.getenv("NODE_LABELS", ""))
    p.add_argument("--log-level", default=os.getenv("NODE_LOG_LEVEL", "INFO"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="[%(asctime)s %(levelname)s %(name)s] %(message)s")

    requested = [w.strip() for w in args.node_worlds.split(",") if w.strip()]
    world_config = json.loads(args.world_config) if args.world_config else {}

    registry = WorldRegistry()
    # Empty NODE_WORLDS (or "all") => host EVERY world declared under worlds_dir.
    if not requested or requested == ["all"]:
        world_ids = registry.load_from_directory(args.worlds_dir)
        logger.info("NODE_WORLDS not set; hosting all discovered worlds: %s", world_ids)
    else:
        registry.load_from_directory(args.worlds_dir, only=requested)
        world_ids = requested
    if not world_ids:
        raise SystemExit(f"no worlds found under {args.worlds_dir!r}")
    manager = NodeSessionManager(
        registry,
        world_ids=world_ids,
        max_slots_per_world=args.max_slots_per_world,
        world_config=world_config,
        idle_ttl_seconds=args.idle_ttl_seconds,
        prewarm=args.prewarm,
        prewarm_concurrency=args.prewarm_concurrency,
        scale_buffer=args.scale_buffer,
        scale_interval=args.scale_interval,
    )
    max_envs = args.max_slots_per_world * len(world_ids)

    app = build_app(manager)

    if args.master_url:
        master_url = args.master_url.rstrip("/")
        node_id = args.node_id or f"node-{uuid.uuid4().hex[:12]}"
        node_url = args.node_url.rstrip("/") if args.node_url else f"http://{_get_local_ip()}:{args.port}"
        labels = _parse_labels(args.node_labels)
        _register_with_master(master_url, node_id, node_url, args.node_secret, manager, max_envs, labels)
        threading.Thread(
            target=_heartbeat_loop,
            args=(master_url, node_id, node_url, args.node_secret, manager,
                  args.heartbeat_interval, max_envs, labels),
            daemon=True,
        ).start()
        atexit.register(_deregister_from_master, master_url, node_id, args.node_secret)

    # Maintenance loop (always on): reaps idle slots past their TTL (when a TTL
    # is configured — the safety net for envs orphaned by a master restart) AND
    # reaps idle slots whose adapter went unhealthy (e.g. a vanished container),
    # keeping the slot accounting in sync with the real resources. The interval
    # follows the TTL when set, else a gentle default.
    maintenance_interval = max(5.0, args.idle_ttl_seconds / 2.0) if args.idle_ttl_seconds and args.idle_ttl_seconds > 0 else 45.0
    threading.Thread(
        target=_maintenance_loop,
        args=(manager, maintenance_interval),
        daemon=True,
    ).start()

    logger.info("World node serving %s on %s:%s", world_ids, args.host, args.port)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
