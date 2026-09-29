"""Shared Docker helpers for docker-driver worlds.

Multiple worlds run their environments in Docker containers (OSWorld via its
DesktopEnv docker provider, MobileWorld via its mobile_world launcher, ...). The
docker operations they need are the same — list containers, read published
ports, stop+remove a container, reclaim leaked containers — only the access
style differs (the ``docker`` CLI vs the ``docker`` Python SDK). This module is
the single home for that shared logic so no world re-implements it and no world
imports another world's private module.

Placement rules (mirrors the rest of ``cluster/worlds/base``):
- Lives in the world-side ``base`` package, NOT in the neutral master/node layer
  — the platform core never inspects docker (see ``cluster/node/diagnostics.py``,
  which is psutil-only). Only docker-driver adapters import this.
- Imports no benchmark SDK; depends only on the stdlib + (optionally, for the
  SDK helpers) the ``docker`` package, imported lazily so CLI-only callers and
  non-docker machines don't need it.

Two access styles, same semantics:
- **CLI helpers** (subprocess ``docker ps``/``docker rm``): scan/remove by IMAGE,
  i.e. operate on *every* container of an image regardless of who created it.
  Used by orphan reclaim, which must see leaked containers this process never
  tracked.
- **SDK helpers** (``docker`` Python lib): operate on a concrete container handle
  or a name-prefix query. Used by adopt (re-discover running containers) and by
  an adapter tearing down the one container it owns.
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
import re
import subprocess
import time
from typing import Any

logger = logging.getLogger("cluster.worlds.docker")

DOCKER_TIMEOUT_SECONDS = 3.0


# ===========================================================================
# CLI helpers — scan/remove by image (subprocess `docker ps -a` / `docker rm`)
# ===========================================================================

def _round(value: float | int | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _infer_state(status: str) -> str:
    text = (status or "").strip().lower()
    if text.startswith("up "):
        return "running"
    if text.startswith("exited "):
        return "exited"
    if text.startswith("created"):
        return "created"
    if text.startswith("dead"):
        return "dead"
    if text.startswith("restarting"):
        return "restarting"
    if text.startswith("paused"):
        return "paused"
    if text.startswith("removing"):
        return "removing"
    return "unknown"


_CREATED_AT_RE = re.compile(
    r"(?P<stamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (?P<offset>[+-]\d{4})"
)


def _parse_created_ts(created_at: Any) -> float | None:
    text = str(created_at or "").strip()
    if not text:
        return None
    match = _CREATED_AT_RE.search(text)
    if not match:
        return None
    try:
        parsed = datetime.strptime(
            f"{match.group('stamp')} {match.group('offset')}",
            "%Y-%m-%d %H:%M:%S %z",
        )
    except ValueError:
        return None
    return parsed.timestamp()


def parse_port_string(ports: str, port_fields: dict[int, str] | None = None) -> dict[str, int]:
    """Map a ``docker ps`` Ports string to named host-port fields.

    ``port_fields`` ({container_port: field_name}) is supplied by the world being
    scanned. When empty, no named fields are produced.
    """
    if not port_fields:
        return {}
    parsed: dict[str, int] = {}
    for match in _PORT_RE.finditer(ports or ""):
        try:
            host_port = int(match.group("host"))
            container_port = int(match.group("container"))
        except ValueError:
            continue
        field = port_fields.get(container_port)
        if field:
            parsed[field] = host_port
    return parsed


_PORT_RE = re.compile(
    r"(?:(?:0\.0\.0\.0|127\.0\.0\.1|localhost|\[::\]|::):)?(?P<host>\d+)->(?P<container>\d+)/tcp"
)


def _normalize_container(row: dict[str, Any], port_fields: dict[int, str] | None = None) -> dict[str, Any]:
    cid = row.get("id") or row.get("ID") or row.get("ContainerID") or ""
    name = row.get("name") or row.get("Names") or row.get("Name") or ""
    image = row.get("image") or row.get("Image") or ""
    status = row.get("status") or row.get("Status") or ""
    ports = row.get("ports") or row.get("Ports") or ""
    state = row.get("state") or row.get("State") or ""
    created_at = row.get("created_at") or row.get("CreatedAt") or row.get("Created") or ""
    running_for = row.get("running_for") or row.get("RunningFor") or ""
    created_ts = _parse_created_ts(created_at)
    state = str(state or _infer_state(str(status))).strip().lower()
    if not state or state == "unknown":
        state = _infer_state(str(status))
    age_seconds = None
    if created_ts is not None:
        age_seconds = _round(max(0.0, time.time() - created_ts), 3)
    name = str(name).lstrip("/")
    return {
        "id": str(cid)[:12],
        "name": name,
        "image": str(image),
        "state": state,
        "status": str(status),
        "ports": str(ports),
        "created_at": str(created_at),
        "created_ts": created_ts,
        "age_seconds": age_seconds,
        "running_for": str(running_for),
        **parse_port_string(str(ports), port_fields),
    }


def _parse_tab_container(line: str) -> dict[str, Any]:
    parts = line.split("\t")
    keys = ["ID", "Names", "Image", "State", "Status", "Ports"]
    return {key: parts[idx] if idx < len(parts) else "" for idx, key in enumerate(keys)}


def parse_docker_ps_output(stdout: str, port_fields: dict[int, str] | None = None) -> list[dict[str, Any]]:
    """Parse ``docker ps`` output.

    Production uses one JSON object per line. Unit tests and ad-hoc debugging can
    pass tab-separated rows in the order id, name, image, state, status, ports.
    """
    containers: list[dict[str, Any]] = []
    for raw in (stdout or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            row = _parse_tab_container(line)
        containers.append(_normalize_container(row, port_fields))
    return containers


def sample_docker_containers(
    *,
    image: str,
    port_fields: dict[int, str] | None = None,
    timeout: float = DOCKER_TIMEOUT_SECONDS,
) -> tuple[list[dict[str, Any]], str | None]:
    """List all containers of ``image`` (including exited), normalized.

    Returns ``(containers, error)``. Used by orphan reclaim, which must see every
    container of the image — even ones this process never tracked.
    """
    cmd = [
        "docker",
        "ps",
        "-a",
        "--filter",
        f"ancestor={image}",
        "--format",
        "{{json .}}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return [], f"docker ps -a timed out after {timeout:g}s"
    except Exception as exc:
        return [], f"docker ps -a failed: {exc}"

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        return [], stderr or f"docker ps -a exited with {result.returncode}"
    try:
        return parse_docker_ps_output(result.stdout, port_fields), None
    except Exception as exc:
        return [], f"failed to parse docker ps -a output: {exc}"


def remove_docker_containers(
    container_ids: list[str],
    *,
    force: bool = False,
    timeout: float = DOCKER_TIMEOUT_SECONDS,
) -> tuple[int, list[str]]:
    """Remove containers by id via ``docker rm``. Returns ``(removed, errors)``."""
    removed = 0
    errors: list[str] = []
    for container_id in container_ids:
        cid = str(container_id or "").strip()
        if not cid:
            continue
        cmd = ["docker", "rm"]
        if force:
            cmd.append("-f")
        cmd.append(cid)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            errors.append(f"{cid}: {' '.join(cmd)} timed out after {timeout:g}s")
            continue
        except Exception as exc:
            errors.append(f"{cid}: {' '.join(cmd)} failed: {exc}")
            continue
        if result.returncode == 0:
            removed += 1
        else:
            err = (result.stderr or result.stdout or "").strip()
            errors.append(f"{cid}: {err or f'{cmd[0]} {cmd[1]} exited with {result.returncode}'}")
    return removed, errors


def reclaim_orphan_containers(
    *,
    image: str,
    grace_seconds: float,
    live_ids: set[str],
) -> int:
    """Remove containers of ``image`` not owned by any live slot and older than
    ``grace_seconds``. The shared body of every docker world's reclaim hook:
    spares ``live_ids`` and young containers (still being registered),
    force-removes the rest. Returns the number removed.

    ``live_ids`` is whatever each slot's ``resource_id()`` returns — worlds use
    different stable handles (OSWorld: the 12-char container id; MobileWorld: the
    container name). To stay world-agnostic this spares a container if EITHER its
    id-prefix OR its name is in ``live_ids``, so the caller need not normalize to
    one id space."""
    containers, error = sample_docker_containers(image=image)
    if error:
        logger.warning("Orphan reclaim skipped: docker ps failed: %s", error)
        return 0
    now = time.time()
    victims: list[str] = []
    for c in containers:
        cid = (c.get("id") or "")[:12]
        name = c.get("name") or ""
        if not cid:
            continue
        if cid in live_ids or (name and name in live_ids):
            continue  # spare containers a slot still owns (matched by id or name)
        created = c.get("created_ts")
        if created and (now - created) < grace_seconds:
            continue  # too young — may still be registering
        victims.append(cid)
    if not victims:
        return 0
    removed, errors = remove_docker_containers(victims, force=True)
    if errors:
        logger.warning("Orphan reclaim removed %d, errors: %s", removed, "; ".join(errors[:3]))
    return removed


# ===========================================================================
# SDK helpers — operate on concrete handles (docker Python lib, imported lazily)
# ===========================================================================

def _client():
    """Return a docker SDK client (lazy import so CLI-only callers / non-docker
    machines never need the ``docker`` package)."""
    import docker as docker_lib  # lazy

    return docker_lib.from_env()


def list_running_containers(*, name_prefix: str | None = None, image: str | None = None) -> list[Any]:
    """List RUNNING container handles, optionally filtered by name prefix and/or
    image tag. Returns docker SDK Container objects (so callers can read .attrs,
    .name, .short_id). Returns [] on any docker error (logged), never raises."""
    try:
        client = _client()
        out = []
        for c in client.containers.list():
            if c.status != "running":
                continue
            if name_prefix and not c.name.startswith(name_prefix):
                continue
            if image and not any(image in (t or "") for t in (c.image.tags or [])):
                continue
            out.append(c)
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to list docker containers: %s", exc)
        return []


def container_host_ports(container: Any, container_ports: dict[int, str]) -> dict[str, int]:
    """Read a running container's published host ports from its SDK ``attrs``.

    ``container_ports`` maps {internal_container_port: field_name}; returns
    {field_name: host_port} for the mappings present. The SDK counterpart of
    :func:`parse_port_string` (which parses the CLI's Ports text)."""
    ports = (container.attrs.get("NetworkSettings", {}) or {}).get("Ports") or {}
    out: dict[str, int] = {}
    for cp, mappings in ports.items():
        if not mappings:
            continue
        try:
            internal = int(str(cp).split("/")[0])
            host_port = int(mappings[0]["HostPort"])
        except (ValueError, KeyError, IndexError):
            continue
        field = container_ports.get(internal)
        if field:
            out[field] = host_port
    return out


def stop_and_remove(container_name: str, *, stop_timeout: int = 10) -> None:
    """Stop and force-remove a container by name. Best-effort: logs and swallows
    errors (a teardown failure must not crash close())."""
    try:
        client = _client()
        container = client.containers.get(container_name)
        container.stop(timeout=stop_timeout)
        container.remove(force=True)
        logger.info("Stopped container %s", container_name)
    except Exception:  # noqa: BLE001
        logger.warning("Failed to stop container %s", container_name, exc_info=True)


def container_state(container: Any) -> "str | None":
    """Raw docker state of a container handle. The single shared probe behind
    every docker world's liveness + display.

    Returns ``"running"`` / ``"exited"`` / ``"unhealthy"`` / other docker status,
    or None on error. ``unhealthy`` is surfaced as a distinct state (a container
    whose process is up but whose docker HEALTHCHECK is wedged — the env_17 case)
    rather than mislabeled ``running`` or ``exited``:
    - ``status != "running"`` -> that status verbatim (exited/created/dead/...).
    - running + HEALTHCHECK ``unhealthy`` -> ``"unhealthy"``.
    - running + ``starting`` / no HEALTHCHECK / ``healthy`` -> ``"running"``
      (``starting`` counts as running so a freshly created container isn't
      condemned before its first probe completes).
    Callers decide reuse from this: reusable iff state == ``"running"``."""
    try:
        container.reload()
        status = container.status
        if status == "running":
            health = (
                (container.attrs.get("State", {}) or {}).get("Health", {}) or {}
            ).get("Status")
            return "unhealthy" if health == "unhealthy" else "running"
        return status
    except Exception:  # noqa: BLE001
        return None


def container_state_by_name(container_name: str) -> "str | None":
    """Raw docker state of a container BY NAME, for display + reuse decisions.

    Returns ``"running"`` / ``"exited"`` / ``"unhealthy"`` / other docker status;
    ``"gone"`` if the name is not found / docker is unreachable; or None if no
    name was given (nothing to check). Resolves the name to a handle and defers
    to :func:`container_state`. One cached call feeds both an adapter's liveness
    (reuse = state == "running") and its get_info display."""
    if not container_name:
        return None
    try:
        container = _client().containers.get(container_name)
    except Exception:  # noqa: BLE001 - not found / docker down -> dead
        return "gone"
    state = container_state(container)
    return state if state is not None else "gone"
