"""Minimal structured debug-event logging for cluster remote failures."""
from __future__ import annotations

import inspect
import json
import os
import socket
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_WRITE_LOCK = threading.Lock()
_SENSITIVE_KEY_SUFFIXES = ("KEY", "SECRET", "PASSWORD", "TOKEN")
_DEFAULT_EVENT_FILE = "logs/cluster_debug_events.jsonl"


def _event_file() -> Path:
    return Path(os.environ.get("CLUSTER_DEBUG_EVENTS_FILE", _DEFAULT_EVENT_FILE))


def _event_dir() -> Path:
    override = os.environ.get("CLUSTER_DEBUG_EVENTS_DIR")
    if override:
        return Path(override)
    return _event_file().with_suffix(".d")


def _safe_path_component(value: Any, default: str) -> str:
    text = str(value or "").strip() or default
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    return safe.strip("._") or default


def _writer_id() -> str:
    return _safe_path_component(
        os.environ.get("CLUSTER_DEBUG_EVENTS_WRITER_ID"),
        socket.gethostname() or "unknown-writer",
    )


def _event_shard_dir() -> Path:
    return _event_dir() / _writer_id()


def _event_shard_file() -> Path:
    return _event_shard_dir() / f"{os.getpid()}.jsonl"


def _default_node_id(service: str, node_id: str | None) -> str | None:
    if node_id:
        return node_id
    if service == "node":
        return os.environ.get("NODE_ID") or os.environ.get("CLUSTER_DEBUG_EVENTS_WRITER_ID")
    return node_id


def _limit(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _short(value: Any, limit: int | None = None) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\r", "\\r")
    max_len = limit if limit is not None else _limit("CLUSTER_DEBUG_EVENTS_MAX_EXCERPT", 1200)
    return text if len(text) <= max_len else text[:max_len] + "..."


def _is_sensitive(key: str | None) -> bool:
    if not key:
        return False
    upper = key.upper()
    return any(part in upper for part in _SENSITIVE_KEY_SUFFIXES)


def _sanitize(value: Any, key: str | None = None) -> Any:
    if _is_sensitive(key):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): _sanitize(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _source_from_stack(stacklevel: int = 1) -> dict[str, Any] | None:
    this_file = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    remaining = max(1, int(stacklevel or 1))
    for frame in inspect.stack()[2:]:
        path = Path(frame.filename).resolve()
        if path == this_file:
            continue
        remaining -= 1
        if remaining > 0:
            continue
        try:
            file_name = str(path.relative_to(cwd))
        except ValueError:
            file_name = str(path)
        return {"file": file_name, "function": frame.function, "line": frame.lineno}
    return None


def _origin_from_exception(exc: BaseException | None) -> dict[str, Any] | None:
    if exc is None or exc.__traceback__ is None:
        return None
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return None
    frame = frames[-1]
    cwd = Path.cwd().resolve()
    path = Path(frame.filename).resolve()
    try:
        file_name = str(path.relative_to(cwd))
    except ValueError:
        file_name = str(path)
    return {"file": file_name, "function": frame.name, "line": frame.lineno}


def _traceback_excerpt(exc: BaseException | None) -> str | None:
    if exc is None:
        return None
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return _short(text, _limit("CLUSTER_DEBUG_EVENTS_MAX_TRACEBACK", 4000))


def _write_event(event: dict[str, Any]) -> None:
    try:
        path = _event_shard_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        with _WRITE_LOCK:
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            fd = os.open(path, flags, 0o644)
            try:
                os.write(fd, payload)
            finally:
                os.close(fd)
    except Exception:
        return


def record_event(
    *,
    type: str,
    service: str,
    component: str,
    level: str = "error",
    path: str | None = None,
    status_code: int | None = None,
    lease_id: str | None = None,
    env_id: str | None = None,
    episode_id: str | None = None,
    node_id: str | None = None,
    node_url: str | None = None,
    error_type: str | None = None,
    message: Any = None,
    response_excerpt: Any = None,
    traceback_excerpt: Any = None,
    error_origin: dict[str, Any] | None = None,
    stacklevel: int = 1,
    **extra: Any,
) -> None:
    event = {
        "ts": _utc_now(),
        "level": level,
        "type": type,
        "service": service,
        "component": component,
        "path": path,
        "status_code": status_code,
        "lease_id": lease_id,
        "env_id": env_id,
        "episode_id": episode_id,
        "node_id": _default_node_id(service, node_id),
        "node_url": node_url,
        "error_type": error_type,
        "message": _short(message),
        "response_excerpt": _short(response_excerpt),
        "traceback_excerpt": _short(traceback_excerpt, _limit("CLUSTER_DEBUG_EVENTS_MAX_TRACEBACK", 4000)),
        "event_source": _source_from_stack(stacklevel=stacklevel),
        "error_origin": error_origin,
    }
    for key, value in extra.items():
        event[str(key)] = _sanitize(value, str(key))
    _write_event(event)


def record_exception(
    exc: BaseException,
    *,
    type: str = "exception",
    service: str,
    component: str,
    message: Any = None,
    stacklevel: int = 1,
    **fields: Any,
) -> None:
    record_event(
        type=type,
        service=service,
        component=component,
        error_type=exc.__class__.__name__,
        message=message if message is not None else str(exc),
        traceback_excerpt=_traceback_excerpt(exc),
        error_origin=_origin_from_exception(exc),
        stacklevel=stacklevel,
        **fields,
    )


def record_http_non_2xx(
    *,
    service: str,
    component: str,
    status_code: int,
    response_excerpt: Any = None,
    message: Any = None,
    stacklevel: int = 1,
    **fields: Any,
) -> None:
    record_event(
        type="http_non_2xx",
        service=service,
        component=component,
        status_code=status_code,
        response_excerpt=response_excerpt,
        message=message or f"HTTP non-2xx response: {status_code}",
        stacklevel=stacklevel,
        **fields,
    )


def _event_paths() -> list[Path]:
    shard_dir = _event_shard_dir()
    if shard_dir.exists():
        try:
            return sorted(path for path in shard_dir.glob("*.jsonl") if path.is_file())
        except Exception:
            return []
    legacy_file = _event_file()
    return [legacy_file] if legacy_file.exists() else []


def _matches_filters(event: dict[str, Any], filters: dict[str, Any]) -> bool:
    return all(
        str(event.get(key, "")) == str(value)
        for key, value in filters.items()
        if value not in (None, "")
    )


def tail_events(limit: int = 200, **filters: Any) -> list[dict[str, Any]]:
    try:
        limit = max(0, min(int(limit or 0), 2000))
    except (TypeError, ValueError):
        limit = 200
    if limit == 0:
        return []
    items: list[dict[str, Any]] = []
    for path in _event_paths():
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if _matches_filters(event, filters):
                        items.append(event)
        except Exception:
            continue
    items.sort(key=lambda item: str(item.get("ts") or ""))
    return items[-limit:]
