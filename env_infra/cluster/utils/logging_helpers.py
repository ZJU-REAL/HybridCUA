"""Small logging helpers for OSWorld cluster services."""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any


CLUSTER_LOG_FORMAT = (
    "[%(asctime)s %(levelname)s %(name)s %(processName)s/%(threadName)s "
    "%(filename)s:%(lineno)d %(funcName)s] %(message)s"
)

_SENSITIVE_KEY_SUFFIXES = ("_KEY", "_SECRET", "_PASSWORD", "_TOKEN")


def configure_logging(level: str | int) -> None:
    """Configure the shared cluster log format with source location context."""
    resolved_level = getattr(logging, str(level).upper(), level)
    logging.basicConfig(level=resolved_level, format=CLUSTER_LOG_FORMAT)


def safe_body_keys(body: Mapping[str, Any] | None) -> list[str]:
    """Return request body keys only, without exposing secret-looking key names."""
    if not isinstance(body, Mapping):
        return []
    keys: list[str] = []
    for key in body.keys():
        name = str(key)
        if name.upper().endswith(_SENSITIVE_KEY_SUFFIXES):
            name = "<redacted-secret-key>"
        keys.append(name)
    return sorted(keys)


def short_excerpt(value: Any, limit: int = 500) -> str:
    """Keep remote error snippets single-line and bounded."""
    text = "" if value is None else str(value)
    text = text.replace("\r", "\\r").replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."
