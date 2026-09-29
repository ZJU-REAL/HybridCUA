"""Shared utilities for cluster server code."""
from __future__ import annotations

from typing import Any

import requests
from flask import Response, request as flask_request


def json_body() -> dict[str, Any]:
    """Parse JSON body from the current Flask request."""
    return flask_request.get_json(force=True, silent=True) or {}


def proxy_request(target_url: str, timeout: int | None = None) -> Response:
    """Forward the current Flask request (method + body + headers) to ``target_url``
    and stream the response back.

    Used by the live-view proxy chain (master -> node -> container). Supports any
    method so a web viewer's event submits (e.g. Gradio's POST ``/queue/join``)
    reach the upstream, and streams SSE (``text/event-stream``) untouched.

    ``timeout=None`` keeps long-lived streams (SSE long-poll, noVNC) open; a hard
    read timeout would sever them mid-session.
    """
    upstream = requests.request(
        flask_request.method,
        target_url,
        data=flask_request.get_data(),
        headers={k: v for k, v in flask_request.headers
                 if k.lower() not in ("host", "content-length")},
        stream=True,
        timeout=timeout,
    )
    # iter_content() auto-decompresses (gzip/deflate), so the upstream
    # Content-Length (compressed size) no longer matches the streamed bytes —
    # forwarding it truncates the response. Drop it alongside the encoding
    # headers and let WSGI chunk / recompute the length.
    excluded = {"content-encoding", "transfer-encoding", "connection", "content-length"}
    headers = [(k, v) for k, v in upstream.raw.headers.items() if k.lower() not in excluded]
    return Response(upstream.iter_content(chunk_size=4096), status=upstream.status_code, headers=headers)
