"""Shared HTTP session with a tuned connection pool.

Bare ``requests.post()`` builds and discards a one-shot Session per call, so
HTTP keep-alive is never used — every call pays a fresh TCP(+TLS) handshake and
leaves a TIME_WAIT socket. A long-lived Session keeps the urllib3 pool alive so
connections to a host are reused across calls.

``pool_maxsize`` must be >= the per-host concurrency, or excess concurrent calls
still open throwaway connections. Auto-retry is intentionally limited to safe
idempotent methods: never silently replay a POST (``step``/``reset``), which
would double-apply an action.
"""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def make_session(pool_connections: int = 64, pool_maxsize: int = 128) -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
        max_retries=Retry(total=0, allowed_methods=frozenset({"GET", "HEAD", "OPTIONS"})),
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
