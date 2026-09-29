"""AsyncTTLCache: a cached value refreshed off the hot path.

Wraps a slow sampler (docker ps, psutil scan, health probe) so callers always
get a non-blocking snapshot: ``get()`` returns the last value and, when it is
older than ``ttl``, kicks a single background refresh (single-flight — at most
one in-flight refresh at a time). Used by the node to keep ``/slots`` and the
heartbeat off blocking IO.
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Any, Callable


class AsyncTTLCache:
    def __init__(self, sampler: Callable[[], Any], ttl: float, default: Any) -> None:
        self._sampler = sampler
        self._ttl = float(ttl)
        self._value = default
        self._ts = 0.0
        self._refreshing = False
        self._lock = threading.Lock()

    def ready(self) -> bool:
        """True once at least one refresh has populated the value."""
        with self._lock:
            return self._ts > 0

    def _refresh(self) -> None:
        try:
            result = self._sampler()
            with self._lock:
                self._value = result
                self._ts = time.time()
        finally:
            with self._lock:
                self._refreshing = False

    def get(self) -> Any:
        """Return a snapshot of the cached value; trigger a background refresh if
        stale. Never blocks on the sampler."""
        with self._lock:
            stale = (time.time() - self._ts) > self._ttl
            if stale and not self._refreshing:
                self._refreshing = True
                threading.Thread(target=self._refresh, daemon=True).start()
            return copy.copy(self._value)

    def prime(self) -> None:
        """Kick an initial background refresh (e.g. at startup)."""
        with self._lock:
            if self._refreshing:
                return
            self._refreshing = True
        threading.Thread(target=self._refresh, daemon=True).start()
