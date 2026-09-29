from __future__ import annotations

import logging
import threading
from urllib.parse import urlparse, urlunparse

from simple_websocket.errors import ConnectionClosed
from websocket import ABNF, WebSocketConnectionClosedException, WebSocketTimeoutException, create_connection


def http_to_ws_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse((scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))


def close_quietly(conn) -> None:
    try:
        conn.close()
    except Exception:
        pass


def _forward_upstream_to_browser(upstream, browser_ws, stop_event: threading.Event, logger: logging.Logger) -> None:
    try:
        while not stop_event.is_set():
            try:
                opcode, payload = upstream.recv_data()
            except WebSocketTimeoutException:
                continue
            if opcode == ABNF.OPCODE_CLOSE:
                break
            if opcode == ABNF.OPCODE_PING:
                upstream.pong(payload)
                continue
            if opcode == ABNF.OPCODE_PONG:
                continue
            if opcode == ABNF.OPCODE_TEXT and isinstance(payload, bytes):
                payload = payload.decode("utf-8", errors="replace")
            browser_ws.send(payload)
    except (ConnectionClosed, WebSocketConnectionClosedException, OSError):
        pass
    except Exception:
        logger.debug("WebSocket upstream-to-browser bridge failed", exc_info=True)
    finally:
        stop_event.set()
        close_quietly(browser_ws)


def proxy_websocket(browser_ws, upstream_url: str, logger: logging.Logger, label: str = "websocket") -> None:
    upstream = None
    stop_event = threading.Event()
    try:
        upstream = create_connection(upstream_url, timeout=10, enable_multithread=True)
        upstream.settimeout(1)
        reader = threading.Thread(
            target=_forward_upstream_to_browser,
            args=(upstream, browser_ws, stop_event, logger),
            daemon=True,
        )
        reader.start()

        while not stop_event.is_set():
            message = browser_ws.receive()
            if message is None:
                break
            if isinstance(message, (bytes, bytearray)):
                upstream.send_binary(bytes(message))
            else:
                upstream.send(message)
    except (ConnectionClosed, WebSocketConnectionClosedException, OSError):
        pass
    except Exception:
        logger.warning("%s proxy failed", label, exc_info=True)
    finally:
        stop_event.set()
        if upstream is not None:
            close_quietly(upstream)
        close_quietly(browser_ws)
