"""Binary <-> base64 encoding helpers (zero-dependency).

Screenshots and any other binary modality must be base64 strings to travel over
the JSON transport between runner, master, node and sidecar. World-neutral and
free of flask/requests, so a world adapter can import it without pulling in the
server stack.
"""

from __future__ import annotations

import base64
from typing import Union


def encode_screenshot(raw: Union[bytes, bytearray, None]) -> str | None:
    """Encode raw screenshot bytes (PNG) to a base64 ascii string."""
    if raw is None:
        return None
    if isinstance(raw, str):
        # already-encoded; pass through
        return raw
    return base64.b64encode(bytes(raw)).decode("ascii")


def decode_screenshot(b64: str | None) -> bytes | None:
    """Decode a base64 ascii string back to raw screenshot bytes."""
    if b64 is None:
        return None
    if isinstance(b64, (bytes, bytearray)):
        return bytes(b64)
    return base64.b64decode(b64)
