"""Pre-run wiring check: prove the model can read an image before a run starts.

The image half is the one that matters. A broken image channel does not raise — a GUI
agent that cannot see its screenshots produces confident, screenshot-free hallucination,
and every trajectory it writes is silently worthless. Wasting three API calls up front is
strictly cheaper than discovering that from the scores.

``selftest(call)`` is transport-agnostic: the caller passes a function that takes
``(text, image_bytes | None)`` and returns the model's reply text (or "" on failure), so
Kimi's OpenAI-compatible payload and Claude's Anthropic payload both plug in.
"""
from __future__ import annotations

import struct
import zlib
from typing import Callable, Optional

#: (name, RGB) pairs. Two colours, because a model that answers "green" to everything
#: would pass a single-colour check.
_PROBES = (("green", (0, 200, 0)), ("red", (220, 0, 0)))

_COLOUR_QUESTION = "What is the single dominant colour of this image? Answer with one word."


def solid_png(width: int, height: int, rgb) -> bytes:
    """Build a solid-colour PNG in-process (no PIL dependency in the check itself)."""
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def selftest(call: Callable[[str, Optional[bytes]], str], label: str = "selftest") -> bool:
    """Check the text channel, then the image channel. True only if all probes pass."""
    reply = call("Reply with exactly: PONG", None)
    if not reply:
        print(f"{label}: text FAILED (no response)")
        return False
    print(f"{label}: text ok ({reply.strip()[:40]!r})")

    for name, rgb in _PROBES:
        reply = call(_COLOUR_QUESTION, solid_png(240, 160, rgb))
        if not reply:
            print(f"{label}: image ({name}) FAILED (no response)")
            return False
        if name not in reply.lower():
            print(f"{label}: image ({name}) FAILED -- expected {name!r}, "
                  f"got {reply.strip()[:70]!r}")
            print("  The model is NOT seeing screenshots. Refusing to run: a GUI agent\n"
                  "  that cannot see the screen invents UI detail, and every trajectory\n"
                  "  it produces is silently worthless.")
            return False
        print(f"{label}: image ({name}) ok ({reply.strip()[:30]!r})")
    return True
