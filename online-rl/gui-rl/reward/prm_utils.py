"""Shared helpers for PRM (process reward model) reward agents.

Both the external-API reward agent and the local-sglang reward agent parse the
same ``\\boxed{...}`` verdict format and read trajectory images off disk. These
helpers were duplicated verbatim across the two agents; they live here now.
"""

from __future__ import annotations

import re

# A PRM judges each step and emits a verdict as ``\boxed{1}`` (good) or
# ``\boxed{-1}`` (bad). These patterns extract and validate that scalar.
_PRM_BOXED_PATTERN = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)
_PRM_STRICT_NUMBER_PATTERN = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*$")


def extract_prm_sign_from_text(text: str) -> int:
    """Parse a PRM verdict into ``+1`` / ``-1`` / ``0`` (unparseable/neutral)."""
    if not text:
        return 0
    match = _PRM_BOXED_PATTERN.search(text)
    if not match:
        return 0
    boxed_content = match.group(1).strip()
    strict_number_match = _PRM_STRICT_NUMBER_PATTERN.fullmatch(boxed_content)
    if not strict_number_match:
        return 0
    try:
        value = float(strict_number_match.group(1))
    except ValueError:
        return 0
    if abs(value - 1.0) < 1e-9:
        return 1
    if abs(value + 1.0) < 1e-9:
        return -1
    return 0


def read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()
