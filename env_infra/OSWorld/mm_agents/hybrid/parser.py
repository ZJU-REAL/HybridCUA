"""Tool-call extractor for the hybrid agent.

qwen's ``parser.parse_xml_tool_call`` hard-codes ``func == "computer_use"`` and drops
everything else — so it would silently discard ``<function=cli>``. We therefore keep the
SAME XML shape (``<tool_call>…</tool_call>`` / ``<function=NAME>`` / ``<parameter=…>``)
but accept BOTH ``computer_use`` and ``cli``, yielding ordered ``(func_name, params)`` so
the agent can dispatch GUI vs CLI while preserving the model's action order.

This file re-implements the extraction (it does not modify qwen); the value coercion
(JSON for ``[…]``/``{…}`` params, else raw string) matches qwen's behavior so the reused
``parse_base_response`` sees identical ``computer_use`` params.
"""
from __future__ import annotations

import json
import re
from typing import Dict, Iterator, List, Tuple

#: Function names the hybrid agent understands.
HYBRID_FUNCTIONS = ("computer_use", "cli")

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>]+)>")
_PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def _parse_one(xml_content: str, allowed: Tuple[str, ...]) -> Tuple[str, Dict] | None:
    """Parse a single ``<tool_call>`` body into ``(func_name, params)`` or None.

    None means the function name is not in *allowed* (unknown/foreign tool).
    """
    func_match = _FUNCTION_RE.search(xml_content)
    if not func_match:
        return None
    func_name = func_match.group(1).strip()
    if func_name not in allowed:
        return None

    params: Dict = {}
    for match in _PARAM_RE.finditer(xml_content):
        name = match.group(1).strip()
        value = match.group(2).strip()
        # Match qwen: JSON-decode list/object values, else keep raw string.
        if value.startswith("[") or value.startswith("{"):
            try:
                params[name] = json.loads(value)
                continue
            except json.JSONDecodeError:
                pass
        params[name] = value
    return func_name, params


def iter_tool_calls(
    response: str, allowed: Tuple[str, ...] = HYBRID_FUNCTIONS
) -> Iterator[Tuple[str, Dict]]:
    """Yield ``(func_name, params)`` for each recognized ``<tool_call>`` in order."""
    for m in _TOOL_CALL_RE.finditer(response):
        parsed = _parse_one(m.group(1), allowed)
        if parsed is not None:
            yield parsed


def split_tool_calls(
    response: str, allowed: Tuple[str, ...] = HYBRID_FUNCTIONS
) -> List[Tuple[str, Dict]]:
    """List form of :func:`iter_tool_calls` (order preserved)."""
    return list(iter_tool_calls(response, allowed))
