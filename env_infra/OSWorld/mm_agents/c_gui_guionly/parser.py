"""Tool-call extractor for the c-gui agent (single tool, bash surface).

Same XML shape as qwen/hybrid (``<tool_call>`` / ``<function=NAME>`` / ``<parameter=..>``),
but two differences from ``hybrid.parser``:

  * the accepted function name is the agent's configured ``tool_name`` — either
    ``computer_use`` or ``cli`` (the A/B naming variable). Pass it via ``allowed``.
  * **no ``[``/``{`` -> JSON coercion.** c-gui params are all scalars
    (``action`` / ``command`` / ``time`` / ``status`` / ``text``); ``command`` routinely
    holds heredoc/shell text and may start with ``[`` or ``{`` (e.g. ``[ -f x ]``), which
    JSON-decoding would corrupt. Every value is taken as a raw stripped string.
"""
from __future__ import annotations

import re
from typing import Dict, Iterator, List, Tuple

#: Function names c-gui may be configured with (the A/B naming variable).
C_GUI_FUNCTIONS = ("computer_use", "cli")

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
        # Raw string only — NO JSON coercion (see module docstring): `command` may be a
        # heredoc or start with [ / {, which json.loads would mangle.
        params[match.group(1).strip()] = match.group(2).strip()
    return func_name, params


def iter_tool_calls(
    response: str, allowed: Tuple[str, ...] = C_GUI_FUNCTIONS
) -> Iterator[Tuple[str, Dict]]:
    """Yield ``(func_name, params)`` for each recognized ``<tool_call>`` in order."""
    for m in _TOOL_CALL_RE.finditer(response):
        parsed = _parse_one(m.group(1), allowed)
        if parsed is not None:
            yield parsed


def split_tool_calls(
    response: str, allowed: Tuple[str, ...] = C_GUI_FUNCTIONS
) -> List[Tuple[str, Dict]]:
    """List form of :func:`iter_tool_calls` (order preserved)."""
    return list(iter_tool_calls(response, allowed))
