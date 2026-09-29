"""Claude hybrid agent: Claude on a single bash surface with real-pixel coordinates.

Action space, shim and episode loop all come from :mod:`mm_agents.c_gui_pixel` and are
shared byte-for-byte with :mod:`mm_agents.kimi_hybrid`; only the transport and the reply
syntax differ (Claude emits native ``tool_use`` blocks, Kimi a fenced code block).
``run_single_example_claude_hybrid`` is that shared loop under the name the runner
imports — it dispatches on the action dict's ``kind`` and is unaware that this agent
keeps stateful ``tool_use``/``tool_result`` pairing internally.
"""
from __future__ import annotations

from mm_agents.c_gui_pixel import run_single_example_c_gui_pixel

from .prompts import (
    TOOL_NAME,
    build_claude_hybrid_system_prompt,
    build_claude_hybrid_tool_def,
)

#: The episode loop, under the name the runner and its docstring already use.
run_single_example_claude_hybrid = run_single_example_c_gui_pixel

# ClaudeHybridAgent pulls in mm_agents.anthropic (the anthropic SDK, PIL); import lazily
# so the prompt/tool helpers and the loop stay importable where those deps are absent
# (mirrors the pattern in hybrid / c_gui / evocua_hybrid).
try:
    from .agent import ClaudeHybridAgent
except Exception:  # pragma: no cover - optional heavy deps
    ClaudeHybridAgent = None  # type: ignore[assignment]

__all__ = [
    "ClaudeHybridAgent",
    "run_single_example_claude_hybrid",
    "build_claude_hybrid_tool_def",
    "build_claude_hybrid_system_prompt",
    "TOOL_NAME",
]
