"""Kimi hybrid agent: Kimi on a single bash surface with real-pixel coordinates.

Action space, shim and episode loop all come from :mod:`mm_agents.c_gui_pixel` and are
shared byte-for-byte with :mod:`mm_agents.claude_hybrid`; only the transport and the
reply syntax differ. ``run_single_example_kimi_hybrid`` is the shared loop under the name
the runner imports — it dispatches on the action dict's ``kind`` and never inspects the
agent class, so there is nothing Kimi-specific to write.
"""
from __future__ import annotations

from mm_agents.c_gui_pixel import run_single_example_c_gui_pixel

from .prompts import build_kimi_hybrid_system_prompt

#: The episode loop, under the name the runner and its docstring already use.
run_single_example_kimi_hybrid = run_single_example_c_gui_pixel

# KimiHybridAgent pulls in kimi.kimi_agent (httpx, loguru, backoff); import lazily so the
# prompt helper and the loop stay importable where those deps are absent (mirrors the
# pattern in hybrid / c_gui / evocua_hybrid).
try:
    from .agent import KimiHybridAgent
except Exception:  # pragma: no cover - optional heavy deps
    KimiHybridAgent = None  # type: ignore[assignment]

__all__ = [
    "KimiHybridAgent",
    "run_single_example_kimi_hybrid",
    "build_kimi_hybrid_system_prompt",
]
