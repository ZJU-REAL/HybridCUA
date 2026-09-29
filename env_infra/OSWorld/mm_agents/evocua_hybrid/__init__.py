from .prompts import (
    build_evocua_hybrid_system_prompt,
    build_evocua_hybrid_tools_def,
    build_description_prompt,
    EVOCUA_HYBRID_FUNCTIONS,
)

# EvoCUAHybridAgent pulls in evocua_agent (openai client, PIL); import lazily so the
# light helpers above stay importable where those heavy deps are absent (mirrors
# hybrid / c_gui).
try:
    from .agent import EvoCUAHybridAgent
except Exception:  # pragma: no cover - optional heavy deps
    EvoCUAHybridAgent = None  # type: ignore[assignment]

# The episode loop is hybrid's, unchanged: it dispatches on the "channel" key and never
# inspects the agent class. Re-exported here so runners have one import site.
from mm_agents.hybrid.run_loop import run_single_example_hybrid

__all__ = [
    "EvoCUAHybridAgent",
    "run_single_example_hybrid",
    "build_evocua_hybrid_system_prompt",
    "build_evocua_hybrid_tools_def",
    "build_description_prompt",
    "EVOCUA_HYBRID_FUNCTIONS",
]
