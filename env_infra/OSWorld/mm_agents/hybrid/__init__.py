from .cli_tools import build_cli_tool_def, cli_action_to_code, CLI_ACTIONS, CliActionError
from .parser import split_tool_calls, iter_tool_calls, HYBRID_FUNCTIONS
from .prompts import build_hybrid_tools_def, build_hybrid_system_prompt
from .run_loop import run_single_example_hybrid

# HybridAgent pulls in qwen.main (openai client etc.); import lazily so the light
# helpers above stay importable even where those heavy deps are absent.
try:
    from .agent import HybridAgent
except Exception:  # pragma: no cover - optional heavy deps
    HybridAgent = None  # type: ignore[assignment]

__all__ = [
    "HybridAgent",
    "run_single_example_hybrid",
    "build_cli_tool_def",
    "cli_action_to_code",
    "CLI_ACTIONS",
    "CliActionError",
    "split_tool_calls",
    "iter_tool_calls",
    "HYBRID_FUNCTIONS",
    "build_hybrid_tools_def",
    "build_hybrid_system_prompt",
]
