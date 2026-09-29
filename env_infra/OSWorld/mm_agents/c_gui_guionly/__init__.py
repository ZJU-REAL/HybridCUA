from .parser import split_tool_calls, iter_tool_calls, C_GUI_FUNCTIONS
from .prompts import build_c_gui_tool_def, build_c_gui_tools_def, build_c_gui_system_prompt
from .shim import build_provision_command, SHIM_SOURCE, COORD_SCALE_ENV, DEFAULT_COORD_SCALE
from .executor import provision_shim, bash_payload
from .run_loop import run_single_example_c_gui_guionly

# CGuiGuiOnlyAgent pulls in qwen.main (openai client, PIL); import lazily so the light helpers
# above stay importable where those heavy deps are absent (mirrors hybrid).
try:
    from .agent import CGuiGuiOnlyAgent
except Exception:  # pragma: no cover - optional heavy deps
    CGuiGuiOnlyAgent = None  # type: ignore[assignment]

__all__ = [
    "CGuiGuiOnlyAgent",
    "run_single_example_c_gui_guionly",
    "build_c_gui_tool_def",
    "build_c_gui_tools_def",
    "build_c_gui_system_prompt",
    "split_tool_calls",
    "iter_tool_calls",
    "C_GUI_FUNCTIONS",
    "build_provision_command",
    "SHIM_SOURCE",
    "COORD_SCALE_ENV",
    "DEFAULT_COORD_SCALE",
    "provision_shim",
    "bash_payload",
]
