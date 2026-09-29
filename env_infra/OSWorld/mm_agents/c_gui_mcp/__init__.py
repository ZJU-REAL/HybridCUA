"""c_gui + MCP: three arms (off / bash / action) over CGuiAgent."""
from importlib import import_module
from typing import Any

__all__ = [
    "CGuiMcpAgent", "run_single_example_c_gui_mcp", "MCP_MODES",
    "MCP_CALL_SNIPPET", "build_example", "build_mcp_section", "patch_tool_def",
    "render_tool_catalog",
]

LAZY = {
    "CGuiMcpAgent": ".agent",
    "run_single_example_c_gui_mcp": ".run_loop",
    "MCP_MODES": ".prompt",
    "MCP_CALL_SNIPPET": ".prompt",
    "build_example": ".prompt",
    "build_mcp_section": ".prompt",
    "patch_tool_def": ".prompt",
    "render_tool_catalog": ".prompt",
}


def __getattr__(name: str) -> Any:
    target = LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(target, __name__), name)


def __dir__():
    return sorted(__all__)
