"""Qwen3.5 + MCP: two arms (off / action) over the official QwenAgent."""
from importlib import import_module
from typing import Any

__all__ = [
    "QwenMcpAgent", "run_single_example_qwen_mcp",
    "patch_qwen_tools_def", "mcp_action_to_code",
]

LAZY = {
    "QwenMcpAgent": ".agent",
    "patch_qwen_tools_def": ".agent",
    "mcp_action_to_code": ".agent",
    "run_single_example_qwen_mcp": ".run_loop",
}


def __getattr__(name: str) -> Any:
    target = LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(target, __name__), name)


def __dir__():
    return sorted(__all__)
