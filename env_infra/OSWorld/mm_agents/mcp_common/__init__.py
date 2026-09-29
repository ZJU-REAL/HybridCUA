"""Shared MCP plumbing: tool selection, guest bring-up, call execution, TIR."""
from .guest_ops import (MCP_SRC, bring_up_via_run_code, bundle_b64,
                        mcp_is_serving, preflight)
from .runtime import (call_mcp_tool, count_tool_invocations, extract_tool_names,
                      list_tools_for_task, provision_mcp, summarize_episode)
from .tool_selection import (ALL_GROUPS, DISTRACTOR_PREFIXES, audit_groups,
                             filter_tools, groups_for_related_apps, normalize_app,
                             split_inventory, tool_prefixes_for_task)

__all__ = [
    "MCP_SRC", "bring_up_via_run_code", "bundle_b64", "mcp_is_serving", "preflight",
    "call_mcp_tool", "count_tool_invocations", "extract_tool_names",
    "list_tools_for_task", "provision_mcp", "summarize_episode",
    "ALL_GROUPS", "DISTRACTOR_PREFIXES", "audit_groups", "filter_tools",
    "groups_for_related_apps", "normalize_app", "split_inventory",
    "tool_prefixes_for_task",
]
