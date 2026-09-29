"""QwenAgent + MCP. One mode: `action` (enum gains `mcp` + name/params, catalog in schema).

No `off` arm -- both arms need the worked GUI example, which already breaks
byte-identity with stock QwenAgent. Use run_qwen_qwen35_9b_sharded.sh for a
leaderboard-comparable baseline.

No `bash` arm -- QwenAgent's action space has no shell channel, so any way to
express a tool call is itself an action-space change. Use c_gui_mcp for that.
"""
from __future__ import annotations

import copy
import json
import logging
from typing import Dict, List, Optional, Sequence, Tuple

from mm_agents.c_gui_mcp.prompt import MCP_CALL_SNIPPET, render_tool_catalog

logger = logging.getLogger("desktopenv.qwen_mcp_agent")

try:
    from mm_agents.qwen import QwenAgent
except Exception as _exc:
    QwenAgent = None
    _IMPORT_ERROR = _exc
else:
    _IMPORT_ERROR = None


MCP_BULLET = (
    "\n* `mcp`: Call a structured MCP tool instead of driving the GUI. Requires"
    " `name` (full tool name from the catalog below) and `params` (a JSON object;"
    " use {} for no-arg tools). One call replaces several GUI steps and returns a"
    " precise result rather than a screenshot to read. Use it when a listed tool"
    " cleanly matches your intent; otherwise stay with the GUI actions. Some"
    " listed tools are irrelevant to your task -- calling those wastes a step and"
    " may change state you did not intend to touch."
)


EXAMPLES_HEADER = """

# Output format examples"""

GUI_EXAMPLE = """

Action: Click the "File" menu in the top menu bar.

<tool_call>
<function=computer_use>
<parameter=action>
left_click
</parameter>
<parameter=coordinate>
[50, 15]
</parameter>
</function>
</tool_call>"""

MCP_EXAMPLE = """

Action: Set cell A1 to 42 with the calc tool instead of clicking through the UI.

<tool_call>
<function=computer_use>
<parameter=action>
mcp
</parameter>
<parameter=name>
osworld_mcp_libreoffice_calc.set_cell_value
</parameter>
<parameter=params>
{"cell": "A1", "value": "42"}
</parameter>
</function>
</tool_call>"""


def patch_qwen_tools_def(tools_def: Dict, tools: Sequence[Dict],
                         desc_limit: int = 110) -> Dict:
    """Return a copy of QwenAgent's tools_def with the MCP action declared."""
    tools = list(tools or [])
    if not tools:
        return tools_def

    patched = copy.deepcopy(tools_def)
    props = (patched.setdefault("function", {})
                    .setdefault("parameters", {})
                    .setdefault("properties", {}))

    action = props.get("action")
    if not isinstance(action, dict):
        logger.warning("qwen mcp: tools_def has no `action` property; not patching")
        return tools_def

    enum = list(action.get("enum") or [])
    if "mcp" not in enum:
        action["enum"] = enum + ["mcp"]
    if "`mcp`" not in (action.get("description") or ""):
        action["description"] = (action.get("description") or "") + MCP_BULLET

    catalog = render_tool_catalog(tools, desc_limit=desc_limit)
    props["name"] = {
        "type": "string",
        "description": (
            f"Required only by `action=mcp`. The full tool name, copied exactly "
            f"from this catalog of {len(tools)} available tools:\n{catalog}"
        ),
    }
    props["params"] = {
        "type": "object",
        "description": ("Required only by `action=mcp`: the tool's arguments as a "
                        "JSON object. Use {} when the tool takes no arguments."),
    }
    return patched


def mcp_action_to_code(name: str, params: Dict) -> str:
    """Render one MCP call as the python3 one-liner env.step() will execute."""
    return MCP_CALL_SNIPPET.format(name=name, params=repr(dict(params or {})))


if QwenAgent is not None:

    class QwenMcpAgent(QwenAgent):
        """QwenAgent that can be offered an MCP catalog."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.mcp_tools: List[Dict] = []
            self.tool_names: set = set()

        def set_episode_tools(self, tools: Optional[Sequence[Dict]]) -> None:
            """Install this episode's catalog (once per episode, before predict)."""
            self.mcp_tools = list(tools or [])
            self.tool_names = {
                (t["name"] if isinstance(t, dict) else getattr(t, "name", ""))
                for t in self.mcp_tools
            }
            logger.info("qwen mcp agent: tools=%d", len(self.mcp_tools))

        def _build_tools_def(self, processed_width: int, processed_height: int) -> Dict:
            tools_def = super()._build_tools_def(processed_width, processed_height)
            if not self.mcp_tools:
                return tools_def
            return patch_qwen_tools_def(tools_def, self.mcp_tools)

        def _build_system_prompt(self, tools_def: Dict) -> str:
            """Append a GUI example, plus an MCP one when tools are live.

            Stock QwenAgent ships no concrete <tool_call> -- only the
            example_function_name placeholder. Showing just the MCP one would make
            tools the single demonstrated action and inflate TIR for a reason
            unrelated to tools being useful.
            """
            prompt = super()._build_system_prompt(tools_def)
            blocks = EXAMPLES_HEADER + GUI_EXAMPLE + (MCP_EXAMPLE if self.mcp_tools else "")
            return prompt.rstrip() + blocks

        def _parse_response(self, response: str, **kwargs) -> Tuple[str, List[str]]:
            """Delegate to QwenAgent, then add the mcp calls it drops."""
            low_level, codes = super()._parse_response(response, **kwargs)
            if not self.mcp_tools:
                return low_level, codes

            calls = self.parse_mcp_calls(response)
            if not calls:
                return low_level, codes

            if codes == ["DONE"] and "terminate" not in response.lower():
                codes = []

            codes = [mcp_action_to_code(n, p) for n, p in calls] + list(codes)
            suffix = f"{len(calls)} mcp call(s)"
            low_level = f"{low_level} + {suffix}" if low_level else suffix
            return low_level, codes

        def parse_mcp_calls(self, response: str) -> List[Tuple[str, Dict]]:
            """Extract (name, params) for every action=mcp tool_call in order."""
            from mm_agents.qwen.parser import iter_tool_call_params

            out: List[Tuple[str, Dict]] = []
            try:
                param_iter = list(iter_tool_call_params(response))
            except Exception as exc:
                logger.warning("qwen mcp: could not iterate tool calls: %s", exc)
                return out

            for params in param_iter:
                if not isinstance(params, dict) or params.get("action") != "mcp":
                    continue
                name = str(params.get("name") or "").strip()
                if not name:
                    logger.warning("qwen mcp: action=mcp with no name; dropping")
                    continue
                if self.tool_names and name not in self.tool_names:
                    logger.warning("qwen mcp: tool %r not in this episode's "
                                   "catalog; dropping", name)
                    continue
                out.append((name, coerce_params(params.get("params"))))
            return out

else:

    class QwenMcpAgent:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "mm_agents.qwen.QwenAgent is unavailable; QwenMcpAgent cannot be "
                f"constructed. Original import error: {_IMPORT_ERROR}"
            )


def coerce_params(raw) -> Dict:
    """Coerce the hand-written `params` text to a dict; {} if unparseable."""
    if isinstance(raw, dict):
        return raw
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        val = json.loads(text)
    except Exception:
        import ast
        try:
            val = ast.literal_eval(text)
        except Exception:
            logger.warning("qwen mcp: unparseable params %r -> {}", text[:120])
            return {}
    return val if isinstance(val, dict) else {}
