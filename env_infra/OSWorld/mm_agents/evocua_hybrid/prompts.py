"""Prompt + tool surface for the EvoCUA hybrid (GUI+CLI) agent.

EvoCUA ships a single ``computer_use`` tool (14 GUI actions, JSON tool-calls). This
module adds a SECOND tool -- a ``cli`` surface. Which one is chosen by the
``EVOCUA_CLI_SURFACE`` env var (see the import block below): ``bash`` (default) gives
bash only, ``full`` gives the original bash/read/write/edit set from
``mm_agents/hybrid/cli_tools.py``.

Default is bash-only because that is the surface the RL side actually trains on
(gui-rl's hybridcua agent, ``mm_agents/c_gui``): GUI is a ``pyautogui`` heredoc, CLI is a
plain shell command, and ``b(tau)`` in the Stage II reward is defined over exactly that
distinction. See ``cli_tools_bash.py``'s docstring for the full rationale.

Nothing under ``mm_agents/evocua/`` is modified: we import its builders and compose.

``S2_SYSTEM_PROMPT`` already says "You may call one or more functions" and takes a
``{tools_xml}`` placeholder, so multi-tool works with the stock skeleton. The only edit
is the ``# Response format`` block, which hard-codes "A single <tool_call>" describing
the UI only -- reworded here to name both surfaces (same rationale as
``hybrid/prompts.py:152-159``, which reworded qwen's step-1 line for the same reason).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from mm_agents.evocua.prompts import (
    S2_DESCRIPTION_PROMPT_TEMPLATE,
    build_s2_tools_def,
)

# EVOCUA_CLI_SURFACE picks which `cli` tool the model sees:
#   "bash" (default) -- bash only, matching the single-bash surface that gui-rl's
#       hybridcua agent and mm_agents/c_gui use. Required for the A/B against the
#       pure-GUI arm to isolate "does CLI help" from "do structured file tools help".
#   "full" -- the original bash/read/write/edit surface.
if os.environ.get("EVOCUA_CLI_SURFACE", "bash").strip().lower() == "full":
    from mm_agents.hybrid.cli_tools import build_cli_tool_def
else:
    from mm_agents.hybrid.cli_tools_bash import build_cli_tool_def

#: Function names the hybrid EvoCUA agent understands.
EVOCUA_HYBRID_FUNCTIONS = ("computer_use", "cli")


def build_evocua_hybrid_tools_def(description_prompt: str) -> List[Dict[str, Any]]:
    """The two tool schemas the model sees: EvoCUA's ``computer_use`` + hybrid's ``cli``."""
    return [build_s2_tools_def(description_prompt), build_cli_tool_def()]


#: Forked from evocua.prompts.S2_SYSTEM_PROMPT. The "# Tools" half is verbatim; the
#: "# Response format" half names the CLI surface so it isn't under-described, and drops
#: "single" (a turn may legitimately mix one GUI and one CLI call).
S2_HYBRID_SYSTEM_PROMPT = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_xml}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

# Surfaces

You act on ONE machine through two interchangeable surfaces:
- `computer_use`: the graphical desktop (mouse/keyboard on the screenshot).
- `cli`: a shell on that same machine.

Prefer `cli` when the goal is really a file or data operation -- editing a spreadsheet,
rewriting a config, batch-renaming, inspecting logs. It is faster and far more reliable
than driving the GUI for those. Use `computer_use` for anything that only exists on
screen. The password for sudo is {password}.

# Response format

Response format for every step:
1) Action: a short imperative describing what to do -- in the UI (`computer_use`) or in
   the shell (`cli`).
2) One or more <tool_call>...</tool_call> blocks, each containing only the JSON:
   {{"name": <function-name>, "arguments": <args-json-object>}}.

Rules:
- Output exactly in the order: Action, then the <tool_call> block(s).
- Be brief: one sentence for Action.
- Do not output anything else outside those parts.
- If finishing, use action=terminate in a `computer_use` tool call."""


def build_evocua_hybrid_system_prompt(
    tools_def: List[Dict[str, Any]], password: str = "password"
) -> str:
    """Render the hybrid system prompt with both tool schemas inlined."""
    return S2_HYBRID_SYSTEM_PROMPT.format(
        tools_xml=json.dumps(tools_def), password=password
    )


def build_description_prompt(coordinate_type: str, p_width: int, p_height: int) -> str:
    """GUI tool description -- same resolution wording as ``EvoCUAAgent._predict_s2``."""
    if coordinate_type == "absolute":
        resolution_info = f"* The screen's resolution is {p_width}x{p_height}."
    else:
        resolution_info = "* The screen's resolution is 1000x1000."
    return S2_DESCRIPTION_PROMPT_TEMPLATE.format(resolution_info=resolution_info)
