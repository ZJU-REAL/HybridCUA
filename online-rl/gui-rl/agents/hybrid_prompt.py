"""Prompt / tool schema / parser / message-history helpers for the Qwen3.5 hybrid CUA
training agent — byte-identical port of the eval-side c_gui + qwen/hybrid pieces so
TRAINING == EVAL.

Sources (copied verbatim, only import headers changed — online-rl and env_infra live in
separate venvs, so importing ``mm_agents.*`` is not possible):
  * prompt + tool schema  <- env_infra/OSWorld/mm_agents/c_gui/prompts.py
  * XML tool-call parser   <- env_infra/OSWorld/mm_agents/c_gui/parser.py

The instruction template, history folding helpers and message builder are shared with
the GUI-only agent and live in ``agents/utils/qwen_prompts.py`` /
``agents/utils/qwen_history.py``; they are re-exported below for back-compat.

The single tool has 4 actions (bash / wait / terminate / answer); GUI is driven via
``action=bash`` running pyautogui in a quoted heredoc (coords 0-999, scaled by the VM
shim in clients/coord_shim.py). The tool's function NAME is the A/B variable
(``computer_use`` or ``cli``); contents are identical.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Dict, Iterator, List, Tuple

# History folding / message assembly and the instruction template are NOT c_gui-specific
# — the GUI-only agent needs the same ones. They live in agents/utils/ and are re-exported
# here so existing `from agents.hybrid_prompt import ...` call sites keep working.
from agents.utils.qwen_history import (
    build_messages,
    previous_actions_text,
    should_collapse_step,
    update_folding_state,
    wrap_tool_response,
)
from agents.utils.qwen_prompts import build_instruction_prompt

__all__ = [
    "build_c_gui_tool_def",
    "build_c_gui_tools_def",
    "build_c_gui_system_prompt",
    "build_instruction_prompt",
    "C_GUI_FUNCTIONS",
    "iter_tool_calls",
    "split_tool_calls",
    "update_folding_state",
    "should_collapse_step",
    "previous_actions_text",
    "wrap_tool_response",
    "build_hybrid_messages",
]


# =============================================================================
# Prompt + tool schema  (verbatim from c_gui/prompts.py)
# =============================================================================
_BASH_ACTION_DESCRIPTION = (
    "* `bash`: run ONE shell command (requires `command`).\n"
    "  - CLI / file ops: write shell directly (ls, cat, sed, grep, python3 - <<'PY' ... PY, ...).\n"
    "  - GUI: drive the screen with pyautogui inside a QUOTED heredoc, e.g.\n"
    "        python3 <<'PY'\n"
    "        import pyautogui\n"
    "        pyautogui.click(500, 300)      # coordinates are 0-999 (screen treated as 1000x1000)\n"
    "        pyautogui.typewrite('hello', interval=0.02)\n"
    "        PY\n"
    "    pyautogui functions:\n"
    "      pyautogui.click(x, y)                        # left click\n"
    "      pyautogui.doubleClick(x, y)                  # double click\n"
    "      pyautogui.tripleClick(x, y)                  # triple click\n"
    "      pyautogui.rightClick(x, y)                   # right click\n"
    "      pyautogui.middleClick(x, y)                  # middle click\n"
    "      pyautogui.moveTo(x, y)                       # move mouse\n"
    "      pyautogui.dragTo(x, y, duration=0.5)         # drag to position\n"
    "      pyautogui.mouseDown()                        # press mouse button\n"
    "      pyautogui.mouseUp()                          # release mouse button\n"
    "      pyautogui.scroll(-5)                         # scroll down (negative=down)\n"
    "      pyautogui.press('enter')                     # press a key\n"
    "      pyautogui.hotkey('ctrl', 's')                # keyboard shortcut\n"
    "      pyautogui.typewrite('text', interval=0.02)   # type text\n"
    "      pyautogui.keyDown('shift')                   # hold key down\n"
    "      pyautogui.keyUp('shift')                     # release key\n"
    "  - Effects are seen via the NEXT screenshot; use print() for any text you need back (stdout is returned).\n"
    "  - One command may bundle multiple steps (several pyautogui lines in the heredoc; && / pipes in shell).\n"
    "  - Optional `timeout` (seconds) for a slow command; the sandbox cuts commands short after ~30s, so\n"
    "    split anything longer or run it in the background.\n"
    "* `wait`: wait for the screen to settle. Optional `time` (seconds).\n"
    "* `terminate`: finish the task. Requires `status` = success | failure.\n"
    "* `answer`: answer a question-type task. Requires `text`."
)


def build_c_gui_tool_def(tool_name: str) -> Dict:
    """The single tool schema. ``tool_name`` is ``computer_use`` or ``cli`` (A/B variable)."""
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": (
                "Control the machine through a bash terminal. One action per call. The screen is a "
                "GUI desktop: drive it with pyautogui via a quoted heredoc, or run shell commands / "
                "file operations directly."
            ),
            "parameters": {
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {
                        "type": "string",
                        "description": _BASH_ACTION_DESCRIPTION,
                        "enum": ["bash", "wait", "terminate", "answer"],
                    },
                    "command": {
                        "type": "string",
                        "description": (
                            "Required by action=bash. A single shell command. For GUI, run pyautogui "
                            "via a quoted heredoc: python3 <<'PY' ... PY (coordinates 0-999)."
                        ),
                    },
                    "timeout": {
                        "type": "number",
                        "description": (
                            "Optional for action=bash: seconds to allow the command (default 60). "
                            "The sandbox cuts commands short after ~30s regardless."
                        ),
                    },
                    "time": {"type": "number", "description": "Optional for action=wait: seconds."},
                    "status": {
                        "type": "string",
                        "enum": ["success", "failure"],
                        "description": "Required by action=terminate.",
                    },
                    "text": {"type": "string", "description": "Required by action=answer."},
                },
            },
        },
    }


def build_c_gui_tools_def(tool_name: str) -> List[Dict]:
    """List wrapper (mirrors ``build_hybrid_tools_def``) — a single tool for c-gui."""
    return [build_c_gui_tool_def(tool_name)]


def build_c_gui_system_prompt(tools_def: List[Dict], collapse_text: str, password: str = "password") -> str:
    """System prompt v3 — restore JSON schema for anchoring + comprehensive examples."""
    tool_name = tools_def[0]["function"]["name"]
    # Pre-format the JSON so we don't fight with .format() braces
    tools_json = json.dumps(tools_def)
    return (
        "You are a multi-purpose intelligent assistant operating a computer through a bash terminal.\n"
        f"The password of the computer is {password}.\n\n"
        "# Environment\n\n"
        "You face a machine with a graphical desktop AND a bash terminal. You act ONLY by running one "
        "shell command per step (action=bash): write shell for CLI/file operations, or drive the GUI "
        "with pyautogui inside a quoted heredoc `python3 <<'PY' ... PY` (coordinates are 0-999). Each "
        "step you are shown the latest screenshot plus the previous command's output.\n\n"
        "# Tools\n\n"
        "You have access to the following functions:\n\n"
        "<tools>\n"
        f"{tools_json}\n"
        "</tools>\n\n"
        "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
        "<tool_call>\n"
        "<function=example_function_name>\n"
        "<parameter=example_parameter_1>\n"
        "value_1\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "<IMPORTANT>\n"
        "- Function calls MUST follow the specified format\n"
        f"- The `action` parameter MUST be one of: bash, wait, terminate, answer\n"
        "- ALL GUI interactions MUST use action=bash with a pyautogui heredoc\n"
        "- Coordinates are 0-999 (the screen is a 1000x1000 grid)\n"
        "- Heredoc delimiter MUST be quoted: <<'PY' (not <<PY)\n"
        "- Observe effects via the NEXT screenshot; use print() for text output\n"
        "- When finished, use action=terminate (not bash exit commands)\n"
        f"- The current date is {datetime.today().strftime('%A, %B %d, %Y')}\n"
        f"- Collapsed screenshots appear as: {collapse_text}\n"
        "</IMPORTANT>\n\n"
        "# Response format\n\n"
        "Every step:\n"
        "1) Action: one sentence describing your next move.\n"
        "2) A single <tool_call>...</tool_call> block.\n\n"
        "# Output format examples\n\n"
        f"## GUI action (action=bash with pyautogui heredoc)\n\n"
        f"Action: Click the \"File\" menu in the top menu bar.\n\n"
        f"<tool_call>\n"
        f"<function={tool_name}>\n"
        f"<parameter=action>\n"
        f"bash\n"
        f"</parameter>\n"
        f"<parameter=command>\n"
        f"python3 <<'PY'\n"
        f"import pyautogui\n"
        f"pyautogui.click(50, 15)\n"
        f"PY\n"
        f"</parameter>\n"
        f"</function>\n"
        f"</tool_call>\n\n"
        f"## Shell command (action=bash with shell)\n\n"
        f"Action: List files in the Documents folder.\n\n"
        f"<tool_call>\n"
        f"<function={tool_name}>\n"
        f"<parameter=action>\n"
        f"bash\n"
        f"</parameter>\n"
        f"<parameter=command>\n"
        f"ls -la ~/Documents/\n"
        f"</parameter>\n"
        f"</function>\n"
        f"</tool_call>\n\n"
        f"## Wait (action=wait)\n\n"
        f"Action: Wait for the application to finish loading.\n\n"
        f"<tool_call>\n"
        f"<function={tool_name}>\n"
        f"<parameter=action>\n"
        f"wait\n"
        f"</parameter>\n"
        f"<parameter=time>\n"
        f"3\n"
        f"</parameter>\n"
        f"</function>\n"
        f"</tool_call>\n\n"
        f"## Finish (action=terminate)\n\n"
        f"Action: The task is complete.\n\n"
        f"<tool_call>\n"
        f"<function={tool_name}>\n"
        f"<parameter=action>\n"
        f"terminate\n"
        f"</parameter>\n"
        f"<parameter=status>\n"
        f"success\n"
        f"</parameter>\n"
        f"</function>\n"
        f"</tool_call>"
    )


# =============================================================================
# XML tool-call parser  (verbatim from c_gui/parser.py)
# =============================================================================
#: Function names c-gui may be configured with (the A/B naming variable).
C_GUI_FUNCTIONS = ("computer_use", "cli")

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>]+)>")
_PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def _parse_one(xml_content: str, allowed: Tuple[str, ...]) -> "Tuple[str, Dict] | None":
    """Parse a single ``<tool_call>`` body into ``(func_name, params)`` or None.

    None means the function name is not in *allowed* (unknown/foreign tool).
    """
    func_match = _FUNCTION_RE.search(xml_content)
    if not func_match:
        return None
    func_name = func_match.group(1).strip()
    if func_name not in allowed:
        return None

    params: Dict = {}
    for match in _PARAM_RE.finditer(xml_content):
        # Raw string only — NO JSON coercion: `command` may be a heredoc or start with
        # [ / {, which json.loads would mangle.
        params[match.group(1).strip()] = match.group(2).strip()
    return func_name, params


def iter_tool_calls(
    response: str, allowed: Tuple[str, ...] = C_GUI_FUNCTIONS
) -> Iterator[Tuple[str, Dict]]:
    """Yield ``(func_name, params)`` for each recognized ``<tool_call>`` in order."""
    for m in _TOOL_CALL_RE.finditer(response):
        parsed = _parse_one(m.group(1), allowed)
        if parsed is not None:
            yield parsed


def split_tool_calls(
    response: str, allowed: Tuple[str, ...] = C_GUI_FUNCTIONS
) -> List[Tuple[str, Dict]]:
    """List form of :func:`iter_tool_calls` (order preserved)."""
    return list(iter_tool_calls(response, allowed))


# =============================================================================
# Hybrid message builder
# =============================================================================
def build_hybrid_messages(**kwargs) -> List[Dict]:
    """The bash-surface message layout: :func:`build_messages` with a CLI-output turn.

    ``cli_outputs`` is required here (that is the only difference from the GUI-only
    layout), so this is a named alias rather than a separate implementation.
    """
    kwargs.setdefault("cli_outputs", [])
    return build_messages(**kwargs)
