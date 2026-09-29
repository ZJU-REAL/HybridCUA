"""System prompt + tool schema for the c-gui GUI-ONLY agent (single bash-surface tool).

Forked from ``c_gui/prompts.py``. The MECHANISM is unchanged — ONE tool with 4 actions
(``bash`` / ``wait`` / ``terminate`` / ``answer``), GUI driven by pyautogui inside a
quoted heredoc ``python3 <<'PY' ... PY`` (coords 0-999, scaled in the VM shim). What
changed is the AFFORDANCE: every mention of CLI / shell / file operations is removed, so
the model is told it can only act on the screen.

This is a SOFT constraint — prompt-level only. ``run_loop.py`` still pipes whatever the
model emits to ``env.run_code()``, exactly as in ``c_gui``. A model that invents a plain
shell command anyway WILL have it executed. That is deliberate: the experiment measures
whether the policy spontaneously stays GUI-only when only GUI is advertised, so the
residual CLI% in ``score_report.py`` is itself the result, not a bug.

Diff vs ``c_gui/prompts.py`` (3 places):
  1. ``_BASH_ACTION_DESCRIPTION`` — dropped the "CLI / file ops: write shell directly"
     line and the shell-bundling hint; kept the full pyautogui function list.
  2. ``build_c_gui_tool_def`` — tool/param descriptions no longer offer shell commands.
  3. ``build_c_gui_system_prompt`` — Environment section describes a GUI-only desktop
     (no "AND a bash terminal"), and the ``## Shell command`` worked example is gone.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List

from mm_agents.qwen.prompts import build_instruction_prompt  # reused read-only from qwen

__all__ = [
    "build_c_gui_tool_def",
    "build_c_gui_tools_def",
    "build_c_gui_system_prompt",
    "build_instruction_prompt",
]


_BASH_ACTION_DESCRIPTION = (
    "* `bash`: perform ONE GUI interaction (requires `command`).\n"
    "  - Drive the screen with pyautogui inside a QUOTED heredoc, e.g.\n"
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
    "  - Effects are seen via the NEXT screenshot.\n"
    "  - One command may bundle multiple steps (several pyautogui lines in the same heredoc).\n"
    "  - Optional `timeout` (seconds) for a slow interaction; the sandbox cuts commands short\n"
    "    after ~30s, so split anything longer.\n"
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
                "Control the machine's graphical desktop. One action per call. Drive the "
                "screen with pyautogui via a quoted heredoc."
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
                            "Required by action=bash. A pyautogui heredoc driving the screen: "
                            "python3 <<'PY' ... PY (coordinates 0-999)."
                        ),
                    },
                    "timeout": {
                        "type": "number",
                        "description": (
                            "Optional for action=bash: seconds to allow the interaction (default 60). "
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
        "You are a multi-purpose intelligent assistant operating a computer's graphical desktop.\n"
        f"The password of the computer is {password}.\n\n"
        "# Environment\n\n"
        "You face a machine with a graphical desktop. You act ONLY by driving the screen with "
        "pyautogui inside a quoted heredoc `python3 <<'PY' ... PY` (coordinates are 0-999), one "
        "interaction per step (action=bash). You do NOT have a terminal: there is no shell, no "
        "file manipulation and no command-line access — everything must be accomplished through "
        "the graphical interface, exactly as a human user would. Each step you are shown the "
        "latest screenshot.\n\n"
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
        "- ALL interactions MUST use action=bash with a pyautogui heredoc\n"
        "- You have NO terminal: do not attempt shell commands, file operations or CLI tools\n"
        "- Coordinates are 0-999 (the screen is a 1000x1000 grid)\n"
        "- Heredoc delimiter MUST be quoted: <<'PY' (not <<PY)\n"
        "- Observe effects via the NEXT screenshot\n"
        "- When finished, use action=terminate\n"
        f"- The current date is {datetime.today().strftime('%A, %B %d, %Y')}\n"
        f"- Collapsed screenshots appear as: {collapse_text}\n"
        "</IMPORTANT>\n\n"
        "# Response format\n\n"
        "Every step:\n"
        "1) Action: one sentence describing your next move.\n"
        "2) A single <tool_call>...</tool_call> block.\n\n"
        "# Output format examples\n\n"
        f"## Click (action=bash with pyautogui heredoc)\n\n"
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
        f"## Type text (action=bash with pyautogui heredoc)\n\n"
        f"Action: Type the filename into the focused text field.\n\n"
        f"<tool_call>\n"
        f"<function={tool_name}>\n"
        f"<parameter=action>\n"
        f"bash\n"
        f"</parameter>\n"
        f"<parameter=command>\n"
        f"python3 <<'PY'\n"
        f"import pyautogui\n"
        f"pyautogui.typewrite('report.txt', interval=0.02)\n"
        f"pyautogui.press('enter')\n"
        f"PY\n"
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
