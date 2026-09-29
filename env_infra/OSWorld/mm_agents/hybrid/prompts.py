"""System prompt assembly for the hybrid agent: computer_use (GUI) + cli (CLI).

The ``computer_use`` tool definition is **copied from** ``mm_agents.qwen.prompts``
(``BASE_ACTION_DESCRIPTION_PROMPT`` + ``build_description_prompt`` + ``build_base_tools_def``)
rather than imported, for ONE reason: qwen's GUI description hard-codes
"You do not have access to a terminal ... You must click on desktop icons", which
directly contradicts the ``cli`` tool we add. Owning the text lets us fix that cleanly
(no fragile runtime string-replace). See ``_HYBRID_GUI_INTRO``.

Copying forks the GUI action list: if qwen adds/renames a computer_use action upstream,
this copy will NOT auto-track it — update ``build_gui_tool_def`` here to match.

The system-prompt skeleton (``build_base_system_prompt``) is ALSO copied here, for a
second reason: its ``# Response format`` line said "describing what to do in the UI",
which under-describes the CLI. We reword it to name both surfaces. This forks the
skeleton too — keep it in sync if qwen changes the tool-call format upstream.

Still reused from qwen (no fork):
  - ``build_instruction_prompt`` — the per-step user prompt.
The GUI *execution* mapping stays reused too (``qwen.actions.parse_base_response`` in agent.py).
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List

from mm_agents.qwen.prompts import (  # reused read-only from qwen (no contradiction there)
    build_instruction_prompt,  # re-exported for the agent
)

from .cli_tools import build_cli_tool_def

__all__ = [
    "build_gui_tool_def",
    "build_hybrid_tools_def",
    "build_hybrid_system_prompt",
    "build_instruction_prompt",
]


# --- copied from qwen.prompts.BASE_ACTION_DESCRIPTION_PROMPT (base action list) -------
# Kept verbatim; the GUI actions themselves are unchanged. Only the *intro* below is
# edited to remove the "no terminal" contradiction.
_BASE_ACTION_DESCRIPTION_PROMPT = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys (e.g., "ctrl", "shift", "ctrl+shift") that will be held during the click.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) coordinate.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys that will be held during the click.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys that will be held during the click.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys that will be held during the click.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it's the closest action). Optional `text` parameter can specify modifier keys that will be held during the click.
* `scroll`: Performs a scroll of the mouse scroll wheel. Optional `text` parameter can specify a modifier key (e.g., "shift", "ctrl") that will be held during scrolling.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll). Optional `text` parameter can specify a modifier key that will be held during scrolling.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question."""


def _build_gui_description(processed_width: int, processed_height: int, coordinate_type: str) -> str:
    """Copied from qwen.prompts.build_description_prompt, with the terminal
    contradiction fixed (line 2) so it does not conflict with the ``cli`` tool.
    """
    resolution = (
        f"* The screen's resolution is {processed_width}x{processed_height}."
        if coordinate_type == "absolute"
        else "* The screen's resolution is 1000x1000."
    )
    return "\n".join(
        [
            "Use a mouse and keyboard to interact with a computer, and take screenshots.",
            # CHANGED from qwen's "You do not have access to a terminal ... click on
            # desktop icons": we DO have a terminal via the separate `cli` tool.
            "* This is an interface to a desktop GUI. For terminal commands and file "
            "operations, use the separate `cli` tool (it runs on the same machine); use "
            "this GUI for anything that must be seen or clicked on screen. You can also "
            "launch applications from the terminal via `cli` or by clicking desktop icons.",
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.",
            resolution,
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
            "* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
        ]
    )


def build_gui_tool_def(processed_width: int, processed_height: int, coordinate_type: str) -> Dict:
    """The ``computer_use`` tool schema — copied from qwen.prompts.build_base_tools_def.

    Identical action enum/params to qwen's base tool; only the description intro differs
    (terminal contradiction removed). Keep this in sync with qwen upstream if its base
    action set changes.
    """
    return {
        "type": "function",
        "function": {
            "name": "computer_use",
            "description": _build_gui_description(processed_width, processed_height, coordinate_type),
            "parameters": {
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {
                        "type": "string",
                        "description": _BASE_ACTION_DESCRIPTION_PROMPT,
                        "enum": [
                            "key",
                            "type",
                            "mouse_move",
                            "left_click",
                            "left_click_drag",
                            "right_click",
                            "middle_click",
                            "double_click",
                            "triple_click",
                            "scroll",
                            "hscroll",
                            "wait",
                            "terminate",
                            "answer",
                        ],
                    },
                    "keys": {"type": "array", "description": "Required only by `action=key`."},
                    "text": {
                        "type": "string",
                        "description": "Required by `action=type` and `action=answer`. Optional for click actions (left_click, right_click, middle_click, double_click, triple_click) to specify modifier keys (e.g., 'ctrl', 'shift', 'ctrl+shift'). Optional for scroll actions (scroll, hscroll) to specify a modifier key (e.g., 'shift', 'ctrl') to hold during scrolling.",
                    },
                    "coordinate": {"type": "array", "description": "(x, y) coordinates."},
                    "pixels": {"type": "number", "description": "Scroll amount."},
                    "time": {"type": "number", "description": "Seconds to wait."},
                    "status": {
                        "type": "string",
                        "description": "Task status for terminate.",
                        "enum": ["success", "failure"],
                    },
                },
            },
        },
    }


def build_hybrid_tools_def(processed_width: int, processed_height: int, coordinate_type: str) -> List[Dict]:
    """The two tool schemas the model sees: our ``computer_use`` (fixed) + ``cli``."""
    return [
        build_gui_tool_def(processed_width, processed_height, coordinate_type),
        build_cli_tool_def(),
    ]


def build_hybrid_system_prompt(tools_def: List[Dict], collapse_text: str, password: str = "password") -> str:
    """System-prompt skeleton — copied from qwen.prompts.build_base_system_prompt.

    Verbatim except two edits: (1) the ``# Response format`` step-1 line, which qwen
    phrased as "a short imperative describing what to do in the UI", is reworded to name
    both the GUI (computer_use) and the CLI (cli) so the CLI surface isn't under-described;
    (2) the sudo ``password`` is stated up front (mirrors ``kimi_hybrid``) so the model
    doesn't have to guess it for ``sudo`` — the value comes from ``DesktopEnv.client_password``.
    """
    return (
        "You are a multi-purpose intelligent assistant. Based on my requests, you can use tools to help me complete various tasks.\n"
        f"The password of the computer is {password}.\n\n"
        "# Tools\n\n"
        "You have access to the following functions:\n\n"
        "<tools>\n"
        + json.dumps(tools_def)
        + "\n</tools>\n\n"
        "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
        "<tool_call>\n"
        "<function=example_function_name>\n"
        "<parameter=example_parameter_1>\n"
        "value_1\n"
        "</parameter>\n"
        "<parameter=example_parameter_2>\n"
        "This is the value for the second parameter\n"
        "that can span\n"
        "multiple lines\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "<IMPORTANT>\n"
        "Reminder:\n"
        "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
        "- Required parameters MUST be specified\n"
        "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
        "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
        f"- The current date is {datetime.today().strftime('%A, %B %d, %Y')}.\n"
        f"- Collapsed screenshots appear as text: {collapse_text}\n"
        "</IMPORTANT>\n\n"
        "# Response format\n\n"
        "Response format for every step:\n"
        # CHANGED from qwen's "describing what to do in the UI": name both surfaces.
        "1) Action: a short imperative describing the next step — a GUI interaction (computer_use) or a terminal/file operation (cli).\n"
        "2) A single <tool_call>...</tool_call> block.\n\n"
        "Rules:\n"
        "- Output exactly in the order: Action, <tool_call>.\n"
        "- Be brief: one sentence for Action.\n"
        "- Do not output anything else outside those parts.\n"
        "- If finishing, use action=terminate in the tool call."
    )
