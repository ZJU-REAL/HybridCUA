"""Tool schema for the Claude single-bash-surface agent.

ONE custom function tool with c-gui's four actions (``bash`` / ``wait`` / ``terminate`` /
``answer``). This is a REAL API tool passed in ``tools=[...]``, not a text-described
surface — Claude emits ``tool_use`` blocks and the agent pairs ``tool_result`` back by id.

Deliberately NOT used here:

  * the native ``computer_2025xxxx`` tool. The whole point of this surface is that GUI
    and CLI are the same channel, and the native tool would reintroduce a second one
    (plus its beta flag, its 1280x720 display contract, and the coordinate rescaling
    that contract implies). ``mm_agents.anthropic`` remains the agent for that design.
  * the native ``bash_20250124`` / ``text_editor_20250124`` tools. They carry Anthropic's
    own execution semantics, which do not match "run this in the OSWorld VM", and their
    availability varies by gateway.

Because there is no native computer tool, the screenshot needs no resize: coordinates are
real screen pixels at the screenshot's own resolution.
"""
from __future__ import annotations

from typing import Dict, List

from mm_agents.c_gui_pixel.actions import ACTION_ENUM, build_action_description

__all__ = [
    "TOOL_NAME",
    "build_claude_hybrid_tool_def",
    "build_claude_hybrid_system_prompt",
]

#: The tool's name, as Claude will spell it in ``tool_use.name``.
TOOL_NAME = "computer_use"


def build_claude_hybrid_tool_def(
    screen_width: int = 1920, screen_height: int = 1080, tool_name: str = TOOL_NAME
) -> Dict:
    """The single tool schema, in Anthropic's custom-tool shape (flat ``input_schema``)."""
    return {
        "name": tool_name,
        "description": (
            "Control the machine through a bash terminal. One action per call. The screen is a "
            "GUI desktop: drive it with pyautogui via a quoted heredoc, or run shell commands / "
            "file operations directly."
        ),
        "input_schema": {
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {
                    "type": "string",
                    "description": build_action_description(screen_width, screen_height),
                    "enum": ACTION_ENUM,
                },
                "command": {
                    "type": "string",
                    "description": (
                        "Required by action=bash. A single shell command. For GUI, run pyautogui "
                        "via a quoted heredoc: python3 <<'PY' ... PY (coordinates are real "
                        "screen pixels)."
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
    }


def build_claude_hybrid_system_prompt(
    password: str = "osworld-public-evaluation",
    screen_width: int = 1920,
    screen_height: int = 1080,
    max_steps: int = 50,
    tool_name: str = TOOL_NAME,
    suffix: str = "",
) -> str:
    """The system prompt.

    Written here rather than reusing ``anthropic.utils.SYSTEM_PROMPT``: that text is built
    around the native computer tool and a separate bash tool ("Using bash tool you can
    start GUI applications", "use str_replace_editor", ...), which contradicts the single
    surface. What IS carried over from it, because it is environment fact rather than tool
    description: the Ubuntu/DISPLAY notes, the home directory, the sudo password, and the
    ``[INFEASIBLE]`` escape hatch that OSWorld's infeasible-task grading depends on.
    """
    parts: List[str] = [
        "<SYSTEM_CAPABILITY>",
        "* You are operating an Ubuntu virtual machine (x86_64) with internet access, "
        "through a bash terminal.",
        f"* The screen is {screen_width}x{screen_height} and the screenshot you are shown is that "
        "exact size. Coordinates are REAL SCREEN PIXELS read straight off the image.",
        "* You act ONLY by calling the `" + tool_name + "` tool. Every GUI interaction is "
        "action=bash with a pyautogui heredoc; every CLI or file operation is action=bash "
        "with shell. There is no separate GUI channel.",
        "* GUI applications started from the shell need DISPLAY set and a subshell, e.g. "
        "(DISPLAY=:1 xterm &). They may take a few seconds to appear — take another "
        "screenshot to confirm.",
        "* For commands that produce very large output, redirect to a temp file and inspect "
        "it with grep -n -B/-A rather than dumping it all back.",
        "* Feel free to install Ubuntu packages with apt/pip. Use curl instead of wget.",
        "* DO NOT ask the user for clarification during execution. Always take action with "
        "the tools available.",
        "* Effects of a command are seen in the NEXT screenshot; use print() or normal shell "
        "output for any text you need returned to you.",
        "* TASK FEASIBILITY: you may declare a task infeasible at any point — immediately "
        "after the first screenshot, or later once you hit a barrier. If the task cannot be "
        "completed because of missing applications that cannot be installed, insufficient "
        "permissions, contradictory requirements, or any other fundamental blocker, output "
        'exactly "[INFEASIBLE]" (with the brackets) anywhere in your response. That pattern '
        "is detected and fails the task deliberately, which is the correct outcome for an "
        "impossible task.",
        "* Home directory of this Ubuntu system is '/home/user'.",
        f"* If you need a password for sudo, the password of the computer is '{password}'.",
        f"* You have a maximum of {max_steps} steps. Use them wisely.",
        "</SYSTEM_CAPABILITY>",
        "",
        "<IMPORTANT>",
        "* A GUI heredoc delimiter MUST be quoted: python3 <<'PY' (not <<PY), or the shell "
        "will expand the python source before it runs.",
        "* When finished, call action=terminate — not a shell exit command.",
        "* Batch what you can: one bash command may hold several pyautogui lines, or a shell "
        "pipeline. Each tool call costs a round trip, so do not split predictable sequences "
        "across calls. Only stop to take a screenshot when the next step genuinely depends "
        "on seeing the result.",
        "</IMPORTANT>",
    ]
    prompt = "\n".join(parts)
    return f"{prompt}\n\n{suffix}" if suffix else prompt
