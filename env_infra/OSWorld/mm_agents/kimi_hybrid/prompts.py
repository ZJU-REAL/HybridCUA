"""System prompt for the Kimi single-bash-surface agent.

Forked from ``mm_agents.kimi.kimi_agent.SYSTEM_PROMPT_{THINKING,NON_THINKING}`` rather
than imported, because the fork is the whole point: Kimi's stock prompt says the code
block is "either pyautogui code or one of the following functions", i.e. the code block
IS python running on OSWorld's own channel. Here the code block is a SHELL command, and
GUI work happens through a pyautogui heredoc inside it. Nothing short of owning the text
expresses that.

What is preserved from Kimi's prompt (so the model stays on familiar ground):
  * the ``## Thought / ## Action: / ## Code:`` response skeleton, verbatim,
  * the ```` ```python ```` / ```` ```code ```` fenced block as the action carrier,
  * the ``computer.*`` pseudo-function style for control actions,
  * the password statement.

What changes:
  * the code block's content is a shell command, not python,
  * ``computer.answer`` joins ``computer.wait`` / ``computer.terminate``,
  * coordinates are REAL SCREEN PIXELS (Kimi natively prefers 0-1.0 relative coords, so
    this is stated twice and the agent does NOT run Kimi's coordinate projection).
"""
from __future__ import annotations

from mm_agents.c_gui_pixel.actions import build_action_description

__all__ = ["build_kimi_hybrid_system_prompt"]


_RESPONSE_SKELETON_THINKING = """For each step, provide your response in this format:
{thought}
## Action:
{action}
## Code:
{code}"""

_RESPONSE_SKELETON_NON_THINKING = """For each step, provide your response in this format:
## Thought
{thought}
## Action:
{action}
## Code:
{code}"""


#: The control pseudo-functions, in Kimi's own JSON-schema documentation style (the stock
#: prompt lists ``computer.wait`` and ``computer.terminate`` exactly like this).
_CONTROL_FUNCTIONS = """- {"name": "computer.wait", "description": "Wait for the screen to settle (installation, loading, running code, etc.).", "parameters": {"type": "object", "properties": {"time": {"type": "number", "description": "Seconds to wait"}}, "required": []}}
- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}, "answer": {"type": "string", "description": "The answer of the task"}}, "required": ["status"]}}
- {"name": "computer.answer", "description": "Answer a question-type task", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "The answer text"}}, "required": ["text"]}}"""


_EXAMPLES = """# Examples

## GUI action — pyautogui heredoc in the code block

## Thought
The File menu sits at the top-left of the window; I need to open it to reach "Save As".
## Action:
Click the "File" menu in the top menu bar.
## Code:
```bash
python3 <<'PY'
import pyautogui
pyautogui.click(96, 29)
PY
```

## Shell command — write shell directly

## Thought
Reading the file directly is faster and more reliable than scrolling the editor.
## Action:
Show the contents of the task file.
## Code:
```bash
cat ~/Documents/report.csv
```

## Wait

## Thought
The window is still painting, so clicking now would hit nothing.
## Action:
Wait for the application to finish loading.
## Code:
```code
computer.wait(time=3)
```

## Finish

## Thought
The saved file now shows the expected values and the last command confirmed the write.
## Action:
The task is complete.
## Code:
```code
computer.terminate(status="success")
```"""


def build_kimi_hybrid_system_prompt(
    password: str = "osworld-public-evaluation",
    thinking: bool = True,
    screen_width: int = 1920,
    screen_height: int = 1080,
) -> str:
    """Assemble the system prompt for :class:`KimiHybridAgent`."""
    skeleton = _RESPONSE_SKELETON_THINKING if thinking else _RESPONSE_SKELETON_NON_THINKING
    return "\n".join([
        "You are a GUI agent operating a computer through a bash terminal. You are given an "
        "instruction, a screenshot of the screen and your previous interactions with the "
        "computer. You need to perform a series of actions to complete the task. "
        f"The password of the computer is {password}.",
        "",
        "# Environment",
        "",
        "You face ONE machine with a graphical desktop AND a bash terminal. You act by "
        "running one shell command per step: write shell directly for CLI and file "
        "operations, or drive the GUI with pyautogui inside a quoted heredoc "
        "`python3 <<'PY' ... PY`. Each step you are shown the latest screenshot plus the "
        "previous command's output.",
        "",
        f"The screen is {screen_width}x{screen_height} and the screenshot you are shown is that "
        "exact size. Coordinates are REAL SCREEN PIXELS read straight off the image — do "
        "NOT normalize them to 0-1 or to any other grid.",
        "",
        skeleton,
        "",
        "In the code section, the code block holds EITHER a shell command (```bash) OR one of "
        "the following control functions (```code):",
        _CONTROL_FUNCTIONS,
        "",
        "# Action reference",
        "",
        build_action_description(screen_width, screen_height),
        "",
        _EXAMPLES,
    ])
