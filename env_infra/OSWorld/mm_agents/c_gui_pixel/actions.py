"""The shared action space: c-gui's four actions, real screen pixels.

ONE surface. Every GUI interaction and every shell/file operation is the same thing —
``action=bash``, one shell command. GUI means running pyautogui inside a quoted heredoc;
CLI means writing shell directly. The other three actions are control-only.

    bash      -> {"kind": "bash", "command": <shell str>, "timeout"?: <seconds>}
    wait      -> {"kind": "control", "control": "WAIT"}
    terminate -> {"kind": "control", "control": "DONE"|"FAIL"}
    answer    -> {"kind": "control", "control": "DONE", "answer": <text>}

``run_loop`` consumes exactly these dicts and never inspects the agent class, so both
kimi_hybrid and claude_hybrid drive it unchanged.

Coordinates are REAL SCREEN PIXELS. The screenshot reaches the model at full resolution
and no shim rescales anything (see ``shim.py``), so what the model measures on the image
is what pyautogui clicks. This is the one substantive difference from
``mm_agents.c_gui``, whose description text specifies a 0-999 grid — keep the two
straight when comparing runs.
"""
from __future__ import annotations

from typing import Dict, List, Optional

#: The four actions, in the order they appear in the tool schema's enum.
ACTION_ENUM: List[str] = ["bash", "wait", "terminate", "answer"]

#: Wall-clock default the executor applies when the model names no ``timeout``.
DEFAULT_BASH_TIMEOUT = 60


def build_action_description(screen_width: int = 1920, screen_height: int = 1080) -> str:
    """The ``action`` parameter's description: the full pyautogui/bash reference.

    Kept in step with ``scripts/python/cua_gym/gateway_c_gui_agent.py`` (the other
    real-pixel c-gui surface) so trajectories stay comparable. The screen size is
    stated explicitly because the model must reason in absolute pixels.
    """
    return (
        "* `bash`: run ONE shell command (requires `command`).\n"
        "  - CLI / file ops: write shell directly (ls, cat, sed, grep, python3 - <<'PY' ... PY, ...).\n"
        "  - GUI: drive the screen with pyautogui inside a QUOTED heredoc, e.g.\n"
        "        python3 <<'PY'\n"
        "        import pyautogui\n"
        "        pyautogui.click(500, 300)      # coordinates are real screen pixels\n"
        "        pyautogui.typewrite('hello', interval=0.02)\n"
        "        PY\n"
        f"    The screen is {screen_width}x{screen_height} and the screenshot you are shown is\n"
        "    that exact size, so read coordinates straight off the image — no scaling.\n"
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


def parse_timeout(value) -> Optional[float]:
    """Parse a bash action's optional ``timeout``; None when absent or unusable.

    A malformed timeout must not cost us the command, so anything unparseable or
    non-positive falls through to None (== the executor's default).
    """
    if value is None or str(value).strip() == "":
        return None
    try:
        seconds = float(str(value).strip())
    except ValueError:
        return None
    return seconds if seconds > 0 else None


def bash_action(command: str, timeout=None) -> Dict:
    """Build a ``bash`` action dict, attaching ``timeout`` only when it parses."""
    action: Dict = {"kind": "bash", "command": command}
    parsed = parse_timeout(timeout)
    if parsed is not None:
        action["timeout"] = parsed
    return action


def control_action(control: str, answer: Optional[str] = None) -> Dict:
    """Build a control action dict (``WAIT`` / ``DONE`` / ``FAIL``)."""
    action: Dict = {"kind": "control", "control": control}
    if answer is not None:
        action["answer"] = answer
    return action


def typed_action(action_name: str, params: Dict) -> Optional[Dict]:
    """Map one parsed tool call (``action`` name + params) to an action dict.

    Returns None for an unknown action or a ``bash`` with no command, so callers can
    skip it and keep the rest of the batch — a single malformed call should cost one
    action, not the whole turn.
    """
    action = str(action_name or "").strip().lower()

    if action == "bash":
        command = params.get("command")
        return bash_action(command, params.get("timeout")) if command else None
    if action == "wait":
        return control_action("WAIT")
    if action == "terminate":
        status = str(params.get("status") or "").strip().lower()
        return control_action("FAIL" if status == "failure" else "DONE")
    if action == "answer":
        return control_action("DONE", answer=params.get("text", ""))
    return None


def describe(actions: List[Dict], max_command_chars: int = 2000) -> str:
    """One-line history label for a turn: the actual commands, not just a count.

    The command text is what makes a text history useful to the model ("I already tried
    wc -l and got 48"), so it is kept rather than summarized away.
    """
    if not actions:
        return ""
    parts: List[str] = []
    for action in actions:
        if action["kind"] == "bash":
            command = action["command"]
            if len(command) > max_command_chars:
                command = command[:max_command_chars] + "...(truncated)..."
            parts.append(f"bash: {command}")
        else:
            label = action["control"]
            if action.get("answer"):
                label += f" (answer: {action['answer']})"
            parts.append(label)
    return "\n".join(parts)
