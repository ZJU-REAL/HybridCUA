"""Bash-only CLI tool -- the minimal-surface variant of ``cli_tools.py``.

Same integration contract as ``cli_tools.py`` (``build_cli_tool_def`` +
``cli_action_to_code``), so it is a drop-in replacement at the two call sites in
``evocua_hybrid`` (``prompts.py`` tool schema, ``agent.py`` action dispatch).

WHY a second module instead of trimming ``cli_tools.py``: this exists to make an
A/B interpretable, not because read/write/edit are bad tools.

  * ``cli_tools.py`` exposes bash + read/write/edit. Those three are pure sugar
    over bash (``sed -n``, ``cat > f <<EOF``, ``sed -i``) borrowed from
    agentic-coding ergonomics -- structured params beat fighting shell quoting.
  * But the RL side we are trying to validate (gui-rl's hybridcua agent, and
    ``mm_agents/c_gui``) has a SINGLE bash surface: GUI is a ``pyautogui``
    heredoc, CLI is a plain shell command, and only the command text tells them
    apart. That is exactly what ``b(tau)`` in the Stage II reward is defined
    over.
  * Running the treatment arm with 4 CLI actions moves two variables at once
    (CLI capability AND structured file tools), so a positive delta could not be
    attributed to "CLI helps" -- which is the only question the A/B is asked.

Also fixes a real bug inherited from ``cli_tools.py``: its bash wrapper prints
stdout/stderr but never inspects ``_r.returncode`` and always exits 0, so a
shell-level failure is invisible downstream (the identical bug this repo already
fixed in ``cluster/worlds/osworld/adapter.py``). Here the wrapper re-raises the
child's exit code via ``sys.exit``, so a non-zero rc actually surfaces.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

#: The only action this surface dispatches on.
CLI_ACTIONS: List[str] = ["bash"]

#: Default wall-clock timeout (seconds) when the model omits one.
DEFAULT_BASH_TIMEOUT = 60


def build_cli_tool_def() -> Dict[str, Any]:
    """OpenAI-compatible function schema for the single-action ``cli`` tool."""
    return {
        "type": "function",
        "function": {
            "name": "cli",
            "description": (
                "Run a shell command inside the machine (the same VM the GUI acts on). "
                "Requires `command`; optional `timeout` (seconds, default "
                f"{DEFAULT_BASH_TIMEOUT}). Runs through /bin/sh -c, so pipes, "
                "redirection and heredocs all work -- use `sed -n '5,20p' f` to read a "
                "slice, `cat > f <<'EOF' ... EOF` to write a file, `sed -i` to edit one. "
                "Output (stdout/stderr) is returned and also reflected in the next "
                "screenshot."
            ),
            "parameters": {
                "type": "object",
                "required": ["action", "command"],
                "properties": {
                    "action": {"type": "string", "enum": CLI_ACTIONS},
                    "command": {
                        "type": "string",
                        "description": "The shell command to run.",
                    },
                    "timeout": {
                        "type": "number",
                        "description": f"Optional timeout in seconds (default {DEFAULT_BASH_TIMEOUT}).",
                    },
                },
            },
        },
    }


def _lit(value: Any) -> str:
    """Render *value* as a python literal safe to embed in a code string."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(str(value), ensure_ascii=False)


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class CliActionError(ValueError):
    """Raised when a ``cli`` action is missing a required parameter."""


def cli_action_to_code(params: Dict[str, Any]) -> str:
    """Translate one ``cli`` tool call into a python code string for the VM.

    Raises:
        CliActionError: unknown action, or ``command`` missing.
    """
    action = str(params.get("action") or "bash").strip().lower()
    if action != "bash":
        raise CliActionError(
            f"unknown cli action: {action!r} (this surface only supports 'bash')"
        )

    command = params.get("command")
    if not command:
        raise CliActionError("cli action=bash requires 'command'")
    timeout = _int(params.get("timeout"), DEFAULT_BASH_TIMEOUT)

    # sys.exit(returncode) is the point: without it every command looks like it
    # succeeded, no matter what the shell actually returned.
    return (
        "import subprocess as _sp, sys as _sys\n"
        f"_r = _sp.run({_lit(command)}, shell=True, capture_output=True, text=True, timeout={timeout})\n"
        "print(_r.stdout, end='')\n"
        "print(_r.stderr, end='', file=_sys.stderr)\n"
        "_sys.exit(_r.returncode)"
    )
