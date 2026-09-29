"""CLI tool for the hybrid GUI+CLI agent.

Exposes a SINGLE ``cli`` function-tool (mirroring the ``computer_use`` single-tool +
``action`` dispatch pattern used by ``mm_agents/qwen``) and translates each CLI action
into a **python code string** that OSWorld's ``DesktopEnv.execute_python_command`` runs
inside the VM. GUI (pyautogui strings) and CLI (python strings) therefore share the one
``actions`` list consumed by ``lib_run_single.py`` — no envelope, no cluster client.

Actions dispatched by the ``action`` field:
    - bash  {command, timeout?}          -> subprocess.run in the VM
    - read  {path, offset?, limit?}      -> print file (optionally a line slice)
    - write {path, content}              -> overwrite file
    - edit  {path, old, new, replace_all?}-> read, str.replace, write back

Every user-supplied string is injected via ``json.dumps`` (== a valid python literal for
str/int), so quotes, newlines and backslashes can never break the generated code string.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

#: Actions the ``cli`` tool dispatches on.
CLI_ACTIONS: List[str] = ["bash", "read", "write", "edit"]

#: Default wall-clock timeout (seconds) for ``bash`` when the model omits one.
DEFAULT_BASH_TIMEOUT = 60


def build_cli_tool_def() -> Dict[str, Any]:
    """OpenAI-compatible function schema for the single ``cli`` tool.

    Flat params with per-action conditional requirement (same style as qwen's
    ``computer_use``: ``coordinate``/``text``/``keys`` are flat and the description
    says which action needs which field).
    """
    return {
        "type": "function",
        "function": {
            "name": "cli",
            "description": (
                "Run a command-line / file operation inside the machine (the same VM the "
                "GUI acts on). Pick one `action`:\n"
                "* `bash`: run a shell command. Requires `command`; optional `timeout` (seconds).\n"
                "* `read`: print a file's contents. Requires `path`; optional `offset`/`limit` "
                "(1-based line range).\n"
                "* `write`: overwrite a file. Requires `path` and `content`.\n"
                "* `edit`: replace text in a file. Requires `path`, `old`, `new`; optional "
                "`replace_all` (default false = first match only).\n"
                "Output (stdout/stderr) is returned and also reflected in the next screenshot."
            ),
            "parameters": {
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {"type": "string", "enum": CLI_ACTIONS},
                    "command": {"type": "string", "description": "Required by `action=bash`: the shell command."},
                    "path": {"type": "string", "description": "Required by `action=read`/`write`/`edit`: file path."},
                    "content": {"type": "string", "description": "Required by `action=write`: full file contents."},
                    "old": {"type": "string", "description": "Required by `action=edit`: text to replace."},
                    "new": {"type": "string", "description": "Required by `action=edit`: replacement text."},
                    "offset": {"type": "number", "description": "Optional for `action=read`: 1-based start line."},
                    "limit": {"type": "number", "description": "Optional for `action=read`: number of lines."},
                    "replace_all": {"type": "boolean", "description": "Optional for `action=edit`: replace every match."},
                    "timeout": {"type": "number", "description": "Optional for `action=bash`: timeout in seconds."},
                },
            },
        },
    }


def _lit(value: Any) -> str:
    """Render *value* as a python literal safe to embed in a code string.

    ``json.dumps`` produces a valid python literal for str/int/float/bool/None
    (JSON ``true``/``false``/``null`` differ, so booleans/None are handled first).
    """
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

    The returned string is meant to be appended to the ``actions`` list and executed
    by ``DesktopEnv.execute_python_command`` (action_space="pyautogui" treats any
    non-control string as python).

    Raises:
        CliActionError: unknown action or a missing required parameter.
    """
    action = str(params.get("action") or "").strip().lower()

    if action == "bash":
        command = params.get("command")
        if not command:
            raise CliActionError("cli action=bash requires 'command'")
        timeout = _int(params.get("timeout"), DEFAULT_BASH_TIMEOUT)
        # Print stdout then stderr; never raise so the episode continues.
        return (
            "import subprocess as _sp\n"
            f"_r = _sp.run({_lit(command)}, shell=True, capture_output=True, text=True, timeout={timeout})\n"
            "print(_r.stdout, end='')\n"
            "import sys as _sys\n"
            "print(_r.stderr, end='', file=_sys.stderr)"
        )

    if action == "read":
        path = params.get("path")
        if not path:
            raise CliActionError("cli action=read requires 'path'")
        has_slice = params.get("offset") is not None or params.get("limit") is not None
        if not has_slice:
            return f"print(open({_lit(path)}, encoding='utf-8', errors='replace').read(), end='')"
        offset = max(1, _int(params.get("offset"), 1))
        # limit omitted -> read to EOF (very large slice end).
        limit = _int(params.get("limit"), 10 ** 9)
        start = offset - 1  # 1-based -> 0-based
        end = start + max(0, limit)
        return (
            f"_lines = open({_lit(path)}, encoding='utf-8', errors='replace').read().splitlines(keepends=True)\n"
            f"print(''.join(_lines[{start}:{end}]), end='')"
        )

    if action == "write":
        path = params.get("path")
        if not path:
            raise CliActionError("cli action=write requires 'path'")
        if "content" not in params:
            raise CliActionError("cli action=write requires 'content'")
        content = params.get("content") or ""
        return (
            f"open({_lit(path)}, 'w', encoding='utf-8').write({_lit(content)})\n"
            f"print('WROTE ' + str(len({_lit(content)})) + ' chars to ' + {_lit(path)})"
        )

    if action == "edit":
        path = params.get("path")
        if not path:
            raise CliActionError("cli action=edit requires 'path'")
        if "old" not in params:
            raise CliActionError("cli action=edit requires 'old'")
        if "new" not in params:
            raise CliActionError("cli action=edit requires 'new'")
        old = params.get("old") or ""
        new = params.get("new") or ""
        count = "-1" if params.get("replace_all") else "1"
        return (
            f"_p = {_lit(path)}\n"
            "_s = open(_p, encoding='utf-8', errors='replace').read()\n"
            f"_old = {_lit(old)}\n"
            f"_n = _s.count(_old)\n"
            "if _n == 0:\n"
            "    import sys as _sys; print('EDIT FAILED: old text not found in ' + _p, file=_sys.stderr)\n"
            "else:\n"
            f"    open(_p, 'w', encoding='utf-8').write(_s.replace(_old, {_lit(new)}, {count}))\n"
            f"    print('EDITED ' + _p + ' (' + str({count} if {count} != -1 else _n) + ' replacement(s))')"
        )

    raise CliActionError(f"unknown cli action: {action!r} (expected one of {CLI_ACTIONS})")
