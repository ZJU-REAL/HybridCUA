"""Episode-level MCP runtime: provision, list, call, and TIR accounting.

Everything here talks to the guest through a DesktopEnv-like ``env`` that offers
``run_code(code, lang=..., timeout=...)``. That is deliberately the same channel
c_gui's own shim provisioning uses, so this works against a local DesktopEnv and
against env_infra's remote OSWorldSessionClient without either knowing about MCP.

Why not reuse OSWorld-MCP's own desktop_env.py hooks (get_mcp_tool_list /
call_mcp_tool): those live on an overlay copy of DesktopEnv, require
action_space='mcp', and re-fetch the tool list on EVERY observation by probing the
focused window. We want the list fixed per episode (see list_tools_for_task) and
we must not require a patched env class.
"""
from __future__ import annotations

import ast
import json
import logging
import os
import re
import shlex
from typing import Dict, Iterable, List, Optional, Sequence

from mm_agents.mcp_common.tool_selection import (DISTRACTOR_PREFIXES, filter_tools,
                            groups_for_related_apps, split_inventory)

logger = logging.getLogger("desktopenv.mcp_runtime")

__all__ = [
    "provision_mcp", "list_tools_for_task", "call_mcp_tool",
    "count_tool_invocations", "extract_tool_names", "MCP_TOOL_CALL_RE",
]

GUEST_HOME = "/home/user"
NODE_BIN = f"{GUEST_HOME}/.nvm/versions/node/v22.18.0/bin"
UV_BIN = f"{GUEST_HOME}/.local/bin"
GUEST_PROXY = "http://star-proxy.oa.com:3128"
ENV_PREFIX = (
    f'export PATH="{NODE_BIN}:{UV_BIN}:$PATH" '
    f'http_proxy={GUEST_PROXY} https_proxy={GUEST_PROXY} '
    f'no_proxy=localhost,127.0.0.1,::1; '
)


def provision_mcp(env, logger_=None, timeout: int = 600) -> bool:
    """Ensure the MCP stack is up in the guest. Idempotent; call AFTER env.reset().

    Mirrors c_gui's provision_shim contract: never raises, returns success, and is
    safe to re-run every episode (reset may roll the snapshot back and wipe the
    guest-side install, and the container pool recycles VMs underneath us).

    Delegates to setup_mcp_guest's helpers by driving them over env.run_code, so
    there is exactly one description of how the stack comes up.
    """
    log = logger_ or logger
    try:
        from mm_agents.mcp_common.guest_ops import bring_up_via_run_code
        ok = bring_up_via_run_code(env, log=log, timeout=timeout)
        log.info("mcp provision: %s", "ok" if ok else "FAILED")
        return ok
    except Exception as exc:
        log.warning("mcp provision failed: %s", exc)
        return False


def list_tools_for_task(env, task_config: Dict, keep_distractors: bool = True,
                        budget: int = 0, logger_=None) -> List[Dict]:
    """The tool list for one episode, filtered by the task's ``related_apps``.

    Fetched ONCE per episode and held constant for its duration. OSWorld-MCP's own
    client re-derives the list from whatever window happens to be focused, which
    makes the system prompt change every step and defeats prefix caching; a task's
    related_apps is known up front and does not move.

    `budget > 0` truncates the app tools (distractors are kept first, since
    dropping them would inflate TIR relative to the paper).
    """
    log = logger_ or logger
    all_tools = list_all_tools(env)
    if not all_tools:
        log.warning("mcp: server returned no tools")
        return []

    groups = groups_for_related_apps(task_config.get("related_apps"))
    kept = filter_tools(all_tools, groups, keep_distractors=keep_distractors)
    app, dis = split_inventory(kept)

    if budget and len(kept) > budget:
        room = max(budget - len(dis), 0)
        kept = dis + app[:room] if keep_distractors else app[:budget]
        app, dis = split_inventory(kept)

    log.info("mcp tools: %d of %d (groups=%s, app=%d, distractor=%d)",
             len(kept), len(all_tools), ",".join(groups), len(app), len(dis))
    return kept


def list_all_tools(env) -> List[Dict]:
    """Full inventory from the guest, unfiltered (rag=False)."""
    code = (
        "import json\n"
        "from osworld_mcp_client import OsworldMcpClient as C\n"
        "t = C.list_tools(tool_name=None, shuffle=False, rag=False)\n"
        "print('<<<MCPTOOLS>>>' + json.dumps(t))\n"
    )
    out = run_guest_python(env, code, timeout=180)
    return parse_marked_json(out, "<<<MCPTOOLS>>>") or []


def call_mcp_tool(env, name: str, params: Optional[Dict] = None,
                  timeout: int = 120) -> Dict:
    """Invoke one MCP tool in the guest.

    Returns {"ok": bool, "data": Any, "text": str} where `text` is what gets fed
    back to the model as CLI output.
    """
    params = params or {}
    code = (
        "import json\n"
        "from osworld_mcp_client import OsworldMcpClient as C\n"
        f"r = C.call_tool({name!r}, {params!r})\n"
        "payload = {'ok': not bool(getattr(r, 'is_error', False)),\n"
        "           'data': getattr(r, 'data', None)}\n"
        "try:\n"
        "    print('<<<MCPCALL>>>' + json.dumps(payload, default=str))\n"
        "except Exception:\n"
        "    print('<<<MCPCALL>>>' + json.dumps({'ok': payload['ok'],\n"
        "                                        'data': str(payload['data'])}))\n"
    )
    out = run_guest_python(env, code, timeout=timeout)
    parsed = parse_marked_json(out, "<<<MCPCALL>>>")

    if parsed is None:
        tail = (out or "").strip().splitlines()
        detail = tail[-1] if tail else "no output"
        return {"ok": False, "data": None,
                "text": f"MCP call {name} failed: {detail[:400]}"}

    ok = bool(parsed.get("ok"))
    data = parsed.get("data")
    body = data if isinstance(data, str) else json.dumps(data, default=str)
    return {"ok": ok, "data": data,
            "text": f"{'MCP result' if ok else 'MCP error'} [{name}]: {body}"}


def run_guest_python(env, code: str, timeout: int = 120) -> str:
    """Run python in the guest with cwd=/home/user so osworld_mcp_client imports.

    Goes through bash (not lang='python') because we need the PATH/proxy exports
    and the cd; the heredoc is quoted so nothing in `code` is expanded by the shell.
    """
    cmd = (f"{ENV_PREFIX}cd {GUEST_HOME} && python3 - <<'MCP_PY_EOF'\n"
           f"{code}\nMCP_PY_EOF")
    try:
        res = env.run_code(cmd, lang="bash", timeout=timeout) or {}
    except TypeError:
        res = env.run_code(cmd, lang="bash") or {}
    except Exception as exc:
        logger.warning("mcp guest python failed: %s", exc)
        return ""
    return ((res.get("output") or "") + (res.get("error") or ""))


def parse_marked_json(out: str, marker: str):
    """Pull our marked JSON line out of guest stdout.

    A marker is necessary because the guest prints unrelated noise around it:
    fastmcp emits a "Client failed to connect:" banner for each distractor server
    it cannot reach, and osworld_mcp_client.py itself print()s the raw result.
    """
    if not out or marker not in out:
        return None
    tail = out.rsplit(marker, 1)[1]
    line = tail.splitlines()[0] if tail.splitlines() else ""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(line)
        except (ValueError, SyntaxError):
            logger.warning("mcp: could not parse marked payload: %s", line[:200])
            return None


MCP_TOOL_CALL_RE = re.compile(r"""call_tool\(\s*\\?['"]([\w.\-]+)\\?['"]""")


def extract_tool_names(actions: Sequence[Dict]) -> List[str]:
    """Tool names invoked by one predict()'s actions, across both modes.

    Single definition on purpose: `action` mode can be counted exactly while
    `bash` mode has to be inferred from the command text, and if those two lived
    apart the A/B TIR numbers would not be comparable.
    """
    names: List[str] = []
    for a in actions or []:
        kind = a.get("kind")
        if kind == "mcp":
            if a.get("name"):
                names.append(str(a["name"]))
        elif kind == "bash":
            names.extend(MCP_TOOL_CALL_RE.findall(a.get("command") or ""))
    return names


def count_tool_invocations(actions: Sequence[Dict]) -> int:
    return len(extract_tool_names(actions))


def summarize_episode(step_records: Iterable[Dict]) -> Dict:
    """Aggregate per-episode tool usage for the result dir.

    `step_records` is an iterable of {"actions": [...]} in step order. Reports the
    raw counts only -- TIR as the paper defines it (was using a tool the *right*
    call?) needs has_tool.json ground truth, which the caller joins in.

    `used_distractor` is tracked separately because reaching for a filesystem_*/
    git_*/calculator tool is never the right call on a desktop task: those 26 exist
    purely as bait, so the rate at which a model takes it is the cleanest available
    read on tool-selection discipline.
    """
    calls, steps_with_calls, names = 0, 0, []
    total_steps = 0
    for rec in step_records or []:
        total_steps += 1
        got = extract_tool_names(rec.get("actions") or [])
        if got:
            steps_with_calls += 1
            calls += len(got)
            names.extend(got)
    distractors = sorted({n for n in names if n.startswith(DISTRACTOR_PREFIXES)})
    return {
        "steps": total_steps,
        "tool_calls": calls,
        "steps_with_tool_calls": steps_with_calls,
        "distinct_tools": sorted(set(names)),
        "used_tool": calls > 0,
        "distractor_calls": sum(1 for n in names if n.startswith(DISTRACTOR_PREFIXES)),
        "used_distractor": distractors,
    }
