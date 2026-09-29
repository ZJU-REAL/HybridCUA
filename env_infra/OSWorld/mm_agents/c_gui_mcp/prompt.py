"""System-prompt fragments and tool-schema patching for MCP-enabled c-gui runs.

Three exposure modes, selected by ``MCP_MODE``:

  off     no MCP at all -- the agent's prompt and tool schema are untouched.
          This is the A/B baseline and must reproduce plain run_c_gui.sh.

  bash    tools are described in prose; the model calls them through the
          EXISTING ``action=bash`` channel with a python3 one-liner. Zero
          changes to the action space, so nothing about the agent's
          tool-call contract shifts -- only the prompt grows.

  action  the single tool's ``action`` enum gains an ``mcp`` member plus
          ``name``/``params`` properties. The model fills structured fields
          instead of hand-quoting a shell command, which is markedly easier
          to emit correctly, and tool invocations become exactly countable
          for TIR rather than regex-inferred.

Both live modes render the SAME catalog text so the only difference between
them is call syntax -- otherwise an A/B comparison would confound "how tools
are described" with "how tools are invoked".

Why prose and not raw JSON Schema: dumping all ~63 tool schemas a task sees
costs ~28k chars, while the compact rendering below costs roughly 40% of that
and reads better. The model still gets every parameter name and its required
flag; what it loses is JSON boilerplate it does not need.
"""
from __future__ import annotations

import copy
import json
import logging
from typing import Dict, Iterable, List

__all__ = [
    "MCP_MODES", "render_tool_catalog", "build_mcp_section", "build_example",
    "patch_tool_def", "MCP_CALL_SNIPPET",
]

MCP_MODES = ("off", "bash", "action")

MCP_CALL_SNIPPET = (
    "python3 -c \"from osworld_mcp_client import OsworldMcpClient as C; "
    "print(C.call_tool('{name}', {params}))\""
)


def params_of(tool: Dict) -> tuple[list[str], list[str]]:
    """Return (required, optional) parameter names for one tool dict."""
    schema = tool.get("parameters") or {}
    props = list((schema.get("properties") or {}).keys())
    required = [p for p in (schema.get("required") or []) if p in props]
    optional = [p for p in props if p not in required]
    return required, optional


def one_line(text: str, limit: int = 110) -> str:
    """Collapse a tool description to a single short line."""
    line = " ".join(str(text or "").split())
    if len(line) <= limit:
        return line
    cut = line[:limit].rsplit(" ", 1)[0]
    return cut + "…"


def group_of(name: str) -> str:
    """Presentational group heading for one tool name.

    App tools are `osworld_mcp_<app>.<method>` -> group on the app. The distractor
    servers name their tools `filesystem_read_file` / `git_git_status` with NO dot,
    so splitting on '.' would give each one its own heading and produce 27 one-line
    sections; group those by their server prefix instead.
    """
    if name.startswith("filesystem_"):
        return "filesystem"
    if name.startswith("git_"):
        return "git"
    if "." in name:
        return name.split(".", 1)[0]
    return name


def render_tool_catalog(tools: Iterable[Dict], desc_limit: int = 110) -> str:
    """Group tools by namespace and render a compact signature catalog.

    Every tool keeps its FULL name in the signature line, because that full name is
    what must be passed to call_tool -- the headings are navigation only.
    """
    groups: Dict[str, List[Dict]] = {}
    for t in tools:
        name = t["name"] if isinstance(t, dict) else getattr(t, "name", "")
        groups.setdefault(group_of(name), []).append(t)

    out: List[str] = []
    for head in sorted(groups):
        out.append(f"### {head}")
        for t in sorted(groups[head], key=lambda x: x["name"]):
            req, opt = params_of(t)
            sig = ", ".join(req + [f"{p}?" for p in opt])
            desc = one_line(t.get("description"), desc_limit)
            out.append(f"- {t['name']}({sig})" + (f" — {desc}" if desc else ""))
        out.append("")
    return "\n".join(out).rstrip()


ACTION_USAGE = """\
To call one, use action=mcp with the tool's full name and a JSON object of
arguments:

<tool_call>
<function={tool_name}>
<parameter=action>
mcp
</parameter>
<parameter=name>
osworld_mcp_libreoffice_calc.set_cell_value
</parameter>
<parameter=params>
{{"cell": "A1", "value": "42"}}
</parameter>
</function>
</tool_call>

- `params` must be a JSON object; use {{}} when the tool takes no arguments.
- The tool's JSON result comes back to you as CLI output.
- One call per action. Check the result before chaining another.\
"""

PREAMBLE = """\
# MCP tools

This machine also exposes structured tools over MCP. One call replaces several
GUI steps and returns a precise result instead of a screenshot to read.

Use one when it cleanly matches your intent; otherwise stay with the GUI. Some
listed tools are irrelevant to your task -- calling those wastes a step and may
change state you did not intend to touch.\
"""


BASH_EXAMPLE = """\
## Call an MCP tool

Action: Set cell A1 to 42 with the calc tool instead of clicking through the UI.

<tool_call>
<function={tool_name}>
<parameter=action>
bash
</parameter>
<parameter=command>
{snippet}
</parameter>
</function>
</tool_call>
"""

ACTION_EXAMPLE = """\
## Call an MCP tool

Action: Set cell A1 to 42 with the calc tool instead of clicking through the UI.

<tool_call>
<function={tool_name}>
<parameter=action>
mcp
</parameter>
<parameter=name>
osworld_mcp_libreoffice_calc.set_cell_value
</parameter>
<parameter=params>
{{"cell": "A1", "value": "42"}}
</parameter>
</function>
</tool_call>
"""


def build_example(mode: str, tool_name: str = "computer_use") -> str:
    """The worked MCP example for `mode`, or '' when there is nothing to add."""
    if mode not in MCP_MODES:
        raise ValueError(f"mode must be one of {MCP_MODES}, got {mode!r}")
    if mode == "off":
        return ""
    if mode == "bash":
        return BASH_EXAMPLE.format(
            tool_name=tool_name,
            snippet=MCP_CALL_SNIPPET.format(
                name="osworld_mcp_libreoffice_calc.set_cell_value",
                params="{'cell': 'A1', 'value': '42'}"))
    return ACTION_EXAMPLE.format(tool_name=tool_name)


def build_mcp_section(tools: Iterable[Dict], mode: str, tool_name: str = "computer_use",
                      desc_limit: int = 110) -> str:
    """Build the '# MCP tools' block to splice into the system prompt.

    Returns '' for off, for bash, and when there are no tools.

    bash returns '' because everything it would say now lives in the `bash`
    action's schema description (see patch_tool_def). Emitting both would render
    the catalog twice (+8188 chars on calc), making bash 24396 vs action 15152 --
    a gap that would confound the bash-vs-action TIR comparison.
    """
    if mode not in MCP_MODES:
        raise ValueError(f"mode must be one of {MCP_MODES}, got {mode!r}")
    tools = list(tools or [])
    if mode in ("off", "bash") or not tools:
        return ""

    catalog = render_tool_catalog(tools, desc_limit=desc_limit)
    usage = ACTION_USAGE.format(tool_name=tool_name)
    return f"{PREAMBLE}\n\n{usage}\n\n## Available tools ({len(tools)})\n\n{catalog}"


MCP_ACTION_BULLET = (
    "\n* `mcp`: call a structured MCP tool (requires `name`, and `params` unless the "
    "tool takes no arguments). See the MCP tools section for the catalog."
)

BASH_GUI_ANCHOR = "\n  - GUI: drive the screen with pyautogui"

BASH_MCP_SUBUSE = (
    "\n  - MCP tools: structured tools, one call replacing several GUI steps and"
    " returning a\n"
    "    precise result instead of a screenshot to read. Use one when it cleanly"
    " matches your\n"
    "    intent; otherwise stay with the GUI. Some listed tools are irrelevant to"
    " your task --\n"
    "    calling those wastes a step and may change state you did not intend to"
    " touch.\n"
    "    Call one with a single python3 command:\n"
    "        {snippet}\n"
    "    `params` is a python dict literal ({{}} if the tool takes none). Mind the"
    " quoting:\n"
    "    the command is inside double quotes, so use single quotes in the dict."
    " One call\n"
    "    per bash action; check the result before chaining another. Available tools"
    " ({n}):"
)


def indent_catalog(catalog: str, pad: str = "      ") -> str:
    """Indent the rendered catalog to sit under the MCP sub-use bullet."""
    return "\n".join(pad + ln if ln.strip() else ln for ln in catalog.split("\n"))


def patch_bash_description(action: Dict, catalog: str, n_tools: int) -> bool:
    """Insert the MCP sub-use (command form + catalog) into the stock bash
    description, in place. Returns True if inserted.

    Anchors on the GUI line, not append: appending would land after the
    wait/terminate/answer bullets, outside the action=bash block. If c_gui's
    wording changes the anchor misses and this returns False -- the caller then
    leaves the def alone and warns, rather than guessing a position.
    """
    desc = action.get("description") or ""
    if "- MCP tools:" in desc:
        return True
    if BASH_GUI_ANCHOR not in desc:
        return False
    head = BASH_MCP_SUBUSE.format(
        snippet=MCP_CALL_SNIPPET.format(
            name="osworld_mcp_libreoffice_calc.set_cell_value",
            params="{'cell': 'A1', 'value': '42'}"),
        n=n_tools)
    block = head + "\n" + indent_catalog(catalog)
    action["description"] = desc.replace(
        BASH_GUI_ANCHOR, block + BASH_GUI_ANCHOR, 1)
    return True


def patch_tool_def(tool_def: Dict, mode: str, tools: Iterable[Dict] = (),
                   desc_limit: int = 110) -> Dict:
    """Return a copy of c_gui's single tool definition adjusted for `mode`.

    off     unchanged (byte-identical to stock -- this is the A/B baseline).

    bash    enum and properties untouched (action space really is unchanged).
            Only the `bash` action's description gains a third sub-use next to the
            stock "CLI / file ops" and "GUI" lines, carrying the command form and
            the full catalog -- the model reads it where it decides what to emit.

    action  adds 'mcp' to the action enum and declares `name`/`params`.

    `tools` is required for bash mode and ignored otherwise.
    """
    if mode not in MCP_MODES:
        raise ValueError(f"mode must be one of {MCP_MODES}, got {mode!r}")
    if mode == "off":
        return tool_def

    if mode == "bash":
        tools = list(tools or [])
        if not tools:
            return tool_def
        patched = copy.deepcopy(tool_def)
        props = (patched.setdefault("function", {})
                        .setdefault("parameters", {})
                        .setdefault("properties", {}))
        action = props.get("action")
        catalog = render_tool_catalog(tools, desc_limit=desc_limit)
        if not isinstance(action, dict) or not patch_bash_description(
                action, catalog, len(tools)):
            logging.getLogger("desktopenv.experiment").warning(
                "mcp bash mode: could not anchor the MCP sub-use into c_gui's "
                "bash description; schema-level catalog omitted. Stock wording "
                "may have changed -- check BASH_GUI_ANCHOR in mcp_prompt.py.")
            return tool_def
        return patched

    patched = copy.deepcopy(tool_def)
    params = patched.setdefault("function", {}).setdefault("parameters", {})
    props = params.setdefault("properties", {})

    action = props.setdefault("action", {"type": "string"})
    enum = list(action.get("enum") or [])
    if "mcp" not in enum:
        action["enum"] = enum + ["mcp"]
    action["description"] = (action.get("description", "") + MCP_ACTION_BULLET)

    props["name"] = {
        "type": "string",
        "description": ("Required by action=mcp. Full tool name from the MCP tools "
                        "catalog, e.g. osworld_mcp_libreoffice_calc.set_cell_value."),
    }
    props["params"] = {
        "type": "object",
        "description": ("Arguments for action=mcp as a JSON object; omit or use {} "
                        "when the tool takes no arguments."),
    }
    return patched


def patch_tools_def(tools_def: List[Dict], mode: str,
                    tools: Iterable[Dict] = ()) -> List[Dict]:
    """List wrapper, matching c_gui's build_c_gui_tools_def shape."""
    tools = list(tools or [])
    return [patch_tool_def(t, mode, tools) for t in tools_def]


def section_stats(tools: Iterable[Dict], mode: str) -> Dict[str, int]:
    """Cheap size accounting, for logging how much prompt a mode costs."""
    text = build_mcp_section(tools, mode)
    tools = list(tools or [])
    return {
        "tools": len(tools),
        "chars": len(text),
        "approx_tokens": len(text) // 4,
        "raw_schema_chars": sum(len(json.dumps(t)) for t in tools),
    }
