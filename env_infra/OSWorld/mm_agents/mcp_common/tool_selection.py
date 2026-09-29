"""Map an OSWorld task's `related_apps` to the MCP tool groups it should see.

WHY A MAPPING TABLE IS NEEDED
-----------------------------
`related_apps` in the upstream task JSONs is free-form and dirty. Measured over all
369 task files:

    chrome 88, os 85, libreoffice_calc 58, libreoffice_impress 55,
    libreoffice_writer 47, gimp 36, vscode 28, thunderbird 27, vlc 23,
    "libreoffice calc" 18 (note the SPACE), terminal 7, vs_code 6, pdf 5,
    image 3, calc 2, writer 2, browser 1, OS 1, Chrome 1, picard 1,
    Writer 1, ubuntu_media_player 1, libreoffice 1
    ... plus 1 task with no `related_apps` key at all.

So the same app appears as `vscode` / `vs_code`, `libreoffice_calc` /
`libreoffice calc` / `calc`, `os` / `OS` / `terminal`, `chrome` / `Chrome` /
`browser`. Naive string matching against MCP group names silently drops ~60 tasks
worth of tools. Hence: normalize, then map.

MCP GROUPS (from mcp/mcp_server/server.py meta_tools, 131 registered tools)
    code(11) code2(9) google_chrome(12) libreoffice_calc(26) libreoffice_calc2(4)
    libreoffice_impress(21) libreoffice_impress2(1) libreoffice_writer(15)
    os(20) vlc(12)

Apps with NO MCP tools at all: gimp(36 tasks), thunderbird(27), pdf, image,
picard, ubuntu_media_player. Those tasks get the `os` group only -- which matches
OSWorld-MCP's own fallback in osworld_mcp_client.py:50-66, where a task whose app
has no tools falls back to the non-app-specific groups.
"""
from __future__ import annotations

__all__ = [
    "normalize_app", "groups_for_related_apps", "tool_prefixes_for_task",
    "filter_tools", "split_inventory", "ALL_GROUPS", "DISTRACTOR_PREFIXES",
    "audit_groups",
]

NS = "osworld_mcp_"

ALL_GROUPS: tuple[str, ...] = (
    "code", "code2", "google_chrome",
    "libreoffice_calc", "libreoffice_calc2",
    "libreoffice_impress", "libreoffice_impress2",
    "libreoffice_writer", "os", "vlc",
)

DISTRACTOR_PREFIXES: tuple[str, ...] = ("filesystem_", "git_", "osworld_mcp_test.")

ALWAYS_GROUPS = ("os",)

APP_TO_GROUPS: dict[str, tuple[str, ...]] = {
    "libreoffice_calc": ("libreoffice_calc", "libreoffice_calc2"),
    "libreoffice_impress": ("libreoffice_impress", "libreoffice_impress2"),
    "libreoffice_writer": ("libreoffice_writer",),
    "libreoffice": ("libreoffice_calc", "libreoffice_calc2", "libreoffice_impress",
                    "libreoffice_impress2", "libreoffice_writer"),
    "vscode": ("code", "code2"),
    "chrome": ("google_chrome",),
    "vlc": ("vlc",),
    "os": ("os",),
    "gimp": (),
    "thunderbird": (),
    "pdf": (),
    "image": (),
    "picard": (),
}

ALIASES: dict[str, str] = {
    "vs_code": "vscode",
    "code": "vscode",
    "libreoffice calc": "libreoffice_calc",
    "calc": "libreoffice_calc",
    "libreoffice writer": "libreoffice_writer",
    "writer": "libreoffice_writer",
    "libreoffice impress": "libreoffice_impress",
    "impress": "libreoffice_impress",
    "browser": "chrome",
    "google_chrome": "chrome",
    "terminal": "os",
    "ubuntu": "os",
    "ubuntu_media_player": "vlc",
    "multi_apps": "",
}


def normalize_app(raw: str) -> str:
    """Canonicalize one raw `related_apps` entry. Returns '' if unrecognized."""
    key = str(raw).strip().lower().replace("-", "_")
    key = " ".join(key.split())
    key = ALIASES.get(key, key)
    return key if key in APP_TO_GROUPS else ""


def groups_for_related_apps(related_apps, fallback_all: bool = True) -> list[str]:
    """Map a task's `related_apps` list to the MCP groups to expose.

    `fallback_all=True` (default): when a task names no recognizable app -- missing
    key, or only tool-less apps like gimp/thunderbird -- fall back to ALWAYS_GROUPS ('os')
    rather than to the full 131, so a gimp task doesn't get LibreOffice tools it
    cannot use. Set False to return [] and let the caller decide.
    """
    groups: list[str] = []
    for raw in (related_apps or []):
        app = normalize_app(raw)
        if not app:
            continue
        for grp in APP_TO_GROUPS[app]:
            if grp not in groups:
                groups.append(grp)

    for grp in ALWAYS_GROUPS:
        if grp not in groups:
            groups.append(grp)

    if not fallback_all:
        return groups
    return [g for g in ALL_GROUPS if g in groups]


def tool_prefixes_for_task(task_config: dict) -> list[str]:
    """Convenience: read `related_apps` straight off a loaded task JSON."""
    return groups_for_related_apps(task_config.get("related_apps"))


def filter_tools(tools, groups, keep_distractors: bool = True) -> list:
    """Keep tools belonging to `groups`, plus (by default) all distractor tools.

    `tools` items may be dicts (osworld_mcp_client.list_tools output) or objects
    with a `.name`.

    `keep_distractors=True` matches the paper's setting: the 28 filesystem_*/git_*/
    calculator tools stay in the prompt regardless of the task's related_apps,
    because resisting them is precisely what TIR scores. Set False only to measure
    how much the distractors cost.
    """
    allow = tuple(f"{NS}{g}." for g in groups)
    out = []
    for t in tools:
        name = t["name"] if isinstance(t, dict) else getattr(t, "name", "")
        if name.startswith(allow):
            out.append(t)
        elif keep_distractors and name.startswith(DISTRACTOR_PREFIXES):
            out.append(t)
    return out


def audit_groups(tools) -> dict:
    """Cross-check ALL_GROUPS against a real tool inventory.

    This exists because a group name that does not match anything is INVISIBLE at
    runtime: filter_tools just returns fewer tools, nothing raises. That is how
    `os` (which the live server does not have -- it is os_ours + os_ours_2) once
    left 65 of the 361 tasks with distractor tools only.

    Returns {"declared_unused": [...], "live_unclaimed": [...], "ok": bool}:
      declared_unused  in ALL_GROUPS but matching no live tool -> stale/guessed
      live_unclaimed   app tools no declared group claims -> silently dropped

    Call it once per run with the inventory from the live server; log both lists.
    """
    names = [t["name"] if isinstance(t, dict) else getattr(t, "name", "")
             for t in tools]
    app = [n for n in names if not n.startswith(DISTRACTOR_PREFIXES)]

    declared_unused = [g for g in ALL_GROUPS
                       if not any(n.startswith(f"{NS}{g}.") for n in app)]
    allow = tuple(f"{NS}{g}." for g in ALL_GROUPS)
    live_unclaimed = sorted({n.rsplit(".", 1)[0] for n in app
                             if not n.startswith(allow)})
    return {"declared_unused": declared_unused,
            "live_unclaimed": live_unclaimed,
            "ok": not declared_unused and not live_unclaimed}


def split_inventory(tools) -> tuple[list, list]:
    """Partition a tool list into (app_tools, distractor_tools). Useful for logging
    and for asserting the 130/28 split that the benchmark expects."""
    app, dis = [], []
    for t in tools:
        name = t["name"] if isinstance(t, dict) else getattr(t, "name", "")
        (dis if name.startswith(DISTRACTOR_PREFIXES) else app).append(t)
    return app, dis
