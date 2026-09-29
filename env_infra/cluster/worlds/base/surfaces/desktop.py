"""Desktop surface: screenshot + accessibility tree + terminal, GUI actions."""

from __future__ import annotations

from cluster.schemas import ActionKind
from cluster.worlds.base.surfaces.base import SurfaceAdapter


class DesktopWorld(SurfaceAdapter):
    """Base for desktop GUI worlds (OSWorld, WindowsAgentArena, ...).

    Concrete adapters subclass this and implement reset/step/observe/evaluate/
    health_check/close by translating to the official benchmark API.
    """

    surface = "desktop"
    default_modalities = ("screenshot", "accessibility_tree", "terminal", "instruction")
    # TOOL covers code/CLI execution (kind="tool", type="cli"), run via the
    # adapter's controller rather than the GUI — see OSWorldWorldAdapter._run_code.
    allowed_action_kinds = (ActionKind.GUI.value, ActionKind.CONTROL.value, ActionKind.TOOL.value)
