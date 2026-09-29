"""Web surface: screenshot + DOM + network state, GUI + tool actions."""

from __future__ import annotations

from cluster.schemas import ActionKind
from cluster.worlds.base.surfaces.base import SurfaceAdapter


class WebWorld(SurfaceAdapter):
    """Base for browser worlds (BrowserGym, WebArena, ...).

    Browser interactions arrive as ``gui`` (click/type/nav) or ``tool``
    (api_call) actions; the concrete adapter maps them to the official web env.
    """

    surface = "web"
    default_modalities = ("screenshot", "dom", "network_state", "instruction")
    allowed_action_kinds = (ActionKind.GUI.value, ActionKind.TOOL.value, ActionKind.CONTROL.value)
