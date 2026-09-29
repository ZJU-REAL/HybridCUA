"""Mobile surface: screenshot + UI tree + device state, device gesture actions."""

from __future__ import annotations

from cluster.schemas import ActionKind
from cluster.worlds.base.surfaces.base import SurfaceAdapter


class MobileWorld(SurfaceAdapter):
    """Base for mobile worlds (MobileWorld, AndroidWorld, ...).

    Device gestures (tap/swipe/text/key/adb) arrive as ``Action(kind="device")``;
    the concrete adapter translates them into the official mobile env's API.
    """

    surface = "mobile"
    default_modalities = ("screenshot", "ui_tree", "device_state", "instruction")
    allowed_action_kinds = (ActionKind.DEVICE.value, ActionKind.CONTROL.value)
