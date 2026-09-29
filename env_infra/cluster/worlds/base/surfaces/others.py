"""Others surface: catch-all for remote/hosted/API-only environments.

Covers worlds that don't fit desktop/mobile/web — hosted online benchmarks,
API-only services, and the synthetic Fake world used in conformance tests.
Accepts GUI, tool and control actions, and imposes no fixed modality set.
"""

from __future__ import annotations

from cluster.schemas import ActionKind
from cluster.worlds.base.surfaces.base import SurfaceAdapter


class OthersWorld(SurfaceAdapter):
    surface = "others"
    default_modalities = ("screenshot",)
    allowed_action_kinds = (
        ActionKind.GUI.value,
        ActionKind.TOOL.value,
        ActionKind.DEVICE.value,
        ActionKind.CONTROL.value,
    )
