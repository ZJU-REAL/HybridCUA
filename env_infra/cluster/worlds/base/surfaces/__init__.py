"""Surface base classes: one per environment surface (desktop/mobile/web/others).

Each surface base sits between :class:`~cluster.worlds.base.adapter.WorldAdapter`
and a concrete world adapter. It fixes that surface's observation modality set
and action-kind vocabulary, and offers shared encoding helpers, so a concrete
world adapter only has to translate calls to/from the official benchmark API.

Surfaces are platform-level and benchmark-neutral; concrete adapters live in
``cluster/worlds/<name>/adapter.py`` and subclass the matching surface here.
"""

from __future__ import annotations

from cluster.worlds.base.surfaces.base import SurfaceAdapter
from cluster.worlds.base.surfaces.desktop import DesktopWorld
from cluster.worlds.base.surfaces.mobile import MobileWorld
from cluster.worlds.base.surfaces.others import OthersWorld
from cluster.worlds.base.surfaces.web import WebWorld

#: surface name -> base class, used for validation / docs
SURFACE_REGISTRY = {
    "desktop": DesktopWorld,
    "mobile": MobileWorld,
    "web": WebWorld,
    "others": OthersWorld,
}

__all__ = [
    "SurfaceAdapter",
    "DesktopWorld",
    "MobileWorld",
    "WebWorld",
    "OthersWorld",
    "SURFACE_REGISTRY",
]
