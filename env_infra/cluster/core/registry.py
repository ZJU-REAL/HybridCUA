"""WorldRegistry: discover and hold world manifests from ``world.yaml`` files.

The registry scans a worlds directory (``cluster/worlds/<id>/world.yaml``) and
loads each :class:`~cluster.worlds.base.manifest.WorldManifest`. The master uses it
to answer ``GET /v1/runtimes`` and to know which capabilities a world advertises;
the node uses it to know which worlds it can host and to build an adapter.

Loading a manifest does NOT import the world's adapter code — capability
discovery is pure data, so the master stays free of benchmark dependencies.
"""

from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional

from cluster.worlds.base.manifest import WorldManifest

try:  # PyYAML is already a project dependency
    import yaml
except Exception:  # pragma: no cover - yaml is declared in pyproject
    yaml = None  # type: ignore


class WorldRegistry:
    """Thread-safe registry of world manifests keyed by ``world_id``."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._worlds: Dict[str, WorldManifest] = {}

    def register(self, manifest: WorldManifest) -> None:
        with self._lock:
            self._worlds[manifest.world_id] = manifest

    def unregister(self, world_id: str) -> None:
        with self._lock:
            self._worlds.pop(world_id, None)

    def has(self, world_id: str) -> bool:
        with self._lock:
            return world_id in self._worlds

    def get(self, world_id: str) -> WorldManifest:
        with self._lock:
            if world_id not in self._worlds:
                raise KeyError(f"world {world_id!r} is not registered")
            return self._worlds[world_id]

    def list_all(self) -> List[WorldManifest]:
        with self._lock:
            return list(self._worlds.values())

    def ids(self) -> List[str]:
        with self._lock:
            return list(self._worlds.keys())

    def load_from_directory(self, worlds_dir: str, only: Optional[List[str]] = None) -> List[str]:
        """Scan ``worlds_dir`` for ``<id>/world.yaml`` and register each.

        ``only`` optionally restricts to a subset of world ids (e.g. the worlds a
        node is configured to host). Returns the list of registered ids.
        """
        if yaml is None:  # pragma: no cover
            raise RuntimeError("PyYAML is required to load world manifests")
        if not os.path.isdir(worlds_dir):
            return []

        loaded: List[str] = []
        for entry in sorted(os.listdir(worlds_dir)):
            world_path = os.path.join(worlds_dir, entry)
            manifest_path = os.path.join(world_path, "world.yaml")
            if not os.path.isfile(manifest_path):
                continue
            with open(manifest_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            data.setdefault("world_id", entry)
            manifest = WorldManifest.from_dict(data)
            if only is not None and manifest.world_id not in only:
                continue
            self.register(manifest)
            loaded.append(manifest.world_id)
        return loaded
