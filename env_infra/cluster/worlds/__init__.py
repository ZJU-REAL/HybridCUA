"""worlds: concrete benchmark plugins.

Each subdirectory is one world: a ``world.yaml`` manifest, an ``adapter.py``
subclassing a surface base class, and a ``sidecar.py`` entrypoint. Worlds may
import their official benchmark SDK; the platform core never imports worlds.
"""
