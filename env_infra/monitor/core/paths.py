"""Path helpers shared by the monitor apps.

Both apps serve files out of user-supplied directories, so every filesystem
read is gated on a containment check — a caller must never be able to walk
out of the experiment directory it named.
"""

import os


def safe_path(path):
    """Return `path` as an absolute path if it is an existing directory, else None."""
    if not path:
        return None
    path = os.path.abspath(path)
    return path if os.path.isdir(path) else None


def is_contained(base, target, strict=True):
    """True when `target` resolves inside `base`.

    Symlinks are resolved first, so a link pointing outside `base` is rejected
    rather than followed. `strict` requires a proper descendant; pass False to
    also accept `target == base`.
    """
    try:
        base_real = os.path.realpath(base)
        target_real = os.path.realpath(target)
    except OSError:
        return False
    if not base_real or not target_real:
        return False
    if target_real == base_real:
        return not strict
    return target_real.startswith(base_real + os.sep)
