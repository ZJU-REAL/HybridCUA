"""VM-side pyautogui coordinate shim + bash payload wrapper (execution side).

Mirror of the eval-side ``mm_agents/c_gui/shim.py`` + ``mm_agents/c_gui/executor.py``
so the TRAINING rollout drives the OSWorld VM byte-identically to c_gui eval:

  * ``SHIM_SOURCE`` / ``build_provision_command()`` — the ``usercustomize.py`` written
    into the VM once per episode. It lazily patches pyautogui so 0-999 normalized coords
    map to real pixels, GATED on env ``CUA_COORD_SCALE`` (OSWorld's own /execute channel
    never sets it, so existing benchmarks are unaffected).
  * ``bash_payload(command)`` — prepend ``export CUA_COORD_SCALE=999`` so the model's
    ``python3 <<'PY' ... pyautogui ... PY`` heredoc gets scaled.

Kept on the CLIENT side (imported by clients/osworld_remote_async.py and the rollout),
NOT by the agent — matching c_gui's split where run_loop/executor own execution and the
agent only emits action dicts. Nothing here imports the agent package.
"""
from __future__ import annotations

#: Environment variable that gates coordinate scaling inside the shim. The env client
#: sets it (via ``bash_payload``, e.g. "999") only for bash commands.
COORD_SCALE_ENV = "CUA_COORD_SCALE"

#: Default normalization base (coords are 0..999 => screen treated as 1000x1000).
DEFAULT_COORD_SCALE = "999"

#: usercustomize.py source, written verbatim into the VM user-site. Raw string: keep it
#: self-contained and free of triple single-quotes. Byte-identical to c_gui/shim.py.
SHIM_SOURCE = r'''# Auto-generated coordinate shim — patches pyautogui for the c-gui bash surface.
import os, sys, importlib.abc, importlib.util

_SCALE_ENV = "CUA_COORD_SCALE"
_POINT_FNS = ("click", "doubleClick", "tripleClick", "rightClick", "middleClick",
              "moveTo", "dragTo", "mouseDown", "mouseUp", "moveRel", "dragRel")


def _patch(pyautogui):
    try:
        pyautogui.FAILSAFE = False
    except Exception:
        pass
    # OSWorld Linux shift-char fix (mirror of desktop_env pkgs_prefix).
    try:
        import platform
        _sh = "~!@#$%^&*()_+" + chr(123) + chr(125) + "|:" + chr(34) + "<>?"
        _shl = "~!@#$%^&*()_+" + chr(123) + chr(125) + "|:" + chr(34) + ">?"
        pyautogui.isShiftCharacter = (lambda c: c.isupper() or
            c in (_shl if platform.system() == "Linux" else _sh))
    except Exception:
        pass
    # Coordinate scaling — only when the gating env is set (keeps OSWorld setup intact).
    scale = os.environ.get(_SCALE_ENV)
    if not scale:
        return
    try:
        base = float(scale)
        _W, _H = pyautogui.size()
        _sxf, _syf = _W / base, _H / base
    except Exception:
        return

    def _sx(v):
        return v if v is None else int(round(v * _sxf))

    def _sy(v):
        return v if v is None else int(round(v * _syf))

    def _wrap_point(fn):
        def g(*a, **k):
            a = list(a)
            if len(a) >= 1:
                a[0] = _sx(a[0])
            if len(a) >= 2:
                a[1] = _sy(a[1])
            if "x" in k:
                k["x"] = _sx(k["x"])
            if "y" in k:
                k["y"] = _sy(k["y"])
            return fn(*a, **k)
        return g

    for _name in _POINT_FNS:
        _fn = getattr(pyautogui, _name, None)
        if _fn is not None:
            setattr(pyautogui, _name, _wrap_point(_fn))

    def _wrap_scroll(fn):
        # scroll(clicks, x=None, y=None) / hscroll — only x,y are coordinates.
        def g(clicks, x=None, y=None, *a, **k):
            if "x" in k:
                k["x"] = _sx(k["x"])
            if "y" in k:
                k["y"] = _sy(k["y"])
            return fn(clicks, _sx(x), _sy(y), *a, **k)
        return g

    for _name in ("scroll", "hscroll"):
        _fn = getattr(pyautogui, _name, None)
        if _fn is not None:
            setattr(pyautogui, _name, _wrap_scroll(_fn))


class _PyAutoGUIFinder(importlib.abc.MetaPathFinder):
    """Lazy finder: patches pyautogui exactly once, on its first import."""
    _armed = True

    def find_spec(self, name, path, target=None):
        if name != "pyautogui" or not self._armed:
            return None
        self._armed = False  # prevent recursion in find_spec below and re-patching
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            self._armed = True
            return None
        _orig_exec = spec.loader.exec_module

        def exec_module(module, _orig_exec=_orig_exec):
            _orig_exec(module)
            try:
                _patch(module)
            except Exception:
                pass

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _PyAutoGUIFinder())
'''

#: heredoc delimiter for writing the shim (unlikely to appear in SHIM_SOURCE).
_PROVISION_EOF = "__CGUI_SHIM_EOF__"


def build_provision_command() -> str:
    """Bash that writes ``SHIM_SOURCE`` to the VM user-site as ``usercustomize.py``.

    Uses the SAME ``python3`` the model will use so the shim lands on that interpreter's
    path. ``python3 -m site --user-site`` returns a path even when user-site is disabled;
    ``mkdir -p`` + write then makes it usable (a fresh dir is added to ``sys.path`` at
    interpreter startup). Idempotent — safe to re-run every episode.
    """
    return "\n".join([
        "set -e",
        'DIR="$(python3 -m site --user-site)"',
        'mkdir -p "$DIR"',
        "cat > \"$DIR/usercustomize.py\" <<'" + _PROVISION_EOF + "'",
        SHIM_SOURCE,
        _PROVISION_EOF,
        # sanity: importing it must not blow up (pyautogui not imported yet -> no-op).
        'python3 -c "import usercustomize" 2>/dev/null || true',
        'echo "c-gui shim installed to $DIR/usercustomize.py"',
    ])


def bash_payload(command: str, scale: str = DEFAULT_COORD_SCALE) -> str:
    """Prepend the coord-scale export so the VM's python3 (pyautogui) scales 0-999."""
    return f"export {COORD_SCALE_ENV}={scale}\n{command}"
