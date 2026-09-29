"""Executor helpers for c-gui: shim provisioning + coord-scale injection.

Thin by design (the real dispatch lives in ``run_loop.py``). Two helpers:
  * ``provision_shim(env)`` — install the VM pyautogui shim once, AFTER ``env.reset()``.
  * ``bash_payload(command)`` — prepend the ``CUA_COORD_SCALE`` export so the VM python3
    subprocess launched by a GUI heredoc scales 0-999 coords. Prepended at EXECUTION
    time only; the raw model command is what gets stored in traj.
"""
from __future__ import annotations

from .shim import build_provision_command, COORD_SCALE_ENV, DEFAULT_COORD_SCALE


def provision_shim(env, logger=None) -> None:
    """Install the pyautogui shim into the VM user-site (idempotent; run AFTER reset)."""
    try:
        res = env.run_code(build_provision_command(), lang="bash") or {}
        if logger:
            logger.info("c-gui shim: %s", str(res.get("output", "")).strip()[:200])
    except Exception as exc:  # non-fatal; real issues surface on the first GUI step
        if logger:
            logger.warning("c-gui shim provision failed: %s", exc)


def bash_payload(command: str, scale: str = DEFAULT_COORD_SCALE) -> str:
    """Prepend the coord-scale export so the VM's python3 (pyautogui) scales 0-999.

    OSWorld's own setup channel never sets this env, so its absolute-pixel pyautogui is
    unaffected — only c-gui bash commands enable scaling.
    """
    return f"export {COORD_SCALE_ENV}={scale}\n{command}"
