"""Shared plumbing for real-pixel, single-bash-surface OSWorld agents.

Two agents build on this — ``mm_agents.kimi_hybrid`` and ``mm_agents.claude_hybrid``.
Both offer the SAME action space (c-gui's four actions: ``bash`` / ``wait`` /
``terminate`` / ``answer``, one bash channel for GUI *and* CLI) and differ only in the
model transport and the reply syntax the model uses to express a call. Everything that
does not depend on those two things lives here:

  * ``shim`` — the VM ``usercustomize.py`` (FAILSAFE off + the Linux shift-char fix).
    **No coordinate scaling**: the model is shown the screenshot at full resolution and
    reports real screen pixels, so pyautogui's own coordinates are already correct.
    This is the difference from ``mm_agents.c_gui``, whose shim rescales a 0-999 grid.
  * ``actions`` — the pyautogui/bash action reference text (real pixels) and the typed
    action dicts the loop executes.
  * ``run_loop`` — the episode loop: one bash channel, control actions via ``env.step``.
  * ``selftest`` — the pre-run check that the model can actually read an image.

Nothing here imports ``kimi``/``anthropic``/``qwen``, so it stays importable without any
model SDK installed.
"""
from __future__ import annotations

from .actions import (
    ACTION_ENUM,
    bash_action,
    build_action_description,
    control_action,
    describe,
    parse_timeout,
    typed_action,
)
from .run_loop import build_bash_payload, run_single_example_c_gui_pixel
from .selftest import selftest, solid_png
from .shim import SHIM_SOURCE, build_provision_command, provision_shim

__all__ = [
    "ACTION_ENUM",
    "bash_action",
    "build_action_description",
    "control_action",
    "describe",
    "parse_timeout",
    "typed_action",
    "build_bash_payload",
    "run_single_example_c_gui_pixel",
    "selftest",
    "solid_png",
    "SHIM_SOURCE",
    "build_provision_command",
    "provision_shim",
]
