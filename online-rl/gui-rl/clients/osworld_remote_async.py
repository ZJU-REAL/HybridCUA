"""OSWorld async remote client — DesktopEnv-compatible, asyncio.

Self-contained copy from env_infra (no ``cluster.*`` imports): the obs/action
conversion helpers that the original imported from ``osworld_remote`` are inlined
here, so the async path carries no dependency on the sync client.

Same DesktopEnv-style obs/action shapes (so an async episode loop sees identical
observations), backed by AsyncClusterSessionClient. Lets one event loop drive
many concurrent OSWorld episodes in a single process.
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, Optional, Tuple

from .async_session_client import AsyncClusterSessionClient
from .coord_shim import bash_payload, build_provision_command

logger = logging.getLogger("cluster.client.osworld_remote_async")


# -- format conversion (pure; inlined from osworld_remote) -------------------

def to_osworld_obs(obs_data: Dict[str, Any], instruction: Optional[str] = None) -> Dict[str, Any]:
    """Cluster observation → OSWorld dict format (screenshot decoded to bytes)."""
    modalities = obs_data.get("modalities", obs_data)
    screenshot_b64 = modalities.get("screenshot")
    screenshot_bytes = base64.b64decode(screenshot_b64) if screenshot_b64 else None
    obs = {
        "screenshot": screenshot_bytes,
        "accessibility_tree": modalities.get("accessibility_tree"),
        "terminal": modalities.get("terminal"),
    }
    if instruction is not None:
        obs["instruction"] = instruction
    return obs


def to_cluster_action(action, action_space: str = "pyautogui") -> Dict[str, Any]:
    """OSWorld action (pyautogui str / computer_13 dict / control token / bash dict) → cluster Action."""
    # Bash surface (c_gui hybrid agent): a dict tagged action_type=bash/cli/shell runs one
    # shell command in the VM via the TOOL/cli channel (adapter._run_code). bash_payload
    # prepends `export CUA_COORD_SCALE=999` so the pyautogui heredoc's 0-999 coords scale.
    if isinstance(action, dict) and action.get("action_type") in ("bash", "cli", "shell"):
        payload: Dict[str, Any] = {"code": bash_payload(action["command"]), "lang": "bash"}
        if action.get("timeout") is not None:
            payload["timeout"] = action["timeout"]
        return {"kind": "tool", "type": "cli", "payload": payload}
    if isinstance(action, str):
        action_stripped = action.strip()
        if action_stripped in ("DONE", "FAIL", "WAIT"):
            return {"kind": "control", "type": action_stripped, "payload": {}}
        return {"kind": "gui", "type": "pyautogui", "payload": {"command": action}}
    elif isinstance(action, dict):
        action_type = action.get("action_type", action_space)
        return {"kind": "gui", "type": action_type, "payload": action}
    return {"kind": "gui", "type": "pyautogui", "payload": {"command": str(action)}}


class _NoOpRecorder:
    """No-op controller — cluster doesn't support VNC recording."""

    def start_recording(self) -> None:
        pass

    def end_recording(self, path: str) -> None:
        pass

    def run_bash_script(self, script: str, timeout: int = 30, working_dir: str | None = None):
        logger.warning("run_bash_script is not supported on OSWorldRemoteClient")
        return ""


class OSWorldAsyncRemoteClient(AsyncClusterSessionClient):
    """Async DesktopEnv-compatible env client backed by the cluster session API.

    Async mirror of OSWorldRemoteClient. Methods are coroutines:
        await env.acquire()                       # or `async with`
        await env.reset(task_config) → dict obs
        await env.get_obs() → dict
        await env.step(action) → (obs, reward, done, info)
        await env.evaluate() → float
    """

    def __init__(
        self,
        cluster_url: str | None = None,
        action_space: str = "pyautogui",
        screen_size: Tuple[int, int] = (1920, 1080),
        **kwargs,
    ):
        super().__init__(cluster_url=cluster_url, runtime="osworld")
        self.action_space = action_space
        self.screen_width, self.screen_height = screen_size
        self.instruction: Optional[str] = None
        self.is_environment_used = False
        self.task_id: Optional[str] = None
        self.controller = _NoOpRecorder()
        self._traj_no = -1
        self._step_no = 0

    async def reset(self, task_config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        self._traj_no += 1
        self._step_no = 0
        self.is_environment_used = False
        task_config = task_config or {}
        self.task_id = task_config.get("id")
        self.instruction = task_config.get("instruction")
        obs_data = await super().reset(task_config)
        # Coordinate shim for the bash-surface (c_gui hybrid) agent: install the VM-side
        # usercustomize.py once per episode (reset may roll back the snapshot). Gated on
        # GUI_COORD_SHIM=1 so non-bash agents are unaffected; scaling itself is further
        # gated on CUA_COORD_SCALE (only bash_payload sets it), so this is a cheap no-op
        # for any interpreter that never imports pyautogui.
        if os.getenv("GUI_COORD_SHIM") == "1":
            try:
                await self.run_code(build_provision_command(), lang="bash")
            except Exception:  # provisioning must never abort an episode
                logger.warning("coord shim provisioning failed (continuing)", exc_info=True)
        return to_osworld_obs(obs_data, self.instruction)

    async def run_code(self, code: str, lang: str = "bash", timeout: float | None = None) -> Dict[str, Any]:
        """Run code in the VM via the TOOL/cli channel (adapter._run_code). Returns the
        raw StepResponse dict; exec output is under info['exec_result']."""
        payload: Dict[str, Any] = {"code": code, "lang": lang}
        if timeout is not None:
            payload["timeout"] = timeout
        return await super().step({"kind": "tool", "type": "cli", "payload": payload})

    async def get_obs(self) -> Dict[str, Any]:
        obs_data = await self.observe()
        return to_osworld_obs(obs_data, self.instruction)

    async def step(self, action, pause: float | None = None) -> Tuple[Dict[str, Any], float, bool, Dict]:
        # The node settles server-side and returns the post-settle screenshot in one
        # response (no client-side sleep / second observe). `pause` overrides the
        # adapter's default settle when provided; None keeps the adapter default.
        self._step_no += 1
        self.is_environment_used = True
        typed_action = to_cluster_action(action, self.action_space)
        data = await super().step(typed_action, pause=pause)
        obs = to_osworld_obs(data.get("observation", {}), self.instruction)
        done = bool(data.get("done", False))
        reward = float(data.get("reward", 0))
        info = data.get("info", {}) or {}
        return obs, reward, done, info

    async def evaluate(self) -> float:
        data = await super().evaluate()
        return float(data.get("score", 0.0))
