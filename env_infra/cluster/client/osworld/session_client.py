"""OSWorld session client — DesktopEnv-compatible, zero OSWorld dependency.

Drop-in replacement for a local OSWorld ``DesktopEnv`` (formerly named
``RemoteDesktopEnv``). Implements the exact interface that OSWorld's
lib_run_single.py expects (reset/step/_get_obs/evaluate/
controller.start_recording/end_recording), backed entirely by the generic
:class:`ClusterSessionClient` protocol.

No imports from desktop_env, DesktopEnv, or DockerServerProvider.
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, Optional, Tuple

from cluster.client.base.session_client import ClusterSessionClient

logger = logging.getLogger("cluster.client.osworld.session")


# -- format conversion (pure) ------------------------------------------------

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
    """OSWorld action (pyautogui str / computer_13 dict / control token) → cluster Action.

    cli/code execution does NOT go through here — it has its own entry point.
    """
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
        logger.warning("run_bash_script is not supported on OSWorldSessionClient")
        return ""


class OSWorldSessionClient(ClusterSessionClient):
    """DesktopEnv-compatible env client backed by the cluster session API.

    Implements the duck-typed interface that OSWorld evaluation scripts expect:
        env.reset(task_config) → dict obs
        env._get_obs() → dict
        env.step(action, pause) → (obs, reward, done, info)
        env.evaluate() → float
        env.controller.start_recording() / end_recording(path)
    """

    def __init__(
        self,
        cluster_url: str | None = None,
        action_space: str = "pyautogui",
        screen_size: Tuple[int, int] = (1920, 1080),
        headless: bool = True,
        enable_proxy: bool = True,
        client_password: str = "password",
        **kwargs,
    ):
        super().__init__(cluster_url=cluster_url, runtime="osworld")
        self.action_space = action_space
        self.screen_width, self.screen_height = screen_size
        self.headless = headless
        self.enable_proxy = enable_proxy
        self.client_password = client_password
        self.instruction: Optional[str] = None
        self.is_environment_used = False
        self.task_id: Optional[str] = None
        self.controller = _NoOpRecorder()
        self._traj_no = -1
        self._step_no = 0

    # -- DesktopEnv-compatible interface ------------------------------------

    def reset(self, task_config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        self._traj_no += 1
        self._step_no = 0
        self.is_environment_used = False
        task_config = task_config or {}
        self.task_id = task_config.get("id")
        self.instruction = task_config.get("instruction")
        obs_data = super().reset(task_config)
        return self._to_osworld_obs(obs_data)

    def _get_obs(self) -> Dict[str, Any]:
        obs_data = self.observe()
        return self._to_osworld_obs(obs_data)

    def step(self, action, pause: float | None = None) -> Tuple[Dict[str, Any], float, bool, Dict]:
        # The node settles server-side and returns the post-settle screenshot in one
        # response (no client-side sleep / second observe). `pause` overrides the
        # adapter's default settle when provided; None keeps the adapter default.
        self._step_no += 1
        self.is_environment_used = True
        typed_action = self._to_cluster_action(action)
        data = super().step(typed_action, pause=pause)
        obs = self._to_osworld_obs(data.get("observation", {}))
        done = bool(data.get("done", False))
        reward = float(data.get("reward", 0))
        info = data.get("info", {}) or {}
        return obs, reward, done, info

    def evaluate(self) -> float:
        data = super().evaluate()
        return float(data.get("score", 0.0))

    def close(self):
        self.release()

    # -- format conversion (delegate to shared module functions) ------------

    def _to_osworld_obs(self, obs_data: Dict[str, Any]) -> Dict[str, Any]:
        return to_osworld_obs(obs_data, self.instruction)

    def _to_cluster_action(self, action) -> Dict[str, Any]:
        return to_cluster_action(action, self.action_space)

    def run_code(
        self, code: str, lang: str = "python", timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Execute code in the remote env and return {status, output, error}.

        The coding agent's executor: sends one cli step through the normal step
        channel (client→node→adapter._run_code) and returns its exec_result.

        ``timeout`` (seconds) applies to ``lang="bash"`` only; omitted -> the
        adapter's default. The adapter clamps it below the sandbox's own caps.
        """
        payload: Dict[str, Any] = {"code": code, "lang": lang}
        if timeout is not None:
            payload["timeout"] = timeout
        action = {"kind": "tool", "type": "cli", "payload": payload}
        data = super().step(action)
        return (data.get("info", {}) or {}).get("exec_result", {}) or {}


# Backwards-compatible alias (old code imported OSWorldRemoteClient).
OSWorldRemoteClient = OSWorldSessionClient
