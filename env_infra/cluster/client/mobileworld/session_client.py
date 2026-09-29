"""MobileWorld session client — drop-in replacement for AndroidEnvClient.

Routes all calls through the cluster session API. Compatible with MobileWorld's
runner (run_agent_with_evaluation) without modifying MobileWorld source code.
"""
from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Any

from PIL import Image

from cluster.client.base.session_client import ClusterSessionClient

logger = logging.getLogger("cluster.client.mobileworld.session")


class MobileWorldSessionClient(ClusterSessionClient):
    """AndroidEnvClient-compatible interface backed by the cluster session API."""

    def __init__(
        self,
        cluster_url: str | None = None,
        device: str = "emulator-5554",
        step_wait_time: float = 1.0,
    ):
        super().__init__(cluster_url=cluster_url, runtime="mobileworld")
        self.device = device
        self.step_wait_time = step_wait_time
        self.base_url = f"cluster-session://{self.session_id}"
        self.tools = []

    def switch_suite_family(self, target_family: str) -> dict:
        """No-op for remote — suite family is pre-configured on the node."""
        return {"switched": False, "note": "remote session, suite already configured"}

    def initialize_task(self, task_name: str):
        """Reset the session with a task — equivalent to AndroidEnvClient.initialize_task.

        Stateless: every call performs a real reset on the node (which both
        initializes the task and returns its instruction). No per-task caching,
        so the client can never return a stale previous task's state.
        """
        return self._to_observation(self.reset({"task_name": task_name}))

    def execute_action(self, action):
        """Execute a JSONAction — translates to cluster Action protocol."""
        action_dict = action.model_dump(exclude_none=True) if hasattr(action, "model_dump") else dict(action)
        action_type = action_dict.pop("action_type", "unknown")
        typed_action = {"kind": "device", "type": action_type, "payload": action_dict}
        data = self.step(typed_action)
        obs_data = data.get("observation", {})
        obs = self._to_observation(obs_data)
        # Check for ask_user response in info
        info = data.get("info", {})
        if hasattr(obs, "ask_user_response") and info.get("action_type") == "ask_user":
            obs.ask_user_response = info.get("ask_user_response")
        return obs

    def get_observation(self, type="screenshot", wait_to_stabilize: bool = True) -> dict:
        """Get current observation."""
        obs_data = self.observe()
        screenshot = self._decode_screenshot(obs_data)
        return {"screenshot": screenshot, "accessibility_tree": None}

    def get_screenshot(self, wait_to_stabilize: bool = False) -> Image.Image:
        """Get current screenshot as PIL Image."""
        obs_data = self.observe()
        return self._decode_screenshot(obs_data)

    def get_task_score(self, task_type: str) -> tuple[float, str]:
        """Evaluate the current task."""
        data = self.evaluate()
        score = float(data.get("score", 0.0))
        reason = data.get("reason", "")
        return score, reason

    def get_task_goal(self, task_type: str) -> str:
        """Get task goal/instruction for a task from the node.

        Stateless: resets the session to ``task_type`` and reads the instruction
        the node returns in the observation. Always reflects the requested task —
        no caching, so it can never return a previous task's goal.
        """
        obs_data = self.reset({"task_name": task_type})
        modalities = obs_data.get("modalities", obs_data)
        return modalities.get("instruction") or ""

    def tear_down_task(self, task_type: str):
        """No-op for remote — the node handles teardown on next reset or release."""
        pass

    def health(self) -> bool:
        """Check if session is active."""
        return self.session_id is not None

    def get_suite_task_list(self, enable_mcp: bool = False, enable_user_interaction: bool = False) -> list[str]:
        """Get available task list from local MobileWorld task registry.

        Task definitions live in the codebase (not in the remote container),
        so we load them locally — same pattern as OSWorld reading its test JSON.
        """
        try:
            from mobile_world.tasks.registry import TaskRegistry
            registry = TaskRegistry()
            filtered = []
            for name, task_cls in registry.tasks.items():
                tags = getattr(task_cls, "task_tags", set())
                if not enable_mcp and "agent-mcp" in tags:
                    continue
                if not enable_user_interaction and "agent-user-interaction" in tags:
                    continue
                filtered.append(name)
            return filtered
        except Exception as exc:
            logger.warning("Failed to load local task registry: %s", exc)
            return []

    def _to_observation(self, obs_data: dict):
        """Convert cluster observation format to MobileWorld Observation."""
        from mobile_world.runtime.utils.models import Observation
        modalities = obs_data.get("modalities", obs_data)
        screenshot = self._decode_screenshot_from_modalities(modalities)
        ask_user_response = modalities.get("ask_user_response")
        return Observation(screenshot=screenshot, ask_user_response=ask_user_response)

    def _decode_screenshot(self, obs_data: dict) -> Image.Image | None:
        modalities = obs_data.get("modalities", obs_data)
        return self._decode_screenshot_from_modalities(modalities)

    @staticmethod
    def _decode_screenshot_from_modalities(modalities: dict) -> Image.Image | None:
        b64 = modalities.get("screenshot")
        if not b64:
            return None
        try:
            img_bytes = base64.b64decode(b64)
            return Image.open(BytesIO(img_bytes))
        except Exception:
            return None


# Backwards-compatible alias (old code imported MobileWorldRemoteClient).
MobileWorldRemoteClient = MobileWorldSessionClient
