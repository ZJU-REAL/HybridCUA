"""CUA-Gym session client — OSWorld-identical, but acquires the ``cua_gym`` runtime.

CUA-Gym desktop tasks use the same VM image, action space, and wire protocol as
OSWorld, so every method (reset / step / _get_obs / evaluate / run_code / the
observation & action conversions) is inherited from :class:`OSWorldSessionClient`
unchanged. Only two things differ, and both are handled elsewhere:
  * the acquired **runtime** is ``cua_gym`` (routes to CuaGymWorldAdapter), and
  * the ``task_config`` passed to ``reset()`` carries the loader's inlined
    ``reward_code`` field, which the node hands to the adapter.

``OSWorldSessionClient.__init__`` hardcodes ``runtime="osworld"`` and acquires the
session *inside* ``__init__`` (via ``ClusterSessionClient.__init__``), so we
cannot flip the runtime after construction. We therefore bypass it and call
``ClusterSessionClient.__init__`` directly with ``runtime="cua_gym"``, replicating
OSWorldSessionClient's small, stable field setup. No existing file is modified.
"""
from __future__ import annotations

from typing import Tuple

from cluster.client.base.session_client import ClusterSessionClient
from cluster.client.osworld.session_client import OSWorldSessionClient, _NoOpRecorder


class CuaGymSessionClient(OSWorldSessionClient):
    """DesktopEnv-compatible cluster client that acquires the ``cua_gym`` runtime."""

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
        # Bypass OSWorldSessionClient.__init__ (hardcodes runtime="osworld" and
        # acquires in __init__); replicate its field setup but acquire cua_gym.
        ClusterSessionClient.__init__(self, cluster_url=cluster_url, runtime="cua_gym")
        self.action_space = action_space
        self.screen_width, self.screen_height = screen_size
        self.headless = headless
        self.enable_proxy = enable_proxy
        self.client_password = client_password
        self.instruction = None
        self.is_environment_used = False
        self.task_id = None
        self.controller = _NoOpRecorder()
        self._traj_no = -1
        self._step_no = 0


# Backwards-compatible alias, mirroring OSWorldRemoteClient.
CuaGymRemoteClient = CuaGymSessionClient
