"""MobileWorld adapter: the mobile surface's reference implementation.

Wraps the official MobileWorld ``AndroidEnvClient`` (Tongyi-MAI/MobileWorld)
behind the platform's WorldAdapter contract. This is the second surface
(alongside OSWorld's desktop), proving the platform's two orthogonal axes —
Surface (mobile) x Driver (docker) — compose without touching master/node core.

The official ``mobile_world`` package is imported *lazily* inside ``make_driver``
so this module imports on any machine (no Android/emulator deps needed to load
it). All MobileWorld-specific knowledge stays here:

- ``AndroidEnvClient(url, device, step_wait_time)`` talks HTTP to a MobileWorld
  server that drives the emulator. A "session" = one initialized task on a device.
- Actions: generic ``Action(kind="device", type=...)`` -> a ``JSONAction``
  (action_type + x/y/text/...) passed to ``execute_action``. Control DONE/FAIL/WAIT
  -> the native ``finished``/``error_env``/``wait`` action types.
- Observation: ``get_observation()`` returns a PIL screenshot (+ ask_user_response,
  tool_call) -> generic ``Observation`` (screenshot base64'd).
- Evaluation: ``get_task_score(task) -> (score, reason)`` -> ``EvaluationResult``.

The official action vocabulary (``_ACTION_TYPES``) is:
click, double_tap, long_press, scroll, swipe, drag, input_text, keyboard_enter,
navigate_home, navigate_back, open_app, wait, answer, ask_user, status, finished,
unknown, mcp.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Callable, Dict

from cluster.utils.encoding import encode_screenshot
from cluster.schemas import Action, ControlType, EvaluationResult, Observation, StepResponse
from cluster.worlds.base.surfaces.mobile import MobileWorld

logger = logging.getLogger("cluster.worlds.mobileworld")

#: Native MobileWorld action types an agent may emit as Action(kind="device").
#: (Mirrors mobile_world.runtime.utils.models._ACTION_TYPES minus the control
#: verbs, which arrive as Action(kind="control").)
_DEVICE_ACTION_TYPES = frozenset(
    {
        "click",
        "double_tap",
        "long_press",
        "scroll",
        "swipe",
        "drag",
        "input_text",
        "keyboard_enter",
        "navigate_home",
        "navigate_back",
        "open_app",
        "answer",
        "ask_user",
        "status",
        "mcp",
    }
)

#: Native action_type values that terminate the episode.
_TERMINAL_NATIVE_TYPES = frozenset({"finished", "error_env", "unknown"})

#: Native action_types the agent may emit directly (the remote client forwards
#: every action as kind="device" with type=<action_type>, never re-mapping the
#: terminal/wait verbs to kind="control"). MobileWorld's vocabulary is flat, so
#: these pass straight through as JSONActions: wait (non-terminal) + the terminal
#: verbs. Accepting them here is what makes "finished"/"wait"/"unknown"/"error_env"
#: work when they arrive as device actions instead of control verbs.
_NATIVE_PASSTHROUGH_TYPES = frozenset({"wait"}) | _TERMINAL_NATIVE_TYPES

#: Control verb -> native MobileWorld action_type.
_CONTROL_TO_NATIVE = {
    ControlType.DONE.value: "finished",
    ControlType.FAIL.value: "error_env",
    ControlType.WAIT.value: "wait",
}


class MobileWorldAdapter(MobileWorld):
    """Adapt the official MobileWorld ``AndroidEnvClient`` to WorldAdapter."""

    def __init__(self, client: Any, *, task_name: str | None = None, container_name: str | None = None, vnc_port: int | None = None, viewer_port: int | None = None) -> None:
        self._client = client
        self._task_name = task_name
        self._container_name = container_name
        self._vnc_port = vnc_port
        self._viewer_port = viewer_port
        self._step = 0
        self._closed = False
        # Cache the (slow) docker container-state check shared by liveness()
        # (reuse decision) and get_info() (display). Holds the raw state string
        # ("running"/"exited"/"unhealthy"/"gone") or None. Mirrors OSWorld.
        self._container_alive_cache: "tuple[float, str | None] | None" = None

    # -- driver ------------------------------------------------------------
    @classmethod
    def make_driver(cls, config: Dict[str, Any] | None = None):
        """Return a RuntimeDriver encapsulating all lifecycle hooks.

        The driver supports:
        - create(): launch a single container + wait ready
        - create_batch(n): serial launch N containers + unified ready wait (mw env run behavior)
        - adopt(): discover and reclaim existing running containers
        """
        import threading
        import time as _time

        from cluster.node.pool import RuntimeDriver

        from pathlib import Path

        config = dict(config or {})
        # Image is the world's driver-contract image, injected from
        # manifest.driver.image by _load_adapter_driver — never a config_schema
        # tunable. No fallback default here: a missing image means the manifest
        # is malformed, which should fail loudly rather than launch a guessed tag.
        image = config.get("image")
        if not image:
            raise ValueError("MobileWorld make_driver: no image (expected manifest.driver.image)")
        http_proxy = config.get("http_proxy", "")
        device = config.get("device", "emulator-5554")
        step_wait_time = float(config.get("step_wait_time", 1.0))
        default_task = config.get("task_name")
        use_mcp = bool(config.get("enable_mcp", False))
        enable_vnc = bool(config.get("enable_vnc", False))
        ready_timeout = int(config.get("ready_timeout", 600))
        launch_interval = int(config.get("launch_interval", 20))
        _launch_lock = threading.Lock()

        import os
        _mw_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "MobileWorld",
        )
        _env_file = Path(_mw_root) / ".env"
        env_file_path = _env_file if _env_file.is_file() else None

        # Optionally mount a host src dir over the image's /app/service/src so the
        # container runs live task code (e.g. generated query variants) without
        # rebuilding the image. The path must be visible on the NODE host that
        # launches the container (shared mount). Defaults to the node-local
        # MobileWorld/src next to this cluster checkout; set MW_DEV_SRC_PATH to a
        # different absolute path, or MW_MOUNT_SRC=0 to disable and use the
        # image's baked-in src.
        dev_src_path = None
        if os.environ.get("MW_MOUNT_SRC", "1") != "0":
            _src_override = os.environ.get("MW_DEV_SRC_PATH")
            _src_candidate = Path(_src_override) if _src_override else Path(_mw_root) / "src"
            if _src_candidate.is_dir():
                dev_src_path = _src_candidate
                logger.info("MobileWorld mount-src enabled: %s -> /app/service/src", dev_src_path)
            elif _src_override:
                logger.warning("MW_DEV_SRC_PATH not a directory, mount-src disabled: %s", _src_override)

        def _ensure_mw_path():
            import os, sys
            mw_src = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "MobileWorld", "src",
            )
            if os.path.isdir(mw_src) and mw_src not in sys.path:
                sys.path.insert(0, mw_src)

        def _create() -> "MobileWorldAdapter":
            _ensure_mw_path()
            from mobile_world.core.api.env import (
                build_container_config, find_available_ports, get_container_info,
                launch_container, wait_for_container_ready,
            )
            from mobile_world.runtime.client import AndroidEnvClient  # type: ignore

            with _launch_lock:
                ports = find_available_ports(count=1)
                if not ports:
                    raise RuntimeError("No available ports for MobileWorld container")
                backend_port, viewer_port, vnc_port, adb_port = ports[0]
                container_config = build_container_config(
                    image=image,
                    backend_port=backend_port,
                    viewer_port=viewer_port,
                    vnc_port=vnc_port,
                    adb_port=adb_port,
                    enable_vnc=enable_vnc,
                    http_proxy=http_proxy or None,
                    env_file_path=env_file_path,
                    dev_src_path=dev_src_path,
                )
                result = launch_container(container_config, wait_ready=False)
                if not result or not result.success:
                    err = result.error_message if result else "no result"
                    raise RuntimeError(f"Failed to launch MobileWorld container: {err}")
                # Hold the lock until the container is visible to docker. The
                # container name is chosen by find_next_container_index() (docker
                # ps max+1); launch_container(wait_ready=False) returns before the
                # container registers, so releasing here would let a concurrent
                # _create() pick the SAME index and build a duplicate. Wait until
                # `docker inspect` sees it, then the next create's max+1 is correct.
                for _ in range(50):  # ~5s; registration is near-instant after docker run
                    if get_container_info(result.name) is not None:
                        break
                    _time.sleep(0.1)
                else:
                    raise RuntimeError(f"Container {result.name} did not register with docker")

            if not wait_for_container_ready(result.backend_port, timeout=ready_timeout):
                raise RuntimeError(f"Container {result.name} did not become ready in {ready_timeout}s")

            if not enable_vnc:
                import subprocess
                subprocess.Popen(
                    ["docker", "exec", "-d", result.name, "bash", "-c",
                     "cd /app/service && uv run mobile-world viewer --port 7860"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )

            url = f"http://localhost:{result.backend_port}"
            client_cls = AndroidEnvClient
            if use_mcp:
                from mobile_world.runtime.client import AndroidMCPEnvClient  # type: ignore
                client_cls = AndroidMCPEnvClient
            client = client_cls(url=url, device=device, step_wait_time=step_wait_time)
            return cls(client, task_name=default_task, container_name=result.name,
                       vnc_port=result.vnc_port if enable_vnc else None,
                       viewer_port=result.viewer_port if not enable_vnc else None)

        def _adopt() -> list:
            _ensure_mw_path()
            from cluster.worlds.base import docker_util

            containers = docker_util.list_running_containers(name_prefix="mobile_world_env_")
            if not containers:
                return []

            logger.info("Found %d running MobileWorld container(s), attempting to adopt", len(containers))
            # MobileWorld's published container ports -> our field names.
            _PORTS = {6800: "server_port", 5800: "vnc_port", 7860: "viewer_port"}
            adopted = []
            for container in containers:
                try:
                    mapped = docker_util.container_host_ports(container, _PORTS)
                    server_port = mapped.get("server_port")
                    vnc_port = mapped.get("vnc_port")
                    viewer_port = mapped.get("viewer_port")
                    if not server_port:
                        logger.warning("Container %s has no server port, skipping", container.short_id)
                        continue

                    url = f"http://localhost:{server_port}"
                    from mobile_world.runtime.client import AndroidEnvClient  # type: ignore
                    client_cls = AndroidEnvClient
                    if use_mcp:
                        from mobile_world.runtime.client import AndroidMCPEnvClient  # type: ignore
                        client_cls = AndroidMCPEnvClient

                    env_client = client_cls(url=url, device=device, step_wait_time=step_wait_time)
                    adapter = cls(env_client, task_name=default_task, container_name=container.name,
                                  vnc_port=vnc_port if enable_vnc else None,
                                  viewer_port=viewer_port if not enable_vnc else None)
                    adopted.append(adapter)
                    logger.info("Adopted MobileWorld container %s (server:%d)", container.short_id, server_port)
                except Exception:
                    logger.warning("Failed to adopt container %s", container.short_id, exc_info=True)

            return adopted

        return RuntimeDriver(create=_create, adopt=_adopt, reclaim_orphans=cls._make_reclaim_fn(image))

    @classmethod
    def _make_reclaim_fn(cls, image: str | None):
        """Build the orphan-container reclaimer for MobileWorld. Delegates to the
        shared docker helper, which removes containers of this world's image not
        owned by any live slot (``live_ids`` — here the container NAMEs each slot
        reports via :meth:`resource_id`) and older than ``grace``. No-op without
        an image. Mirrors the OSWorld adapter's reclaim wiring."""
        if not image:
            return None

        def _reclaim(grace_seconds: float, live_ids: set) -> int:
            from cluster.worlds.base import docker_util

            return docker_util.reclaim_orphan_containers(
                image=image, grace_seconds=grace_seconds, live_ids=live_ids
            )

        return _reclaim

    # -- action translation (mobile device dialect) -----------------------
    def _build_json_action(self, action: Action):
        """Translate a generic Action into the official ``JSONAction``.

        This is the mobile surface's equivalent of OSWorld's pyautogui
        translation: the native action vocabulary stays inside this adapter.
        """
        from mobile_world.runtime.utils.models import JSONAction  # type: ignore

        if action.is_control():
            native_type = _CONTROL_TO_NATIVE.get(action.type)
            if native_type is None:
                raise ValueError(f"unsupported control action: {action.type!r}")
            if native_type == "wait":
                return JSONAction(action_type="wait")
            return JSONAction(action_type=native_type)

        # MobileWorld's official action vocabulary is FLAT: wait/finished/
        # error_env/unknown are plain action_type values, not a separate kind.
        # The remote client forwards every agent action as kind="device" with
        # type=<action_type> (it does not re-map control verbs), so accept the
        # full native vocabulary here — including the terminal/wait verbs — and
        # pass them straight through as JSONActions. This is what the control
        # branch builds anyway (JSONAction(action_type="wait")), so the result
        # is identical; step() still derives `done` from _TERMINAL_NATIVE_TYPES.
        if action.type not in _DEVICE_ACTION_TYPES and action.type not in _NATIVE_PASSTHROUGH_TYPES:
            raise ValueError(
                f"unsupported mobile device action type: {action.type!r}; "
                f"expected one of {sorted(_DEVICE_ACTION_TYPES)}"
            )
        return JSONAction(action_type=action.type, **dict(action.payload))

    # -- observation translation ------------------------------------------
    @staticmethod
    def _screenshot_to_b64(screenshot: Any) -> str | None:
        """Encode a PIL.Image (or raw bytes / already-b64 str) to base64 PNG."""
        if screenshot is None:
            return None
        if isinstance(screenshot, str):
            return screenshot
        if isinstance(screenshot, (bytes, bytearray)):
            return encode_screenshot(bytes(screenshot))
        # assume PIL.Image-like with .save
        if hasattr(screenshot, "save"):
            buf = io.BytesIO()
            screenshot.save(buf, format="PNG")
            return encode_screenshot(buf.getvalue())
        raise TypeError(f"cannot encode screenshot of type {type(screenshot)!r}")

    def _wrap_obs(self, obs: Any) -> Observation:
        """Wrap a MobileWorld ``Observation`` (or dict) into a generic Observation."""
        if obs is None:
            data: Dict[str, Any] = {}
        elif isinstance(obs, dict):
            data = obs
        else:
            # pydantic Observation: screenshot / accessibility_tree / ask_user_response / tool_call
            data = {
                "screenshot": getattr(obs, "screenshot", None),
                "accessibility_tree": getattr(obs, "accessibility_tree", None),
                "ask_user_response": getattr(obs, "ask_user_response", None),
                "tool_call": getattr(obs, "tool_call", None),
            }
        modalities: Dict[str, Any] = {
            "screenshot": self._screenshot_to_b64(data.get("screenshot")),
            "ui_tree": data.get("accessibility_tree"),
            "device_state": None,
            "instruction": getattr(self, "_task_goal", None) or self._task_name,
        }
        # carry interaction extras when present
        if data.get("ask_user_response") is not None:
            modalities["ask_user_response"] = data.get("ask_user_response")
        if data.get("tool_call") is not None:
            modalities["tool_call"] = data.get("tool_call")
        return self.make_observation(modalities, step=self._step, runtime="mobileworld")

    # -- lifecycle ---------------------------------------------------------
    def reset(self, task_payload: Dict[str, Any]) -> Observation:
        self._step = 0
        task_payload = task_payload or {}
        task_name = task_payload.get("task_name") or task_payload.get("id") or self._task_name
        if not task_name:
            return self._wrap_obs(self._client.get_observation())
        self._task_name = task_name
        obs = self._client.initialize_task(task_name)
        try:
            self._task_goal = self._client.get_task_goal(task_name)
        except Exception:
            self._task_goal = task_name
        return self._wrap_obs(obs)

    def step(self, action: Action, pause: float | None = None) -> StepResponse:
        # MobileWorld has no client-tunable post-action settle; pause is ignored.
        self.validate_action(action)
        # WAIT is a non-terminal control verb; everything else may terminate.
        json_action = self._build_json_action(action)
        native_type = getattr(json_action, "action_type", None)
        obs = self._client.execute_action(json_action)
        self._step += 1
        done = native_type in _TERMINAL_NATIVE_TYPES
        return StepResponse(observation=self._wrap_obs(obs), done=done, info={"action_type": native_type})

    def observe(self) -> Observation:
        return self._wrap_obs(self._client.get_observation())

    def evaluate(self) -> EvaluationResult:
        if not self._task_name:
            raise ValueError("cannot evaluate before reset(): no task_name")
        score, reason = self._client.get_task_score(self._task_name)
        score = float(score)
        return EvaluationResult(
            score=score,
            success=score >= 1.0,
            metrics={"raw_score": score},
            reason=str(reason),
        )

    def health_check(self) -> bool:
        if self._closed:
            return False
        try:
            return bool(self._client.health())
        except Exception:  # noqa: BLE001 - health probe must never raise
            return False

    def liveness(self) -> bool:
        """Deep pre-reuse check, authoritative on the DOCKER CONTAINER state.

        The container is the source of truth: only a dead container (exited /
        gone) makes a slot unreusable. A transient HTTP health blip on a
        still-running container must NOT condemn the slot — that would destroy a
        reusable warm env and force a costly cold rebuild. So: container not
        running -> False; container running -> True. With no container name to
        consult, fall back to the HTTP health probe. State is TTL-cached so
        repeated reuse probes don't fork docker. Mirrors the OSWorld adapter."""
        if self._closed:
            return False
        state = self._container_state_cached()
        if state is None:
            return self.health_check()  # no container name — defer to HTTP probe
        # Reusable iff running AND not unhealthy (docker_util folds unhealthy into
        # a distinct state, so a wedged-but-running container is not reused).
        return state == "running"

    def _container_state_cached(self) -> "str | None":
        """TTL-cached raw docker container state by name: ``"running"`` /
        ``"exited"`` / ``"unhealthy"`` / ``"gone"`` / ..., or None (no container
        name). One probe feeds both liveness (reuse) and get_info (display).
        Mirrors the OSWorld adapter."""
        import os
        import time as _time

        ttl = float(os.environ.get("MOBILEWORLD_LIVENESS_CACHE_TTL", "2"))
        now = _time.time()
        if self._container_alive_cache and now - self._container_alive_cache[0] < ttl:
            return self._container_alive_cache[1]
        from cluster.worlds.base import docker_util

        state = docker_util.container_state_by_name(self._container_name)
        if state is not None:
            self._container_alive_cache = (now, state)
        return state

    def resource_id(self) -> "str | None":
        """Container NAME — the reclaimable resource this slot holds. MobileWorld
        identifies its containers by name (``mobile_world_env_*``), so the pool's
        orphan reclaim spares this slot's container by matching this name in
        ``live_ids`` (docker_util.reclaim_orphan_containers spares by id OR name)."""
        return self._container_name

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._task_name is not None:
                try:
                    self._client.tear_down_task(self._task_name)
                except Exception:  # noqa: BLE001
                    logger.warning("tear_down_task(%s) failed during close", self._task_name)
            self._client.close()
            if self._container_name:
                from cluster.worlds.base import docker_util
                docker_util.stop_and_remove(self._container_name)
        finally:
            self._closed = True

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({
            "world_id": "mobileworld",
            "task_name": self._task_name,
            "server_url": getattr(self._client, "base_url", None),
            "device": getattr(self._client, "device", None),
            "container_name": self._container_name,
        })
        # Self-report raw container state for the dashboard (passthrough; the
        # neutral node never runs docker ps). Reuses the liveness TTL cache.
        if self._container_name:
            state = self._container_state_cached()
            if state is not None:
                info["docker_state"] = state
        if self._vnc_port:
            info["live_view"] = {"protocol": "vnc", "port": self._vnc_port}
        elif self._viewer_port:
            info["live_view"] = {"protocol": "http", "port": self._viewer_port}
        return info
