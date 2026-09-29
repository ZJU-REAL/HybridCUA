"""OSWorld world adapter: wraps DesktopEnv behind the WorldAdapter contract.

This is the single coupling point between the platform and OSWorld. It is the
desktop surface's reference implementation. ``DesktopEnv`` is imported *lazily*
inside ``make_driver``/``create_adapter`` so this module stays importable on
machines without OSWorld's heavy dependency tree (torch, gymnasium, ...), which
is what lets the platform core and conformance tests run OSWorld-free.

Translation responsibilities (all isolated here):
- Action: generic ``Action`` -> legacy pyautogui str / computer_13 dict / control
  token, via the module-level ``action_to_legacy`` below (OSWorld-specific).
- Observation: DesktopEnv's ``{screenshot: bytes, accessibility_tree, terminal,
  instruction}`` -> generic ``Observation`` (screenshot base64'd), via the
  DesktopWorld surface helpers.
- Evaluation: ``DesktopEnv.evaluate()`` float score -> ``EvaluationResult``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, Union

import requests

from cluster.utils.encoding import encode_screenshot
from cluster.schemas import Action, ActionKind, ControlType, EvaluationResult, Observation, StepResponse
from cluster.worlds.base.surfaces.desktop import DesktopWorld

logger = logging.getLogger("cluster.worlds.osworld")

#: pyautogui-family action types DesktopEnv consumes as a python command string.
_PYAUTOGUI_TYPES = ("pyautogui", "claude_computer_use", "autoglm_computer_use")

#: Wall-clock seconds allowed for a ``lang="bash"`` command when the caller names none.
_BASH_TIMEOUT_DEFAULT = 60
#: Ceiling for a caller-requested bash timeout. Two OSWorld-internal caps sit above us
#: and are read-only: the VM's ``/run_python`` runs each script with
#: ``subprocess.run(timeout=30)``, and ``run_python_script`` uses a 200s HTTP timeout.
#: Staying under the HTTP timeout means our inner subprocess raises a clean
#: ``TimeoutExpired`` (captured as stderr) instead of the request dying. NOTE: the VM's
#: own 30s cap still wins in practice — a longer timeout cannot actually be honored.
_BASH_TIMEOUT_MAX = 180

#: Settle seconds after ``reset()`` before frame 0 is captured, so a slow-launching
#: app (Thunderbird with a full mail profile) is on screen by the time the agent looks.
_RESET_SETTLE = float(os.environ.get("CUA_RESET_SETTLE", "30"))


#: cv2 (opencv-python, non-headless) imported by the in-container OSWorld server
#: exports these two vars pointing at its own bundled Qt plugins. Every GUI app the
#: server launches via ``subprocess`` inherits them, so Qt apps like VLC pick up the
#: incompatible xcb plugin, fail to initialize, and exit immediately — a task's
#: ``launch`` setup step then reports success while no window ever appears. We cannot
#: restart the already-running server from here, so strip the vars per launch instead
#: (a harmless no-op for non-Qt apps).
_QT_POISON_VARS = ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR")


def sanitize_launch_env(task_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``task_payload`` whose ``launch`` setup steps run with the
    cv2 Qt-plugin env vars stripped, so system Qt apps (VLC, ...) start cleanly.

    Prepending ``env -u NAME ...`` works whether the launch command is a shell
    string (``shell=True``) or an argv list, and ``env``'s ``NAME=value`` passthrough
    keeps prefixes like ``VLC_VERBOSE=-1`` intact. ``google-chrome``/``chromium`` are
    left untouched: they are not Qt apps (so unaffected) and OSWorld's setup keys its
    proxy-flag injection on ``command[0]``, which the wrapper would otherwise hide.
    """
    config = (task_payload or {}).get("config")
    if not isinstance(config, list):
        return task_payload

    def _first_token(cmd: Union[str, list]) -> str:
        tokens = cmd if isinstance(cmd, list) else cmd.split()
        return tokens[0] if tokens else ""

    prefix_str = "env " + " ".join(f"-u {v}" for v in _QT_POISON_VARS) + " "
    prefix_list = ["env"] + [a for v in _QT_POISON_VARS for a in ("-u", v)]
    new_config = []
    changed = False
    for step in config:
        if isinstance(step, dict) and step.get("type") == "launch":
            params = dict(step.get("parameters") or {})
            cmd = params.get("command")
            first = _first_token(cmd) if cmd else ""
            if first not in ("google-chrome", "chromium", "env"):
                if isinstance(cmd, str):
                    params["command"] = prefix_str + cmd
                    step = {**step, "parameters": params}
                    changed = True
                elif isinstance(cmd, list) and cmd:
                    params["command"] = prefix_list + list(cmd)
                    step = {**step, "parameters": params}
                    changed = True
        new_config.append(step)
    if not changed:
        return task_payload
    return {**task_payload, "config": new_config}


def action_to_legacy(action: Action) -> Union[str, Dict[str, Any]]:
    """Translate a generic :class:`Action` into what ``DesktopEnv.step`` expects.

    OSWorld-specific, so it lives with the OSWorld adapter (not in the
    world-neutral protocol layer):
    - control DONE/FAIL/WAIT -> bare uppercase token string
    - gui/pyautogui          -> the python command string in ``payload['command']``
    - gui/computer_13        -> the structured dict payload
    - anything else          -> raise ValueError (surface mismatch)
    """
    if action.kind == ActionKind.CONTROL.value:
        if action.type in (ControlType.DONE.value, ControlType.FAIL.value, ControlType.WAIT.value):
            return action.type
        raise ValueError(f"unknown control action type: {action.type!r}")
    if action.kind == ActionKind.GUI.value:
        if action.type in _PYAUTOGUI_TYPES:
            command = action.payload.get("command")
            if command is None:
                raise ValueError("gui/pyautogui action missing payload['command']")
            return command
        if action.type == "computer_13":
            return dict(action.payload)
        raise ValueError(f"unsupported gui action type: {action.type!r}")
    raise ValueError(
        f"action kind {action.kind!r} is not a desktop/OSWorld action; "
        "non-desktop surfaces must translate in their own adapter"
    )


class OSWorldWorldAdapter(DesktopWorld):
    """Adapt an OSWorld ``DesktopEnv`` to the platform's WorldAdapter contract."""

    def __init__(self, env: Any, *, action_space: str = "pyautogui", pause: float = 3.0) -> None:
        self._env = env
        self._action_space = action_space
        self._pause = pause
        self._step = 0
        self._closed = False
        # Cache the (slow) docker container-state check shared by liveness()
        # (reuse decision) and get_info() (display), so a burst of reuse probes
        # and /slots calls doesn't hammer the docker daemon. Holds the raw state
        # string ("running"/"exited"/"unhealthy"/...) or None (non-docker).
        self._container_alive_cache: "tuple[float, str | None] | None" = None

    # -- driver ------------------------------------------------------------
    @classmethod
    def make_driver(
        cls,
        env_kwargs: Dict[str, Any] | None = None,
        *,
        action_space: str = "pyautogui",
        pause: float = 3.0,
    ):
        """Return a RuntimeDriver for OSWorld.

        DesktopEnv is imported lazily so importing this module never pulls
        in OSWorld's heavy dependency tree.
        """
        from cluster.node.pool import RuntimeDriver

        env_kwargs = dict(env_kwargs or {})
        env_kwargs.setdefault("action_space", action_space)
        # a11y tree / terminal are heavy per-step observations (a full AT-SPI walk
        # each _get_obs). Default them OFF so screenshot-only eval doesn't pay for
        # observations it never reads; a world that needs them sets the flag.
        env_kwargs.setdefault("require_a11y_tree", False)
        env_kwargs.setdefault("require_terminal", False)

        def _create() -> "OSWorldWorldAdapter":
            import os
            import sys

            osworld_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "OSWorld",
            )
            if os.path.isdir(osworld_dir) and sys.path[:1] != [osworld_dir]:
                sys.path.insert(0, osworld_dir)

            from desktop_env.desktop_env import DesktopEnv  # lazy: OSWorld dep

            import desktop_env as _de
            de_path = getattr(_de, "__file__", "") or ""
            if osworld_dir and os.path.isdir(osworld_dir) and not de_path.startswith(osworld_dir):
                raise RuntimeError(
                    f"'desktop_env' resolved to {de_path!r}, not this repo's "
                    f"{osworld_dir!r}. A sibling repo's desktop_env shadows ours; "
                    f"remove it from PYTHONPATH or start the node from a clean env."
                )

            env = DesktopEnv(**env_kwargs)
            return cls(env, action_space=env_kwargs.get("action_space", action_space), pause=pause)

        adopt_fn = cls._make_adopt_fn(env_kwargs, action_space=action_space, pause=pause)
        reclaim_fn = cls._make_reclaim_fn(env_kwargs.get("image"))
        return RuntimeDriver(create=_create, adopt=adopt_fn, reclaim_orphans=reclaim_fn)

    @classmethod
    def _make_reclaim_fn(cls, image: str | None):
        """Build the orphan-container reclaimer for the docker provider. Delegates
        to the shared docker helper, which removes containers of this world's
        image that are not mapped to any live slot (``live_ids``) and have existed
        longer than ``grace``. The pool/node only invoke the hook. No-op without
        an image."""
        if not image:
            return None

        def _reclaim(grace_seconds: float, live_ids: set) -> int:
            from cluster.worlds.base import docker_util

            return docker_util.reclaim_orphan_containers(
                image=image, grace_seconds=grace_seconds, live_ids=live_ids
            )

        return _reclaim

    @classmethod
    def _make_adopt_fn(
        cls,
        env_kwargs: Dict[str, Any] | None = None,
        *,
        action_space: str = "pyautogui",
        pause: float = 3.0,
    ) -> Callable[[], list]:
        """Return a callable that discovers running OSWorld Docker containers and
        wraps each in an adapter for the RuntimeDriver's adopt hook."""
        env_kwargs = dict(env_kwargs or {})
        env_kwargs.setdefault("action_space", action_space)

        def _adopt() -> list:
            import os
            import sys

            osworld_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "OSWorld",
            )
            if os.path.isdir(osworld_dir) and sys.path[:1] != [osworld_dir]:
                sys.path.insert(0, osworld_dir)

            from desktop_env.desktop_env import DesktopEnv

            provider_name = env_kwargs.get("provider_name", "docker")
            if "docker" not in str(provider_name):
                return []

            import docker as docker_lib
            from cluster.worlds.base import docker_util

            containers = docker_util.list_running_containers(image="happysixd/osworld-docker")
            if not containers:
                return []

            logger.info("Found %d running osworld container(s), attempting to adopt", len(containers))
            # OSWorld's published container ports -> our field names.
            _PORTS = {5000: "server_port", 9222: "chromium_port", 8006: "vnc_port", 8080: "vlc_port"}
            adopted = []
            for container in containers:
                try:
                    mapped = docker_util.container_host_ports(container, _PORTS)
                    server_port = mapped.get("server_port")
                    chromium_port = mapped.get("chromium_port")
                    vnc_port = mapped.get("vnc_port")
                    vlc_port = mapped.get("vlc_port")

                    if not server_port:
                        logger.warning("Container %s has no server port mapping, skipping", container.short_id)
                        continue

                    # Build a DesktopEnv shell without calling __init__ (which
                    # would boot a new VM). We manually wire the fields that the
                    # adapter needs for reset/step/observe/evaluate.
                    env = object.__new__(DesktopEnv)
                    env.vm_ip = "localhost"
                    env.server_port = server_port
                    env.chromium_port = chromium_port or 9222
                    env.vnc_port = vnc_port or 8006
                    env.vlc_port = vlc_port or 8080
                    env.action_space = env_kwargs.get("action_space", "pyautogui")
                    env.screen_width = int(env_kwargs.get("screen_width", 1920))
                    env.screen_height = int(env_kwargs.get("screen_height", 1080))
                    env.headless = env_kwargs.get("headless", True)
                    env.os_type = env_kwargs.get("os_type", "Ubuntu")
                    env.path_to_vm = env_kwargs.get("path_to_vm", "")
                    env.cache_dir_base = env_kwargs.get("cache_dir", "cache")
                    env.cache_dir = env.cache_dir_base
                    env.task_id = None
                    env.client_password = env_kwargs.get("client_password", "password")
                    # Episodic state + flags that DesktopEnv.__init__ creates and
                    # reset()/step()/_get_obs() read. object.__new__ skipped __init__,
                    # so set them here or the first reset() AttributeErrors on _traj_no.
                    env._traj_no = -1            # reset(): self._traj_no += 1
                    env._step_no = 0             # step(): self._step_no += 1 (NOT _step)
                    env.action_history = []      # reset().clear() / step().append()
                    env.require_a11y_tree = bool(env_kwargs.get("require_a11y_tree", False))
                    env.require_terminal = bool(env_kwargs.get("require_terminal", False))
                    env.enable_proxy = bool(env_kwargs.get("enable_proxy", False))
                    env.current_use_proxy = False  # adopted container starts proxy-free
                    env.is_environment_used = False  # docker provider starts clean

                    # Wire the provider with the existing container
                    from desktop_env.providers.docker.provider import DockerProvider
                    provider = object.__new__(DockerProvider)
                    provider.client = docker_lib.from_env()
                    provider.container = container
                    provider.server_port = server_port
                    provider.vnc_port = vnc_port
                    provider.chromium_port = chromium_port
                    provider.vlc_port = vlc_port
                    provider.environment = {"DISK_SIZE": "32G", "RAM_SIZE": "4G", "CPU_CORES": "4"}
                    env.provider = provider
                    env.provider_name = "docker"

                    # Re-wire both controllers against the adopted container's
                    # ports. DesktopEnv keeps two distinct controllers:
                    #   - env.controller       = PythonController (obs/step; has get_screenshot)
                    #   - env.setup_controller = SetupController  (task init only)
                    # They must NOT be aliased — reset() calls _get_obs() ->
                    # env.controller.get_screenshot(), which SetupController lacks.
                    from desktop_env.controllers.python import PythonController
                    from desktop_env.controllers.setup import SetupController
                    env.controller = PythonController(
                        vm_ip=env.vm_ip,
                        server_port=env.server_port,
                    )
                    env.setup_controller = SetupController(
                        vm_ip=env.vm_ip,
                        server_port=env.server_port,
                        chromium_port=env.chromium_port,
                        vlc_port=env.vlc_port,
                        cache_dir=env.cache_dir_base,
                        client_password=env.client_password,
                        screen_width=env.screen_width,
                        screen_height=env.screen_height,
                    )

                    adapter = cls(env, action_space=env_kwargs.get("action_space", action_space), pause=pause)
                    adopted.append(adapter)
                    logger.info("Adopted container %s (server:%d)", container.short_id, server_port)
                except Exception:
                    logger.warning("Failed to adopt container %s", container.short_id, exc_info=True)

            return adopted

        return _adopt

    # -- observation translation ------------------------------------------
    def _wrap_obs(self, raw: Dict[str, Any]) -> Observation:
        raw = raw or {}
        modalities: Dict[str, Any] = {
            "screenshot": encode_screenshot(raw.get("screenshot")),
            "accessibility_tree": raw.get("accessibility_tree"),
            "terminal": raw.get("terminal"),
            "instruction": raw.get("instruction"),
        }
        return self.make_observation(modalities, step=self._step, runtime="osworld")

    # -- lifecycle ---------------------------------------------------------
    def reset(self, task_payload: Dict[str, Any]) -> Observation:
        self._step = 0
        self._env.reset(task_config=sanitize_launch_env(task_payload) or None)
        time.sleep(_RESET_SETTLE)
        return self._wrap_obs(self._env._get_obs())

    def step(self, action: Action, pause: float | None = None) -> StepResponse:
        self.validate_action(action)
        if action.kind == ActionKind.TOOL.value and action.type == "cli":
            return self._run_code(action)
        legacy = action_to_legacy(action)
        settle = self._pause if pause is None else pause
        raw, reward, done, info = self._env.step(legacy, settle)
        self._step += 1
        obs = self._wrap_obs(raw)
        return StepResponse(observation=obs, done=bool(done), info=dict(info or {}), reward=float(reward or 0.0))

    def _run_code(self, action: Action) -> StepResponse:
        """Execute coder-produced code in the container via the controller's
        run_python_script (the in-container /run_python endpoint, verified working).

        bash is wrapped in a python subprocess rather than calling the controller's
        run_bash_script: the current image's /run_bash_script is broken — it returns
        ``{"status":"error","output":"Failed to execute script: name '_append_event'
        is not defined","returncode":-1}`` (a bug in the image's server, which we must
        not patch). The subprocess wrapper avoids it and touches no benchmark code.
        If a fixed image ships, this can call run_bash_script directly.

        The result goes in info['exec_result']; observation carries the latest
        screenshot so the next routing step sees the post-exec state. Episode is
        not ended."""
        payload = action.payload or {}
        code = payload.get("code", "")
        lang = (payload.get("lang") or "python").lower()
        if lang in ("bash", "shell", "sh"):
            timeout = self._bash_timeout(payload.get("timeout"))
            code = (
                "import subprocess\n"
                f"_r = subprocess.run({code!r}, shell=True, executable='/bin/bash', "
                # Kept under run_python's 90s HTTP timeout, so the subprocess times out
                # first with a clean TimeoutExpired instead of an HTTP ReadTimeout.
                f"capture_output=True, text=True, timeout={timeout})\n"
                "print(_r.stdout)\n"
                "import sys as _sys\n"
                "print(_r.stderr, file=_sys.stderr)\n"
                # Propagate the command's exit code as the wrapper's own, else the
                # wrapper always exits 0 and the in-VM server reports status="success"
                # for every command (main.py derives status from the wrapper's rc).
                "_sys.exit(_r.returncode)"
            )
        res = self._env.controller.run_python_script(code) or {}
        self._step += 1
        # Settle before the screenshot, mirroring DesktopEnv.step's post-action
        # sleep(pause): code that touches the GUI (opens a file, changes a window)
        # needs a beat to stabilize, else _get_obs() may capture a mid-transition
        # frame. Same self._pause the GUI step path uses.
        time.sleep(self._pause)
        obs = self._wrap_obs(self._env._get_obs())
        return StepResponse(
            observation=obs,
            done=False,
            reward=0.0,
            info={"exec_result": {
                "status": res.get("status"),
                "output": res.get("output"),
                "error": res.get("error"),
                "return_code": res.get("return_code"),
            }},
        )

    @staticmethod
    def _bash_timeout(requested: Any) -> int:
        """Clamp a caller-requested bash timeout into the sandbox's usable range.

        Non-numeric / non-positive input falls back to the default rather than
        raising: a malformed timeout should not abort the agent's command.
        """
        try:
            timeout = int(float(requested))
        except (TypeError, ValueError):
            return _BASH_TIMEOUT_DEFAULT
        if timeout <= 0:
            return _BASH_TIMEOUT_DEFAULT
        if timeout > _BASH_TIMEOUT_MAX:
            logger.info(
                "bash timeout %ss exceeds the sandbox ceiling; clamping to %ss",
                timeout, _BASH_TIMEOUT_MAX,
            )
            return _BASH_TIMEOUT_MAX
        return timeout

    def observe(self) -> Observation:
        raw = self._env._get_obs()
        return self._wrap_obs(raw)

    def evaluate(self) -> EvaluationResult:
        score = float(self._env.evaluate())
        return EvaluationResult(
            score=score,
            success=score >= 1.0,
            metrics={"raw_score": score},
            reason="osworld evaluator",
        )

    def health_check(self) -> bool:
        if self._closed:
            return False
        env = self._env
        ip = getattr(env, "vm_ip", None)
        port = getattr(env, "server_port", None)
        if not ip or not port:
            return False
        try:
            # /platform is the cheapest real endpoint: the in-container server
            # returns platform.system() with zero VM interaction, so a 200 proves
            # container + server + HTTP link are alive. Single shot with a short
            # timeout — never retry/block: health_check runs under the pool lock.
            return requests.get(f"http://{ip}:{port}/platform", timeout=2).ok
        except Exception:  # noqa: BLE001 - a probe must never raise
            return False

    def liveness(self) -> bool:
        """Deep pre-reuse check, authoritative on the DOCKER CONTAINER state.

        The container is the source of truth: only a dead container (exited /
        gone) makes a slot unreusable. A transient HTTP ``/platform`` blip on a
        *still-running* container must NOT condemn the slot — that误杀 would
        destroy a perfectly reusable warm env and force a costly cold rebuild.
        So: container not running -> False; container running -> True even if the
        HTTP probe is momentarily unreachable (it will recover, and reset() will
        re-establish state). For non-docker providers there's no container to
        check, so fall back to the HTTP health probe. Docker state is TTL-cached
        so repeated reuse probes don't fork ``docker``."""
        if self._closed:
            return False
        state = self._container_state_cached()
        if state is None:
            # No container to consult (e.g. vmware provider): defer to HTTP probe.
            return self.health_check()
        # Reusable iff running AND not unhealthy (a wedged-but-running container
        # must not be handed to a new session). Matches docker_util's judgement.
        return state == "running"

    def _container_state_cached(self) -> "str | None":
        """Cached raw docker container state string: ``"running"`` / ``"exited"``
        / ``"unhealthy"`` / other, or None (no container — non-docker provider).
        One probe feeds both liveness (reuse decision) and get_info (display)."""
        ttl = float(os.environ.get("OSWORLD_LIVENESS_CACHE_TTL", "2"))
        now = time.time()
        if self._container_alive_cache and now - self._container_alive_cache[0] < ttl:
            return self._container_alive_cache[1]
        state = self._container_state()
        self._container_alive_cache = (now, state)
        return state

    def _container_state(self) -> "str | None":
        provider = getattr(self._env, "provider", None)
        container = getattr(provider, "container", None) if provider else None
        if container is None:
            return None  # non-docker provider (e.g. vmware): nothing to check
        from cluster.worlds.base import docker_util

        return docker_util.container_state(container)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._env.close()
        finally:
            self._closed = True

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        vnc_port = getattr(self._env, "vnc_port", None)
        info.update(
            {
                "world_id": "osworld",
                "action_space": self._action_space,
                "provider_name": getattr(self._env, "provider_name", None),
                "os_type": getattr(self._env, "os_type", None),
                "vm_ip": getattr(self._env, "vm_ip", None),
                "server_port": getattr(self._env, "server_port", None),
                "chromium_port": getattr(self._env, "chromium_port", None),
                "vlc_port": getattr(self._env, "vlc_port", None),
            }
        )
        if vnc_port:
            info["live_view"] = {"protocol": "vnc", "port": vnc_port}
        provider = getattr(self._env, "provider", None)
        container = getattr(provider, "container", None) if provider else None
        if container is not None:
            info["container_id"] = (getattr(container, "id", None) or "")[:12]
            info["container_name"] = getattr(container, "name", None)
            # Self-report the raw container state so the dashboard's per-slot
            # display works WITHOUT the neutral node ever running `docker ps`.
            # Reuses the same TTL-cached probe as liveness() — running/exited/
            # unhealthy passed through verbatim for the frontend to colour.
            state = self._container_state_cached()  # str | None
            if state is not None:
                info["docker_state"] = state
        return info

    def resource_id(self) -> "str | None":
        """Docker container id (12-char) — the reclaimable resource this slot holds.
        Matches the id space used by the orphan reclaimer's docker-ps scan."""
        provider = getattr(self._env, "provider", None)
        container = getattr(provider, "container", None) if provider else None
        cid = getattr(container, "id", None) if container is not None else None
        return cid[:12] if cid else None
