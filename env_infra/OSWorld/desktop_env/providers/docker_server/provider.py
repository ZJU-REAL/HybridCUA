from __future__ import annotations

import base64
import logging
import os
import threading
import time
from urllib.parse import urlparse

import requests

from cluster.protocol.codec import action_from_legacy
from cluster.utils.debug_events import record_exception
from desktop_env.providers.base import Provider

logger = logging.getLogger("desktopenv.providers.docker_server.DockerServerProvider")
logger.setLevel(logging.INFO)

ALLOCATE_RETRIES = 10
ALLOCATE_RETRY_INTERVAL = 3
HEARTBEAT_INTERVAL = float(os.environ.get("DOCKER_SERVER_HEARTBEAT_INTERVAL", "30"))


class DockerServerProvider(Provider):

    supports_remote_ops = True

    def __init__(self, region: str = None, server_url: str | None = None):
        super().__init__(region)
        self.server_url = (
            server_url
            or os.environ.get("GUI_ENV_SERVER_URL")
            or "http://127.0.0.1:18080"
        ).rstrip("/")
        parsed = urlparse(self.server_url)
        self.server_host = parsed.hostname or "127.0.0.1"

        self.node_url: str | None = None

        self.lease_id: str | None = None
        self.env_id: str | None = None
        self.vm_ip: str | None = None
        self.server_port: int | None = None
        self.chromium_port: int | None = None
        self.vnc_port: int | None = None
        self.vlc_port: int | None = None
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    @staticmethod
    def _decode_obs(observation: dict) -> dict:
        # The node speaks the neutral protocol: the observation is
        # ``{"modalities": {...}, "artifacts": [...], "metadata": {...}}`` and
        # the screenshot lives at ``modalities["screenshot"]`` as a base64 str.
        # RemoteDesktopEnv expects a FLAT dict with raw ``screenshot`` bytes and
        # the other modalities (accessibility_tree/terminal/instruction/...) at
        # the top level, so we translate here (OSWorld-specific, client-side).
        observation = observation or {}
        modalities = observation.get("modalities", observation) or {}
        out: dict = {k: v for k, v in modalities.items() if k != "screenshot"}
        b64 = modalities.get("screenshot")
        out["screenshot"] = base64.b64decode(b64) if b64 else None
        return out

    def _data_base_url(self) -> str:
        # Prefer the allocated node for data-path calls once the master has
        # returned node_url. Before allocation, or in standalone-node mode, use
        # the configured server URL.
        return self.node_url or self.server_url

    def _record_post_exception(
        self,
        exc: BaseException,
        *,
        type: str,
        path: str,
        base: str,
        payload: dict,
        timeout: float,
        response: requests.Response | None = None,
    ) -> None:
        record_exception(
            exc,
            type=type,
            service="runner",
            component="docker_server_provider._post",
            path=path,
            status_code=getattr(response, "status_code", None),
            lease_id=self.lease_id or payload.get("lease_id"),
            env_id=self.env_id,
            node_url=base,
            response_excerpt=getattr(response, "text", None),
            timeout_seconds=timeout,
            body_keys=sorted(str(k) for k in payload.keys()),
            stacklevel=2,
        )

    def _request(self, method: str, path: str, json: dict | None = None, timeout: float = 60, use_master: bool = False) -> dict:
        base = self.server_url if use_master else self._data_base_url()
        payload = json or {}
        resp: requests.Response | None = None
        try:
            resp = requests.request(method, f"{base}{path}", json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("ok"):
                raise RuntimeError(f"Server {path} failed: {data.get('error', data)}")
            return data
        except requests.exceptions.Timeout as exc:
            self._record_post_exception(exc, type="timeout", path=path, base=base, payload=payload, timeout=timeout)
            raise
        except requests.exceptions.HTTPError as exc:
            self._record_post_exception(
                exc,
                type="http_non_2xx",
                path=path,
                base=base,
                payload=payload,
                timeout=timeout,
                response=exc.response if exc.response is not None else resp,
            )
            raise
        except requests.exceptions.RequestException as exc:
            self._record_post_exception(
                exc,
                type="exception",
                path=path,
                base=base,
                payload=payload,
                timeout=timeout,
                response=resp,
            )
            raise
        except (RuntimeError, ValueError) as exc:
            self._record_post_exception(
                exc,
                type="exception",
                path=path,
                base=base,
                payload=payload,
                timeout=timeout,
                response=resp,
            )
            raise

    def _post(self, path: str, json: dict | None = None, timeout: float = 60, use_master: bool = False) -> dict:
        return self._request("POST", path, json=json, timeout=timeout, use_master=use_master)

    def _delete(self, path: str, timeout: float = 60, use_master: bool = False) -> dict:
        return self._request("DELETE", path, json={}, timeout=timeout, use_master=use_master)

    def _allocate(self) -> None:
        last_err = None
        alloc_body = {
            "runtime": os.environ.get("OSWORLD_RUNTIME", "osworld"),
            "user_id": os.environ.get("OSWORLD_USER_ID", "anonymous"),
            "task_type": os.environ.get("OSWORLD_TASK_TYPE", "evaluation"),
            "mode": "eval",
        }
        for attempt in range(1, ALLOCATE_RETRIES + 1):
            try:
                data = self._post("/v1/sessions", alloc_body, use_master=True)
                sessions = data.get("sessions", [])
                if not sessions:
                    raise RuntimeError(data.get("error", "no session returned"))
                sess = sessions[0]
                self.lease_id = sess["session_id"]
                self.env_id = sess.get("env_id") or self.lease_id

                if "node_url" in sess and sess["node_url"]:
                    self.node_url = sess["node_url"].rstrip("/")
                    node_parsed = urlparse(self.node_url)
                    self.server_host = node_parsed.hostname or self.server_host

                self.vm_ip = self.server_host
                self.server_port = sess.get("server_port")
                self.chromium_port = sess.get("chromium_port")
                self.vnc_port = sess.get("vnc_port")
                self.vlc_port = sess.get("vlc_port")
                self._start_heartbeat()
                logger.info(
                    "Allocated session %s (env %s) via %s node_url=%s",
                    self.lease_id, self.env_id, self.server_url, self.node_url,
                )
                return
            except Exception as exc:
                last_err = exc
                logger.warning(
                    "Allocate attempt %d/%d failed: %s", attempt, ALLOCATE_RETRIES, exc,
                )
                time.sleep(ALLOCATE_RETRY_INTERVAL)
        raise RuntimeError(f"Failed to allocate after {ALLOCATE_RETRIES} attempts: {last_err}")

    def _release(self) -> None:
        if self.lease_id is None:
            return
        self._stop_heartbeat()
        try:
            self._delete(f"/v1/sessions/{self.lease_id}", timeout=120, use_master=True)
            logger.info("Released session %s (env %s)", self.lease_id, self.env_id)
        except Exception as exc:
            logger.warning("Failed to release session %s: %s", self.lease_id, exc)
        finally:
            self.lease_id = None
            self.env_id = None
            self.node_url = None
            self.vm_ip = None
            self.server_port = None
            self.chromium_port = None
            self.vnc_port = None
            self.vlc_port = None

    def _start_heartbeat(self) -> None:
        if HEARTBEAT_INTERVAL <= 0 or self.lease_id is None:
            return
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"DockerServerHeartbeat-{self.lease_id}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        thread = self._heartbeat_thread
        if thread and thread.is_alive():
            thread.join(timeout=1)
        self._heartbeat_thread = None

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(HEARTBEAT_INTERVAL):
            session_id = self.lease_id
            if session_id is None:
                return
            try:
                self._post(f"/v1/sessions/{session_id}/heartbeat", {}, timeout=10, use_master=True)
            except Exception as exc:
                logger.debug("Session heartbeat failed for %s: %s", session_id, exc)

    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str = "Ubuntu"):
        self._allocate()

    def get_ip_address(self, path_to_vm: str) -> str:
        if self.server_port is None:
            raise RuntimeError("Remote docker_server provider does not expose direct VM ports")
        host = self.server_host
        return f"{host}:{self.server_port}:{self.chromium_port}:{self.vnc_port}:{self.vlc_port}"

    def start_recording(self):
        # The neutral /v1/sessions protocol has no recording op. Recording is a
        # world-specific capability that the OSWorld node no longer exposes on
        # this surface, so this is a no-op (RemoteDesktopEnv ignores the return).
        if self.lease_id is None:
            raise RuntimeError("No active lease — call start_emulator first")
        logger.warning("start_recording is a no-op on /v1/sessions (no recording op)")

    def end_recording(self, out_path: str):
        if self.lease_id is None:
            raise RuntimeError("No active lease — call start_emulator first")
        logger.warning(
            "end_recording is a no-op on /v1/sessions; no video written to %s", out_path,
        )

    def reset(self, task_config: dict | None = None) -> dict:
        if self.lease_id is None:
            raise RuntimeError("No active lease — call start_emulator first")
        data = self._post(
            f"/v1/sessions/{self.lease_id}/reset",
            {"task_payload": task_config or {}},
            timeout=300,
        )
        return self._decode_obs(data.get("observation", {}))

    def evaluate(self) -> float:
        if self.lease_id is None:
            raise RuntimeError("No active lease — call start_emulator first")
        data = self._post(
            f"/v1/sessions/{self.lease_id}/evaluate",
            {},
            timeout=120,
        )
        return float(data.get("score", 0.0))

    def get_obs(self) -> dict:
        if self.lease_id is None:
            raise RuntimeError("No active lease — call start_emulator first")
        data = self._post(
            f"/v1/sessions/{self.lease_id}/observe",
            {},
            timeout=60,
        )
        return self._decode_obs(data.get("observation", {}))

    def step(self, action, sleep_after_execution: float = 0.0) -> dict:
        if self.lease_id is None:
            raise RuntimeError("No active lease — call start_emulator first")
        # OSWorld agents emit a pyautogui code string / a computer_13 dict / a
        # bare WAIT|DONE|FAIL token. Wrap it in the neutral typed Action here
        # (OSWorld-specific translation belongs on the client, not the node).
        typed_action = action_from_legacy(action).to_dict()
        data = self._post(
            f"/v1/sessions/{self.lease_id}/step",
            {"action": typed_action},
            timeout=120,
        )
        observation = data.get("observation", {})
        # The neutral /step has no `sleep_after_execution`; reproduce the old
        # "let the screen settle, then re-capture" semantics on the client by
        # sleeping and re-observing.
        if sleep_after_execution and sleep_after_execution > 0:
            time.sleep(sleep_after_execution)
            try:
                obs_data = self._post(
                    f"/v1/sessions/{self.lease_id}/observe",
                    {},
                    timeout=60,
                )
                observation = obs_data.get("observation", observation)
            except Exception as exc:  # settle re-observe is best-effort
                logger.warning("Post-step re-observe failed for %s: %s", self.lease_id, exc)
        return {
            "observation": self._decode_obs(observation),
            "reward": data.get("reward", 0),
            "done": bool(data.get("done", False)),
            "info": data.get("info", {}) or {},
        }

    def run_bash_script(
        self,
        script: str,
        timeout: int = 30,
        working_dir: str | None = None,
    ) -> dict:
        # Not part of the neutral /v1/sessions protocol. Only coder/executor
        # agents call this; standard pyautogui evaluation never does.
        raise NotImplementedError(
            "run_bash_script is not supported on the /v1/sessions protocol; "
            "use a world that exposes a shell tool"
        )

    def save_state(self, path_to_vm: str, snapshot_name: str):
        raise NotImplementedError("Snapshots not supported for docker_server provider")

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str):
        self._release()

    def stop_emulator(self, path_to_vm: str, region=None, *args, **kwargs):
        self._release()
