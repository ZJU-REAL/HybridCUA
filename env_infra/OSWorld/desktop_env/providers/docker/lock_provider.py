"""Fast Docker provider with non-overlapping port ranges and minimal locking.

Solves two problems with the original DockerProvider:
1. Port ranges overlap (VNC 8006+ collides with VLC 8080+ at 74 containers)
2. File lock held during container creation (seconds), causing timeout for other threads

This module provides:
- PortAllocator: atomic counter-based port allocation (<1ms lock hold)
- FastDockerProvider: subclass of DockerProvider with lock-free container creation
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from filelock import FileLock

from desktop_env.providers.docker.provider import DockerProvider

logger = logging.getLogger("desktopenv.providers.docker.FastDockerProvider")

CONTAINER_IMAGE = os.getenv("OSWORLD_DOCKER_IMAGE", "happysixd/osworld-docker")

# Non-overlapping port ranges — each service gets 1000 ports
PORT_RANGES = {
    "server": 5000,    # 5000-5999
    "vnc": 6000,       # 6000-6999
    "vlc": 7000,       # 7000-7999
    "chromium": 9000,  # 9000-9999
}
MAX_OFFSET = 1000
START_ATTEMPTS = int(os.getenv("OSWORLD_DOCKER_START_ATTEMPTS", "8"))

REGISTRY_PATH = Path("/tmp/docker_port_registry.json")
LOCK_PATH = Path("/tmp/docker_port_allocation.lck")


class PortAllocator:
    """Thread/process-safe port allocator using file-based atomic counter.

    Lock is held only for JSON read+write (<1ms), not during container creation.
    """

    def __init__(
        self,
        registry_path: Path = REGISTRY_PATH,
        lock_path: Path = LOCK_PATH,
    ):
        self._registry_path = registry_path
        self._lock = FileLock(str(lock_path), timeout=30)
        self._initialized = False

    def _ensure_initialized(self, used_ports: set[int]) -> None:
        """On first use, initialize counters from already-collected used ports."""
        if self._initialized:
            return
        self._initialized = True
        if self._registry_path.exists():
            return
        offsets = {}
        for name, base in PORT_RANGES.items():
            max_offset = 0
            for port in used_ports:
                if base <= port < base + MAX_OFFSET:
                    max_offset = max(max_offset, port - base + 1)
            offsets[name] = max_offset
        self._write_offsets(offsets)
        logger.info("Port registry initialized from system: %s", offsets)

    def _collect_used_ports(self) -> set[int]:
        used_ports: set[int] = set()
        try:
            import docker
            client = docker.from_env()
            for container in client.containers.list():
                ports = container.attrs.get("NetworkSettings", {}).get("Ports") or {}
                for bindings in ports.values():
                    if bindings:
                        for b in bindings:
                            used_ports.add(int(b["HostPort"]))
        except Exception as exc:
            logger.warning("Failed to collect Docker ports: %s", exc)

        try:
            import psutil
            used_ports.update(conn.laddr.port for conn in psutil.net_connections() if conn.laddr)
        except Exception as exc:
            logger.warning("Failed to collect system ports: %s", exc)
        return used_ports

    @staticmethod
    def _normalize_offset(value: int | str | None) -> int:
        try:
            return int(value or 0) % MAX_OFFSET
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _bundle_for_offset(offset: int) -> dict[str, int]:
        return {name: base + offset for name, base in PORT_RANGES.items()}

    def allocate(self) -> dict[str, int]:
        """Allocate 4 non-conflicting ports. Returns {"vnc": N, "server": N, ...}."""
        used_ports = self._collect_used_ports()
        with self._lock:
            self._ensure_initialized(used_ports)
            offsets = self._read_offsets()
            start_offset = self._normalize_offset(offsets.get("server", 0))
            for step in range(MAX_OFFSET):
                offset = (start_offset + step) % MAX_OFFSET
                ports = self._bundle_for_offset(offset)
                if any(port in used_ports for port in ports.values()):
                    continue
                next_offset = (offset + 1) % MAX_OFFSET
                self._write_offsets({name: next_offset for name in PORT_RANGES})
                return ports
        raise RuntimeError("No available Docker port bundle")

    def sync_from_containers(self, port_mappings: list[dict[str, int]]) -> None:
        """After adopting existing containers, advance counters past used ports."""
        if not port_mappings:
            return
        with self._lock:
            offsets = self._read_offsets()
            for mapping in port_mappings:
                for name, base in PORT_RANGES.items():
                    port = mapping.get(name, 0)
                    if port >= base:
                        used_offset = port - base + 1
                        if used_offset > offsets.get(name, 0):
                            offsets[name] = used_offset
            self._write_offsets(offsets)
        logger.info("Port registry synced: %s", offsets)

    def _read_offsets(self) -> dict[str, int]:
        if self._registry_path.exists():
            try:
                return json.loads(self._registry_path.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _write_offsets(self, offsets: dict[str, int]) -> None:
        self._registry_path.write_text(json.dumps(offsets))


_allocator: PortAllocator | None = None


def get_port_allocator() -> PortAllocator:
    global _allocator
    if _allocator is None:
        _allocator = PortAllocator()
    return _allocator


class FastDockerProvider(DockerProvider):
    """DockerProvider with fast port allocation and lock-free container creation.

    Differences from DockerProvider:
    - Ports allocated via atomic registry; Docker/system scans happen outside the file lock
    - Lock held <1ms (counter read/write only)
    - Container creation happens outside the lock (parallel-safe)
    - Port ranges don't overlap (supports 1000 containers per range)
    """

    @staticmethod
    def _is_port_conflict(exc: Exception) -> bool:
        text = str(exc).lower()
        return "address already in use" in text or "port is already allocated" in text

    def _clear_ports(self) -> None:
        self.vnc_port = None
        self.server_port = None
        self.chromium_port = None
        self.vlc_port = None

    def _assign_ports(self, ports: dict[str, int]) -> None:
        self.vnc_port = ports["vnc"]
        self.server_port = ports["server"]
        self.chromium_port = ports["chromium"]
        self.vlc_port = ports["vlc"]

    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str):
        # 1. Check KVM once; port-conflict retries only change host port bindings.
        devices = []
        if os.path.exists("/dev/kvm"):
            devices.append("/dev/kvm")
            logger.info("KVM device found, using hardware acceleration")
        else:
            self.environment["KVM"] = "N"
            logger.warning("KVM device not found, running without hardware acceleration")

        for attempt in range(1, START_ATTEMPTS + 1):
            self._assign_ports(get_port_allocator().allocate())

            # 2. Create container — outside lock, multiple threads can do this in parallel
            try:
                self.container = self.client.containers.run(
                    CONTAINER_IMAGE,
                    environment=self.environment,
                    cap_add=["NET_ADMIN"],
                    devices=devices,
                    volumes={
                        os.path.abspath(path_to_vm): {
                            "bind": "/System.qcow2",
                            "mode": "ro",
                        }
                    },
                    ports={
                        "8006/tcp": ("0.0.0.0", self.vnc_port),
                        "5000/tcp": ("0.0.0.0", self.server_port),
                        "9222/tcp": ("0.0.0.0", self.chromium_port),
                        "8080/tcp": ("0.0.0.0", self.vlc_port),
                    },
                    detach=True,
                )
                break
            except Exception as e:
                self.container = None
                self._clear_ports()
                if self._is_port_conflict(e) and attempt < START_ATTEMPTS:
                    logger.warning(
                        "Docker port conflict on attempt %d/%d; retrying with new ports: %s",
                        attempt,
                        START_ATTEMPTS,
                        e,
                    )
                    continue
                raise

        logger.info(
            "Started container with ports - VNC: %d, Server: %d, Chrome: %d, VLC: %d",
            self.vnc_port, self.server_port, self.chromium_port, self.vlc_port,
        )

        # 3. Wait for VM to be ready — also outside lock
        self._wait_for_vm_ready()
