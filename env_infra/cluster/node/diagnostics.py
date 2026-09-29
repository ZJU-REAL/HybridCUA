"""Host resource diagnostics — psutil only, docker-free.

The world-neutral node layer never inspects docker (no ``docker ps``); docker
scanning/removal lives in the world-side shared helper
``cluster/worlds/base/docker_util.py``, used only by docker-driver adapters.
This module keeps only host-level psutil metrics: CPU/mem/swap/disk/load and
the qemu process counters.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

try:
    import psutil
except ModuleNotFoundError:
    class _MissingPsutil:
        STATUS_DISK_SLEEP = "disk-sleep"

        @staticmethod
        def cpu_count() -> int:
            return 1

        @staticmethod
        def process_iter(_attrs=None) -> list[Any]:
            return []

        @staticmethod
        def cpu_percent(interval=None) -> None:
            return None

        @staticmethod
        def virtual_memory() -> Any:
            return type("Usage", (), {"percent": None})()

        @staticmethod
        def swap_memory() -> Any:
            return type("Usage", (), {"percent": None})()

        @staticmethod
        def disk_usage(_path: str) -> Any:
            return type("Usage", (), {"percent": None})()

    psutil = _MissingPsutil()

logger = logging.getLogger("cluster.diagnostics")


def _round(value: float | int | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _safe_percent(obj: Any) -> float | None:
    try:
        return _round(obj.percent, 2)
    except Exception:
        return None


def _is_qemu_process(proc_info: dict[str, Any]) -> bool:
    name = str(proc_info.get("name") or "").lower()
    if name.startswith("qemu"):
        return True
    cmdline = proc_info.get("cmdline") or []
    try:
        cmd = " ".join(str(p).lower() for p in cmdline)
    except Exception:
        cmd = ""
    return "qemu-system" in cmd or "qemu-kvm" in cmd


def sample_resources() -> dict[str, Any]:
    """Collect host resource counters without shelling out."""
    load1 = load5 = load15 = None
    try:
        load1, load5, load15 = os.getloadavg()
    except (AttributeError, OSError):
        pass

    cores = psutil.cpu_count() or 1
    qemu_count = 0
    qemu_d_state_count = 0
    disk_sleep = getattr(psutil, "STATUS_DISK_SLEEP", "disk-sleep")
    try:
        for proc in psutil.process_iter(["name", "status", "cmdline"]):
            info = getattr(proc, "info", {}) or {}
            if not _is_qemu_process(info):
                continue
            qemu_count += 1
            if info.get("status") == disk_sleep:
                qemu_d_state_count += 1
    except Exception as exc:
        logger.debug("Failed to inspect processes for qemu diagnostics: %s", exc)

    try:
        cpu_percent = _round(psutil.cpu_percent(interval=None), 2)
    except Exception:
        cpu_percent = None

    try:
        mem_percent = _safe_percent(psutil.virtual_memory())
    except Exception:
        mem_percent = None

    try:
        swap_percent = _safe_percent(psutil.swap_memory())
    except Exception:
        swap_percent = None

    try:
        disk_root_percent = _safe_percent(psutil.disk_usage("/"))
    except Exception:
        disk_root_percent = None

    return {
        "sampled_ts": time.time(),
        "cpu_percent": cpu_percent,
        "cpu_count": cores,
        "load1": _round(load1, 3),
        "load5": _round(load5, 3),
        "load15": _round(load15, 3),
        "load_per_core": _round(load1 / cores, 3) if load1 is not None else None,
        "mem_percent": mem_percent,
        "swap_percent": swap_percent,
        "disk_root_percent": disk_root_percent,
        "qemu_count": qemu_count,
        "qemu_d_state_count": qemu_d_state_count,
    }
