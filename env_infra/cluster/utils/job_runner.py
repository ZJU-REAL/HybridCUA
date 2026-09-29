"""Shared job execution logic for node and master servers."""
from __future__ import annotations

import os
import subprocess
import threading
from typing import Callable

MAX_LOG_LINES = 5000


def run_job(
    *,
    script_path: str,
    env: dict[str, str] | None,
    cwd: str,
    log_list: list[str],
    log_lock: threading.RLock,
    on_finish: Callable[[int | None], None] | None = None,
) -> None:
    """Run a bash script in a subprocess, streaming output to log_list.

    Args:
        script_path: Path to the temporary bash script file (deleted after execution).
        env: Environment variables for the subprocess.
        cwd: Working directory for the subprocess.
        log_list: Mutable list to append output lines to.
        log_lock: Lock protecting log_list mutations.
        on_finish: Optional callback invoked with exit code (or None on error).
    """
    exit_code = None
    try:
        proc = subprocess.Popen(
            ["bash", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=cwd,
            preexec_fn=os.setsid,
        )
        for raw_line in iter(proc.stdout.readline, b""):
            line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
            with log_lock:
                log_list.append(line)
                if len(log_list) > MAX_LOG_LINES:
                    del log_list[:len(log_list) - MAX_LOG_LINES]
        proc.wait()
        exit_code = proc.returncode
    except Exception as exc:
        with log_lock:
            log_list.append(f"[ERROR] {exc}")
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass
        if on_finish:
            on_finish(exit_code)
