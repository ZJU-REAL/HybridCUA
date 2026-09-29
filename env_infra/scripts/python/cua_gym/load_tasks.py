"""CLI + re-export for the CUA-Gym task loader.

The loader itself lives in the client package (``cluster.client.cua_gym.tasks``)
so ``eval.py`` can import it with a stable absolute path. This thin module
re-exports it and keeps a ``python load_tasks.py <bundle_or_root>`` smoke entry.
"""
from __future__ import annotations

import json
import os
import sys

# Allow running this file directly: put the env_infra root on sys.path so
# ``cluster`` imports resolve without requiring PYTHONPATH to be set.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from cluster.client.cua_gym.tasks import load_task, load_tasks  # noqa: E402,F401

__all__ = ["load_task", "load_tasks"]


if __name__ == "__main__":
    from pathlib import Path

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if target and (target / "task.json").exists():
        t = load_task(target)
        print(json.dumps({k: v for k, v in t.items() if k != "reward_code"}, indent=2, ensure_ascii=False))
        print("reward_code:", len(t["reward_code"] or ""), "chars")
    elif target and target.is_dir():
        tasks = load_tasks(target)
        print(f"loaded {len(tasks)} tasks; first id={tasks[0]['id']!r}" if tasks else "no tasks found")
    else:
        print("usage: python load_tasks.py <bundle_dir | tasks_root>")
