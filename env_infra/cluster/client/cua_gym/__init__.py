"""client.cua_gym: CUA-Gym benchmark client + eval task source.

CUA-Gym reuses OSWorld's VM and protocol; these adapt only the runtime name and
the task model (bundle dirs + inlined python reward). See tasks.py / eval.py /
session_client.py for details.
"""
from __future__ import annotations

from cluster.client.cua_gym.eval import CuaGymEvalSource
from cluster.client.cua_gym.session_client import CuaGymRemoteClient, CuaGymSessionClient
from cluster.client.cua_gym.tasks import load_task, load_tasks

__all__ = [
    "CuaGymSessionClient",
    "CuaGymRemoteClient",
    "CuaGymEvalSource",
    "load_task",
    "load_tasks",
]
