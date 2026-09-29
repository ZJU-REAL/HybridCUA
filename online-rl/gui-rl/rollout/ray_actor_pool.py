"""Ray-actor rollout pool: one long-lived actor per slot, one trajectory at a time.

Keeps per-trajectory synchronous CPU work (tokenize, image base64,
``build_train_data``) off the RolloutManager event loop. Results go back through
plasma (zero-copy on the local node) instead of a single result pipe, which is
what keeps big multimodal payloads off the critical path — so plasma must fit the
in-flight results (``OBJECT_STORE_GB``) or it spills.

Pool size is ``GUI_FAST_ROLLOUT_PROCS`` (``config.rollout_pool_size``).
"""

from __future__ import annotations

import asyncio
import logging
import os

import ray

from slime.utils.async_utils import run
from slime.utils.types import Sample

import config

logger = logging.getLogger(__name__)

_env_client = None


def _get_env_client():
    """Per-process singleton env client (httpx connection pool reused across tasks).

    The lease is still allocate -> reset -> close per trajectory.
    """
    global _env_client
    if _env_client is None:
        base_url = config.env_server_url()
        if config.env_client_kind() == "legacy":
            from env_client import GuiEnvClient

            _env_client = GuiEnvClient(base_url)
        else:
            from clients import SessionGuiEnvClient

            _env_client = SessionGuiEnvClient(base_url)
        logger.info("rollout worker env client = %s (%s)", type(_env_client).__name__, base_url)
    return _env_client


def _worker_init(args) -> None:
    """Prepare a worker: slime HTTP client + env client.

    ``use_distributed_post`` is forced off — the worker does its own POSTs;
    routing them back through Ray actors would re-add the hop we removed.
    """
    os.environ[config.ROLLOUT_WORKER_ENV_FLAG] = "1"

    from slime.utils.http_utils import init_http_client

    try:
        args.use_distributed_post = False
    except AttributeError:  # frozen/namespace args — nothing to override
        pass
    init_http_client(args)
    _get_env_client()


def _run_one(args, payload):
    """Run one trajectory to completion on this worker's event loop."""
    sample, sampling_params, evaluation = payload
    from rollout.trajectory_runner import run_trajectory

    return run(run_trajectory(args, sample, sampling_params, evaluation))


@ray.remote
class _RolloutActor:
    """One long-lived worker == one pool slot."""

    def __init__(self, args) -> None:
        self._args = args
        _worker_init(args)  # http client (no distributed post) + env client

    def run_one(self, payload):
        return _run_one(self._args, payload)  # returns Sample | list[Sample]


class RayActorPool:
    """N actors + a free-actor queue: exactly N concurrent, work-stealing reuse."""

    _instance: "RayActorPool | None" = None

    def __init__(self, args) -> None:
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        n = config.rollout_pool_size()
        # Forward the full env (GUI_* knobs, env-server url, cuDNN LD_LIBRARY_PATH).
        runtime_env = {"env_vars": dict(os.environ)}
        # Co-locate with this (RolloutManager) node so plasma reads are zero-copy.
        sched = NodeAffinitySchedulingStrategy(ray.get_runtime_context().get_node_id(), soft=True)
        Actor = _RolloutActor.options(
            num_cpus=config.ray_actor_cpus(), runtime_env=runtime_env, scheduling_strategy=sched
        )

        logger.info("starting rollout actor pool: %d actors", n)
        self._actors = [Actor.remote(args) for _ in range(n)]
        self._free: asyncio.Queue = asyncio.Queue()
        for a in self._actors:
            self._free.put_nowait(a)

    @classmethod
    def get(cls, args) -> "RayActorPool":
        if cls._instance is None:
            cls._instance = cls(args)
        return cls._instance

    async def submit(self, sample: Sample, sampling_params: dict, evaluation: bool = False):
        payload = (sample, dict(sampling_params), evaluation)
        actor = await self._free.get()  # blocks (backpressure) until a worker is free
        try:
            return await actor.run_one.remote(payload)  # ObjectRef awaited; result via plasma
        finally:
            self._free.put_nowait(actor)

    def shutdown(self) -> None:
        for a in self._actors:
            ray.kill(a)
        type(self)._instance = None
