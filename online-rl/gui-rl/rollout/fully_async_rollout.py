"""Fully-async rollout for slime.

Decouples ``max_concurrent_tasks`` from ``rollout_batch_size``: a background
asyncio worker keeps a fixed pool of in-flight trajectories across rollout
boundaries, so the next training step doesn't have to wait for the slowest
in-flight sample to finish.

Use with ``--rollout-function-path rollout.fully_async_rollout.generate_rollout_fully_async``.
Plug in per-sample logic via ``--custom-generate-function-path`` and
per-sample reward via ``--custom-rm-path`` — the worker calls slime's stock
:func:`generate_and_rm_group` which dispatches to those.

Concurrency is sourced from ``args.sglang_server_concurrency`` and scaled by
the number of sglang engines to match the per-sample semaphore cap in
:mod:`slime.rollout.sglang_rollout`.

The worker is intentionally oblivious to slime's higher-level pause /
weight-update signalling (e.g. ``GenerateState.aborted``). Each in-flight
generation short-circuits on those signals on its own and surfaces
:data:`Sample.Status.ABORTED`; the only piece the worker owns is
**redirecting ABORTED groups back to ``data_buffer``** instead of shipping
them to training, so the next rollout (with refreshed weights) can pick
them up.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path

import requests

from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from slime.utils.async_utils import run
from slime.utils.http_utils import get_rollout_num_engines
from slime.utils.types import Sample

import config

__all__ = [
    "AsyncRolloutWorker",
    "generate_rollout_fully_async",
    "eval_rollout_fully_async",
]

logger = logging.getLogger("slime.rollout.fully_async")


# ===================== FIX: nested-group flatten helper =====================
# dynamic_history (and other fan-out custom-generate paths) make a group be
# list[list[Sample]] instead of list[Sample]. Callers that do getattr(s,...)
# on a bare list hit list.index (a builtin method, not a field) -> TypeError.
# Flatten to real Samples before reading any field.
def _flatten_samples(obj):
    """Yield every Sample in a possibly-nested group (list[Sample] | list[list[Sample]])."""
    if isinstance(obj, Sample):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _flatten_samples(item)
# ===========================================================================


# ===================== STALENESS FILTER (consumer-side) =====================
# Drop groups whose birth version is too far behind the current engine version.
# Mirrors ROLL's get_batch min_step skip. Disabled when max_staleness is None.
def _group_head_version(group) -> int | None:
    """Group birth version = min over all samples' weight_versions (oldest step)."""
    versions = [int(v) for s in _flatten_samples(group) for v in (s.weight_versions or [])]
    return min(versions) if versions else None


def _fetch_current_version(args) -> int | None:
    """Live engine version via a 1-token /generate probe.

    The router forwards /generate but not /get_weight_version (curling the latter
    returns an empty body -> JSONDecodeError). meta_info.weight_version is the same
    stamp, so /generate works. Returns None on failure (caller skips the filter).
    """
    try:
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        payload = {"input_ids": [0], "sampling_params": {"max_new_tokens": 1}, "return_logprob": False}
        resp = requests.post(url, json=payload, timeout=30)
        wv = resp.json().get("meta_info", {}).get("weight_version")
        return int(wv) if wv is not None else None
    except Exception as e:  # noqa: BLE001 - probe must never break rollout
        logger.warning("staleness: /generate version probe failed (%s); skipping filter", e)
        return None


def _is_too_stale(group, current_version: int, max_staleness: int | None) -> bool:
    """Drop if current_version - birth_version > max_staleness. None disables."""
    if max_staleness is None:
        return False
    v_gen = _group_head_version(group)
    return v_gen is not None and current_version - v_gen > max_staleness


def _dump_staleness(rollout_id, gid, birth, current, max_staleness, dropped) -> None:
    """DEBUG: append one line per group decision to staleness.jsonl under GUI_RESULT_DIR."""
    try:
        out_dir = Path(os.getenv("GUI_RESULT_DIR", "/tmp")) / "_staleness"
        out_dir.mkdir(parents=True, exist_ok=True)
        rec = {"rollout_id": rollout_id, "gid": gid, "birth": birth,
               "current": current, "staleness": None if birth is None else current - birth,
               "max_staleness": max_staleness, "dropped": dropped}
        with open(out_dir / "staleness.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:  # noqa: BLE001 - debug dump must never break rollout
        pass
# ============================================================================


# Global worker, shared across rollout calls so the queue stays warm.
_global_worker: AsyncRolloutWorker | None = None
_worker_lock = threading.Lock()


def _get_global_worker(args, data_buffer) -> AsyncRolloutWorker:
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            logger.info("starting fully-async rollout worker")
            _global_worker = AsyncRolloutWorker(
                args, data_buffer, concurrency=args.sglang_server_concurrency * get_rollout_num_engines(args)
            )
            _global_worker.start()
        return _global_worker


def _stop_global_worker() -> None:
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


atexit.register(_stop_global_worker)


class AsyncRolloutWorker:
    """Background thread + asyncio loop that continuously consumes groups
    from ``data_buffer`` and runs :func:`generate_and_rm_group` on each."""

    def __init__(self, args, data_buffer, concurrency: int = 10):
        self.args = args
        self.data_buffer = data_buffer
        self.concurrency = concurrency
        self.running = True
        self.output_queue: queue.Queue[tuple[int, list[Sample]]] = queue.Queue(maxsize=1000)
        self.worker_thread: threading.Thread | None = None
        self.state = GenerateState(args)

    # -- public --------------------------------------------------------------

    def start(self) -> None:
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._thread_main, name="fully-async-rollout", daemon=True)
            self.worker_thread.start()

    def stop(self) -> None:
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def get_completed_groups(self) -> list[tuple[int, list[Sample]]]:
        completed: list[tuple[int, list[Sample]]] = []
        while True:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def queue_size(self) -> int:
        return self.output_queue.qsize()

    # -- internals -----------------------------------------------------------

    def _thread_main(self) -> None:
        asyncio.run(self._loop())

    async def _loop(self) -> None:
        active_tasks: set[asyncio.Task] = set()
        max_concurrent = self.concurrency
        gid_counter = 0

        while self.running:
            try:
                # Reap done tasks
                if active_tasks:
                    done = {t for t in active_tasks if t.done()}
                    for t in done:
                        try:
                            t.result()  # results already handled in callback
                        except Exception as e:  # noqa: BLE001
                            logger.warning("fully-async task crashed: %r", e)
                    active_tasks -= done

                # Top up.
                while len(active_tasks) < max_concurrent and self.running:
                    groups = self.data_buffer.get_samples(1)
                    if not groups:
                        break
                    for group in groups:
                        gid = gid_counter
                        gid_counter += 1
                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )
                        task.add_done_callback(self._make_done_cb(gid))
                        active_tasks.add(task)

                await asyncio.sleep(1)
            except Exception as e:  # noqa: BLE001
                logger.exception("fully-async loop iteration error: %s", e)
                await asyncio.sleep(1)

        if active_tasks:
            logger.info(
                "fully-async: waiting for %d in-flight tasks to drain",
                len(active_tasks),
            )
            try:
                await asyncio.wait(active_tasks, timeout=30)
            except Exception:  # noqa: BLE001
                pass

    def _make_done_cb(self, gid: int):
        def _cb(done_task: asyncio.Task) -> None:
            try:
                result = done_task.result()
            except Exception:  # noqa: BLE001
                logger.exception("fully-async: process task raised")
                return
            if not isinstance(result, list):
                logger.warning(
                    "fully-async: generate_and_rm_group returned %r, expected list[Sample]; dropping",
                    type(result).__name__,
                )
                return
            
            self.output_queue.put((gid, result))

        return _cb


async def _generate_rollout_async(args, rollout_id: int, data_buffer) -> list[list[Sample]]:
    assert args.rollout_global_dataset
    worker = _get_global_worker(args, data_buffer)

    target = args.rollout_batch_size
    logger.info(
        "fully-async rollout %d: target=%d queue_warm=%d",
        rollout_id,
        target,
        worker.queue_size(),
    )

    collected: dict[int, list[Sample]] = {}
    started = time.time()
    last_log = started
    LOG_EVERY = 30.0

    max_staleness = getattr(args, "rollout_max_staleness", None)

    while len(collected) < target:
        # ===== STALENESS FILTER: refresh version each round (None = off or probe failed) =====
        current_version = _fetch_current_version(args) if max_staleness is not None else None
        filter_on = max_staleness is not None and current_version is not None
        # =====================================================================================
        # Pull whatever's done.
        drained = 0
        for gid, group in worker.get_completed_groups():
            # ===== STALENESS FILTER: drop too-old group, wait for producer to refill =====
            if filter_on:
                drop = _is_too_stale(group, current_version, max_staleness)
                _dump_staleness(rollout_id, gid, _group_head_version(group),
                                current_version, max_staleness, drop)
                if drop:
                    continue
            # =============================================================================
            collected[gid] = group
            drained += 1

        if not drained:
            await asyncio.sleep(0.05)

        now = time.time()
        if now - last_log > LOG_EVERY:
            logger.info(
                "fully-async rollout %d: collected %d/%d, queue=%d, elapsed=%.1fs",
                rollout_id,
                len(collected),
                target,
                worker.queue_size(),
                now - started,
            )
            last_log = now

    # Order by sample.index for determinism (slime convention). The group may be
    # list[Sample] or list[list[Sample]] (fan-out), so flatten before reading index.
    # ===== FIX: was getattr(s,"index") on a bare list -> list.index method -> int() TypeError =====
    def _key(group) -> int:
        return next((int(s.index) for s in _flatten_samples(group) if s.index is not None), 0)
    # ===========================================================================

    out = sorted(collected.values(), key=_key)[:target]
    logger.info(
        "fully-async rollout %d: done in %.1fs, queue_left=%d",
        rollout_id,
        time.time() - started,
        worker.queue_size(),
    )
    return out


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation: bool = False):
    """Slime ``--rollout-function-path`` entrypoint."""

    if evaluation:
        raise ValueError("fully-async rollout doesn't support evaluation mode")
    return run(_generate_rollout_async(args, rollout_id, data_buffer))


def eval_rollout_fully_async(args, rollout_id, data_buffer, evaluation: bool = False):
    """Slime ``--eval-function-path`` for fully-async runs; train delegates to
    :func:`generate_rollout_fully_async`.

    Eval is opt-in: it contends with the in-flight pool for worker/env slots.
    Set ``GUI_FULLY_ASYNC_ALLOW_EVAL=1`` to run it anyway.
    """
    if not evaluation:
        return generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation=False)

    if not config.get_bool("GUI_FULLY_ASYNC_ALLOW_EVAL", False):
        raise ValueError(
            "In-training eval is disabled under fully-async rollout: it contends with the "
            "in-flight pool for worker/env slots. Leave GUI_EVAL_INTERVAL=0 and eval saved "
            "ckpts offline, or set GUI_FULLY_ASYNC_ALLOW_EVAL=1 to run it anyway."
        )

    from rollout.partial_async_gui_rollout import _fast_eval_rollout_async

    output, _ = run(_fast_eval_rollout_async(args))
    return output
