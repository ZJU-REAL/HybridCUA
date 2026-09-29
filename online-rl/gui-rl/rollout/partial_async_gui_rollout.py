"""slime entry points for the GUI rollout (train + eval).

Wire these via the launch script:
- ``--custom-generate-function-path rollout.partial_async_gui_rollout.generate``
- ``--eval-function-path           rollout.fully_async_rollout.eval_rollout_fully_async``
  (that entry reuses :func:`_fast_eval_rollout_async` below as the eval coroutine)

Both converge on :class:`rollout.ray_actor_pool.RayActorPool`, which runs each
trajectory on its own long-lived worker actor.

Why eval still goes through ``generate_and_rm``: that wrapper owns reward-model
dispatch. For the non-PRM GUI path the trajectory pre-fills ``sample.reward`` so
RM is skipped; for the PRM path it leaves ``reward=None`` and ``generate_and_rm``
runs ``reward_func`` via ``async_rm``/``batched_async_rm``. Crucially,
``generate_and_rm`` resolves ``args.custom_generate_function_path`` — which the
fast scripts point at :func:`generate` below — so it calls into our pool for the
actual trajectory while keeping RM handling correct. We only replace the
*single-trajectory fan-out*, exactly as planned.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from slime.utils.types import Sample
from config import ROLLOUT_WORKER_ENV_FLAG
from utils.rollout_helpers import _load_meta_pairs
from utils.utils import load_task_config

logger = logging.getLogger(__name__)


async def generate(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample | list[Sample]:
    """Train/eval single-trajectory entry: dispatch to the worker pool.

    Called by slime's ``generate_and_rm`` (via ``custom_generate_function_path``)
    on the RolloutManager event loop; suspends here while a worker actor runs the
    trajectory. Inside a worker we never recurse — the worker calls
    ``run_trajectory`` directly, not this function.
    """
    if os.getenv(ROLLOUT_WORKER_ENV_FLAG):
        # Defensive: a worker should never re-enter the slime dispatch layer.
        from rollout.trajectory_runner import run_trajectory

        return await run_trajectory(args, sample, sampling_params, evaluation)

    from rollout.ray_actor_pool import RayActorPool

    return await RayActorPool.get(args).submit(sample, sampling_params, evaluation)


def fast_eval_rollout(args, rollout_id, data_source, evaluation: bool = False):
    """Half-async eval entry (``--eval-function-path``). Train falls back to slime's
    default ``generate_rollout``; eval runs our pool-backed organizer on the global loop."""
    if not evaluation:
        from slime.rollout.sglang_rollout import generate_rollout

        return generate_rollout(args, rollout_id, data_source, evaluation=False)

    from slime.utils.async_utils import run

    output, _ = run(_fast_eval_rollout_async(args))
    return output


async def _fast_eval_rollout_async(args):
    """Build every eval task and run them through ``generate_and_rm``.

    Replicates ``rollout/partial_async_rollout_gui.py::_gui_eval_rollout`` exactly;
    the only behavioral change is upstream — ``generate_and_rm`` dispatches each
    trajectory to the worker pool via :func:`generate` instead of the legacy
    single-loop ``_generate_local``.
    """
    from slime.rollout.base_types import RolloutFnEvalOutput
    from slime.rollout.sglang_rollout import generate_and_rm

    base_dir = os.getenv(
        "GUI_TEST_CONFIG_BASE_DIR",
        # rollout/ -> parent.parent is the gui-rl/ root (see _sample_task_info).
        str(Path(__file__).resolve().parent.parent / "evaluation_examples"),
    )
    meta_path = os.getenv("GUI_EVAL_META_PATH", str(Path(base_dir) / "test_nochrome.json"))
    pairs = _load_meta_pairs(meta_path)
    if not pairs:
        raise RuntimeError(f"No eval tasks loaded from {meta_path}")

    eval_max_response_len = getattr(args, "eval_max_response_len", None)
    if eval_max_response_len is None:
        eval_max_response_len = getattr(args, "rollout_max_response_len", 512)

    eval_top_p = getattr(args, "eval_top_p", None)
    if eval_top_p is None:
        eval_top_p = getattr(args, "rollout_top_p", 1.0)
    eval_top_k = getattr(args, "eval_top_k", None)
    if eval_top_k is None:
        eval_top_k = getattr(args, "rollout_top_k", -1)

    sampling_params = dict(
        temperature=float(getattr(args, "eval_temperature", 0.0) or 0.0),
        top_p=float(eval_top_p),
        top_k=int(eval_top_k),
        max_new_tokens=int(eval_max_response_len),
        stop=getattr(args, "rollout_stop", None),
        stop_token_ids=getattr(args, "rollout_stop_token_ids", None),
        skip_special_tokens=getattr(args, "rollout_skip_special_tokens", False),
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    n_samples = int(getattr(args, "n_samples_per_eval_prompt", 1) or 1)
    tasks = []
    sample_index = 0
    for domain, example_id in pairs:
        cfg_path = Path(base_dir) / "examples" / str(domain) / f"{example_id}.json"
        if not cfg_path.exists():
            continue
        task_config = load_task_config(base_dir, domain, example_id)
        instruction = str(task_config.get("instruction", ""))
        group_index = sample_index // n_samples
        for _ in range(n_samples):
            sample = Sample(
                prompt=instruction,
                label="",
                metadata={
                    "domain": domain,
                    "example_id": example_id,
                    "instruction": instruction,
                    "task_config": task_config,
                },
            )
            sample.index = sample_index
            sample.group_index = group_index
            sample_index += 1
            tasks.append(
                asyncio.create_task(
                    generate_and_rm(args, sample, sampling_params=sampling_params, evaluation=True)
                )
            )

    if not tasks:
        raise RuntimeError(f"No valid eval tasks found from {meta_path}")

    data = []
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if isinstance(sample, list):
            data.extend(sample)
        else:
            data.append(sample)

    data.sort(key=lambda s: s.index)
    reward_key = getattr(args, "eval_reward_key", None) or getattr(args, "reward_key", "score")
    rewards = []
    for sample in data:
        if isinstance(sample.reward, dict):
            rewards.append(float(sample.reward.get(reward_key, 0.0)))
        elif sample.reward is not None:
            rewards.append(float(sample.reward))
        else:
            rewards.append(0.0)

    return RolloutFnEvalOutput(
        data={
            "gui_eval": {
                "rewards": rewards,
                "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
                "samples": data,
            }
        }
    ), []
