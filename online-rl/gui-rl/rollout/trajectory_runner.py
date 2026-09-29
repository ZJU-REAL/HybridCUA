"""Single-trajectory logic for the pooled rollout path.

``run_trajectory`` is the worker-side counterpart of the legacy
``rollout/partial_async_rollout_gui.py::_generate_local``. The business logic is
preserved verbatim (task resolution, agent creation, ``Trajectory.run()``,
``build_train_data``, dynamic-history / PRM / timing / abort marking), with two
intentional differences for the pooled model:

1. **No trajectory semaphore.** In the legacy single-loop world many trajectories
   share one process, so ``_get_gui_trajectory_semaphore`` bounds concurrent env
   sessions. Here the pool size *is* the concurrency cap and each worker runs
   exactly one trajectory at a time, so the env-session count is already bounded
   by the pool size. The semaphore would always be uncontended; we drop it to
   keep the hot path clean.

2. **Process-level env-client reuse.** A per-process singleton from
   :func:`rollout.ray_actor_pool._get_env_client` reuses its httpx connection
   pool across tasks. The lease is still allocate→reset→close per task.

The shared helpers (`_sample_task_info`, `_build_result_dir`,
`_clear_sample_result_dir`, `_create_gui_agent`, `_attach_gui_timings`) are
imported from ``utils.rollout_helpers`` — the same module the legacy path uses, so
there is no logic fork.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import config
from config import EpisodeConfig
from reward.reward_func import cli_aware_enabled, mark_aborted_samples
from rollout.trajectory import Trajectory, build_dynamic_history_samples
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample
from utils.rollout_helpers import (
    _attach_gui_timings,
    _build_result_dir,
    _clear_sample_result_dir,
    _create_gui_agent,
    _sample_task_info,
)

logger = logging.getLogger(__name__)


async def run_trajectory(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample | list[Sample]:
    """Run one GUI trajectory and populate ``sample`` with training data.

    Mirrors ``_generate_local`` minus the env-session semaphore (the pool bounds
    concurrency) and using a process-level reused env client.
    """
    assert not args.partial_rollout, "Partial rollout is not supported for GUI rollout."

    # Imported lazily to avoid a circular import at module load
    # (pool -> trajectory_runner -> pool).
    from rollout.ray_actor_pool import _get_env_client

    profile = config.gui_profile()
    # A-group external timings (outside ep.run()). Each measured by perf_counter
    # deltas guarded by `profile`, so there is zero overhead when disabled.
    ext: dict[str, float] = {}
    setup_t0 = time.perf_counter() if profile else 0.0
    instruction, task_config, domain, example_id = _sample_task_info(sample, evaluation=evaluation)
    result_dir = _build_result_dir(args, domain, example_id, sample, evaluation=evaluation)
    _clear_sample_result_dir(result_dir)

    # Per-process singleton env client (connection pool reused across tasks).
    # GUI_ENV_CLIENT=session uses the self-contained /v1/sessions adapter
    # (clients/), default keeps the legacy lease-HTTP GuiEnvClient.
    env_client = _get_env_client()
    state = GenerateState(args)

    ep_cfg = EpisodeConfig.resolve(args, evaluation=evaluation)
    max_steps = ep_cfg.max_steps
    max_image_history_length = ep_cfg.max_image_history_length

    sampling_params = dict(sampling_params)
    if evaluation and getattr(args, "eval_temperature", None) is not None:
        sampling_params["temperature"] = float(args.eval_temperature)
    elif (not evaluation) and getattr(args, "rollout_temperature", None) is not None:
        sampling_params["temperature"] = float(args.rollout_temperature)
    parser = _create_gui_agent(
        args,
        max_steps=max_steps,
        max_image_history_length=max_image_history_length,
        result_dir=result_dir,
        agent_class_path=(sample.metadata or {}).get("agent_class_path"),
    )
    parser.reset(logging.getLogger("desktopenv.gui_agent.rollout"))

    ep = Trajectory(
        args=args,
        agent=parser,
        env_client=env_client,
        state=state,
        ep_cfg=ep_cfg,
        sampling_params=sampling_params,
        sample=sample,
        instruction=instruction,
        task_config=task_config,
        result_dir=result_dir,
        evaluation=evaluation,
    )
    if profile:
        ext["setup"] = round(time.perf_counter() - setup_t0, 4)
        run_t0 = time.perf_counter()
    res = await ep.run()
    if profile:
        ext["episode_run"] = round(time.perf_counter() - run_t0, 4)

    final_status = res.status
    eval_score = res.eval_score
    step_snapshots = res.step_snapshots
    assistant_responses = res.assistant_responses
    prm_step_scores = res.prm.step_scores
    prm_step_details = res.prm.step_details
    if res.error_stage is not None:
        sample.metadata = sample.metadata or {}
        sample.metadata["gui_invalid_reason"] = f"{res.error_stage}_failed"
        sample.metadata["gui_error_stage"] = res.error_stage
        sample.metadata["gui_error_message"] = res.error_message
    train_messages_for_loss = res.train_messages_for_loss
    tool_spec_for_loss = res.tool_spec_for_loss
    if profile:
        btd_t0 = time.perf_counter()
    input_ids, loss_mask, mm_train = parser.build_train_data(
        args=args,
        state=state,
        train_messages=train_messages_for_loss,
        tool_spec=tool_spec_for_loss,
    )
    if profile:
        ext["build_train_data"] = round(time.perf_counter() - btd_t0, 4)
    response_start = None
    active_positions = [i for i in range(len(loss_mask)) if i < len(input_ids) and int(loss_mask[i]) == 1]
    if active_positions:
        response_start = active_positions[0]
        response_length = len(input_ids) - response_start
        loss_mask = [int(loss_mask[i]) if i < len(loss_mask) else 0 for i in range(response_start, len(input_ids))]
    else:
        response_length = 0
        loss_mask = []

    sample.tokens = input_ids
    sample.loss_mask = loss_mask
    sample.response = "\n".join(assistant_responses)
    sample.response_length = response_length
    sample.multimodal_train_inputs = mm_train
    sample.status = final_status
    sample.metadata = sample.metadata or {}
    sample.metadata["gui_result_dir"] = str(result_dir)
    sample.metadata["gui_score"] = eval_score
    # CLI-aware signals (paper eq. 6/7), composed into the reward by reward_func.
    cli_ref = sample.metadata.get("cli_preferred")
    sample.metadata["gui_used_cli"] = res.used_cli
    # R_CLI = I[Success] * I[b(tau) == b*]; unlabeled b* -> 0 (neither
    # rewarded nor penalized).
    sample.metadata["gui_cli_reward"] = float(
        eval_score >= 1.0 and cli_ref is not None and res.used_cli == bool(cli_ref)
    )
    # The non-dynamic path trains only the last step, so take that step's e_t.
    last_step = int(step_snapshots[-1]["step_idx"]) if step_snapshots else -1
    sample.metadata["gui_exec_penalty"] = res.exec_penalty_by_step.get(last_step, 0.0)
    if getattr(args, "prm_enable", False):
        sample.metadata["prm"] = {
            "enabled": True,
            "step_scores": prm_step_scores,
            "step_mean_score": (sum(prm_step_scores) / len(prm_step_scores)) if prm_step_scores else 0.0,
            "step_details": prm_step_details,
        }
        # Current GUI non-dynamic path trains the suffix of one step response;
        # align step_wise metadata to that suffix span.
        if response_start is not None and response_length > 0:
            last_step_idx = int(step_snapshots[-1]["step_idx"]) if step_snapshots else 0
            prm_score_by_step = {int(d.get("step_index", i)): float(d.get("mean_score", 0.0)) for i, d in enumerate(prm_step_details)}
            sample.metadata["step_wise"] = {
                "step_scores": [float(prm_score_by_step.get(last_step_idx, 0.0))],
                "step_indices": [int(last_step_idx)],
                "step_token_spans": [[0, int(response_length)]],
            }
    # gui_reward = 1.0 if eval_score == 1 else -1.0
    gui_reward = float(eval_score)
    # Keep training reward on `score` (GRPO expects this),
    # and expose raw task accuracy on `acc` for eval logging.
    # Important:
    # - PRM path must go through reward_func so step-wise PRM composition
    #   and prm_example_eval are visible in rollout logs.
    # - CLI-aware terms are composed there too, so leaving reward=None is what
    #   makes generate_and_rm dispatch to the RM at all.
    # - Otherwise keep old behavior with prefilled reward.
    if getattr(args, "prm_enable", False) or cli_aware_enabled(args):
        sample.reward = None
    else:
        sample.reward = {"score": gui_reward, "acc": float(eval_score)}
    if getattr(args, "dynamic_history", False) and not evaluation:
        prm_score_by_step = None
        if getattr(args, "prm_enable", False) and isinstance(sample.metadata.get("prm"), dict):
            prm_score_by_step = {}
            for item in sample.metadata["prm"].get("step_details", []):
                if isinstance(item, dict) and "step_index" in item:
                    prm_score_by_step[int(item["step_index"])] = float(item.get("mean_score", 0.0))
        if profile:
            bdh_t0 = time.perf_counter()
        dynamic_samples = build_dynamic_history_samples(
            args=args,
            state=state,
            agent=parser,
            base_sample=sample,
            step_snapshots=step_snapshots,
            outcome_reward=gui_reward,
            prm_score_by_step=prm_score_by_step,
            exec_penalty_by_step=res.exec_penalty_by_step,
        )
        if profile:
            ext["build_dynamic_history"] = round(time.perf_counter() - bdh_t0, 4)
        _attach_gui_timings(sample, ext, ep, profile)
        result = dynamic_samples if dynamic_samples else [sample]
        mark_aborted_samples(result)
        # >>> CHANGED (per-sample / cua-style): 不设 rollout_id（保持 None）。
        # 每个 step 样本作为独立训练样本（按样本归一化 + 按样本切 step），对齐
        # computeruseagent。slime 侧 _convert 会对 rollout_id 为 None 的样本自动
        # 分配每样本唯一 id（list(range(len))），_validate_rollout_id_annotated /
        # rollout_mask_sums 聚合已禁用，故无需共享 rollout_id。
        # <<< CHANGED
        return result
    _attach_gui_timings(sample, ext, ep, profile)
    mark_aborted_samples([sample])
    return sample
