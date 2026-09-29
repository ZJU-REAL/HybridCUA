"""MobileWorld eval task source.

The MobileWorld task model: bare ``task_name`` strings, task definitions live
in the local ``mobile_world.tasks.registry`` (not in the remote container),
results are written by :class:`TrajLogger` under ``{log_file_root}/{task_name}/``
keyed by a score file, and resume is via :func:`scan_finished_tasks`.

The episode loop mirrors MobileWorld's official ``_execute_single_task`` 1:1
(same reset→predict→execute→score sequence, same terminal-action handling).
The EvalRunner owns env lifecycle (build once, reuse across tasks, close in
finally) and agent construction (``agent_factory(args, env)``), so this source
holds neither an env queue nor ``create_agent`` — only the per-task episode.
"""
from __future__ import annotations

import argparse
import logging
import time
from typing import List

from cluster.client.base.eval_worker import EvalTaskSource

logger = logging.getLogger("cluster.client.mobileworld.eval")


def load_task_list(enable_mcp: bool = False, enable_user_interaction: bool = False) -> List[str]:
    """Flat task-name list from the local MobileWorld TaskRegistry (no network).

    Same filtering as MobileWorldSessionClient.get_suite_task_list, lifted to a
    module helper so the eval source does not spin a cluster session just to
    enumerate tasks.
    """
    from mobile_world.tasks.registry import TaskRegistry
    registry = TaskRegistry()
    filtered = []
    for name, task_cls in registry.tasks.items():
        tags = getattr(task_cls, "task_tags", set())
        if not enable_mcp and "agent-mcp" in tags:
            continue
        if not enable_user_interaction and "agent-user-interaction" in tags:
            continue
        filtered.append(name)
    return filtered


class MobileWorldEvalSource(EvalTaskSource):
    """MobileWorld task model: task_name strings + TrajLogger + scan_finished_tasks.

    Args:
        model_name: Unused for path layout (MobileWorld uses log_file_root), kept
            for EvalRunner API symmetry / args.json.
    """

    def __init__(self, model_name: str = ""):
        self.model_name = model_name

    # -- EvalTaskSource --

    def load_tasks(self, args: argparse.Namespace) -> list:
        if args.task and args.task != "ALL":
            return args.task.split(",")
        return load_task_list(getattr(args, "enable_mcp", False),
                              getattr(args, "enable_user_interaction", False))

    def pending_tasks(self, args: argparse.Namespace, all_tasks: list) -> list:
        from mobile_world.runtime.client import scan_finished_tasks
        finished, _ = scan_finished_tasks(args.log_file_root, all_tasks)
        pending = [t for t in all_tasks if t not in finished]
        if getattr(args, "shuffle_tasks", False):
            import random
            random.shuffle(pending)
        return pending

    def prepare_example(self, args: argparse.Namespace, task_item) -> tuple:
        # task_item IS the task_name string. Synthesize a minimal example dict;
        # TrajLogger (constructed in run_episode) owns the result subdir.
        return {"id": task_item}, args.log_file_root

    def run_episode(self, agent, env, example, args: argparse.Namespace, result_dir: str, shared_scores: list) -> None:
        from mobile_world.runtime.utils.models import ANSWER, ENV_FAIL, FINISHED, UNKNOWN
        from mobile_world.runtime.utils.trajectory_logger import TrajLogger

        task_name = example["id"]
        traj_logger = TrajLogger(result_dir, task_name)

        # One reset: initializes the task AND returns its instruction in the obs.
        obs_data = env.reset({"task_name": task_name})
        modalities = obs_data.get("modalities", obs_data)
        task_goal = modalities.get("instruction") or ""
        obs = env._to_observation(obs_data)
        agent.initialize(task_goal)
        logger.info("Task '%s' goal: %s", task_name, task_goal)

        step = 0
        start = time.time()
        while True:
            step += 1
            prediction, action = agent.predict(
                {
                    "screenshot": obs.screenshot,
                    "tool_call": getattr(obs, "tool_call", None),
                    "ask_user_response": obs.ask_user_response,
                }
            )
            if prediction is None:
                logger.warning("Agent prediction failed at step %d of task '%s'", step, task_name)
                break

            traj_logger.log_traj(
                task_name, task_goal, step, prediction,
                action.model_dump(exclude_none=True), obs,
                agent.get_total_token_usage(),
            )

            if action.action_type in (ENV_FAIL, FINISHED, UNKNOWN):
                break
            obs = env.execute_action(action)  # ANSWER and all device actions execute
            if action.action_type == ANSWER:
                break
            if step >= args.max_round:
                logger.debug("Task '%s' reached max_round %s", task_name, args.max_round)
                break

        score, reason = env.get_task_score(task_name)
        traj_logger.log_score(score=score, reason=reason)
        agent.done()
        shared_scores.append(score)
        logger.info("Task '%s' done: score=%s, steps=%s, %.1fs", task_name, score, step, time.time() - start)

    def item_label(self, task_item) -> str:
        return f"task={task_item}"

    def error_dump(self, args: argparse.Namespace, task_item, result_dir: str, exc: BaseException) -> None:
        # On failure, write NO score file -> scan_finished_tasks will retry the
        # task next attempt. Just log; the EvalRunner already logged the error.
        logger.exception("Error executing task '%s'", task_item)

    def save_args(self, args: argparse.Namespace, model_name: str = "") -> None:
        import json
        import os
        os.makedirs(args.log_file_root, exist_ok=True)
        with open(os.path.join(args.log_file_root, "args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
