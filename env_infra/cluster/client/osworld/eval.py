"""OSWorld eval task source + helpers.

The OSWorld task model: a ``test_all_meta`` JSON mapping ``{domain: [example_id]}``,
one JSON config file per task under ``examples/{domain}/{example_id}.json``, and
results under ``{result_dir}/{action_space}/{observation_type}/{model}/{domain}/{example_id}/``
keyed by a ``result.txt`` (resume via :func:`get_unfinished`).

The episode loop delegates to OSWorld's ``lib_run_single.run_single_example``
(lazy-imported — it lives in the read-only OSWorld submodule and pulls heavy
deps). Agents with a distinct ``run_single_example_*`` (e.g. kimi) subclass
and override :meth:`run_episode`.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from typing import Dict, List

from cluster.client.base.eval_worker import EvalTaskSource

logger = logging.getLogger("desktopenv.experiment")


# ---------------------------------------------------------------------------
# OSWorld task-model helpers (moved here from the old run_multienv_remote.py).
# ---------------------------------------------------------------------------
def distribute_tasks(test_all_meta: Dict[str, List[str]]) -> List[tuple]:
    return [(domain, eid) for domain, examples in test_all_meta.items() for eid in examples]


def build_config_path(args: argparse.Namespace, domain: str, example_id: str) -> str:
    return os.path.join(args.test_config_base_dir, args.examples_subdir, domain, f"{example_id}.json")


def build_result_dir(args: argparse.Namespace, domain: str, example_id: str, model: str = "") -> str:
    if args.simple_path:
        return os.path.join(args.result_dir, domain, example_id)
    return os.path.join(args.result_dir, args.action_space, args.observation_type, model, domain, example_id)


def get_unfinished(args: argparse.Namespace, total_meta: Dict[str, List[str]], model: str = "") -> Dict[str, List[str]]:
    """Filter out already-completed tasks (those with result.txt)."""
    if args.simple_path:
        target_dir = args.result_dir
    else:
        target_dir = os.path.join(args.result_dir, args.action_space, args.observation_type, model)

    if not os.path.exists(target_dir):
        return total_meta

    finished: Dict[str, List[str]] = {}
    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if not os.path.isdir(domain_path):
            continue
        finished[domain] = []
        for example_id in os.listdir(domain_path):
            example_path = os.path.join(domain_path, example_id)
            if not os.path.isdir(example_path):
                continue
            if os.path.exists(os.path.join(example_path, "result.txt")):
                finished[domain].append(example_id)
                continue
            # Clean incomplete results
            for item in os.listdir(example_path):
                item_path = os.path.join(example_path, item)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                except Exception:
                    pass

    remaining = {}
    for domain, examples in total_meta.items():
        done_ids = set(finished.get(domain, []))
        left = [eid for eid in examples if eid not in done_ids]
        if left:
            remaining[domain] = left
    return remaining


def get_result(args: argparse.Namespace, model: str = "") -> list[float] | None:
    """Print current success rate from completed results."""
    if args.simple_path:
        target_dir = args.result_dir
    else:
        target_dir = os.path.join(args.result_dir, args.action_space, args.observation_type, model)

    if not os.path.exists(target_dir):
        return None

    all_result = []
    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if not os.path.isdir(domain_path):
            continue
        for example_id in os.listdir(domain_path):
            result_file = os.path.join(domain_path, example_id, "result.txt")
            if not os.path.exists(result_file):
                continue
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    all_result.append(float(f.read()))
            except Exception:
                all_result.append(0.0)

    if all_result:
        logger.info("Current Success Rate: %.2f%% (%d tasks)", sum(all_result) / len(all_result) * 100, len(all_result))
    return all_result or None


# ---------------------------------------------------------------------------
# EvalTaskSource for OSWorld.
# ---------------------------------------------------------------------------
class OSWorldEvalSource(EvalTaskSource):
    """OSWorld task model: meta JSON + per-task JSON config + result.txt layout.

    Args:
        model_name: Model identifier used in the result-dir path and args.json.
        run_single_fn: The OSWorld ``run_single_example`` variant to call per
            episode. Defaults to ``lib_run_single.run_single_example``; agents
            with a distinct variant (kimi) pass theirs or subclass instead.
    """

    def __init__(self, model_name: str = "", run_single_fn=None):
        self.model_name = model_name
        self._run_single_fn = run_single_fn

    def _run_single(self):
        if self._run_single_fn is not None:
            return self._run_single_fn
        import lib_run_single
        return lib_run_single.run_single_example

    # -- EvalTaskSource --

    def load_tasks(self, args: argparse.Namespace) -> list:
        with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
            test_all_meta = json.load(f)
        if args.domain != "all":
            # Comma-separated multi-domain selection (e.g. "libreoffice_calc,os").
            # Backward-compatible with a single domain name.
            domains = [d.strip() for d in args.domain.split(",") if d.strip()]
            unknown = [d for d in domains if d not in test_all_meta]
            if unknown:
                raise KeyError(
                    f"Unknown domain(s) {unknown}; available: {sorted(test_all_meta)}"
                )
            test_all_meta = {d: test_all_meta[d] for d in domains}
        return distribute_tasks(test_all_meta)

    def pending_tasks(self, args: argparse.Namespace, all_tasks: list) -> list:
        meta: Dict[str, List[str]] = {}
        for domain, eid in all_tasks:
            meta.setdefault(domain, []).append(eid)
        remaining = get_unfinished(args, meta, model=self.model_name)
        return distribute_tasks(remaining)

    def prepare_example(self, args: argparse.Namespace, task_item) -> tuple:
        domain, example_id = task_item
        config_file = build_config_path(args, domain, example_id)
        with open(config_file, "r", encoding="utf-8") as f:
            example = json.load(f)
        result_dir = build_result_dir(args, domain, example_id, model=self.model_name)
        os.makedirs(result_dir, exist_ok=True)
        return example, result_dir

    def run_episode(self, agent, env, example, args: argparse.Namespace, result_dir: str, shared_scores: list) -> None:
        self._run_single()(
            agent, env, example, args.max_steps,
            example["instruction"], args, result_dir, shared_scores,
        )

    def item_label(self, task_item) -> str:
        domain, example_id = task_item
        return f"{domain}/{example_id}"

    def error_dump(self, args: argparse.Namespace, task_item, result_dir: str, exc: BaseException) -> None:
        domain, example_id = task_item
        path = os.path.join(build_result_dir(args, domain, example_id, model=self.model_name), "traj.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"Error": f"{domain}/{example_id} - {exc}"}, ensure_ascii=False) + "\n")

    def save_args(self, args: argparse.Namespace, model_name: str = "") -> None:
        if args.simple_path:
            args_path = os.path.join(args.result_dir, "args.json")
        else:
            args_path = os.path.join(args.result_dir, args.action_space,
                                     args.observation_type, self.model_name or model_name, "args.json")
        os.makedirs(os.path.dirname(args_path), exist_ok=True)
        with open(args_path, "w", encoding="utf-8") as f:
            safe_args = {
                key: "[REDACTED]" if any(part in key.lower() for part in ("api_key", "password", "secret", "token")) else value
                for key, value in vars(args).items()
            }
            json.dump(safe_args, f, indent=2, ensure_ascii=False)

    def summarize(self, args: argparse.Namespace, model_name: str = "") -> None:
        get_result(args, model=self.model_name or model_name)


class KimiOSWorldEvalSource(OSWorldEvalSource):
    """OSWorld eval source using ``run_single_example_kimi`` per episode."""

    def __init__(self, model_name: str = ""):
        super().__init__(model_name=model_name)

    def run_episode(self, agent, env, example, args: argparse.Namespace, result_dir: str, shared_scores: list) -> None:
        import lib_run_single
        lib_run_single.run_single_example_kimi(
            agent, env, example, args.max_steps,
            example["instruction"], args, result_dir, shared_scores,
        )
