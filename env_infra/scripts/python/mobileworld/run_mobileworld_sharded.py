"""MobileWorld sharded sync remote evaluation.

Drives concurrent episodes via the generic ``EvalRunner`` (process-per-env +
shared ``Manager.Queue`` work-stealing + env reuse across tasks — the proven
gui-env model) and spreads model requests across MULTIPLE vLLM endpoints,
round-robin by worker index.

MobileWorld's agent takes the endpoint as a constructor argument via
``create_agent(..., llm_base_url, ...)`` (unlike OSWorld, which reads
``OPENAI_BASE_URL`` from the env). So each worker pins its endpoint inside
``build_agent`` from its process index — the same round-robin sharding pattern
OSWorld applies in ``env_setup``, just consumed at agent-construction time.

The blocking ``agent.predict`` runs synchronously inside the worker process
(no asyncio, no thread pool): concurrency comes from multiple OS processes,
which also gives clean isolation (crash → auto-restart) and OS-level resource
reclamation on exit.

Usage:
    python scripts/python/mobileworld/run_mobileworld_sharded.py \
        --cluster_url http://127.0.0.1:19000 \
        --openai_base_urls http://h:8000/v1,...,http://h:8007/v1 \
        --num_envs 8 --model_name Qwen3-VL-8B-Instruct
"""
import argparse
import os
import sys
from multiprocessing import current_process

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../MobileWorld/src"))

from loguru import logger

from cluster.client import EvalRunner
from cluster.client.mobileworld import MobileWorldEvalSource, MobileWorldSessionClient


def _parse_base_urls(args: argparse.Namespace) -> list[str]:
    """Comma-separated endpoints from --openai_base_urls/env, else single fallback."""
    raw = args.openai_base_urls or os.environ.get("OPENAI_BASE_URLS", "")
    urls = [u.strip() for u in raw.split(",") if u.strip()]
    if not urls:
        urls = [args.llm_base_url or os.environ.get("LLM_BASE_URL", "https://api.llm.mioffice.cn/v1")]
    return urls


def _worker_index() -> int:
    """Derive the worker index from the EvalRunner process name (Env-1, Env-Restart-1, ...)."""
    name = current_process().name
    try:
        return int(name.rsplit("-", 1)[-1]) - 1
    except (ValueError, IndexError):
        return 0


def build_env(args: argparse.Namespace) -> MobileWorldSessionClient:
    """One cluster session per worker, reused across all tasks it drains."""
    return MobileWorldSessionClient(
        cluster_url=args.cluster_url,
        device=args.device,
        step_wait_time=args.step_wait_time,
    )


def build_agent(args: argparse.Namespace, env) -> "object":
    """Build a MobileWorld agent pinned to this worker's endpoint (round-robin).

    ``env`` is required: create_agent reads ``env.tools`` at construction.
    """
    from mobile_world.agents.registry import create_agent
    urls = _parse_base_urls(args)
    idx = _worker_index()
    selected = urls[idx % len(urls)]
    logger.info("Worker {} using endpoint: {}", idx, selected)
    return create_agent(
        args.agent_type, args.model_name, selected, args.api_key,
        env=env,
        enable_mcp=args.enable_mcp,
    )


def main():
    parser = argparse.ArgumentParser(description="MobileWorld sharded sync remote evaluation")
    parser.add_argument("--cluster_url", default=os.environ.get("GUI_ENV_SERVER_URL", "http://127.0.0.1:19000"))
    parser.add_argument("--agent_type", default="general_e2e")
    parser.add_argument("--model_name", default="gemini-3.1-pro-preview-ai-train")
    parser.add_argument("--llm_base_url", default=os.environ.get("LLM_BASE_URL", ""),
                        help="Single-endpoint fallback when --openai_base_urls is unset")
    parser.add_argument("--openai_base_urls", default=os.environ.get("OPENAI_BASE_URLS"),
                        help="Comma-separated LLM endpoints; sharded round-robin by worker index")
    parser.add_argument("--api_key", default=os.environ.get("API_KEY", ""))
    parser.add_argument("--num_envs", type=int, default=8,
                        help="Parallel worker processes (= total cluster slots demanded)")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--max_concurrency", type=int, default=None,
                        help="(Deprecated, ignored) Kept for backward compatibility with the "
                             "old joblib runner. Concurrency is exactly --num_envs (one session "
                             "per worker process).")
    parser.add_argument("--max_round", type=int, default=50)
    parser.add_argument("--step_wait_time", type=float, default=3.0)
    parser.add_argument("--task", default="ALL")
    parser.add_argument("--log_file_root", default="traj_logs/mobileworld_sharded")
    parser.add_argument("--enable_mcp", action="store_true")
    parser.add_argument("--enable_user_interaction", action="store_true")
    parser.add_argument("--shuffle_tasks", action="store_true")
    parser.add_argument("--auto_retry", type=int, default=10)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--device", default="emulator-5554")
    args = parser.parse_args()

    from mobile_world.runtime.client import scan_finished_tasks

    urls = _parse_base_urls(args)
    logger.info("Endpoints: {} | num_envs={}", urls, args.num_envs)

    if args.task and args.task != "ALL":
        task_list = args.task.split(",")
    else:
        task_list = MobileWorldEvalSource().load_tasks(args)
    logger.info("Task list: {} tasks", len(task_list))

    spec = MobileWorldEvalSource(model_name=args.model_name)
    max_attempts = min(1 + args.auto_retry, 10)
    for attempt in range(max_attempts):
        finished, _ = scan_finished_tasks(args.log_file_root, task_list)
        pending = [t for t in task_list if t not in finished]
        logger.info("Attempt {}/{}: {} pending tasks", attempt + 1, max_attempts, len(pending))
        if not pending:
            break
        if args.dry_run:
            logger.info("Dry run: skipping execution")
            break
        # --shuffle_tasks is honored inside MobileWorldEvalSource.pending_tasks
        # (the order EvalRunner drains the shared queue from).

        runner = EvalRunner(
            args,
            agent_factory=build_agent,
            build_env=build_env,
            task_source=spec,
            model_name=args.model_name,
            log_prefix="mobileworld-sharded",
        )
        runner.run()

    # Authoritative results from disk (includes retries).
    finished, scores = scan_finished_tasks(args.log_file_root, task_list)
    no_results = [t for t in task_list if t not in finished]
    print(f"\nCompleted: {len(finished)} tasks with results, {len(no_results)} without")
    if scores:
        passed = sum(1 for s in scores if s > 0)
        print(f"Pass rate: {passed}/{len(scores)} ({passed / len(scores) * 100:.1f}%)")


if __name__ == "__main__":
    main()
