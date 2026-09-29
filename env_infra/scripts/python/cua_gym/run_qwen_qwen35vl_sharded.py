"""Qwen3.5-VL sharded CUA-Gym evaluation, built on the official mm_agents.qwen.QwenAgent.

Combines two existing entry points:
  - scripts/python/osworld/run_qwen_qwen35vl_sharded.py — the QwenAgent construction,
    param names (--coord / --history_n / --image_max / --fold_size) and the
    round-robin sharding of model requests across multiple vLLM endpoints (each
    worker process pins one endpoint by worker index, passed to QwenAgent(base_url=...)).
  - scripts/python/cua_gym/run_kimi_remote.py — the CUA-Gym task source
    (a directory of <uuid>/ bundles via CuaGymEvalSource) and the cua_gym runtime
    session client (CuaGymSessionClient).

Unlike the Kimi entry point, QwenAgent.predict returns 2 values (response, actions),
so it drives episodes with the generic lib_run_single.run_single_example — no
run_single_fn override is needed (CuaGymEvalSource defaults to it).

Usage: see scripts/bash/cua_gym/run_qwen_qwen35vl_sharded.sh (needs a node hosting
the cua_gym world, a downloaded CUA-Gym bundle dir, and vLLM endpoints serving Qwen3.5-VL).
"""
import argparse
import logging
import os
import sys
from multiprocessing import current_process

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))          # env_infra root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../OSWorld"))  # lib_run_single, mm_agents

from cluster.client import EvalRunner, add_common_args
from cluster.client.cua_gym import CuaGymEvalSource, CuaGymSessionClient
from mm_agents.qwen import QwenAgent


def _parse_base_urls(args: argparse.Namespace) -> list[str]:
    raw = args.openai_base_urls or os.environ.get("OPENAI_BASE_URLS", "")
    urls = [u.strip() for u in raw.split(",") if u.strip()]
    if not urls:
        urls = [args.base_url or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")]
    return urls


def _worker_index() -> int:
    name = current_process().name
    try:
        return int(name.rsplit("-", 1)[-1]) - 1
    except (ValueError, IndexError):
        return 0


def build_agent(args: argparse.Namespace, env=None) -> QwenAgent:
    # Same construction as the OSWorld sharded entry point: base_url is picked
    # per-worker (round-robin over the shard list) and passed directly to
    # QwenAgent (it does NOT read the env var when base_url is given).
    urls = _parse_base_urls(args)
    idx = _worker_index()
    selected = urls[idx % len(urls)]
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "dummy")
    logging.getLogger("desktopenv.experiment").info(
        "Worker %d using endpoint: %s", idx, selected
    )
    return QwenAgent(
        model=args.model,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        action_space=args.action_space,
        observation_type=args.observation_type,
        coordinate_type=args.coord,
        add_thought_prefix=args.add_thought_prefix,
        history_n=args.history_n,
        image_max=args.image_max,
        fold_size=args.fold_size,
        enable_thinking=args.enable_thinking,
        base_url=selected,
        api_key=api_key,
    )


def build_env(args: argparse.Namespace) -> CuaGymSessionClient:
    """One cua_gym cluster session per worker (reused across tasks)."""
    cluster_url = args.cluster_url or os.environ.get("GUI_ENV_SERVER_URL", "http://127.0.0.1:19000")
    os.environ["GUI_ENV_SERVER_URL"] = cluster_url
    return CuaGymSessionClient(
        cluster_url=cluster_url,
        action_space=args.action_space,
        screen_size=(args.screen_width, args.screen_height),
        headless=args.headless,
        enable_proxy=os.environ.get("ENABLE_PROXY", "1") == "1",
        client_password=args.client_password,
    )


def main():
    parser = add_common_args(argparse.ArgumentParser(
        description="Qwen3.5-VL sharded CUA-Gym evaluation (official mm_agents.qwen.QwenAgent)"
    ))
    # CUA-Gym task source: a directory of <uuid>/ bundles (replaces test_all_meta).
    parser.add_argument("--tasks_root", required=True,
                        help="Directory of CUA-Gym <uuid>/ bundles (task.json + reward.py + initial_setup.*)")
    # Optional OSWorld-style meta JSON pinning an exact subset: {app_type: [uuid, ...]}.
    parser.add_argument("--tasks_meta", default=None,
                        help="JSON {app_type: [uuid, ...]} selecting specific tasks (default: all bundles under --tasks_root)")
    # Agent params — mirror the OSWorld sharded entry point names/defaults.
    parser.add_argument("--model", default="qwen35-vl")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Max output tokens. Must be < model context length to leave room for input "
                             "(vLLM rejects requests where max_tokens >= context_length).")
    parser.add_argument("--history_n", type=int, default=100,
                        help="Number of recent steps kept in the prompt")
    parser.add_argument("--coord", type=str, default="relative", choices=["absolute", "relative"],
                        help="Coordinate system for agent outputs")
    parser.add_argument("--image_max", type=int, default=20,
                        help="Max screenshots kept as images before old ones are collapsed to text")
    parser.add_argument("--fold_size", type=int, default=10,
                        help="Number of screenshots collapsed at a time once image_max is exceeded")
    parser.add_argument("--add_thought_prefix", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Enable model-side thinking (only effective for dashscope base_url)")
    # Endpoint sharding.
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL"),
                        help="Single-endpoint fallback when --openai_base_urls is unset")
    parser.add_argument("--openai_base_urls", default=os.environ.get("OPENAI_BASE_URLS"),
                        help="Comma-separated vLLM endpoints for round-robin sharding")
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--password", default="password")
    args = parser.parse_args()

    EvalRunner(
        args,
        agent_factory=build_agent,
        task_source=CuaGymEvalSource(model_name=args.model),
        build_env=build_env,
        model_name=args.model,
        log_prefix="cua_gym-qwen35vl-sharded",
    ).run()


if __name__ == "__main__":
    main()
