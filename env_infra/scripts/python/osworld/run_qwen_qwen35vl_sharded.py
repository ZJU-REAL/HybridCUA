"""Qwen3.5-VL sharded remote evaluation, built on the official mm_agents.qwen.QwenAgent.

Defaults the model to Qwen3.5-VL.
Mirrors the official ``scripts/python/run_multienv_qwen.py`` (same QwenAgent
construction, same param names: --coord / --history_n / --image_max / --fold_size,
same lib_run_single.run_single_example episode loop) but drives the platform's
cluster ``EvalRunner`` + ``OSWorldEvalSource`` (remote OSWorldSessionClient
instead of a local DesktopEnv) and shards model requests across MULTIPLE vLLM
endpoints — each worker process pins one endpoint round-robin by worker index,
passed directly to QwenAgent(base_url=...).

Usage:
    python scripts/python/osworld/run_qwen_qwen35vl_sharded.py \
        --openai_base_urls http://h:8000/v1,...,http://h:8007/v1 \
        --num_envs 8 --model qwen35-vl
"""
import argparse
import os
import sys
from multiprocessing import current_process

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../OSWorld"))

from cluster.client import EvalRunner, add_common_args
from cluster.client.osworld import OSWorldEvalSource
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
    # Same construction as official run_multienv_qwen.py, except base_url is
    # picked per-worker (round-robin over the shard list) instead of a single
    # --base_url. Passed directly to QwenAgent (it does NOT read the env var
    # when base_url is given).
    urls = _parse_base_urls(args)
    idx = _worker_index()
    selected = urls[idx % len(urls)]
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "dummy")
    import logging
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


def main():
    parser = add_common_args(argparse.ArgumentParser(
        description="Qwen3.5-VL sharded remote evaluation (official mm_agents.qwen.QwenAgent)"
    ))
    # Agent params — mirror official run_multienv_qwen.py names/defaults.
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

    runner = EvalRunner(
        args,
        agent_factory=build_agent,
        task_source=OSWorldEvalSource(model_name=args.model),
        model_name=args.model,
        log_prefix="qwen-qwen35vl-sharded",
    )
    runner.run()


if __name__ == "__main__":
    main()
