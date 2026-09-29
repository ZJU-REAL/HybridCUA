"""Qwen3.7-Plus (API) remote evaluation, built on the official mm_agents.qwen.QwenAgent.

Targets a single hosted Qwen API endpoint (one base_url + one api_key shared by
all workers) instead of sharding requests across local vLLM endpoints. Mirrors
the official ``scripts/python/run_multienv_qwen.py`` (same QwenAgent construction,
same param names: --coord / --history_n / --image_max / --fold_size, same
lib_run_single.run_single_example episode loop) but drives the platform's cluster
``EvalRunner`` + ``OSWorldEvalSource`` (remote OSWorldSessionClient instead of a
local DesktopEnv).

Usage:
    python scripts/python/osworld/run_qwen_qwen37plus.py \
        --base_url https://api.llm.mioffice.cn/v1 \
        --api_key sk-... \
        --num_envs 8 --model tongyi/qwen3.7-plus
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../OSWorld"))

from cluster.client import EvalRunner, add_common_args
from cluster.client.osworld import OSWorldEvalSource
from mm_agents.qwen import QwenAgent


def build_agent(args: argparse.Namespace, env=None) -> QwenAgent:
    # Same construction as official run_multienv_qwen.py. base_url/api_key are
    # passed directly to QwenAgent (it does NOT read the env var when base_url is
    # given) and shared by every worker — single hosted endpoint, no sharding.
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "dummy")
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
        base_url=base_url,
        api_key=api_key,
    )


def main():
    parser = add_common_args(argparse.ArgumentParser(
        description="Qwen3.7-Plus API remote evaluation (official mm_agents.qwen.QwenAgent)"
    ))
    # Agent params — mirror official run_multienv_qwen.py names/defaults.
    parser.add_argument("--model", default="tongyi/qwen3.7-plus")
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
    # Single hosted endpoint.
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "https://api.llm.mioffice.cn/v1"),
                        help="Qwen API endpoint shared by all workers")
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--password", default="password")
    args = parser.parse_args()

    runner = EvalRunner(
        args,
        agent_factory=build_agent,
        task_source=OSWorldEvalSource(model_name=args.model),
        model_name=args.model,
        log_prefix="qwen-qwen37plus",
    )
    runner.run()


if __name__ == "__main__":
    main()
