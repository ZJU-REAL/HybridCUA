"""Qwen3-VL remote evaluation entry point.

Thin wrapper: injects Qwen3VLAgent into the generic EvalRunner framework.
All parallel execution, task distribution, and result management is handled
by cluster.client.EvalRunner.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../OSWorld"))

from cluster.client import EvalRunner, add_common_args
from cluster.client.osworld import OSWorldEvalSource
from mm_agents.qwen3vl_agent import Qwen3VLAgent


def build_agent(args: argparse.Namespace, env=None) -> Qwen3VLAgent:
    return Qwen3VLAgent(
        model=args.model,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        action_space=args.action_space,
        observation_type=args.observation_type,
        coordinate_type=args.coordinate_type,
        history_n=args.max_image_history_length,
        add_thought_prefix=args.add_thought_prefix,
        api_backend="openai",
        enable_thinking=args.enable_thinking,
        thinking_budget=args.thinking_budget,
    )


def env_setup(args: argparse.Namespace) -> None:
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key


def main():
    parser = add_common_args(argparse.ArgumentParser(description="Qwen3-VL remote evaluation"))
    parser.add_argument("--model", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", "sk-local"))
    parser.add_argument("--coordinate_type", default="relative", choices=["relative", "absolute"])
    parser.add_argument("--max_image_history_length", type=int, default=4)
    parser.add_argument("--add_thought_prefix", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--thinking_budget", type=int, default=32768)
    parser.add_argument("--password", default="password")
    args = parser.parse_args()

    runner = EvalRunner(
        args,
        agent_factory=build_agent,
        task_source=OSWorldEvalSource(model_name=args.model),
        model_name=args.model,
        log_prefix="qwen3vl-remote",
        env_setup=env_setup,
    )
    runner.run()


if __name__ == "__main__":
    main()
