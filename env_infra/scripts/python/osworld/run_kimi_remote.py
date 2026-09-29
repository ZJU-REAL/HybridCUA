"""Kimi K2.5/K2.6 remote evaluation entry point.

Thin wrapper: injects KimiAgent into the generic EvalRunner framework.
All parallel execution, task distribution, and result management is handled
by cluster.client.EvalRunner.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../OSWorld"))

from cluster.client import EvalRunner, add_common_args
from cluster.client.osworld import KimiOSWorldEvalSource
from mm_agents.kimi import KimiAgent


def build_agent(args: argparse.Namespace, env=None) -> KimiAgent:
    return KimiAgent(
        model=args.model,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        action_space=args.action_space,
        observation_type=args.observation_type,
        screen_size=(args.screen_width, args.screen_height),
        coordinate_type=args.coordinate_type,
        max_image_history_length=args.max_image_history_length,
        max_steps=args.max_steps,
        thinking=args.thinking,
        password=args.password,
    )


def env_setup(args: argparse.Namespace) -> None:
    if args.base_url:
        os.environ["KIMI_BASE_URL"] = args.base_url
    if args.api_key:
        os.environ["KIMI_API_KEY"] = args.api_key


def main():
    parser = add_common_args(argparse.ArgumentParser(description="Kimi remote evaluation"))
    parser.add_argument("--model", default="moonshot/kimi-k2.6")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--base_url", default=os.environ.get("KIMI_BASE_URL"))
    parser.add_argument("--api_key", default=os.environ.get("KIMI_API_KEY"))
    parser.add_argument("--coordinate_type", default="relative", choices=["relative", "absolute", "qwen25"])
    parser.add_argument("--max_image_history_length", type=int, default=3)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--password", default="osworld-public-evaluation")
    args = parser.parse_args()

    runner = EvalRunner(
        args,
        agent_factory=build_agent,
        task_source=KimiOSWorldEvalSource(model_name=args.model),
        model_name=args.model,
        log_prefix="kimi-remote",
        env_setup=env_setup,
    )
    runner.run()


if __name__ == "__main__":
    main()
