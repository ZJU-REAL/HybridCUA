"""EvoCUA-32B hybrid (GUI + CLI) sharded remote evaluation.

Same model and sharding as ``run_evocua_sharded.py``; the difference is the tool
surface. ``EvoCUAHybridAgent`` (mm_agents/evocua_hybrid) exposes EvoCUA's native
``computer_use`` GUI tool PLUS the ``cli`` tool from ``mm_agents/hybrid/cli_tools.py``
(bash / read / write / edit), and returns channel-tagged action dicts driven by
``run_single_example_hybrid``.

This is the treatment arm of an A/B: run ``run_evocua.sh`` (pure GUI) and this script
over the same task set with the same settings, and the score delta answers whether a
CLI surface helps a model trained purely on GUI computer-use.

Usage:
    python scripts/python/osworld/run_evocua_hybrid_sharded.py \\
        --openai_base_urls http://h:8000/v1,...,http://h:8003/v1 \\
        --num_envs 32 --model EvoCUA
"""
import argparse
import os
import sys
from multiprocessing import current_process

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../OSWorld"))

from cluster.client import EvalRunner, add_common_args
from cluster.client.osworld import OSWorldEvalSource
from mm_agents.evocua_hybrid import EvoCUAHybridAgent, run_single_example_hybrid


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


def build_agent(args: argparse.Namespace, env=None) -> EvoCUAHybridAgent:
    # Endpoint pinned per-worker via OPENAI_BASE_URL — the only channel EvoCUA's
    # call_llm reads. Safe because agent_factory runs inside the forked worker
    # (cluster/client/base/eval_worker.py:277). See run_evocua_sharded.py for detail.
    urls = _parse_base_urls(args)
    idx = _worker_index()
    selected = urls[idx % len(urls)]
    os.environ["OPENAI_BASE_URL"] = selected
    os.environ["OPENAI_API_KEY"] = args.api_key or os.environ.get("OPENAI_API_KEY", "dummy")

    import logging
    logging.getLogger("desktopenv.experiment").info(
        "Worker %d using endpoint: %s (hybrid GUI+CLI)", idx, selected
    )
    return EvoCUAHybridAgent(
        model=args.model,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        action_space=args.action_space,
        observation_type=args.observation_type,
        max_steps=args.max_steps,
        prompt_style="S2",  # hybrid extends the S2 tool surface; S1 has none
        max_history_turns=args.history_n,
        screen_size=(args.screen_width, args.screen_height),
        coordinate_type=args.coord,
        password=args.password,
    )


def main():
    parser = add_common_args(argparse.ArgumentParser(
        description="EvoCUA-32B hybrid GUI+CLI sharded remote evaluation"
    ))
    parser.add_argument("--model", default="EvoCUA",
                        help="Must match vLLM --served-model-name, else every request 404s")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="Max output tokens; stay well below the served --max-model-len")
    parser.add_argument("--history_n", type=int, default=4,
                        help="Screenshot turns kept in the prompt (EvoCUA max_history_turns)")
    parser.add_argument("--coord", type=str, default="relative", choices=["absolute", "relative"])
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--openai_base_urls", default=os.environ.get("OPENAI_BASE_URLS"),
                        help="Comma-separated vLLM endpoints for round-robin sharding")
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--password", default="password",
                        help="sudo password stated in the system prompt (for action=bash)")
    args = parser.parse_args()

    runner = EvalRunner(
        args,
        agent_factory=build_agent,
        # Channel-tagged dicts need hybrid's loop (GUI -> env.step, CLI -> env.run_code).
        task_source=OSWorldEvalSource(model_name=args.model, run_single_fn=run_single_example_hybrid),
        model_name=args.model,
        log_prefix="evocua-hybrid-sharded",
    )
    runner.run()


if __name__ == "__main__":
    main()
