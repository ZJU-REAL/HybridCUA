"""EvoCUA-32B (pure-GUI) sharded remote evaluation.

Built on ``mm_agents.evocua.evocua_agent.EvoCUAAgent`` -- meituan/EvoCUA-32B-20260105,
a Qwen3-VL derivative trained for computer use (native ``computer_use`` tool, JSON
tool-calls, 14 GUI actions). Same cluster ``EvalRunner`` + ``OSWorldEvalSource`` as the
other sharded runners, with round-robin endpoint sharding.

Two differences from ``run_c_gui_sharded.py``:

  * **No custom run_loop.** ``EvoCUAAgent._predict_s2`` returns ``(response, pyautogui_code)``
    -- exactly the contract ``lib_run_single.run_single_example`` expects at line 35 --
    so ``OSWorldEvalSource`` keeps its default ``run_single_fn``. c-gui needed its own
    loop only because it returns channel-tagged action dicts.

  * **Sharding via process env, not a constructor arg.** ``EvoCUAAgent.call_llm``
    (evocua_agent.py:627-631) reads ``OPENAI_BASE_URL`` from ``os.environ`` and takes no
    ``base_url`` parameter. Rather than patch the vendored agent (CLAUDE.md: OSWorld
    internals are read-only), ``build_agent`` sets the env var per worker. This is safe
    because ``EvalRunner`` calls ``agent_factory`` inside ``_worker``
    (cluster/client/base/eval_worker.py:277), i.e. in the forked child -- each worker
    mutates only its own copy of the environment.

    Caveat: that ``call_llm`` has NO retry (it re-raises on any exception), unlike
    ``mm_agents/qwen/client.py`` which retries 5x. A transient endpoint hiccup kills the
    episode. Verify all shards are healthy before a full run.

Usage:
    python scripts/python/osworld/run_evocua_sharded.py \\
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
from mm_agents.evocua.evocua_agent import EvoCUAAgent


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


def build_agent(args: argparse.Namespace, env=None) -> EvoCUAAgent:
    # Endpoint is pinned per-worker (round-robin over the shard list) by exporting
    # OPENAI_BASE_URL, which is the only channel EvoCUAAgent.call_llm reads. We are in
    # the worker's own process here, so this does not leak across workers.
    urls = _parse_base_urls(args)
    idx = _worker_index()
    selected = urls[idx % len(urls)]
    os.environ["OPENAI_BASE_URL"] = selected
    os.environ["OPENAI_API_KEY"] = args.api_key or os.environ.get("OPENAI_API_KEY", "dummy")

    import logging
    logging.getLogger("desktopenv.experiment").info(
        "Worker %d using endpoint: %s (prompt_style=%s)", idx, selected, args.prompt_style
    )
    return EvoCUAAgent(
        model=args.model,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        action_space=args.action_space,
        observation_type=args.observation_type,
        max_steps=args.max_steps,
        prompt_style=args.prompt_style,
        # EvoCUA names its history window max_history_turns (screenshot turns kept in
        # the prompt); --history_n is the repo-wide spelling used by every other runner.
        max_history_turns=args.history_n,
        screen_size=(args.screen_width, args.screen_height),
        coordinate_type=args.coord,
        password=args.password,
    )


def main():
    parser = add_common_args(argparse.ArgumentParser(
        description="EvoCUA-32B pure-GUI sharded remote evaluation (mm_agents.evocua.EvoCUAAgent)"
    ))
    # Agent params.
    parser.add_argument("--model", default="EvoCUA",
                        help="Must match vLLM --served-model-name, else every request 404s")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="Max output tokens. Must stay well below the served "
                             "--max-model-len (32768) to leave room for screenshots.")
    parser.add_argument("--history_n", type=int, default=4,
                        help="Screenshot turns kept in the prompt (EvoCUA max_history_turns; "
                             "its own default is 4)")
    parser.add_argument("--coord", type=str, default="relative", choices=["absolute", "relative"],
                        help="EvoCUA rescales tool-call coordinates to the real screen using this")
    parser.add_argument("--prompt_style", default="S2", choices=["S1", "S2"],
                        help="S2 is the released prompt format (JSON tool_call in <tool_call> tags)")
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
        # Default run_single_fn: EvoCUA returns (response, pyautogui_code), which is what
        # OSWorld's stock episode loop consumes.
        task_source=OSWorldEvalSource(model_name=args.model),
        model_name=args.model,
        log_prefix="evocua-sharded",
    )
    runner.run()


if __name__ == "__main__":
    main()
