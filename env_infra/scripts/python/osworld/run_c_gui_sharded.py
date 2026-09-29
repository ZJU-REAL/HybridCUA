"""c-gui (single bash-surface GUI+CLI) sharded remote evaluation.

Built on ``mm_agents.c_gui.CGuiAgent``. Structurally identical to
``run_hybrid_sharded.py`` — same cluster ``EvalRunner`` + ``OSWorldEvalSource`` (remote
OSWorldSessionClient) and round-robin endpoint sharding. The agent exposes ONE tool
(name = ``--tool_name``: ``computer_use`` or ``cli`` — the A/B naming variable) with a
single ``bash`` action; GUI is pyautogui inside a quoted heredoc (coords 0-999, scaled by
the VM shim). Constructor signature matches HybridAgent/QwenAgent plus ``--tool_name``, so
the sharded factory is otherwise unchanged. See ``cua-h/docs/codeact_harness_design.md``.

Usage:
    python scripts/python/osworld/run_c_gui_sharded.py \
        --openai_base_urls http://h:8000/v1,...,http://h:8007/v1 \
        --num_envs 8 --model qwen35-vl --tool_name computer_use
"""
import argparse
import os
import sys
from multiprocessing import current_process

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../OSWorld"))

from cluster.client import EvalRunner, add_common_args
from cluster.client.osworld import OSWorldEvalSource
from mm_agents.c_gui import CGuiAgent, run_single_example_c_gui


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


def build_agent(args: argparse.Namespace, env=None) -> CGuiAgent:
    # base_url picked per-worker (round-robin over the shard list). Same construction as
    # the hybrid sharded runner; the agent class differs (CGuiAgent = single bash tool)
    # and we pass --tool_name (the A/B naming variable).
    urls = _parse_base_urls(args)
    idx = _worker_index()
    selected = urls[idx % len(urls)]
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "dummy")
    import logging
    logging.getLogger("desktopenv.experiment").info(
        "Worker %d using endpoint: %s (tool_name=%s)", idx, selected, args.tool_name
    )
    return CGuiAgent(
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
        password=args.password,
        tool_name=args.tool_name,
    )


def main():
    parser = add_common_args(argparse.ArgumentParser(
        description="c-gui (single bash surface) sharded remote evaluation (mm_agents.c_gui.CGuiAgent)"
    ))
    # Agent params — mirror the hybrid sharded runner names/defaults.
    parser.add_argument("--model", default="qwen35-vl")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Max output tokens. Must be < model context length to leave room for input "
                             "(vLLM rejects requests where max_tokens >= context_length). 2048 truncated "
                             "~10% of episodes mid-tool_call, which the loop sees as an empty action list "
                             "and treats as a hard stop.")
    parser.add_argument("--history_n", type=int, default=100,
                        help="Number of recent steps kept in the prompt")
    parser.add_argument("--coord", type=str, default="relative", choices=["absolute", "relative"],
                        help="Kept for signature parity with hybrid; c-gui coords are 0-999 scaled in the VM shim")
    parser.add_argument("--image_max", type=int, default=20,
                        help="Max screenshots kept as images before old ones are collapsed to text")
    parser.add_argument("--fold_size", type=int, default=10,
                        help="Number of screenshots collapsed at a time once image_max is exceeded")
    parser.add_argument("--add_thought_prefix", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Enable model-side thinking (only effective for dashscope base_url)")
    parser.add_argument("--tool_name", default="computer_use", choices=["computer_use", "cli"],
                        help="Function-tool name — the A/B naming variable. Tool CONTENTS are identical.")
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
        task_source=OSWorldEvalSource(model_name=args.model, run_single_fn=run_single_example_c_gui),
        model_name=args.model,
        log_prefix="c-gui-sharded",
    )
    runner.run()


if __name__ == "__main__":
    main()
