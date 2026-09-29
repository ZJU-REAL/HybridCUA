"""Qwen3.5 + MCP sharded remote evaluation.

Same shape as ``run_qwen_qwen35vl_sharded.py`` (cluster ``EvalRunner`` +
``OSWorldEvalSource``, round-robin endpoint sharding) with the agent swapped for
``QwenMcpAgent`` and the episode loop for ``run_single_example_qwen_mcp``, which
brings up the guest MCP server and records tool usage.

Tools are always on: the action enum gains `mcp` plus name/params and the catalog is
embedded in the schema. There is no off/bash arm --
  off:  both arms would need the worked GUI example (stock QwenAgent ships no concrete
        <tool_call>, and showing only the MCP one would inflate its TIR), so `off`
        could never be byte-identical to stock. Use run_qwen_qwen35_9b_sharded.sh for
        a leaderboard-comparable baseline instead.
  bash: QwenAgent's action space is GUI primitives with no shell channel, so any way
        to express a tool call is itself an action-space change -- the same thing as
        `action`. Use run_c_gui_mcp_sharded.py for the three-arm study.

Usage:
    python scripts/python/osworld/run_qwen35_mcp_sharded.py \
        --openai_base_urls http://h:8000/v1,...,http://h:8007/v1 \
        --num_envs 64 --model Qwen3.5-9B
"""
import argparse
import logging
import os
import sys
from multiprocessing import current_process

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../OSWorld"))

from cluster.client import EvalRunner, add_common_args
from cluster.client.osworld import OSWorldEvalSource
from mm_agents.mcp_common import MCP_SRC, preflight
from mm_agents.qwen_mcp import QwenMcpAgent, run_single_example_qwen_mcp

#: OSWorld-MCP's 361-task list. Verified identical to upstream test_nogdrive.json.
DEFAULT_META = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../../OSWorld-MCP/evaluation_examples/test_all.json"))


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


def build_agent(args: argparse.Namespace, env=None) -> QwenMcpAgent:
    # Same construction as run_qwen_qwen35vl_sharded.py, except base_url is picked
    # per-worker (round-robin over the shard list).
    urls = _parse_base_urls(args)
    idx = _worker_index()
    selected = urls[idx % len(urls)]
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "dummy")
    logging.getLogger("desktopenv.experiment").info(
        "Worker %d using endpoint: %s", idx, selected
    )
    return QwenMcpAgent(
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
        description="Qwen3.5 + MCP sharded remote evaluation (mm_agents.qwen_mcp)"
    ))
    # Agent params -- mirror run_qwen_qwen35vl_sharded.py names/defaults.
    parser.add_argument("--model", default="Qwen3.5-9B",
                        help="Must match vLLM --served-model-name, else every request 404s.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Max output tokens. Must be < model context length "
                             "(vLLM rejects max_tokens >= context_length).")
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
    # MCP-specific.
    parser.add_argument("--mcp_tool_budget", type=int, default=0,
                        help="Cap on app tools per episode (0 = no cap). Distractors are "
                             "never dropped by the budget -- resisting them is what TIR scores.")
    parser.add_argument("--mcp_no_distractors", action="store_true",
                        help="Drop the 28 filesystem_*/git_* bait tools. NOTE: this makes TIR "
                             "incomparable to the published numbers.")
    args = parser.parse_args()

    # Default to OSWorld-MCP's task list when the caller did not override it.
    if args.test_all_meta_path in ("", None, "evaluation_examples/test_nogdrive.json"):
        if os.path.isfile(DEFAULT_META):
            args.test_all_meta_path = DEFAULT_META

    # Fail here, not per-episode: a missing bundle makes provision_mcp return False
    # and every episode silently degrades to GUI-only with tools=0.
    preflight()

    logging.getLogger("desktopenv.experiment").info(
        "budget=%s distractors=%s meta=%s mcp_src=%s",
        args.mcp_tool_budget or "none",
        "off" if args.mcp_no_distractors else "on", args.test_all_meta_path, MCP_SRC,
    )

    runner = EvalRunner(
        args,
        agent_factory=build_agent,
        task_source=OSWorldEvalSource(model_name=args.model,
                                      run_single_fn=run_single_example_qwen_mcp),
        model_name=args.model,
        log_prefix="qwen35-mcp",
    )
    runner.run()


if __name__ == "__main__":
    main()
