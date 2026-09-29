"""c-gui + MCP sharded remote evaluation: three-arm A/B over MCP tool exposure.

Same shape as ``run_c_gui_sharded.py`` (cluster ``EvalRunner`` + ``OSWorldEvalSource``,
round-robin endpoint sharding) with the agent swapped for ``CGuiMcpAgent`` and the
episode loop for ``run_single_example_c_gui_mcp``, which brings up the guest MCP
server and records tool usage.

The three arms, selected by --mcp_mode:
    off     plain c-gui -- prompt and tool schema BYTE-IDENTICAL to CGuiAgent.
            This is the baseline; MCP_MODE=off must reproduce run_c_gui_sharded.py.
    bash    tools reachable through the EXISTING action=bash channel via a python3
            one-liner. enum/properties unchanged, so the action space really is
            unchanged; the catalog lives in the bash action's schema description.
    action  action enum gains `mcp` plus name/params. The model fills structured
            fields instead of hand-quoting shell.

Why two live arms: off->bash isolates "does the model know tools exist", bash->action
isolates "how hard is it to emit a call". Running only off vs action would confound
the two.

Usage:
    python scripts/python/osworld/run_c_gui_mcp_sharded.py \
        --openai_base_urls http://h:8000/v1,...,http://h:8007/v1 \
        --num_envs 64 --model Qwen3.5-9B --mcp_mode bash
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
from mm_agents.c_gui_mcp import CGuiMcpAgent, MCP_MODES, run_single_example_c_gui_mcp
from mm_agents.mcp_common import MCP_SRC, preflight

#: OSWorld-MCP's 361-task list. Verified identical to upstream test_nogdrive.json,
#: so results are comparable to a plain OSWorld run on the same set.
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


def build_agent(args: argparse.Namespace, env=None) -> CGuiMcpAgent:
    # base_url picked per-worker (round-robin over the shard list), same as
    # run_c_gui_sharded.py; mcp_mode is the only added constructor argument.
    urls = _parse_base_urls(args)
    idx = _worker_index()
    selected = urls[idx % len(urls)]
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "dummy")
    logging.getLogger("desktopenv.experiment").info(
        "Worker %d using endpoint: %s (tool_name=%s mcp_mode=%s)",
        idx, selected, args.tool_name, args.mcp_mode,
    )
    return CGuiMcpAgent(
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
        mcp_mode=args.mcp_mode,
    )


def main():
    parser = add_common_args(argparse.ArgumentParser(
        description="c-gui + MCP sharded remote evaluation (mm_agents.c_gui_mcp)"
    ))
    # Agent params -- mirror run_c_gui_sharded.py names/defaults.
    parser.add_argument("--model", default="Qwen3.5-9B",
                        help="Must match vLLM --served-model-name, else every request 404s.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Max output tokens. Must be < model context length. 2048 truncated "
                             "~10%% of episodes mid-tool_call, which the loop sees as an empty "
                             "action list and treats as a hard stop.")
    parser.add_argument("--history_n", type=int, default=100)
    parser.add_argument("--coord", type=str, default="relative", choices=["absolute", "relative"])
    parser.add_argument("--image_max", type=int, default=20)
    parser.add_argument("--fold_size", type=int, default=10)
    parser.add_argument("--add_thought_prefix", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--tool_name", default="computer_use", choices=["computer_use", "cli"])
    # Endpoint sharding.
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--openai_base_urls", default=os.environ.get("OPENAI_BASE_URLS"),
                        help="Comma-separated vLLM endpoints for round-robin sharding")
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--password", default="password")
    # MCP-specific.
    parser.add_argument("--mcp_mode", choices=list(MCP_MODES), default="bash",
                        help="off = plain c-gui baseline (byte-identical prompt); "
                             "bash = tools via the existing action=bash channel; "
                             "action = adds action=mcp with name/params")
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
    # and every episode silently degrades to GUI-only with tools=0. The off arm has
    # no tools by design, so it does not need the bundle.
    if args.mcp_mode != "off":
        preflight()

    logging.getLogger("desktopenv.experiment").info(
        "mcp_mode=%s budget=%s distractors=%s meta=%s mcp_src=%s",
        args.mcp_mode, args.mcp_tool_budget or "none",
        "off" if args.mcp_no_distractors else "on", args.test_all_meta_path,
        MCP_SRC if args.mcp_mode != "off" else "n/a",
    )

    runner = EvalRunner(
        args,
        agent_factory=build_agent,
        task_source=OSWorldEvalSource(model_name=args.model,
                                      run_single_fn=run_single_example_c_gui_mcp),
        model_name=args.model,
        log_prefix=f"c-gui-mcp-{args.mcp_mode}",
    )
    runner.run()


if __name__ == "__main__":
    main()
