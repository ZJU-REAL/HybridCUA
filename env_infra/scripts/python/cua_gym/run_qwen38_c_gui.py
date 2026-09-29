"""Qwen3.8-27B CUA-Gym sampling: gateway prompt + c_gui folded context, sharded vLLM.

Wiring only -- the agent is in ``qwen38_c_gui_agent.py``, the episode loop in
``qwen38_c_gui_loop.py``. Cluster plumbing matches ``run_qwen_qwen35vl_sharded.py``
(``CuaGymEvalSource`` + ``CuaGymSessionClient``, each worker pinning one vLLM endpoint by
worker index). Sharding exists here but not in the gateway entry point, where the single
local daemon serves every worker.

Usage: see ``scripts/bash/cua_gym/run_qwen38_c_gui.sh``.
"""
import argparse
import logging
import os
import sys
from multiprocessing import current_process

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))          # env_infra root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../OSWorld"))  # mm_agents, lib_*
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                  # sibling modules

from cluster.client import EvalRunner, add_common_args
from cluster.client.cua_gym import CuaGymEvalSource, CuaGymSessionClient

from qwen38_c_gui_agent import Qwen38CGuiAgent
from qwen38_c_gui_loop import run_single_example_qwen38_c_gui

logger = logging.getLogger("desktopenv.experiment")

# VM-side egress proxy for the bash channel: CUA-Gym VMs cannot reach the public
# internet directly (curl returns 000) and CuaGymSessionClient(enable_proxy=True) only
# adds --proxy-server to google-chrome. Set VM_PROXY="" to disable.
DEFAULT_VM_PROXY = "http://127.0.0.1:3128"
DEFAULT_VM_NO_PROXY = ("localhost,127.0.0.1,::1,.oa.com,.woa.com,.tencent.com,"
                       ".myqcloud.com,.tencentcos.cn,.tencentyun.com,.local,"
                       "10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,169.254.0.0/16")


def _parse_base_urls(args: argparse.Namespace) -> list:
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


def build_agent(args: argparse.Namespace, env=None) -> Qwen38CGuiAgent:
    """One agent per worker, pinned to one vLLM endpoint (round-robin by worker index)."""
    urls = _parse_base_urls(args)
    idx = _worker_index()
    selected = urls[idx % len(urls)]
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "dummy")
    logger.info("Worker %d using endpoint: %s", idx, selected)
    return Qwen38CGuiAgent(
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
    )


def build_env(args: argparse.Namespace) -> CuaGymSessionClient:
    """One cua_gym cluster session per worker (reused across that worker's tasks)."""
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
        description="Qwen3.8-27B CUA-Gym sampling (gateway prompt + c_gui folded context)"))
    # CUA-Gym task source: a directory of <uuid>/ bundles (replaces test_all_meta).
    parser.add_argument("--tasks_root", required=True,
                        help="Directory of CUA-Gym <uuid>/ bundles "
                             "(task.json + reward.py + initial_setup.*)")
    parser.add_argument("--tasks_meta", default=None,
                        help="JSON {app_type: [uuid, ...]} selecting specific tasks "
                             "(default: every bundle under --tasks_root)")
    # -- model --
    parser.add_argument("--model", default=os.environ.get("MODEL", "Qwen3.8-27B"),
                        help="--served-model-name of the vLLM servers")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Max OUTPUT tokens, and must stay well below the server's "
                             "--max-model-len (vLLM rejects max_tokens >= context length). "
                             "This model writes long reasoning before the tool_call: at 4096, "
                             "169 of 373 measured episodes were cut mid-tool_call (76%% missing "
                             "the closing tag), which the loop sees as an empty action list and "
                             "treats as a hard stop. Raise MAX_TOKENS to recover those.")
    # -- context (c_gui fold) --
    parser.add_argument("--history_n", type=int, default=50,
                        help="Number of recent steps kept in the prompt")
    parser.add_argument("--image_max", type=int, default=5,
                        help="Max screenshots kept as images before old ones collapse to text. "
                             "Must be <= the server's --limit-mm-per-prompt image cap.")
    parser.add_argument("--fold_size", type=int, default=1,
                        help="Number of screenshots collapsed at a time once image_max is exceeded")
    parser.add_argument("--coord", type=str, default="relative", choices=["absolute", "relative"],
                        help="Signature parity with the qwen runners; c-gui coords are always "
                             "0-999, scaled in the VM shim")
    parser.add_argument("--add_thought_prefix", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Model-side thinking (only effective for dashscope base_url)")
    parser.add_argument("--password", default="password",
                        help="VM sudo password quoted in the system prompt")
    # -- per-domain CLI skills (scripts/python/cua_gym/cli_skills/) --
    parser.add_argument("--cli_skills", dest="cli_skills", action="store_true",
                        default=os.environ.get("CLI_SKILLS", "1") == "1",
                        help="Inject the task app_type's CLI skill block into the system "
                             "prompt (default on; CLI_SKILLS=0 or --no_cli_skills disables)")
    parser.add_argument("--no_cli_skills", dest="cli_skills", action="store_false",
                        help="Disable CLI skill injection (A/B baseline)")
    parser.add_argument("--wait_after_reset", type=float, default=60,
                        help="Seconds to wait after env.reset() before the first observation")
    # -- endpoint sharding --
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL"),
                        help="Single-endpoint fallback when --openai_base_urls is unset")
    parser.add_argument("--openai_base_urls", default=os.environ.get("OPENAI_BASE_URLS"),
                        help="Comma-separated vLLM endpoints for round-robin sharding")
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY"))
    # -- VM egress proxy for the bash channel --
    parser.add_argument("--vm_proxy", default=os.environ.get("VM_PROXY", DEFAULT_VM_PROXY),
                        help="Proxy exported before each VM bash command ('' to disable)")
    parser.add_argument("--vm_no_proxy", default=os.environ.get("VM_NO_PROXY", DEFAULT_VM_NO_PROXY))
    args = parser.parse_args()

    EvalRunner(
        args,
        agent_factory=build_agent,
        task_source=CuaGymEvalSource(model_name=args.model,
                                     run_single_fn=run_single_example_qwen38_c_gui),
        build_env=build_env,
        model_name=args.model,
        log_prefix="cua_gym-qwen38-c-gui",
    ).run()


if __name__ == "__main__":
    main()
