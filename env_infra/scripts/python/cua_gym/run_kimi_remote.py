"""CUA-Gym remote evaluation entry point (Kimi agent).

Mirrors ``scripts/python/osworld/run_kimi_remote.py`` but points the shared
:class:`EvalRunner` at CUA-Gym's task source (a directory of ``<uuid>/`` bundles)
and the ``cua_gym`` runtime env client. The agent and all parallel orchestration
are the same machinery OSWorld uses — only the task model + runtime differ.

Run: see ``scripts/bash/cua_gym/run_kimi_remote.sh`` (needs a node hosting the
``cua_gym`` world and a downloaded CUA-Gym bundle dir passed via --tasks_root).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))          # env_infra root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../OSWorld"))  # lib_run_single, mm_agents

from cluster.client import EvalRunner, add_common_args
from cluster.client.cua_gym import CuaGymEvalSource, CuaGymSessionClient
from mm_agents.kimi import KimiAgent
# KimiAgent.predict returns (response, actions, info_dict) — 3 values. The generic
# run_single_example unpacks only 2, so drive episodes with the kimi-specific loop.
from lib_run_single import run_single_example_kimi


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


def build_env(args: argparse.Namespace) -> CuaGymSessionClient:
    """One cua_gym cluster session per worker (reused across tasks)."""
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


def env_setup(args: argparse.Namespace) -> None:
    if args.base_url:
        os.environ["KIMI_BASE_URL"] = args.base_url
    if args.api_key:
        os.environ["KIMI_API_KEY"] = args.api_key


def main():
    parser = add_common_args(argparse.ArgumentParser(description="CUA-Gym remote evaluation (Kimi)"))
    # CUA-Gym task source: a directory of <uuid>/ bundles (replaces test_all_meta).
    parser.add_argument("--tasks_root", required=True,
                        help="Directory of CUA-Gym <uuid>/ bundles (task.json + reward.py + initial_setup.*)")
    # Optional OSWorld-style meta JSON pinning an exact subset: {app_type: [uuid, ...]}.
    # Omit to run every bundle under --tasks_root. --domain still applies on top.
    parser.add_argument("--tasks_meta", default=None,
                        help="JSON {app_type: [uuid, ...]} selecting specific tasks (default: all bundles under --tasks_root)")
    # --domain filters by app_type (e.g. libreoffice_calc,libreoffice_impress); default all.
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

    EvalRunner(
        args,
        agent_factory=build_agent,
        task_source=CuaGymEvalSource(model_name=args.model, run_single_fn=run_single_example_kimi),
        build_env=build_env,
        model_name=args.model,
        log_prefix="cua_gym-remote",
        env_setup=env_setup,
    ).run()


if __name__ == "__main__":
    main()
