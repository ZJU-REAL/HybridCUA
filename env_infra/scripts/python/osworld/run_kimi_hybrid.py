"""Kimi single-bash-surface (GUI+CLI) OSWorld evaluation entry point.

Same cluster plumbing as the other OSWorld runners — ``EvalRunner`` + ``OSWorldEvalSource``
with a remote ``OSWorldSessionClient``, not a local ``DesktopEnv`` — but the agent is
:class:`mm_agents.kimi_hybrid.KimiHybridAgent`: Kimi driving ONE bash surface where a GUI
interaction is a pyautogui heredoc and a CLI/file operation is plain shell, both through
``env.run_code(lang="bash")``.

The episode loop is the shared ``run_single_example_kimi_hybrid``
(== ``mm_agents.c_gui_pixel.run_single_example_c_gui_pixel``), passed via
``run_single_fn``. It provisions the VM pyautogui shim after reset, executes bash actions,
and threads each command's stdout/stderr into the next ``predict`` so the model reads real
terminal output.

Coordinates are REAL SCREEN PIXELS: the screenshot is sent at full resolution and nothing
rescales it, so ``--screen_width/--screen_height`` must match the VM's actual screen.

Usage:
    # verify the wiring first (~3 calls); do this after ANY change
    python scripts/python/osworld/run_kimi_hybrid.py --selftest_only

    python scripts/python/osworld/run_kimi_hybrid.py \
        --model moonshot/kimi-k2.6 --thinking --num_envs 16 --max_steps 50
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../OSWorld"))

from cluster.client import EvalRunner, add_common_args
from cluster.client.osworld import OSWorldEvalSource
from mm_agents.kimi_hybrid import KimiHybridAgent, run_single_example_kimi_hybrid

#: Egress proxy exported before each VM bash command. The VM cannot reach the public
#: internet directly, and the session client's ``enable_proxy`` only covers chrome.
DEFAULT_VM_PROXY = "http://127.0.0.1:3128"
DEFAULT_VM_NO_PROXY = "localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"


def build_agent(args: argparse.Namespace, env=None) -> KimiHybridAgent:
    return KimiHybridAgent(
        model=args.model,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        action_space=args.action_space,
        observation_type=args.observation_type,
        screen_size=(args.screen_width, args.screen_height),
        # Coordinates are absolute pixels. The parent's projection is never invoked by
        # KimiHybridAgent (the code block is shell, not python), so this only documents
        # intent — but a "relative" value here would be actively misleading.
        coordinate_type="absolute",
        max_image_history_length=args.max_image_history_length,
        max_steps=args.max_steps,
        thinking=args.thinking,
        password=args.password,
        max_output_chars=args.max_output_chars,
    )


def env_setup(args: argparse.Namespace) -> None:
    """Pin the gateway env vars per worker (``KimiAgent.call_llm`` reads them)."""
    if args.base_url:
        os.environ["KIMI_BASE_URL"] = args.base_url
    if args.api_key:
        os.environ["KIMI_API_KEY"] = args.api_key


def main():
    parser = add_common_args(argparse.ArgumentParser(
        description="Kimi single-bash-surface (GUI+CLI) OSWorld evaluation"
    ))
    parser.add_argument("--model", default=os.environ.get("MODEL", "moonshot/kimi-k2.6"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--base_url", default=os.environ.get("KIMI_BASE_URL"))
    parser.add_argument("--api_key", default=os.environ.get("KIMI_API_KEY"))
    parser.add_argument("--max_image_history_length", type=int, default=3,
                        help="how many recent turns keep their screenshot (K2.5: 3, K2.6: 8-10)")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--password", default="osworld-public-evaluation",
                        help="VM sudo password stated in the system prompt")
    parser.add_argument("--max_output_chars", type=int, default=2000,
                        help="per-step command-output budget in the prompt; head-kept "
                             "(the overflow is dropped and marked)")
    parser.add_argument("--wait_after_reset", type=float, default=60,
                        help="seconds to wait after env.reset() before the first observation")
    parser.add_argument("--vm_proxy", default=os.environ.get("VM_PROXY", DEFAULT_VM_PROXY),
                        help="proxy exported before each VM bash command ('' to disable)")
    parser.add_argument("--vm_no_proxy", default=os.environ.get("VM_NO_PROXY", DEFAULT_VM_NO_PROXY))
    parser.add_argument("--selftest_only", action="store_true",
                        help="verify gateway wiring (text + image channel) and exit")
    args = parser.parse_args()

    if args.selftest_only:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                            format="%(levelname)s %(name)s: %(message)s")
        env_setup(args)
        print(f"gateway {os.environ.get('KIMI_BASE_URL', '<default>')}\nmodel   {args.model}")
        ok = build_agent(args).selftest()
        print("\nselftest passed." if ok else "\nselftest FAILED.")
        sys.exit(0 if ok else 3)

    runner = EvalRunner(
        args,
        agent_factory=build_agent,
        task_source=OSWorldEvalSource(
            model_name=args.model,
            run_single_fn=run_single_example_kimi_hybrid,
        ),
        model_name=args.model,
        log_prefix="kimi-hybrid",
        env_setup=env_setup,
    )
    runner.run()


if __name__ == "__main__":
    main()
