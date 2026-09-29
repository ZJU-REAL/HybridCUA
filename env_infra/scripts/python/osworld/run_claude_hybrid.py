"""Claude single-bash-surface (GUI+CLI) OSWorld evaluation entry point.

Same cluster plumbing as the other OSWorld runners — ``EvalRunner`` + ``OSWorldEvalSource``
with a remote ``OSWorldSessionClient``, not a local ``DesktopEnv`` — but the agent is
:class:`mm_agents.claude_hybrid.ClaudeHybridAgent`: Claude driving ONE bash surface where a
GUI interaction is a pyautogui heredoc and a CLI/file operation is plain shell, both through
``env.run_code(lang="bash")``. There is no native ``computer`` tool and no separate bash or
text-editor tool: one custom four-action tool covers everything.

There is no endpoint sharding — Claude is a hosted API, so every worker shares one account
(auth via ``--auth_token`` for a gateway, ``--api_key`` for the first-party API, or AWS_*
env vars for ``--provider bedrock``).

The episode loop is the shared ``run_single_example_claude_hybrid``
(== ``mm_agents.c_gui_pixel.run_single_example_c_gui_pixel``). The agent's message loop is
stateful (``tool_use`` <-> ``tool_result`` pairing by id), but the loop does not need to
know that: it hands back each command's output and the agent does the pairing.

Coordinates are REAL SCREEN PIXELS, so ``--screen_width/--screen_height`` must match the
VM's actual screen. Note that Anthropic downscales images whose long edge exceeds 1568px,
so at 1920x1080 the model sees a ~1568x882 rendering while reasoning in 1920x1080
coordinates.

Usage:
    # verify the wiring first (~3 calls, tool schema included); do this after ANY change
    python scripts/python/osworld/run_claude_hybrid.py --selftest_only

    python scripts/python/osworld/run_claude_hybrid.py \
        --provider anthropic --model claude-opus-4-8 --num_envs 8 --max_steps 50
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../OSWorld"))

from cluster.client import EvalRunner, add_common_args
from cluster.client.osworld import OSWorldEvalSource
from mm_agents.anthropic.utils import APIProvider
from mm_agents.claude_hybrid import ClaudeHybridAgent, run_single_example_claude_hybrid

#: Egress proxy exported before each VM bash command. The VM cannot reach the public
#: internet directly, and the session client's ``enable_proxy`` only covers chrome.
DEFAULT_VM_PROXY = "http://127.0.0.1:3128"
DEFAULT_VM_NO_PROXY = "localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"


def build_agent(args: argparse.Namespace, env=None) -> ClaudeHybridAgent:
    return ClaudeHybridAgent(
        platform=args.platform,
        model=args.model,
        provider=APIProvider(args.provider),
        base_url=args.base_url or os.environ.get("ANTHROPIC_BASE_URL"),
        auth_token=args.auth_token or os.environ.get("ANTHROPIC_AUTH_TOKEN"),
        api_key=args.api_key or os.environ.get("ANTHROPIC_API_KEY"),
        max_tokens=args.max_tokens,
        effort=args.effort,
        no_thinking=args.no_thinking,
        use_isp=args.use_isp,
        only_n_most_recent_images=args.only_n_most_recent_images,
        screen_size=(args.screen_width, args.screen_height),
        max_steps=args.max_steps,
        password=args.password,
        max_output_chars=args.max_output_chars,
    )


def main():
    parser = add_common_args(argparse.ArgumentParser(
        description="Claude single-bash-surface (GUI+CLI) OSWorld evaluation"
    ))
    parser.add_argument("--model", default=os.environ.get("MODEL", "claude-opus-4-8"))
    parser.add_argument("--provider", default=os.environ.get("ANTHROPIC_PROVIDER", "anthropic"),
                        choices=[p.value for p in APIProvider],
                        help="anthropic (gateway/direct via auth_token+base_url) | bedrock (AWS_* env) | vertex")
    parser.add_argument("--base_url", default=os.environ.get("ANTHROPIC_BASE_URL"),
                        help="Anthropic-compatible gateway base URL. Setting it also disables "
                             "the prompt-caching beta, which a gateway may not honour.")
    parser.add_argument("--auth_token", default=os.environ.get("ANTHROPIC_AUTH_TOKEN"),
                        help="Bearer token for the gateway (preferred over --api_key)")
    parser.add_argument("--api_key", default=os.environ.get("ANTHROPIC_API_KEY"),
                        help="x-api-key auth (used only when --auth_token is unset)")
    parser.add_argument("--platform", default="Ubuntu")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"],
                        help="Adaptive-thinking effort (Claude 4.7+ profiles)")
    parser.add_argument("--no_thinking", action="store_true",
                        help="Disable thinking (legacy 4.5/4.6 profiles only)")
    parser.add_argument("--use_isp", action="store_true",
                        help="Interleaved scratchpad thinking (legacy 4.5/4.6 profiles only)")
    parser.add_argument("--only_n_most_recent_images", type=int, default=10,
                        help="Keep only the N most recent screenshots as images (older ones dropped)")
    parser.add_argument("--password", default="osworld-public-evaluation",
                        help="VM sudo password stated in the system prompt")
    parser.add_argument("--max_output_chars", type=int, default=4000,
                        help="per-command output budget in the prompt; head-kept "
                             "(the overflow is dropped and marked)")
    parser.add_argument("--wait_after_reset", type=float, default=60,
                        help="seconds to wait after env.reset() before the first observation")
    parser.add_argument("--vm_proxy", default=os.environ.get("VM_PROXY", DEFAULT_VM_PROXY),
                        help="proxy exported before each VM bash command ('' to disable)")
    parser.add_argument("--vm_no_proxy", default=os.environ.get("VM_NO_PROXY", DEFAULT_VM_NO_PROXY))
    parser.add_argument("--selftest_only", action="store_true",
                        help="verify API wiring (text, image channel, tool schema) and exit")
    args = parser.parse_args()

    if args.provider == "anthropic" and not (args.auth_token or args.api_key):
        parser.error("--provider anthropic needs --auth_token (Bearer, preferred) or --api_key")

    if args.selftest_only:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                            format="%(levelname)s %(name)s: %(message)s")
        print(f"endpoint {args.base_url or '<first-party API>'}\nmodel    {args.model}")
        ok = build_agent(args).selftest()
        print("\nselftest passed." if ok else "\nselftest FAILED.")
        sys.exit(0 if ok else 3)

    runner = EvalRunner(
        args,
        agent_factory=build_agent,
        task_source=OSWorldEvalSource(
            model_name=args.model,
            run_single_fn=run_single_example_claude_hybrid,
        ),
        model_name=args.model,
        log_prefix="claude-hybrid",
    )
    runner.run()


if __name__ == "__main__":
    main()
