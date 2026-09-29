"""OpenCUA sharded remote evaluation (OpenCUA-7B / OpenCUA-32B).

Built on ``scripts/python/osworld/opencua_local.OpenCUALocalAgent`` -- the platform-side
subclass of OSWorld's vendored ``mm_agents.opencua.OpenCUAAgent`` (xlangai/OpenCUA), a
Qwen2.5-VL derivative whose M-RoPE was replaced with 1D RoPE and whose tokenizer is
Kimi-VL's TikTokenV3. Same cluster ``EvalRunner`` + ``OSWorldEvalSource`` as the other
sharded runners, with round-robin endpoint sharding.

Three differences from ``run_evocua_sharded.py``:

  * **The endpoint is a constructor arg, not an env var.** ``EvoCUAAgent.call_llm``
    reads ``OPENAI_BASE_URL`` from ``os.environ``, so that runner exports it per worker.
    ``OpenCUAAgent.call_llm`` instead hardcodes ``https://{model}.app.msh.team/...``
    (opencua_agent.py:451) and ignores the environment entirely, so
    ``OpenCUALocalAgent`` takes ``base_url``/``api_key`` and we pass the shard in. See
    ``opencua_local.py`` for why subclassing beats patching the vendored agent.

  * **A custom run_single_fn.** ``OSWorldEvalSource``'s default is
    ``lib_run_single.run_single_example``, which expects ``predict`` to return a
    2-tuple. OpenCUAAgent returns ``(response, pyautogui_actions, other_cot)``, and its
    matching loop ``run_single_example_opencua`` writes the same traj.jsonl but omits
    ``log_task_completion`` -- which would leave ``summary/results.json`` missing and
    break ``eval_ckpt_serial.sh``'s success check. We pass the wrapper that adds it.

  * **OpenCUA-specific prompt knobs**: ``--cot_level``, ``--history_type``,
    ``--max_image_history_length``, ``--use_old_sys_prompt``.

Usage:
    python scripts/python/osworld/run_opencua_sharded.py \\
        --openai_base_urls http://h:8000/v1,...,http://h:8003/v1 \\
        --num_envs 32 --model OpenCUA-32B --use_old_sys_prompt
"""
import argparse
import os
import sys
from multiprocessing import current_process

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../OSWorld"))

from cluster.client import EvalRunner, add_common_args
from cluster.client.osworld import OSWorldEvalSource
from opencua_local import OpenCUALocalAgent, run_single_example_opencua_logged


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


def build_agent(args: argparse.Namespace, env=None) -> OpenCUALocalAgent:
    # Endpoint is pinned per-worker (round-robin over the shard list). We are in the
    # worker's own process here, so nothing leaks across workers.
    urls = _parse_base_urls(args)
    idx = _worker_index()
    selected = urls[idx % len(urls)]
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "sk-local")

    import logging
    logging.getLogger("desktopenv.experiment").info(
        "Worker %d using endpoint: %s (cot_level=%s, coord=%s)",
        idx, selected, args.cot_level, args.coord,
    )
    return OpenCUALocalAgent(
        base_url=selected,
        api_key=api_key,
        model=args.model,
        # OpenCUAAgent requires history_type positionally (no default); action_history
        # is the upstream default and what --history_type documents.
        history_type=args.history_type,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        action_space=args.action_space,
        observation_type=args.observation_type,
        cot_level=args.cot_level,
        # Repo-wide spelling is --history_n; the agent calls it max_image_history_length
        # ("the max number of images in the history").
        max_image_history_length=args.history_n,
        screen_size=(args.screen_width, args.screen_height),
        coordinate_type=args.coord,
        use_old_sys_prompt=args.use_old_sys_prompt,
        password=args.password,
    )


def main():
    parser = add_common_args(argparse.ArgumentParser(
        description="OpenCUA sharded remote evaluation (mm_agents.opencua.OpenCUAAgent)"
    ))
    # Agent params.
    parser.add_argument("--model", default="OpenCUA-32B",
                        help="Must match vLLM --served-model-name, else every request 404s")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Max output tokens. Must stay well below the served "
                             "--max-model-len (32768) to leave room for 3 screenshots; "
                             "vLLM rejects requests where max_tokens >= context_length. "
                             "OpenCUA emits a long reflective CoT -- raise this if "
                             "trajectories end mid-thought.")
    parser.add_argument("--history_n", type=int, default=3,
                        help="Screenshots kept in the prompt (OpenCUA "
                             "max_image_history_length; its own default is 3)")
    parser.add_argument("--coord", type=str, default="qwen25",
                        choices=["relative", "absolute", "qwen25"],
                        help="How to project the model's coordinates back onto the real "
                             "screen. Upstream's own command uses qwen25 (Qwen2.5-VL "
                             "smart-resize space, /28 factor); if clicks land "
                             "systematically off-target, this is the first knob to try.")
    parser.add_argument("--cot_level", type=str, default="l2", choices=["l1", "l2", "l3"],
                        help="CoT verbosity. l2 = thought + action (upstream default)")
    parser.add_argument("--history_type", type=str, default="action_history",
                        choices=["action_history", "thought_history", "observation_history"],
                        help="How prior steps are rendered into the prompt")
    parser.add_argument("--use_old_sys_prompt", action="store_true",
                        help="Use SYSTEM_PROMPT_V1_* instead of build_sys_prompt(). "
                             "Upstream's docstring puts this on the OpenCUA-7B and 32B "
                             "commands and omits it for 72B -- pass it for 7B/32B.")
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
        # Custom loop: OpenCUAAgent.predict returns a 3-tuple, and the matching vendor
        # loop skips log_task_completion (see module docstring).
        task_source=OSWorldEvalSource(model_name=args.model,
                                      run_single_fn=run_single_example_opencua_logged),
        model_name=args.model,
        log_prefix="opencua-sharded",
    )
    runner.run()


if __name__ == "__main__":
    main()
