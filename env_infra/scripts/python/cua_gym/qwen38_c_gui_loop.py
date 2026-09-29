"""Episode loop for the Qwen3.8-27B c-gui agent on CUA-Gym.

Composes two existing loops rather than forking either, since each has half of what a
0-999 agent on CUA-Gym needs:

  * ``mm_agents.c_gui.shim`` -- the pyautogui shim WITH coordinate scaling
    (``CUA_COORD_SCALE``), which the gateway loop's shim omits (it reports real pixels).
  * ``gateway_c_gui_loop`` -- disk artifacts and the VM egress proxy. CUA-Gym VMs have no
    direct internet (curl returns 000) and the session client's ``enable_proxy`` only
    covers google-chrome, not shell commands; without it a pip install hangs for the full
    timeout instead of failing fast.

Both prefixes are prepended at execution time only, so traj.jsonl keeps the raw command.

One turn == one screenshot: bash captures no frame, so a turn that ran bash re-observes
once at the end, keeping the agent's screenshot/response/cli_output arrays aligned.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

from mm_agents.c_gui.shim import COORD_SCALE_ENV, DEFAULT_COORD_SCALE, build_provision_command

from cli_skills import load_cli_skill
from gateway_c_gui_loop import _now, _record_step, _save_screenshot

logger = logging.getLogger("desktopenv.experiment")

_CONTROL_END = ("DONE", "FAIL")


def provision_shim(env) -> None:
    """Install the 0-999-scaling shim. Idempotent; MUST run after ``env.reset()``, which
    may roll the snapshot back and wipe the user-site write."""
    try:
        res = env.run_code(build_provision_command(), lang="bash") or {}
        logger.info("c-gui shim: %s", str(res.get("output", "")).strip()[:200])
    except Exception as exc:  # non-fatal: real problems surface on the first GUI step
        logger.warning("c-gui shim provision failed: %s", exc)


def build_bash_payload(command: str, vm_proxy: str = "", vm_no_proxy: str = "",
                       coord_scale: str = DEFAULT_COORD_SCALE) -> str:
    """Prefix the coord-scale env (always) and the egress proxy (when set).

    Every bash command is an independent subprocess with no inherited env, so the scale
    gate must be re-exported each time. CUA-Gym's own setup channel never sets it, so its
    absolute-pixel pyautogui is unaffected.
    """
    exports = [f"export {COORD_SCALE_ENV}={coord_scale}"]
    if vm_proxy:
        exports.append(
            f"export http_proxy={vm_proxy} https_proxy={vm_proxy} "
            f"HTTP_PROXY={vm_proxy} HTTPS_PROXY={vm_proxy} "
            f"no_proxy={vm_no_proxy} NO_PROXY={vm_no_proxy}"
        )
    return "\n".join(exports + [command])


def run_single_example_qwen38_c_gui(
    agent, env, example, max_steps, instruction, args, example_result_dir, scores
):
    """Drive one CUA-Gym episode with the Qwen3.8 c-gui agent."""
    vm_proxy = getattr(args, "vm_proxy", "") or ""
    vm_no_proxy = getattr(args, "vm_no_proxy", "") or ""
    sleep_after = getattr(args, "sleep_after_execution", 3)

    env.reset(task_config=example)
    try:
        agent.reset(logger)
    except Exception:
        agent.reset()

    # Per-domain CLI skill injection: the agent is reused across this worker's tasks and
    # the domain (example["app_type"]) changes per task, so set it here, after reset.
    if getattr(args, "cli_skills", True):
        app_type = example.get("app_type")
        agent.cli_skill = load_cli_skill(app_type, instruction)
        logger.info("CLI skill for app_type=%s: %d chars", app_type, len(agent.cli_skill))
    else:
        agent.cli_skill = ""

    provision_shim(env)  # AFTER reset (see provision_shim docstring)

    time.sleep(getattr(args, "wait_after_reset", 60))  # let the environment come up
    obs = env._get_obs()
    _save_screenshot(example_result_dir, f"step_0_{_now()}.png", obs.get("screenshot"))

    done = False
    step_idx = 0
    step_exec_results: Optional[List[Dict]] = None  # threaded into the NEXT predict()
    env.controller.start_recording()

    while not done and step_idx < max_steps:
        response, actions = agent.predict(instruction, obs, step_exec_results)
        if not actions:
            logger.warning("Agent returned no actions at step %d; stopping.", step_idx + 1)
            break

        cli_feedback: List[Dict] = []
        ran_bash = False
        post_cli_file = None  # end-of-turn screenshot shared by this turn's bash lines

        for action in actions:
            ts = _now()

            if action.get("kind") == "control":
                control = action["control"]
                logger.info("Step %d: control %s", step_idx + 1, control)
                obs, reward, done, info = env.step(control, sleep_after)
                if action.get("answer") is not None:
                    info = dict(info or {})
                    info["answer"] = action["answer"]
                _record_step(example_result_dir, step_idx + 1, ts, action, response,
                             reward, done, info, obs)
                if control in _CONTROL_END:
                    # flush any deferred bash screenshot before ending the episode
                    if post_cli_file is not None:
                        _save_screenshot(example_result_dir, post_cli_file, obs.get("screenshot"))
                    done = True
                    break
                continue  # WAIT and other no-ops: keep going

            command = action.get("command")
            if not command:
                continue
            ran_bash = True
            exec_result = env.run_code(
                build_bash_payload(command, vm_proxy, vm_no_proxy),
                lang="bash", timeout=action.get("timeout"),
            ) or {}
            # stdout and stderr must be CONCATENATED, not either-or: a failing command
            # often leaves output="\n" (truthy), which would short-circuit away the
            # stderr that explains the failure.
            out = exec_result.get("output") or ""
            err = exec_result.get("error") or ""
            text = out + (f"\n[stderr]\n{err}" if err.strip() else "")
            cli_feedback.append({"text": text or "(no output)"})
            if post_cli_file is None:
                post_cli_file = f"step_{step_idx + 1}_post_cli_{_now()}.png"
            _record_step(example_result_dir, step_idx + 1, ts, action, response, 0.0,
                         False, {"exec_result": exec_result}, obs,
                         save_screenshot=False, screenshot_file=post_cli_file)

        # Bash mutates the VM without capturing a frame; re-observe once at turn end so
        # the NEXT predict() sees the post-command screen.
        if ran_bash and not done:
            time.sleep(sleep_after)
            obs = env._get_obs()
            _save_screenshot(example_result_dir, post_cli_file, obs.get("screenshot"))

        step_exec_results = cli_feedback
        step_idx += 1

    time.sleep(20)  # let the environment settle before grading
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")

    try:  # optional: OSWorld's aggregate results log
        from lib_results_logger import log_task_completion

        log_task_completion(example, result, example_result_dir, args)
    except Exception:  # pragma: no cover - absent or unwritable is not fatal
        pass
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))
