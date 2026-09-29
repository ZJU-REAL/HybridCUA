"""Episode loop for the real-pixel single-bash-surface agents.

Same 8-arg signature as OSWorld's ``lib_run_single.run_single_example``, so it plugs into
``OSWorldEvalSource`` via ``run_single_fn=``. ONE execution channel: every bash action
(GUI heredoc and CLI shell alike) goes through ``env.run_code(..., lang="bash")``; the
three control actions go through ``env.step('WAIT'/'DONE'/'FAIL')``.

Both ``kimi_hybrid`` and ``claude_hybrid`` use this loop unchanged — it dispatches on the
action dict's ``kind`` and never inspects the agent class (same contract that lets
``evocua_hybrid`` reuse ``hybrid.run_loop``).

Two invariants worth stating, because both are easy to break:

  * ONE TURN == ONE SCREENSHOT. A bash command mutates the VM without capturing a frame,
    so if a turn ran any bash we re-observe ONCE at turn end. The next ``predict`` then
    sees the post-command screen instead of a stale one. All of a turn's bash step lines
    point at that single shared image, so nothing dangles and no extra history slot is
    created.
  * THE SHIM IS PROVISIONED AFTER RESET. ``env.reset`` may roll the VM snapshot back and
    wipe the user-site write, so it goes per episode, not per worker.

Disk artifacts (step PNGs, traj.jsonl, result.txt, recording) match the standard loop's
layout so resume and scoring stay consistent.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import time
from typing import Dict, List, Optional

from .shim import provision_shim

logger = logging.getLogger("desktopenv.experiment")

_CONTROL_END = ("DONE", "FAIL")


def build_bash_payload(command: str, vm_proxy: str = "", vm_no_proxy: str = "") -> str:
    """Prefix the VM egress proxy at EXECUTION time only (traj keeps the raw command).

    The VM cannot reach the public internet directly (curl returns 000), and the session
    client's ``enable_proxy`` only adds ``--proxy-server`` to google-chrome — it does
    nothing for shell commands. Without this a ``pip install`` does not fail fast: it
    hangs for the full timeout and drags the trajectory down with it.

    Empty ``vm_proxy`` disables the prefix entirely (pass ``VM_PROXY=""``).
    """
    if not vm_proxy:
        return command
    return (f"export http_proxy={vm_proxy} https_proxy={vm_proxy} "
            f"HTTP_PROXY={vm_proxy} HTTPS_PROXY={vm_proxy} "
            f"no_proxy={vm_no_proxy} NO_PROXY={vm_no_proxy}\n{command}")


def _now() -> str:
    return datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")


def _save_screenshot(result_dir: str, name: str, screenshot: Optional[bytes]) -> None:
    if screenshot:
        with open(os.path.join(result_dir, name), "wb") as f:
            f.write(screenshot)


def _record_step(result_dir, step_num, ts, action, response, reward, done, info, obs,
                 *, save_screenshot: bool = True, screenshot_file: Optional[str] = None) -> None:
    """Write a traj.jsonl line, optionally saving the step screenshot.

    ``save_screenshot=False`` writes only the traj line — used for bash steps, whose
    inline ``obs`` is the PRE-command screen (no visual value). Their screenshot is the
    turn's shared end-of-turn image instead; pass ``screenshot_file`` to point the line
    at it so replay stays linked.
    """
    screenshot_file = screenshot_file or f"step_{step_num}_{ts}.png"
    if save_screenshot:
        _save_screenshot(result_dir, screenshot_file, (obs or {}).get("screenshot"))
    with open(os.path.join(result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "step_num": step_num,
            "action_timestamp": ts,
            "action": action,
            "response": response,
            "reward": reward,
            "done": done,
            "info": info,
            "screenshot_file": screenshot_file,
        }, ensure_ascii=False))
        f.write("\n")


def run_single_example_c_gui_pixel(
    agent, env, example, max_steps, instruction, args, example_result_dir, scores
):
    """Drive one OSWorld episode with a real-pixel single-bash-surface agent."""
    from lib_run_single import setup_logger  # OSWorld root module (on sys.path)

    vm_proxy = getattr(args, "vm_proxy", "") or ""
    vm_no_proxy = getattr(args, "vm_no_proxy", "") or ""
    sleep_after = getattr(args, "sleep_after_execution", 3)

    runtime_logger = setup_logger(example, example_result_dir)
    env.reset(task_config=example)
    try:
        agent.reset(runtime_logger)
    except Exception:
        agent.reset()

    # AFTER reset: reset may roll the snapshot back and wipe the user-site write.
    provision_shim(env, logger)

    time.sleep(getattr(args, "wait_after_reset", 60))  # let the environment come up
    obs = env._get_obs()
    _save_screenshot(example_result_dir, f"step_0_{_now()}.png", obs.get("screenshot"))

    done = False
    step_idx = 0
    step_exec_results: Optional[List[Dict]] = None  # previous turn's output, fed forward
    env.controller.start_recording()

    while not done and step_idx < max_steps:
        response, actions = agent.predict(instruction, obs, step_exec_results)
        if not actions:
            logger.warning("Agent returned no actions at step %d; stopping.", step_idx + 1)
            break

        cli_feedback: List[Dict] = []   # this turn's command outputs (text only)
        ran_bash = False                # did this turn execute any bash action?
        post_cli_file = None            # end-of-turn screenshot shared by bash step lines

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
                    # Flush any deferred bash screenshot for this turn: the end-of-turn
                    # re-observe below is skipped once done, so those lines would dangle.
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
        # the NEXT predict sees the post-command screen. (One turn == one screenshot:
        # only refresh obs, don't add a slot.)
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
