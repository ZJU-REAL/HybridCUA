"""Episode loop for the c-gui agent — single bash channel + control actions.

Same 8-arg signature as OSWorld's ``lib_run_single.run_single_example`` (plugs into
``OSWorldEvalSource`` via ``run_single_fn=``). Differs from hybrid's loop:
  * ONE channel: every bash action -> ``env.run_code(<cmd>, lang="bash")`` (GUI heredocs
    and CLI shell alike). No separate GUI/pyautogui ``env.step`` path.
  * control actions (wait/terminate/answer) -> ``env.step('WAIT'/'DONE'/'FAIL')``.
  * the pyautogui shim is provisioned into the VM once, AFTER reset (reset may roll back
    the snapshot and wipe the user-site write).

Reuses hybrid's ``_now`` / ``_save_screenshot`` / ``_record_step`` so the traj layout is
identical. One turn == one screenshot: a bash action captures no inline screenshot, so if
the turn ran any bash we re-observe ONCE at turn end — the NEXT predict then sees the
post-command screen, keeping the screenshot/response/cli_output index alignment intact.
"""
from __future__ import annotations

import logging
import os
import time

from mm_agents.hybrid.run_loop import _now, _save_screenshot, _record_step

from .executor import provision_shim, bash_payload

logger = logging.getLogger("desktopenv.experiment")

_CONTROL_END = ("DONE", "FAIL")


def run_single_example_c_gui_guionly(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    from lib_run_single import setup_logger  # OSWorld root module (on sys.path)

    runtime_logger = setup_logger(example, example_result_dir)
    env.reset(task_config=example)
    try:
        agent.reset(runtime_logger)
    except Exception:
        agent.reset()

    # Install the pyautogui shim AFTER reset (reset may roll back the VM snapshot and wipe
    # the user-site write). Idempotent — safe to re-run every episode.
    provision_shim(env, logger)

    time.sleep(60)  # wait for the environment to be ready
    obs = env._get_obs()
    _save_screenshot(example_result_dir, f"step_0_{_now()}.png", obs.get("screenshot"))

    done = False
    step_idx = 0
    step_exec_results = None  # per-turn CLI feedback threaded into the NEXT predict()
    env.controller.start_recording()

    while not done and step_idx < max_steps:
        response, actions = agent.predict(instruction, obs, step_exec_results)
        if not actions:
            logger.warning("Agent returned no actions at step %d; stopping.", step_idx + 1)
            break

        cli_feedback: list = []          # this turn's command outputs (text only)
        ran_bash = False                 # did this turn execute any bash action?
        post_cli_file = None             # end-of-turn screenshot shared by bash step lines

        for action in actions:
            ts = _now()
            kind = action.get("kind")

            if kind == "control":
                control = action["control"]
                logger.info("Step %d: control %s", step_idx + 1, control)
                obs, reward, done, info = env.step(control, args.sleep_after_execution)
                if action.get("answer") is not None:
                    info = dict(info or {})
                    info["answer"] = action["answer"]
                _record_step(example_result_dir, step_idx + 1, ts, action, response, reward, done, info, obs)
                if control in _CONTROL_END:
                    # flush any deferred bash screenshot for this turn before ending
                    if post_cli_file is not None:
                        _save_screenshot(example_result_dir, post_cli_file, obs.get("screenshot"))
                    done = True
                    break
                # WAIT (or other no-op control): keep going
                continue

            # bash channel — GUI heredoc OR CLI shell; both run_code(lang="bash")
            command = action.get("command")
            if not command:
                continue
            ran_bash = True
            exec_result = env.run_code(
                bash_payload(command), lang="bash", timeout=action.get("timeout")
            )
            reward, done, info = 0.0, False, {"exec_result": exec_result}
            out = (exec_result.get("output") or "")
            err = (exec_result.get("error") or "")
            text = out + (f"\n[stderr]\n{err}" if err.strip() else "")
            cli_feedback.append({"text": text or "(no output)"})
            if post_cli_file is None:
                post_cli_file = f"step_{step_idx + 1}_post_cli_{_now()}.png"
            _record_step(example_result_dir, step_idx + 1, ts, action, response, reward,
                         done, info, obs, save_screenshot=False, screenshot_file=post_cli_file)
            if done:
                break

        # A bash action mutates the VM without capturing a screenshot; re-observe once at
        # turn end so the NEXT predict sees the post-command screen. (One turn == one
        # screenshot: only refresh obs, don't add a slot.)
        if ran_bash and not done:
            time.sleep(args.sleep_after_execution)
            obs = env._get_obs()
            _save_screenshot(example_result_dir, post_cli_file, obs.get("screenshot"))

        step_exec_results = cli_feedback
        step_idx += 1

    time.sleep(20)  # let the environment settle
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")

    try:
        from lib_results_logger import log_task_completion
        log_task_completion(example, result, example_result_dir, args)
    except Exception:  # pragma: no cover - optional
        pass
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))
