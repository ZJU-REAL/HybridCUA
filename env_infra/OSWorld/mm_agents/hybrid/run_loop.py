"""Episode loop for the hybrid (GUI+CLI) agent.

``run_single_example_hybrid`` has the same 8-arg signature as OSWorld's
``lib_run_single.run_single_example`` (so it plugs into ``OSWorldEvalSource`` via
``run_single_fn=``), but drives the channel-tagged action dicts HybridAgent returns:

  * GUI action  -> ``env.step(<pyautogui>)`` (pyautogui channel).
  * CLI action  (``channel=="cli"``) -> ``env.run_code(<python>)`` (the cli channel — the
    ONLY one returning real stdout/stderr text). A CLI action captures no screenshot
    inline, so if the turn ran any CLI we re-observe ONCE at the end of the turn
    (``env._get_obs()`` after settling) — the NEXT predict() then sees the post-CLI
    screen, not a stale one. One turn still == one screenshot (we refresh ``obs``, we
    don't add a slot), so qwen's index alignment / image folding stay intact.
  * Control tokens (DONE/FAIL/WAIT/CALL_USER) end or pause the episode.

Per-turn CLI text is collected and threaded into the NEXT ``predict`` as
``step_exec_results`` so the model reads terminal output (HybridAgent renders it as a
``CLI output`` user turn). Disk artifacts (step PNGs, traj.jsonl, result.txt, recording)
match the standard loop, with one difference: a CLI step saves no pre-CLI PNG — its
traj line points at the turn's single post_cli image (the settled post-CLI screen, one
per turn, shared by all CLI steps of that turn and fed to the next predict()).
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import time

logger = logging.getLogger("desktopenv.experiment")

_CONTROL_END = ("DONE", "FAIL")
_CONTROL_NOOP = ("WAIT", "CALL_USER")


def _now() -> str:
    return datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")


def _save_screenshot(result_dir: str, name: str, screenshot) -> None:
    if screenshot:
        with open(os.path.join(result_dir, name), "wb") as f:
            f.write(screenshot)


def run_single_example_hybrid(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    from lib_run_single import setup_logger  # OSWorld root module (on sys.path)

    runtime_logger = setup_logger(example, example_result_dir)
    env.reset(task_config=example)
    try:
        agent.reset(runtime_logger)
    except Exception:
        agent.reset()

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

        cli_feedback: list = []  # this turn's CLI outputs (text only)
        ran_cli = False          # did this turn execute any CLI action?
        post_cli_file: str | None = None  # end-of-turn screenshot shared by CLI step lines

        for action in actions:
            ts = _now()
            control = action.get("control")

            if control in _CONTROL_END:
                logger.info("Step %d: control %s", step_idx + 1, control)
                obs, reward, done, info = env.step(control, args.sleep_after_execution)
                _record_step(example_result_dir, step_idx + 1, ts, action, response, reward, done, info, obs)
                # If earlier CLI steps this turn deferred their screenshot to post_cli_file,
                # save it now (the end-of-turn re-observe below is skipped once done) so
                # those traj lines don't dangle.
                if post_cli_file is not None:
                    _save_screenshot(example_result_dir, post_cli_file, obs.get("screenshot"))
                done = True
                break

            if control in _CONTROL_NOOP:
                logger.info("Step %d: control %s", step_idx + 1, control)
                obs, reward, done, info = env.step(control, args.sleep_after_execution)
                _record_step(example_result_dir, step_idx + 1, ts, action, response, 0.0, False, {}, obs)
                continue

            command = action.get("command")
            if not command:
                continue

            if action["channel"] == "gui":
                obs, reward, done, info = env.step(command, args.sleep_after_execution)
                _record_step(example_result_dir, step_idx + 1, ts, action, response, reward, done, info, obs)
            else:  # cli — run_code, text-only feedback; obs refreshed AFTER the turn
                ran_cli = True
                exec_result = env.run_code(command, lang="python")
                reward, done, info = 0.0, False, {"exec_result": exec_result}
                out = (exec_result.get("output") or "")
                err = (exec_result.get("error") or "")
                text = out + (f"\n[stderr]\n{err}" if err.strip() else "")
                cli_feedback.append({"text": text or "(no output)"})
                # A CLI step's inline obs is the pre-CLI screen (no visual value), so we
                # don't save it. Its screenshot is this turn's single post_cli image;
                # point the line at that shared filename (generated once per turn).
                if post_cli_file is None:
                    post_cli_file = f"step_{step_idx + 1}_post_cli_{_now()}.png"
                _record_step(example_result_dir, step_idx + 1, ts, action, response, reward,
                             done, info, obs, save_screenshot=False, screenshot_file=post_cli_file)

            if done:
                break

        # A CLI action mutates the VM without capturing a screenshot, so `obs` is now
        # stale (it still holds the pre-CLI screen, or the last GUI action's screen).
        # Re-observe once at the end of the turn — after settling — so the NEXT predict()
        # sees the post-CLI screen instead of an outdated one. (One turn == one
        # screenshot; we only refresh, we don't add a slot, so qwen's index alignment /
        # folding are untouched.) Pure-GUI turns already have a fresh obs — skip them.
        if ran_cli and not done:
            time.sleep(args.sleep_after_execution)
            obs = env._get_obs()
            # Save THE post-CLI screen under the filename the turn's CLI step lines already
            # reference — one image per turn, shared by every CLI step of that turn, and
            # the exact image the NEXT predict() consumes.
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


def _record_step(result_dir, step_num, ts, action, response, reward, done, info, obs,
                 *, save_screenshot: bool = True, screenshot_file: str | None = None) -> None:
    """Write a traj.jsonl line (standard-loop layout), optionally saving the screenshot.

    ``save_screenshot=False`` writes only the traj line — used for CLI steps, whose
    inline ``obs`` is the pre-CLI screen (no visual value); their screenshot is the
    end-of-turn post_cli image instead. Pass ``screenshot_file`` to point the line at
    that image so replay stays linked.
    """
    if screenshot_file is None:
        screenshot_file = f"step_{step_num}_{ts}.png"
    if save_screenshot:
        _save_screenshot(result_dir, screenshot_file, obs.get("screenshot"))
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
