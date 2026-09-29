"""Episode loop for c-gui + MCP. A thin fork of ``mm_agents.c_gui.run_loop``.

Same 8-arg signature as OSWorld's ``lib_run_single.run_single_example``, so it drops
into ``OSWorldEvalSource(run_single_fn=...)`` exactly like the stock loop.

Four additions over the stock c-gui loop, all gated on ``args.mcp_mode != "off"``:

  1. provision_mcp(env) next to provision_shim(env) -- both are "install into the
     guest after reset", and reset may have rolled the snapshot back.
  2. the episode's tool catalog is fetched once from the task's related_apps and
     handed to the agent, which renders it into the system prompt.
  3. a `kind == "mcp"` branch dispatches structured tool calls (action mode).
     Results are threaded back as CLI output, reusing the existing feedback channel.
  4. tool usage is summarized into the result dir for TIR rollup.

With mcp_mode="off" this is behaviourally identical to the stock loop -- that is
what makes it a valid A/B baseline.
"""
from __future__ import annotations

import json
import logging
import os
import time

from mm_agents.c_gui.executor import bash_payload, provision_shim
from mm_agents.hybrid.run_loop import _now as timestamp
from mm_agents.hybrid.run_loop import _record_step as record_step
from mm_agents.hybrid.run_loop import _save_screenshot as save_screenshot

from mm_agents.mcp_common.runtime import (call_mcp_tool, list_tools_for_task, provision_mcp,
                         summarize_episode)

logger = logging.getLogger("desktopenv.experiment")

CONTROL_END = ("DONE", "FAIL")


def run_single_example_c_gui_mcp(agent, env, example, max_steps, instruction, args,
                                 example_result_dir, scores):
    from lib_run_single import setup_logger

    runtime_logger = setup_logger(example, example_result_dir)
    mcp_mode = getattr(args, "mcp_mode", "off")

    # Stack goes up BEFORE reset, matching upstream's install-once-then-evaluate
    # flow. Reversed, bring-up runs after the task's setup has opened its document,
    # and anything that restarts soffice destroys it -- that cost libreoffice_calc
    # 24 points (33.3% vs the 57.4% no-MCP baseline). With MCP_BRINGUP=image this is
    # a ~2s probe; the ordering still matters for the runcode fallback.
    mcp_ready = provision_mcp(env, logger) if mcp_mode != "off" else False

    env.reset(task_config=example)
    try:
        agent.reset(runtime_logger)
    except Exception:
        agent.reset()

    provision_shim(env, logger)

    tools = []
    if mcp_mode != "off":
        # Listed after reset: related_apps filtering describes the task we are about
        # to run, and reset may have recycled the guest underneath a warm slot.
        if mcp_ready or provision_mcp(env, logger):
            tools = list_tools_for_task(
                env, example,
                keep_distractors=not getattr(args, "mcp_no_distractors", False),
                budget=int(getattr(args, "mcp_tool_budget", 0) or 0),
                logger_=logger,
            )
        else:
            logger.error("[MCP] provisioning failed; this episode runs GUI-only")
    if hasattr(agent, "set_episode_tools"):
        agent.set_episode_tools(tools)
    logger.info("[MCP] mode=%s tools=%d", mcp_mode, len(tools))

    time.sleep(60)
    obs = env._get_obs()
    save_screenshot(example_result_dir, f"step_0_{timestamp()}.png", obs.get("screenshot"))

    done = False
    step_idx = 0
    step_exec_results = None
    step_records = []
    env.controller.start_recording()

    while not done and step_idx < max_steps:
        response, actions = agent.predict(instruction, obs, step_exec_results)
        if not actions:
            logger.warning("Agent returned no actions at step %d; stopping.", step_idx + 1)
            break
        step_records.append({"actions": actions})

        cli_feedback: list = []
        ran_side_effect = False
        post_cli_file = None

        for action in actions:
            ts = timestamp()
            kind = action.get("kind")

            if kind == "control":
                control = action["control"]
                logger.info("Step %d: control %s", step_idx + 1, control)
                obs, reward, done, info = env.step(control, args.sleep_after_execution)
                if action.get("answer") is not None:
                    info = dict(info or {})
                    info["answer"] = action["answer"]
                record_step(example_result_dir, step_idx + 1, ts, action, response,
                             reward, done, info, obs)
                if control in CONTROL_END:
                    if post_cli_file is not None:
                        save_screenshot(example_result_dir, post_cli_file,
                                         obs.get("screenshot"))
                    done = True
                    break
                continue

            if kind == "mcp":
                name = action.get("name")
                logger.info("Step %d: mcp %s(%s)", step_idx + 1, name,
                            json.dumps(action.get("params") or {})[:200])
                result = call_mcp_tool(env, name, action.get("params") or {})
                ran_side_effect = True
                cli_feedback.append({"text": result["text"] or "(no output)"})
                info = {"mcp_result": {"ok": result["ok"], "name": name}}
                if post_cli_file is None:
                    post_cli_file = f"step_{step_idx + 1}_post_cli_{timestamp()}.png"
                record_step(example_result_dir, step_idx + 1, ts, action, response,
                             0.0, False, info, obs, save_screenshot=False,
                             screenshot_file=post_cli_file)
                continue

            command = action.get("command")
            if not command:
                continue
            ran_side_effect = True
            exec_result = env.run_code(
                bash_payload(command), lang="bash", timeout=action.get("timeout")
            )
            reward, done, info = 0.0, False, {"exec_result": exec_result}
            out = (exec_result.get("output") or "")
            err = (exec_result.get("error") or "")
            text = out + (f"\n[stderr]\n{err}" if err.strip() else "")
            cli_feedback.append({"text": text or "(no output)"})
            if post_cli_file is None:
                post_cli_file = f"step_{step_idx + 1}_post_cli_{timestamp()}.png"
            record_step(example_result_dir, step_idx + 1, ts, action, response, reward,
                         done, info, obs, save_screenshot=False,
                         screenshot_file=post_cli_file)
            if done:
                break

        if ran_side_effect and not done:
            time.sleep(args.sleep_after_execution)
            obs = env._get_obs()
            save_screenshot(example_result_dir, post_cli_file, obs.get("screenshot"))

        step_exec_results = cli_feedback
        step_idx += 1

    time.sleep(20)
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")

    try:
        usage = summarize_episode(step_records)
        usage.update({
            "mcp_mode": mcp_mode,
            "tools_offered": len(tools),
            "score": result,
            "related_apps": example.get("related_apps"),
        })
        with open(os.path.join(example_result_dir, "mcp_usage.json"), "w",
                  encoding="utf-8") as f:
            json.dump(usage, f, indent=2)
        logger.info("[MCP] usage: %d call(s) over %d step(s), %d distractor",
                    usage.get("tool_calls", 0), usage.get("steps", 0),
                    usage.get("distractor_calls", 0))
    except Exception as exc:
        logger.warning("[MCP] could not write mcp_usage.json: %s", exc)

    try:
        from lib_results_logger import log_task_completion
        log_task_completion(example, result, example_result_dir, args)
    except Exception:
        pass
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))
