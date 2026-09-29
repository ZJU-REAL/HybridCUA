"""Episode loop for QwenAgent + MCP."""
from __future__ import annotations

import datetime
import json
import logging
import os
import time
from typing import Dict, List

from mm_agents.c_gui.executor import provision_shim

from mm_agents.mcp_common.runtime import list_tools_for_task, provision_mcp, summarize_episode

logger = logging.getLogger("desktopenv.experiment")


def timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")


def save_screenshot(result_dir: str, filename: str, data) -> None:
    if not data:
        return
    try:
        with open(os.path.join(result_dir, filename), "wb") as f:
            f.write(data)
    except Exception as exc:
        logger.warning("could not write %s: %s", filename, exc)


def is_mcp_call(action: str) -> bool:
    return isinstance(action, str) and "OsworldMcpClient" in action


def run_single_example_qwen_mcp(agent, env, example, max_steps, instruction, args,
                                example_result_dir, scores):
    from lib_run_single import setup_logger

    runtime_logger = setup_logger(example, example_result_dir)

    # Before reset, for the reason spelled out in c_gui_mcp/run_loop.py: bring-up
    # after the task's setup races the document setup just opened.
    mcp_ready = provision_mcp(env, logger)

    env.reset(task_config=example)
    try:
        agent.reset(runtime_logger)
    except Exception:
        agent.reset()

    provision_shim(env, logger)

    tools: List[Dict] = []
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
    logger.info("[MCP] tools=%d", len(tools))

    time.sleep(60)
    obs = env._get_obs()
    save_screenshot(example_result_dir, f"step_0_{timestamp()}.png", obs.get("screenshot"))

    done = False
    step_idx = 0
    step_records: List[Dict] = []
    env.controller.start_recording()

    while not done and step_idx < max_steps:
        response, actions = agent.predict(instruction, obs)
        if not actions:
            logger.warning("Agent returned no actions at step %d; stopping.",
                           step_idx + 1)
            break

        step_records.append({"actions": [
            {"kind": "bash", "command": a} for a in actions if isinstance(a, str)
        ]})

        for action in actions:
            ts = timestamp()
            logger.info("Step %d: %s", step_idx + 1,
                        str(action)[:300] + ("…" if len(str(action)) > 300 else ""))
            obs, reward, done, info = env.step(action, args.sleep_after_execution)
            logger.info("Reward: %.2f  Done: %s", reward, done)

            shot = f"step_{step_idx + 1}_{ts}.png"
            save_screenshot(example_result_dir, shot, obs.get("screenshot"))
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a",
                      encoding="utf-8") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": ts,
                    "action": action,
                    "response": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": shot,
                    "is_mcp_call": is_mcp_call(action),
                }) + "\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1

    time.sleep(20)
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w",
              encoding="utf-8") as f:
        f.write(f"{result}\n")

    try:
        usage = summarize_episode(step_records)
        usage.update({
            "mcp_mode": "action",
            "tools_offered": len(tools),
            "score": result,
            "related_apps": example.get("related_apps"),
            "agent": "qwen",
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
