"""Qwen3.8-27B c-gui agent for CUA-Gym: gateway's PROMPT over c_gui's FOLD context.

  * PROMPT -- ``gateway_c_gui_agent.build_system_prompt``, whose ``<IMPORTANT>`` block
    tells the model to reason from the screenshot AND the previous step's CLI output,
    and to weigh GUI vs CLI every step. c_gui's prompt says neither.
  * CONTEXT -- ``CGuiAgent`` unchanged: native multi-turn history, one screenshot per
    turn, older screenshots folded to text, previous stdout in the same tool_response.

Coordinates are 0-999 AND screenshots are not downscaled -- independent facts, which is
what makes the mix safe. Normalized coords are resolution-free (the shim scales by the
VM's real ``pyautogui.size()/999``), and ``process_image`` caps at 13.1M pixels while
1920x1080 is 2.07M, so it only pads 1080 -> 1088. The gateway agent instead reports REAL
PIXELS, where folding a resized image would silently offset every click -- hence its
prompt text is reused here but its coordinate space is not.

Three prompt deltas, each forced by the context change: 0-999 instead of real pixels,
the "Collapsed screenshots" line restored (the gateway never folds), and an Environment
paragraph describing screenshot history instead of a flat text summary.

A fourth delta restores something ``c_gui`` dropped when it forked from ``qwen``:
``qwen.prompts.build_description_prompt`` states the coordinate space outright ("The
screen's resolution is 1000x1000" when relative), and no c-gui prompt ever did. With no
authoritative resolution to fall back on, Qwen3.8 invented one (1280x720, from prior)
and multiplied every coordinate by 1.5, missing its targets for whole episodes. Measured
on a 32-env run: 14 envs hallucinated 1280x720 and 4 applied the bogus 1.5x; of the
perfect trajectories that survived it, 3.7% still show the confusion and burn 51% more
steps (21.6 vs 14.3) recovering from it.

Simply asserting "1000x1000" is not enough, because c-gui hands the model a terminal
(``qwen`` denies it one) and the shim wraps the click/move functions but NOT ``size()``
or ``position()``. A model that measures therefore finds raw pixels that contradict any
1000x1000 claim -- which is exactly what happened, and what turned a mis-read screenshot
into an invented scale factor. So three bullets state all three numbers and how they
relate: the screen IS 1920x1080, coordinates ARE 0-999, the shim bridges them, and
``size()`` / ``position()`` reporting raw pixels is expected rather than evidence of a
bug. Positions are read by PROPORTION, which also makes the ``process_image`` pad
(1080 -> 1088) irrelevant -- a uniform stretch leaves relative positions unchanged.

The bullets also BAN the derivation, not just correct its inputs: no scaling of your own,
no resolution talk in the response. Stating the facts is necessary but not sufficient,
since the observed failure was the model reasoning from correct facts to a bogus 1.5x. The
misdirected-click clause closes the last exit by naming the real causes (misread
screenshot, moved UI element), so a failed click cannot be re-explained as a scaling bug.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List

from mm_agents.c_gui.agent import CGuiAgent

__all__ = ["Qwen38CGuiAgent", "build_system_prompt"]


def build_system_prompt(tools_def: List[Dict], collapse_text: str, password: str = "password",
                        cli_skill: str = "") -> str:
    """The gateway system prompt, re-pointed at a folded 0-999 context (see module doc).

    ``cli_skill`` (optional) is a per-domain CLI recipe block appended as its own section;
    it is empty for domains with no snippet, and byte-identical across an episode's turns
    so vLLM prefix-caches it (see ``cli_skills/``).
    """
    tool_name = tools_def[0]["function"]["name"]
    tools_json = json.dumps(tools_def)
    skill_section = f"# Domain CLI toolkit\n\n{cli_skill}\n\n" if cli_skill else ""
    return (
        "You are a multi-purpose intelligent assistant operating a computer through a bash terminal.\n"
        f"The password of the computer is {password}.\n\n"
        "# Environment\n\n"
        "You face a machine with a graphical desktop AND a bash terminal. You act ONLY by running one "
        "shell command per step (action=bash): write shell for CLI/file operations, or drive the GUI "
        "with pyautogui inside a quoted heredoc `python3 <<'PY' ... PY` (coordinates are 0-999). Each "
        "step you are shown the latest screenshot plus the previous command's output.\n\n"
        "# Tools\n\n"
        "You have access to the following functions:\n\n"
        "<tools>\n"
        f"{tools_json}\n"
        "</tools>\n\n"
        "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
        "<tool_call>\n"
        "<function=example_function_name>\n"
        "<parameter=example_parameter_1>\n"
        "value_1\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "<IMPORTANT>\n"
        "- Function calls MUST follow the specified format\n"
        "- The `action` parameter MUST be one of: bash, wait, terminate, answer. ALL GUI interactions "
        "MUST use action=bash with a pyautogui heredoc, whose delimiter MUST be quoted (<<'PY', not <<PY)\n"
        "- The screen and the screenshot are both 1920x1080, but the coordinates you emit are 0-999, "
        "NOT pixels: a shim in the VM rescales them by (screen_size / 999). Read positions BY "
        "PROPORTION -- left edge = 0, right edge = 999, top = 0, bottom = 999; an element 30% across "
        "and 60% down is (300, 600)\n"
        "- Pass plain 0-999 literals, e.g. `pyautogui.click(332, 450)`. Never scale or convert them "
        "yourself, and do not discuss resolutions or scale factors in your response\n"
        "- pyautogui.size() and position() report raw pixels (the shim wraps only click/move), so "
        "after moveTo(500, 400) position() says (961, 432). That is expected. If a click lands wrong, "
        "you misread the screenshot or the UI moved -- never the coordinate space\n"

        "- Reason from the CURRENT screenshot and the CLI terminal output of your previous step, then "
        "decide the next step\n"
        "- Weigh GUI vs CLI on EVERY step, but spell out that reasoning only on the key steps where it matters\n"
        "- When finished, use action=terminate (not bash exit commands)\n"
        f"- The current date is {datetime.today().strftime('%A, %B %d, %Y')}\n"
        f"- Collapsed screenshots appear as: {collapse_text}\n"
        "</IMPORTANT>\n\n"
        f"{skill_section}"
        "# Response format\n\n"
        "Every step:\n"
        "1) Action: one sentence describing your next move.\n"
        "2) A single <tool_call>...</tool_call> block.\n\n"
        "# Output format examples\n\n"
        "## GUI action (action=bash with pyautogui heredoc)\n\n"
        "Action: Click the \"File\" menu in the top menu bar.\n\n"
        "<tool_call>\n"
        f"<function={tool_name}>\n"
        "<parameter=action>\n"
        "bash\n"
        "</parameter>\n"
        "<parameter=command>\n"
        "python3 <<'PY'\n"
        "import pyautogui\n"
        "pyautogui.click(50, 15)\n"
        "PY\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "## Shell command (action=bash with shell)\n\n"
        "Action: List files in the Documents folder.\n\n"
        "<tool_call>\n"
        f"<function={tool_name}>\n"
        "<parameter=action>\n"
        "bash\n"
        "</parameter>\n"
        "<parameter=command>\n"
        "ls -la ~/Documents/\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "## Wait (action=wait)\n\n"
        "Action: Wait for the application to finish loading.\n\n"
        "<tool_call>\n"
        f"<function={tool_name}>\n"
        "<parameter=action>\n"
        "wait\n"
        "</parameter>\n"
        "<parameter=time>\n"
        "3\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "## Finish (action=terminate)\n\n"
        "Action: The task is complete.\n\n"
        "<tool_call>\n"
        f"<function={tool_name}>\n"
        "<parameter=action>\n"
        "terminate\n"
        "</parameter>\n"
        "<parameter=status>\n"
        "success\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )


_TOOL_CALL = "<tool_call>"


def add_action_prefix(response: str) -> str:
    """Prefix the post-``</think>`` summary line with ``Action: `` when the model omitted it.

    Qwen3.8-27B drops the prefix on complex instructions (c_gui's own prompt does no better,
    so this is model behaviour, not a prompt bug) and writes the sentence in assistant voice
    instead: ``I'll help create ... <tool_call>``. Downstream SFT keys on ``Action: ``, so the
    label is restored here -- the sentence itself is the model's, only the marker is added.

    Untouched when already prefixed, or when there is no sentence to label.
    """
    head, sep, tail = response.partition(_TOOL_CALL)
    if not sep:
        return response
    reasoning, think_sep, summary = head.rpartition("</think>")
    lines = [ln for ln in summary.split("\n") if ln.strip()]
    if not lines or lines[0].lstrip().startswith("Action:"):
        return response
    summary = summary.replace(lines[0], "Action: " + lines[0].lstrip(), 1)
    return reasoning + think_sep + summary + sep + tail


class Qwen38CGuiAgent(CGuiAgent):
    """``CGuiAgent`` with the gateway system prompt; everything else inherited.

    ``cli_skill`` holds the current task's per-domain CLI recipe block. The agent is
    reused across a worker's tasks and the domain changes per task, so the loop sets
    this at reset time (see ``qwen38_c_gui_loop.run_single_example_qwen38_c_gui``);
    default '' == no skill injected.
    """

    def __init__(self, *args, cli_skill: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.cli_skill = cli_skill

    def _build_system_prompt(self, tools_def: List[Dict]) -> str:  # type: ignore[override]
        return build_system_prompt(tools_def, self.collapse_text, self.password, self.cli_skill)

    def predict(self, instruction: str, obs: Dict, step_exec_results=None):  # type: ignore[override]
        response, actions = super().predict(instruction, obs, step_exec_results)
        fixed = add_action_prefix(response)
        # Rewrite the history slot too, so the prompt the model sees next turn and the
        # response stored in traj.jsonl stay byte-identical.
        if self.responses:
            self.responses[-1] = fixed
        return fixed, actions

    def _debug_message_filename(self, step_idx: int) -> str:
        # pid in the name: workers share ./draft/message_cache, and a step-only name makes
        # them overwrite each other (silently invalidating per-step measurements).
        return f"qwen38_c_gui_messages_pid{os.getpid()}_step_{step_idx}.json"

    def _log_prefix(self) -> str:
        return "Qwen38CGui"
