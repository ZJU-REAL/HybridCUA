"""EvoCUA hybrid (GUI + CLI) agent.

Subclasses ``EvoCUAAgent`` so the vendored agent stays untouched (CLAUDE.md: OSWorld
internals are read-only). Three overrides:

  * ``_predict_s2`` -- same episode turn, but renders the two-tool system prompt and
    returns **channel-tagged action dicts** instead of bare pyautogui strings.
  * ``_build_s2_messages`` -- forked to splice each turn's CLI stdout/stderr into the
    history as a text block, index-aligned with that turn's screenshot.
  * ``predict`` -- takes the extra ``step_exec_results`` argument that
    ``run_single_example_hybrid`` threads in from the previous turn.

The action dicts match ``mm_agents/hybrid/agent.py`` exactly::

    {"channel": "gui", "command": "<pyautogui str>", "control": "DONE"|None}
    {"channel": "cli", "command": "<python str>",    "control": None}

so ``mm_agents.hybrid.run_loop.run_single_example_hybrid`` drives this agent unchanged --
that loop dispatches on ``channel`` and never inspects the agent class.

**GUI conversion is not reimplemented.** The parent's ``_parse_response_s2`` holds a
~150-line JSON-args -> pyautogui table (14 actions, coordinate rescaling, key
normalisation). We reuse it verbatim by handing it ONE synthetic ``<tool_call>`` at a
time and collecting what it emits, so any upstream fix to that table applies here too.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from mm_agents.evocua.evocua_agent import EvoCUAAgent

# Must agree with prompts.py: the schema the model is shown and the translator that
# executes its calls have to come from the same surface. See EVOCUA_CLI_SURFACE there.
if os.environ.get("EVOCUA_CLI_SURFACE", "bash").strip().lower() == "full":
    from mm_agents.hybrid.cli_tools import CliActionError, cli_action_to_code
else:
    from mm_agents.hybrid.cli_tools_bash import CliActionError, cli_action_to_code

from .prompts import (
    build_description_prompt,
    build_evocua_hybrid_system_prompt,
    build_evocua_hybrid_tools_def,
)

logger = logging.getLogger("desktopenv.agent")

#: Bare strings the parent emits in place of pyautogui code for control actions.
_CONTROL_TOKENS = ("DONE", "FAIL", "WAIT", "CALL_USER")

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _iter_tool_calls(response: str) -> List[Dict]:
    """Extract every tool call in order as a parsed JSON dict.

    Mirrors the scan in ``EvoCUAAgent._parse_response_s2`` (tagged blocks first, then
    bare ``{...}`` lines for responses that omit the tags), but returns the decoded
    objects instead of dispatching on them -- we need the ``name`` to route GUI vs CLI,
    which the parent's scan discards for anything that isn't ``computer_use``.
    """
    calls: List[Dict] = []

    def _load(blob: str) -> None:
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            return
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            calls.append(obj)

    tagged = _TOOL_CALL_RE.findall(response)
    for blob in tagged:
        _load(blob.strip())

    if not tagged:
        # Untagged fallback: a response that emitted raw JSON objects, one per line.
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                _load(line)
    return calls


def _low_level_from(response: str) -> str:
    """The ``Action:`` line the model is asked to emit, for traj logging."""
    for line in response.split("\n"):
        line = line.strip()
        if line.lower().startswith("action:"):
            return line.split(":", 1)[-1].strip()
    return ""


class EvoCUAHybridAgent(EvoCUAAgent):
    """EvoCUA with a second ``cli`` tool; returns channel-tagged action dicts."""

    def __init__(self, *args, password: str = "password", **kwargs):
        super().__init__(*args, password=password, **kwargs)
        # Index-aligned with self.screenshots: cli_outputs[i] is the CLI text produced
        # during turn i, rendered back into that turn's history slot.
        self.cli_outputs: List[str] = []

    def reset(self, _logger=None, vm_ip=None):
        super().reset(_logger, vm_ip)
        self.cli_outputs = []

    # --- predict ------------------------------------------------------------
    def predict(  # type: ignore[override]
        self,
        instruction: str,
        obs: Dict,
        step_exec_results: Optional[List[Dict]] = None,
    ) -> Tuple[str, List[Dict]]:
        # Attach the PREVIOUS turn's CLI output to the previous screenshot's slot (this
        # turn's own output isn't known until the loop runs what we return now).
        # Same bookkeeping as hybrid/agent.py:114-124.
        while len(self.cli_outputs) < len(self.screenshots):
            self.cli_outputs.append("")
        if step_exec_results and self.cli_outputs:
            prev_cli = "\n".join(r.get("text", "") for r in step_exec_results if r.get("text"))
            if prev_cli:
                sep = "\n" if self.cli_outputs[-1] else ""
                self.cli_outputs[-1] = self.cli_outputs[-1] + sep + prev_cli

        # The parent's predict() dispatches S1/S2 and does the screenshot preprocessing;
        # reuse it wholesale. S1 has no tool surface to extend, so hybrid requires S2.
        if self.prompt_style != "S2":
            raise ValueError("EvoCUAHybridAgent requires prompt_style='S2'")
        return super().predict(instruction, obs)

    # --- turn: two-tool prompt, channel-tagged output ------------------------
    def _predict_s2(  # type: ignore[override]
        self, instruction, obs, processed_b64, p_width, p_height, original_width, original_height
    ) -> Tuple[str, List[Dict]]:
        current_step = len(self.actions)
        current_history_n = self.max_history_turns

        description_prompt = build_description_prompt(self.coordinate_type, p_width, p_height)
        tools_def = build_evocua_hybrid_tools_def(description_prompt)
        system_prompt = build_evocua_hybrid_system_prompt(tools_def, self.password)

        response = None
        # Same context-shrinking retry as the parent: on a context-length error, drop one
        # history turn and rebuild.
        while True:
            messages = self._build_s2_messages(
                instruction, processed_b64, current_step, current_history_n, system_prompt
            )
            try:
                response = self.call_llm({
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "top_p": self.top_p,
                    "temperature": self.temperature,
                })
                break
            except Exception as e:
                if self._should_giveup_on_context_error(e) and current_history_n > 0:
                    current_history_n -= 1
                    logger.warning(f"Context too large, retrying with history_n={current_history_n}")
                else:
                    logger.error(f"Error in predict: {e}")
                    break

        self.responses.append(response)

        low_level, actions = self._parse_hybrid_response(
            response or "", p_width, p_height, original_width, original_height
        )

        # Force termination at the step ceiling (parent behaviour), unless the model
        # already chose to end the episode.
        current_step = len(self.actions) + 1
        first_control = actions[0].get("control") if actions else None
        if current_step >= self.max_steps and first_control not in ("DONE", "FAIL"):
            logger.warning(f"Reached maximum steps {self.max_steps}. Forcing termination with FAIL.")
            low_level = "Fail the task because reaching the maximum step limit."
            actions = [{"channel": "gui", "command": "FAIL", "control": "FAIL"}]

        logger.info(f"Low level instruction: {low_level}")
        logger.info(f"Hybrid actions: {actions}")

        self.actions.append(low_level)
        return response or "", actions

    # --- parsing: route computer_use -> GUI, cli -> CLI ----------------------
    def _parse_hybrid_response(
        self, response: str, p_width, p_height, original_width, original_height
    ) -> Tuple[str, List[Dict]]:
        actions: List[Dict] = []
        gui_count = cli_count = 0

        for call in _iter_tool_calls(response):
            name = call.get("name")
            if name == "computer_use":
                # Re-wrap this single call and let the PARENT convert it, so the whole
                # JSON-args -> pyautogui table (and coordinate rescaling) is reused
                # rather than copied.
                _, codes = super()._parse_response_s2(
                    f"<tool_call>\n{json.dumps(call)}\n</tool_call>",
                    p_width,
                    p_height,
                    original_width,
                    original_height,
                )
                for code in codes:
                    control = code if code in _CONTROL_TOKENS else None
                    actions.append({"channel": "gui", "command": code, "control": control})
                gui_count += 1
            elif name == "cli":
                try:
                    args = call.get("arguments") or {}
                    actions.append({"channel": "cli", "command": cli_action_to_code(args), "control": None})
                    cli_count += 1
                except CliActionError as exc:
                    # Skip the malformed cli action, keep the rest of the batch.
                    logger.warning("bad cli call: %s", exc)

        low_level = _low_level_from(response)
        if not low_level and actions:
            low_level = f"{gui_count} GUI + {cli_count} CLI action(s)"
        return low_level, actions

    # --- history: splice CLI output into the turn that produced it -----------
    def _build_s2_messages(self, instruction, current_img, step, history_n, system_prompt):
        messages = super()._build_s2_messages(
            instruction, current_img, step, history_n, system_prompt
        )
        if not any(self.cli_outputs):
            return messages

        # The parent emits one assistant message per history turn, in order. Walk them
        # and append the matching turn's CLI text as a user message right after, so the
        # model reads terminal output where it happened.
        history_len = min(history_n, len(self.responses))
        if history_len <= 0:
            return messages
        # cli_outputs is aligned with screenshots/responses; take the same tail window.
        tail = self.cli_outputs[-history_len:] if len(self.cli_outputs) >= history_len else self.cli_outputs

        out: List[Dict] = []
        turn = 0
        for msg in messages:
            out.append(msg)
            if msg.get("role") == "assistant":
                text = tail[turn] if turn < len(tail) else ""
                if text:
                    out.append({
                        "role": "user",
                        "content": [{"type": "text", "text": f"CLI output:\n{text}"}],
                    })
                turn += 1
        return out
