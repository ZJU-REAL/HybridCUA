"""CGuiGuiOnlyAgent — single bash-surface GUI-ONLY OSWorld agent (pyautogui heredoc only).

Subclasses qwen's ``_QwenBaseAgent`` (reusing its OpenAI call, image processing, history
folding, reset) and overrides ``predict`` so the model is offered ONE tool with a single
``bash`` action (plus ``wait``/``terminate``/``answer`` control actions). Unlike ``c_gui``,
the prompt advertises ONLY pyautogui-in-a-heredoc: no shell, no file ops, no CLI. The
mechanism is identical (still ``action=bash``), so this is a SOFT prompt-level constraint —
see ``prompts.py`` for why residual CLI steps are the measurement, not a defect.

The tool's FUNCTION NAME is the A/B experiment variable (``tool_name`` = ``computer_use``
or ``cli``); the tool CONTENTS are byte-identical across both.

``predict`` returns a list of typed action dicts consumed by ``run_loop.py``:
    - bash    -> {"kind": "bash", "command": <shell str>, "timeout"?: <seconds>}
    - wait    -> {"kind": "control", "control": "WAIT"}
    - terminate(success) -> {"kind": "control", "control": "DONE"}
    - terminate(failure) -> {"kind": "control", "control": "FAIL"}
    - answer  -> {"kind": "control", "control": "DONE", "answer": <text>}

History/CLI feedback reuse hybrid's ``build_hybrid_messages`` verbatim: one predict ==
one turn == one screenshot; the previous turn's command stdout/stderr is threaded into
the next predict as ``step_exec_results`` and rendered as a ``CLI output`` user turn,
capped at ``CLI_OUTPUT_MAX`` chars per command (head kept).
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

from mm_agents.qwen.history import previous_actions_text, update_folding_state, dump_debug_messages
from mm_agents.qwen.images import process_image
from mm_agents.qwen.main import _QwenBaseAgent
from mm_agents.hybrid.history import build_hybrid_messages

from .parser import split_tool_calls
from .prompts import build_c_gui_system_prompt, build_c_gui_tools_def, build_instruction_prompt

logger = logging.getLogger("desktopenv.c_gui_agent")

_VALID_TOOL_NAMES = ("computer_use", "cli")
_CONTROL_END = ("DONE", "FAIL")

#: Per-turn cap on CLI text entering the prompt. Output is never reclaimed (image folding
#: drops only images), so uncapped it accumulates until vLLM 400s on context length — one
#: measured block was 907k chars. 98% of observed blocks fall under this cap.
CLI_OUTPUT_MAX = 1000


def _truncate_cli(text: str, limit: int = CLI_OUTPUT_MAX) -> str:
    """Keep the first ``limit`` chars; mark the drop so a cut listing isn't read as whole."""
    return text if len(text) <= limit else text[:limit] + "\n...(truncated)..."


def _parse_timeout(value) -> Optional[float]:
    """Parse the optional ``timeout`` of a bash action; None when absent/garbage.

    A malformed timeout must not cost us the command, so anything unparseable or
    non-positive falls through to None (== the executor's default).
    """
    if value is None or str(value).strip() == "":
        return None
    try:
        timeout = float(str(value).strip())
    except ValueError:
        return None
    return timeout if timeout > 0 else None


class CGuiGuiOnlyAgent(_QwenBaseAgent):
    """Single bash-surface agent. ``predict`` returns typed action dicts and takes an
    extra per-turn CLI-feedback argument (``step_exec_results``)."""

    def __init__(
        self,
        *args,
        enable_thinking: bool = False,
        password: str = "password",
        tool_name: str = "computer_use",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.enable_thinking = enable_thinking
        self.password = password
        if tool_name not in _VALID_TOOL_NAMES:
            raise ValueError(f"tool_name must be one of {_VALID_TOOL_NAMES}, got {tool_name!r}")
        #: The A/B naming variable — 'computer_use' or 'cli'. Only the label differs.
        self.tool_name = tool_name
        #: index-aligned with screenshots; holds CLI text produced leading up to each turn.
        self.cli_outputs: List[str] = []

    def reset(self, _logger=None, *args, **kwargs):
        super().reset(_logger, *args, **kwargs)
        self.cli_outputs = []

    # --- payload / tool surface / prompt overrides ---------------------------
    def _build_payload(self, messages: List[Dict]) -> Dict:
        payload = super()._build_payload(messages)
        base_url = self.base_url or os.environ.get("OPENAI_BASE_URL", "")
        if "dashscope" in base_url.lower():
            extra_body = dict(payload.get("extra_body") or {})
            extra_body["enable_thinking"] = bool(self.enable_thinking)
            payload["extra_body"] = extra_body
        return payload

    def _build_tools_def(self, processed_width: int, processed_height: int) -> List[Dict]:  # type: ignore[override]
        # c-gui coords are 0-999 (scaled in the VM shim), so width/height are unused.
        return build_c_gui_tools_def(self.tool_name)

    def _build_system_prompt(self, tools_def: List[Dict]) -> str:  # type: ignore[override]
        return build_c_gui_system_prompt(tools_def, self.collapse_text, self.password)

    def _debug_message_filename(self, step_idx: int) -> str:
        return f"c_gui_messages_step_{step_idx}.json"

    def _log_prefix(self) -> str:
        return "CGui"

    # --- predict: forked from hybrid for dict output + CLI feedback ----------
    def predict(  # type: ignore[override]
        self,
        instruction: str,
        obs: Dict,
        step_exec_results: Optional[List[Dict]] = None,
    ) -> Tuple[str, List[Dict]]:
        # Attach the PREVIOUS turn's CLI output to the previous screenshot's slot (its
        # screenshot IS the post-command screen). This turn's own output isn't known
        # until run_loop executes the actions we return now.
        while len(self.cli_outputs) < len(self.screenshots):
            self.cli_outputs.append("")
        if step_exec_results and self.cli_outputs:
            # Cap per-result, not the join: a turn may bundle several bash commands, and
            # capping the join would drop the later commands entirely.
            prev_cli = "\n".join(
                _truncate_cli(r.get("text", "")) for r in step_exec_results if r.get("text")
            )
            if prev_cli:
                sep = "\n" if self.cli_outputs[-1] else ""
                self.cli_outputs[-1] = self.cli_outputs[-1] + sep + prev_cli

        # qwen's per-turn screenshot processing (no coordinate sizing needed: coords are
        # 0-999, scaled in the VM shim, so we never map to pixels agent-side).
        processed_b64 = process_image(obs["screenshot"])
        self.screenshots.append(processed_b64)
        total_steps = len(self.screenshots)
        self.folded_prefix_k = update_folding_state(
            total_steps, self.folded_prefix_k, self.image_max, self.fold_size
        )
        self.cli_outputs.append("")  # this turn's (initially empty) slot

        start_step = max(1, total_steps - self.history_n)
        previous_actions_str = previous_actions_text(self.actions, start_step)

        tools_def = self._build_tools_def(0, 0)
        system_prompt = self._build_system_prompt(tools_def)
        instruction_prompt = build_instruction_prompt(instruction, previous_actions_str)

        self.observations.append({"screenshot": processed_b64})
        messages = build_hybrid_messages(
            system_prompt=system_prompt,
            instruction_prompt=instruction_prompt,
            screenshots=self.screenshots,
            responses=self.responses,
            cli_outputs=self.cli_outputs,
            start_step=start_step,
            total_steps=total_steps,
            folded_prefix_k=self.folded_prefix_k,
            collapse_text=self.collapse_text,
            response_transform=self._response_transform,
        )
        dump_debug_messages(messages, self._debug_message_filename(total_steps - 1), logger)

        response = self.call_llm(self._build_payload(messages), self.model)
        if logger:
            logger.info("%s Output: %s", self._log_prefix(), response)
        self.responses.append(response or "")

        low_level, actions = self._parse_response(response or "")
        self.actions.append(low_level)
        return response or "", actions

    # --- response parsing: single tool, 4 actions, order preserved -----------
    def _parse_response(self, response: str) -> Tuple[str, List[Dict]]:  # type: ignore[override]
        actions: List[Dict] = []
        for func_name, params in split_tool_calls(response, allowed=(self.tool_name,)):
            action = str(params.get("action") or "").strip().lower()
            if action == "bash":
                command = params.get("command")
                if command:
                    bash_action = {"kind": "bash", "command": command}
                    timeout = _parse_timeout(params.get("timeout"))
                    if timeout is not None:
                        bash_action["timeout"] = timeout
                    actions.append(bash_action)
            elif action == "wait":
                actions.append({"kind": "control", "control": "WAIT"})
            elif action == "terminate":
                status = str(params.get("status") or "").strip().lower()
                actions.append({"kind": "control", "control": "FAIL" if status == "failure" else "DONE"})
            elif action == "answer":
                actions.append({"kind": "control", "control": "DONE", "answer": params.get("text", "")})
            # unknown action -> skip (keep the rest of the batch)

        n_bash = sum(1 for a in actions if a["kind"] == "bash")
        n_ctrl = len(actions) - n_bash
        low_level = f"{n_bash} bash + {n_ctrl} control action(s)" if actions else ""
        return low_level, actions
