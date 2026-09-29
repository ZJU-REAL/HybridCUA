"""HybridAgent — a GUI+CLI OSWorld agent (plan A: channel-tagged action dicts).

Subclasses qwen's ``_QwenBaseAgent`` (reusing its OpenAI call, image processing, history
folding, and reset) and overrides ``predict`` so the model is offered BOTH
``computer_use`` (GUI) and ``cli`` (CLI), and returns a list of channel-tagged action
dicts that the custom loop (run_loop.py) routes:

    - computer_use -> {channel:"gui", command:<pyautogui/control str>, control:...}
    - cli          -> {channel:"cli", command:<VM python str>}

GUI actions execute via ``env.step(pyautogui)`` (feedback = screenshot); CLI actions via
``env.run_code(python)`` (feedback = stdout/stderr TEXT only). This is the same
architecture as ``claude_hybrid``.

History / CLI feedback: one ``predict`` call == one "turn" == one screenshot. The custom
loop executes ALL of a turn's actions (GUI + CLI, in order) and then hands the latest
``obs`` plus that turn's CLI text (``step_exec_results``) to the NEXT ``predict``. So
``screenshots``, ``responses`` and ``cli_outputs`` grow by exactly one per turn and stay
index-aligned — qwen's per-step folding (which counts only ``len(screenshots)``) is
untouched. The forked ``build_hybrid_messages`` renders ``cli_outputs[i]`` as a
``CLI output`` user turn on step ``i`` so the model reads terminal output.

Ordering is strict: we walk the model's ``<tool_call>`` blocks in source order and
translate each; GUI reuse is per-action via qwen's ``parse_base_response`` on a rewrapped
single-tool string (function-level reuse, no copy of the pyautogui mapping).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

from mm_agents.qwen.actions import parse_base_response  # reused read-only from qwen
from mm_agents.qwen.history import previous_actions_text, update_folding_state
from mm_agents.qwen.images import image_size_from_base64, image_size_from_bytes, process_image
from mm_agents.qwen.main import _QwenBaseAgent
from mm_agents.qwen.prompts import build_instruction_prompt

from .cli_tools import cli_action_to_code, CliActionError
from .history import build_hybrid_messages
from .parser import split_tool_calls
from .prompts import build_hybrid_system_prompt, build_hybrid_tools_def

logger = logging.getLogger("desktopenv.hybrid_agent")

#: Control tokens that parse_base_response emits as bare string entries.
_CONTROL_TOKENS = ("DONE", "FAIL", "WAIT", "CALL_USER")


def _rewrap_computer_use(params: Dict) -> str:
    """Rebuild a minimal single ``<tool_call>`` string for one computer_use call.

    Fed to qwen's ``parse_base_response`` so we reuse its exact GUI->pyautogui mapping
    on exactly one action (preserving inter-action order at the agent level).
    """
    lines = ["<tool_call>", "<function=computer_use>"]
    for name, value in params.items():
        rendered = value if isinstance(value, str) else json.dumps(value)
        lines.append(f"<parameter={name}>")
        lines.append(rendered)
        lines.append("</parameter>")
    lines.append("</function>")
    lines.append("</tool_call>")
    return "\n".join(lines)


class HybridAgent(_QwenBaseAgent):
    """GUI + CLI agent. predict() returns channel-tagged action dicts and takes an extra
    per-turn CLI-feedback argument (``step_exec_results``)."""

    def __init__(self, *args, enable_thinking: bool = False, password: str = "password", **kwargs):
        super().__init__(*args, **kwargs)
        # Mirror QwenAgent: forward DashScope's extra_body.enable_thinking toggle.
        self.enable_thinking = enable_thinking
        # sudo password stated in the system prompt (see build_hybrid_system_prompt); the
        # value comes from DesktopEnv.client_password via the runner (--password).
        self.password = password
        self.cli_outputs: List[str] = []

    def reset(self, _logger=None, *args, **kwargs):
        super().reset(_logger, *args, **kwargs)
        self.cli_outputs = []

    def _build_payload(self, messages: List[Dict]) -> Dict:
        payload = super()._build_payload(messages)
        base_url = self.base_url or os.environ.get("OPENAI_BASE_URL", "")
        if "dashscope" in base_url.lower():
            extra_body = dict(payload.get("extra_body") or {})
            extra_body["enable_thinking"] = bool(self.enable_thinking)
            payload["extra_body"] = extra_body
        return payload

    # --- tool surface / prompt overrides -------------------------------------
    def _build_tools_def(self, processed_width: int, processed_height: int) -> List[Dict]:  # type: ignore[override]
        return build_hybrid_tools_def(processed_width, processed_height, self.coordinate_type)

    def _build_system_prompt(self, tools_def: List[Dict]) -> str:  # type: ignore[override]
        return build_hybrid_system_prompt(tools_def, self.collapse_text, self.password)

    def _debug_message_filename(self, step_idx: int) -> str:
        return f"hybrid_messages_step_{step_idx}.json"

    def _log_prefix(self) -> str:
        return "Hybrid"

    # --- predict: forked from _QwenBaseAgent.predict for dict output + CLI feedback ---
    def predict(  # type: ignore[override]
        self,
        instruction: str,
        obs: Dict,
        step_exec_results: Optional[List[Dict]] = None,
    ) -> Tuple[str, List[Dict]]:
        # cli_outputs is index-aligned with screenshots (one entry per prior turn). Attach
        # the PREVIOUS turn's CLI output to the previous screenshot's slot so
        # build_hybrid_messages renders it on that turn. (This turn's own CLI output isn't
        # known until the loop executes the actions we return now.)
        while len(self.cli_outputs) < len(self.screenshots):
            self.cli_outputs.append("")
        if step_exec_results and self.cli_outputs:
            prev_cli = "\n".join(r.get("text", "") for r in step_exec_results if r.get("text"))
            if prev_cli:
                sep = "\n" if self.cli_outputs[-1] else ""
                self.cli_outputs[-1] = self.cli_outputs[-1] + sep + prev_cli

        # -- qwen's per-turn screenshot processing (verbatim logic) --
        screenshot_bytes = obs["screenshot"]
        original_width, original_height = image_size_from_bytes(screenshot_bytes)
        processed_b64 = process_image(screenshot_bytes)
        processed_width, processed_height = image_size_from_base64(processed_b64)

        self.screenshots.append(processed_b64)
        total_steps = len(self.screenshots)
        self.folded_prefix_k = update_folding_state(
            total_steps, self.folded_prefix_k, self.image_max, self.fold_size
        )
        # add this turn's (initially empty) cli slot, keeping the two lists aligned
        self.cli_outputs.append("")

        start_step = max(1, total_steps - self.history_n)
        previous_actions_str = previous_actions_text(self.actions, start_step)

        tools_def = self._build_tools_def(processed_width, processed_height)
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

        from mm_agents.qwen.history import dump_debug_messages
        dump_debug_messages(messages, self._debug_message_filename(total_steps - 1), logger)

        response = self.call_llm(self._build_payload(messages), self.model)
        if logger:
            logger.info("%s Output: %s", self._log_prefix(), response)
        self.responses.append(response or "")

        low_level, actions = self._parse_response(
            response or "",
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
        )
        self.actions.append(low_level)
        return response or "", actions

    # --- response parsing: GUI + CLI, order preserved, dict output -----------
    def _parse_response(
        self,
        response: str,
        *,
        original_width: int,
        original_height: int,
        processed_width: int,
        processed_height: int,
    ) -> Tuple[str, List[Dict]]:
        actions: List[Dict] = []
        gui_count = cli_count = 0

        for func_name, params in split_tool_calls(response):
            if func_name == "computer_use":
                _, gui_codes = parse_base_response(
                    _rewrap_computer_use(params),
                    coordinate_type=self.coordinate_type,
                    original_width=original_width,
                    original_height=original_height,
                    processed_width=processed_width,
                    processed_height=processed_height,
                )
                for code in gui_codes:
                    control = code if code in _CONTROL_TOKENS else None
                    actions.append({"channel": "gui", "command": code, "control": control})
                gui_count += 1
            elif func_name == "cli":
                try:
                    actions.append({"channel": "cli", "command": cli_action_to_code(params), "control": None})
                    cli_count += 1
                except CliActionError as exc:
                    logger.warning("bad cli call: %s", exc)
                    # skip the malformed cli action; keep the rest of the batch.

        low_level = f"{gui_count} GUI + {cli_count} CLI action(s)" if actions else ""
        return low_level, actions
