"""Qwen3.5 hybrid CUA training agent — bash-surface action space aligned with the
eval-side ``mm_agents/c_gui`` agent.

Subclasses :class:`agents.qwen35_agent.Qwen35VLAgentLocal` and reuses its model-generic
machinery verbatim — ``generate_with_sglang`` (sglang /generate + abort-retry),
``build_train_data`` (loss mask ``qwen3_5`` + multimodal alignment) and
``_align_loss_mask_multimodal``. Only the prompt / tool schema / message layout / response
parsing are overridden to switch from the 14-primitive ``computer_use`` surface to c_gui's
single ``bash`` surface (bash / wait / terminate / answer).

Contract with the rollout (rollout/trajectory.py + trajectory_runner.py) is unchanged:
``reset`` / ``build_policy_messages`` / ``generate_with_sglang`` / ``parse_response`` /
``record_policy_turn`` / ``build_train_data`` / ``build_train_system_message`` /
``get_tool_spec``. One addition — ``record_step_exec_results`` — is an OPTIONAL hook the
rollout calls (guarded by hasattr) to thread the previous command's stdout/stderr back
into the next prompt, mirroring c_gui's ``step_exec_results`` / ``cli_outputs`` design.

GUI is driven via ``action=bash`` running a pyautogui heredoc with 0-999 coordinates; the
0-999 -> real-pixel scaling happens in the VM shim (clients/coord_shim.py), so this agent
does NOT parse or rescale coordinates — the ``command`` string is passed through verbatim.
"""
from __future__ import annotations

import base64
import os
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from agents.qwen35_agent import Qwen35VLAgentLocal, process_image
from agents.hybrid_prompt import (
    build_c_gui_system_prompt,
    build_c_gui_tool_def,
    build_c_gui_tools_def,
    build_hybrid_messages,
    build_instruction_prompt,
    split_tool_calls,
)
from agents.utils.qwen_history import (
    previous_actions_text,
    to_slime_image_parts,
    update_folding_state,
)

_VALID_TOOL_NAMES = ("computer_use", "cli")

#: Per-turn cap on CLI text entering the prompt (aligned with c_gui agent.py). Output is
#: never reclaimed (image folding drops only images), so uncapped it accumulates until the
#: context length blows up. 98% of observed blocks fall under this cap; head kept.
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


class Qwen35HybridCuaAgentLocal(Qwen35VLAgentLocal):
    """Qwen3.5 GUI agent with c_gui's single ``bash`` action surface."""

    def __init__(
        self,
        *args,
        tool_name: str = "computer_use",
        password: str = "password",
        history_n: Optional[int] = None,
        fold_size: int = 1,
        **kwargs,
    ):
        # Verbose-history window (# of most-recent steps rendered in full). Steps older
        # than this are dropped from the message list and kept only as one-line action
        # summaries (hybrid_prompt.previous_actions_text). Default 3 aligns the text
        # window with the image window (image_max/gui_max_image_history_length=3), which
        # keeps total_len — and thus the per-step entropy backward retention (~2*T*V) —
        # bounded on long (up to gui_max_steps) trajectories. Override via GUI_HISTORY_N.
        if history_n is None:
            history_n = int(os.getenv("GUI_HISTORY_N", "3") or "3")
        super().__init__(*args, history_n=history_n, fold_size=fold_size, **kwargs)
        if tool_name not in _VALID_TOOL_NAMES:
            raise ValueError(f"tool_name must be one of {_VALID_TOOL_NAMES}, got {tool_name!r}")
        self.tool_name = tool_name
        self.password = password
        # fold one-at-a-time so the recent-image window stays steady at image_max (and
        # never > image_max, which would collapse the current frame).
        if self.fold_size > self.image_max:
            self.fold_size = self.image_max
        # cli_outputs[i] = stdout/stderr of step i+1's action (filled post-step); "" = none.
        self.cli_outputs: List[str] = []

    def reset(self, _logger=None):
        super().reset(_logger)
        self.cli_outputs = []

    # ------------------------------------------------------------------ prompt / tool
    def get_tool_spec(
        self,
        processed_width: Optional[int] = None,
        processed_height: Optional[int] = None,
    ) -> Dict[str, Any]:
        # c_gui coords are 0-999, so processed width/height are unused.
        return build_c_gui_tool_def(self.tool_name)

    def get_system_prompt(
        self,
        processed_width: Optional[int] = None,
        processed_height: Optional[int] = None,
    ) -> str:
        tools_def = build_c_gui_tools_def(self.tool_name)
        return build_c_gui_system_prompt(tools_def, self.collapse_text, self.password)

    def build_instruction_prompt(self, instruction: str, previous_actions_str: str) -> str:
        return build_instruction_prompt(instruction, previous_actions_str)

    def build_train_system_message(self) -> Dict[str, Any]:
        return {"role": "system", "content": self.get_system_prompt()}

    # ------------------------------------------------------------------ CLI feedback hook
    def record_step_exec_results(self, cli_text: str) -> None:
        """Thread the just-executed turn's command stdout/stderr into the last CLI slot.

        Called by the rollout AFTER the env step (guarded by hasattr), so the NEXT
        build_policy_messages renders it as a ``CLI output`` inside the following turn's
        <tool_response>. No-op when there's no slot yet or no text.
        """
        if cli_text and self.cli_outputs:
            sep = "\n" if self.cli_outputs[-1] else ""
            self.cli_outputs[-1] = self.cli_outputs[-1] + sep + _truncate_cli(cli_text)

    def record_policy_turn(self, *, action_text: str, response: str, screenshot_bytes: bytes) -> None:
        super().record_policy_turn(
            action_text=action_text, response=response, screenshot_bytes=screenshot_bytes
        )
        # This turn's (initially empty) CLI slot; filled by record_step_exec_results.
        self.cli_outputs.append("")

    # ------------------------------------------------------------------ message assembly
    def build_policy_messages(self, instruction: str, obs: Dict) -> Dict[str, Any]:
        step_index = len(self.actions)
        screenshot_bytes: bytes = obs["screenshot"]

        img0 = Image.open(BytesIO(screenshot_bytes))
        original_width, original_height = img0.size

        processed_image_b64 = process_image(screenshot_bytes)
        processed_img = Image.open(BytesIO(base64.b64decode(processed_image_b64)))
        processed_width, processed_height = processed_img.size

        all_screenshots = list(self.screenshots) + [processed_image_b64]
        total_steps = len(all_screenshots)

        # Fold old screenshots to text; keep the recent-image window. Persist across turns.
        self.folded_prefix_k = update_folding_state(
            total_steps, self.folded_prefix_k, self.image_max, self.fold_size
        )
        start_step = max(1, total_steps - self.history_n)
        prev_actions = previous_actions_text(self.actions, start_step)

        system_prompt = self.get_system_prompt()
        tool_spec = self.get_tool_spec()
        instruction_prompt = self.build_instruction_prompt(instruction, prev_actions)

        # cli_outputs aligned to all_screenshots: prior turns + an empty slot for the
        # current (not-yet-executed) turn.
        cli_outputs = list(self.cli_outputs) + [""]

        messages = build_hybrid_messages(
            system_prompt=system_prompt,
            instruction_prompt=instruction_prompt,
            screenshots=all_screenshots,
            responses=self.responses,
            cli_outputs=cli_outputs,
            start_step=start_step,
            total_steps=total_steps,
            folded_prefix_k=self.folded_prefix_k,
            collapse_text=self.collapse_text,
        )
        messages = to_slime_image_parts(messages)

        return {
            "messages": messages,
            "tool_spec": tool_spec,
            "step_index": step_index,
            "processed_image_b64": processed_image_b64,
            "original_width": original_width,
            "original_height": original_height,
            "processed_width": processed_width,
            "processed_height": processed_height,
            "system_prompt": system_prompt,
        }

    # ------------------------------------------------------------------ response parsing
    def parse_response(
        self,
        response: str,
        original_width: int,
        original_height: int,
        processed_width: Optional[int] = None,
        processed_height: Optional[int] = None,
    ) -> Tuple[str, List[Any], Dict[str, Any]]:
        """Parse the bash-surface XML tool call (aligned with c_gui parser/_parse_response).

        Returns ``(natural_action, actions, other)`` where each element of ``actions`` is
        either a bash action dict ``{"action_type":"bash","command":...,"timeout"?:...}``
        (routed to a tool/cli step by clients.osworld_remote_async.to_cluster_action) or a
        control token string ``"WAIT"`` / ``"DONE"`` / ``"FAIL"``. Coordinates inside the
        bash command are 0-999 and scaled by the VM shim — NOT rescaled here.
        """
        other: Dict[str, Any] = {"raw_response": response, "tool_calls": []}
        actions: List[Any] = []
        natural_action = ""

        if response is None or not response.strip():
            other["action"] = ""
            other["code"] = []
            return natural_action, actions, other

        for line in response.split("\n"):
            stripped = line.strip()
            if stripped.lower().startswith("action:"):
                natural_action = stripped.split("Action:", 1)[-1].strip()
                break

        for _func_name, params in split_tool_calls(response, allowed=(self.tool_name,)):
            action = str(params.get("action") or "").strip().lower()
            other["tool_calls"].append(params)
            if action == "bash":
                command = params.get("command")
                if command:
                    bash_action: Dict[str, Any] = {"action_type": "bash", "command": command}
                    timeout = _parse_timeout(params.get("timeout"))
                    if timeout is not None:
                        bash_action["timeout"] = timeout
                    actions.append(bash_action)
            elif action == "wait":
                actions.append("WAIT")
            elif action == "terminate":
                status = str(params.get("status") or "").strip().lower()
                actions.append("FAIL" if status == "failure" else "DONE")
            elif action == "answer":
                actions.append("DONE")
            # unknown action -> skip (keep the rest of the batch)

        if not natural_action:
            if actions and isinstance(actions[0], dict):
                natural_action = "Run bash command"
            elif actions and actions[0] == "WAIT":
                natural_action = "Waiting"
            elif actions and actions[0] in ("DONE", "FAIL"):
                natural_action = "Task completed"
            else:
                natural_action = "Execute action"

        other["action"] = natural_action
        other["code"] = actions
        return natural_action, actions, other
