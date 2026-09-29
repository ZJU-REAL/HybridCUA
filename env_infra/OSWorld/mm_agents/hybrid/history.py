"""Message assembly for the hybrid agent — forks qwen's ``build_messages`` to inject
CLI output turns.

qwen's ``history.build_messages`` interleaves, per step: a ``user`` turn (screenshot or
collapse placeholder) then the ``assistant`` response. It has NO slot for tool/CLI output
text — the qwen model only ever perceives effects via the next screenshot. Our hybrid
adds a ``cli`` tool whose stdout/stderr the model MUST read, so we fork the builder to
splice a ``CLI output`` user message into the turn that carries it.

Design contract (see agent.py): a PURE-CLI step does NOT occupy a ``screenshots[]`` index
(so qwen's strict per-step index alignment and image folding — which count only
``len(screenshots)`` — are untouched). Instead each GUI step carries an aligned
``cli_outputs[i]`` string holding any CLI text produced SINCE the previous GUI step. When
non-empty, that text is rendered as an extra ``user`` message right after the step's
screenshot, inside a ``<tool_response>`` wrapper (reusing qwen's ``wrap_tool_response``).

Only ``build_messages`` is forked (the one place with the contradiction). Everything else
is imported read-only from ``mm_agents.qwen.history``.
"""
from __future__ import annotations

from typing import Callable, Dict, List

from mm_agents.qwen.history import should_collapse_step, wrap_tool_response  # read-only reuse

#: (Deprecated) previously capped CLI text per step; CLI output is now rendered in full
#: (no truncation) inside the following turn's tool_response, alongside the screenshot.
CLI_OUTPUT_MAX = 4000


def build_hybrid_messages(
    *,
    system_prompt: str,
    instruction_prompt: str,
    screenshots: List[str],
    responses: List[str],
    cli_outputs: List[str],
    start_step: int,
    total_steps: int,
    folded_prefix_k: int,
    collapse_text: str,
    response_transform: Callable[[str], str] = lambda text: text,
) -> List[Dict]:
    """Forked from qwen.history.build_messages; adds a per-step CLI-output user turn.

    ``cli_outputs`` is index-aligned with ``screenshots``/``responses`` (one entry per
    GUI step). ``cli_outputs[i]`` is the CLI text accumulated leading up to step ``i+1``;
    empty string means that step had no CLI output.
    """
    messages: List[Dict] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
    ]

    for step_num in range(start_step, total_steps + 1):
        is_first_turn = step_num == start_step
        is_collapsed = should_collapse_step(step_num, folded_prefix_k)

        # HYBRID: the CLI output produced by the PREVIOUS turn's actions is the feedback
        # that belongs WITH this turn's screenshot (this screenshot IS the post-action
        # screen). So we render it INSIDE this turn's tool_response, together with the
        # image: <tool_response>\nCLI output:\n<stdout>\n<image>\n</tool_response>.
        # (prev_idx is this turn's step index minus 1; index 0 for the first turn has no
        # prior CLI output.) Full output — no truncation.
        prev_idx = step_num - 2
        prev_cli = ""
        if not is_first_turn and 0 <= prev_idx < len(cli_outputs) and cli_outputs[prev_idx]:
            prev_cli = "CLI output:\n" + cli_outputs[prev_idx]
        cli_parts = [{"type": "text", "text": prev_cli + "\n"}] if prev_cli else []

        if is_collapsed:
            if is_first_turn:
                user_content = [{"type": "text", "text": instruction_prompt}]
            else:
                user_content = wrap_tool_response(
                    cli_parts + [{"type": "text", "text": collapse_text}]
                )
            messages.append({"role": "user", "content": user_content})
        else:
            img_url = f"data:image/png;base64,{screenshots[step_num - 1]}"
            if is_first_turn:
                user_content = [
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "text", "text": instruction_prompt},
                ]
            else:
                user_content = wrap_tool_response(
                    cli_parts + [{"type": "image_url", "image_url": {"url": img_url}}]
                )
            messages.append({"role": "user", "content": user_content})

        if step_num <= total_steps - 1 and (step_num - 1) < len(responses):
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": response_transform(responses[step_num - 1]),
                        }
                    ],
                }
            )

    return messages
