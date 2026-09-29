"""Message-history helpers shared by every Qwen3.5 GUI agent — byte-identical port of
the eval-side pieces so TRAINING == EVAL.

Source (copied verbatim, only the ``build_messages`` cli_outputs merge is ours):
  env_infra/OSWorld/mm_agents/qwen/history.py

Copied rather than imported: online-rl and env_infra live in separate venvs, so
``mm_agents.*`` is not importable at runtime. Re-copy this file if the upstream one
changes — the byte-for-byte match is what makes train and eval condition on the same
string. ``tests/test_gui_only_eval_parity.py`` guards that by diffing against upstream.

``build_messages`` here unifies two upstream variants that differ in exactly three
places: ``qwen/history.py::build_messages`` (GUI-only) and
``hybrid/history.py::build_hybrid_messages`` (adds a per-step CLI-output turn). Passing
``cli_outputs=None`` reproduces the former byte for byte; passing a list reproduces the
latter.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional

__all__ = [
    "update_folding_state",
    "should_collapse_step",
    "previous_actions_text",
    "wrap_tool_response",
    "ensure_empty_think_prefix",
    "build_messages",
    "to_slime_image_parts",
]


def update_folding_state(total_screenshots: int, folded_prefix_k: int, image_max: int, fold_size: int) -> int:
    """Advance the folded-prefix ratchet. Monotone: once a step folds it never unfolds."""
    while (total_screenshots - folded_prefix_k) > image_max:
        folded_prefix_k += fold_size
    if folded_prefix_k > total_screenshots:
        folded_prefix_k = total_screenshots
    return folded_prefix_k


def should_collapse_step(step_num_1based: int, folded_prefix_k: int) -> bool:
    return step_num_1based <= folded_prefix_k


def previous_actions_text(actions: List[str], start_step: int) -> str:
    """One-line summaries of the turns DROPPED by the ``history_n`` window.

    Folded-but-retained steps are NOT listed here — they keep their real turn (with a
    collapse placeholder in place of the image).
    """
    previous_actions = [
        f"Step {i + 1}: {actions[i]}"
        for i in range(0, min(start_step - 1, len(actions)))
    ]
    return "\n".join(previous_actions) if previous_actions else "None"


def wrap_tool_response(parts: List[Dict]) -> List[Dict]:
    return (
        [{"type": "text", "text": "<tool_response>\n"}]
        + parts
        + [{"type": "text", "text": "\n</tool_response>"}]
    )


def ensure_empty_think_prefix(response: str) -> str:
    """Prepend an empty ``<think></think>`` block unless one is already there.

    The eval-side ``QwenAgent`` applies this to EVERY replayed assistant turn
    (``mm_agents/qwen/main.py:262``) unconditionally — ``enable_thinking`` only sets
    ``extra_body`` and does not gate it.
    """
    text = response or ""
    if re.match(r"^\s*<think>.*?</think>\s*", text, re.DOTALL):
        return text
    return "<think>\n\n</think>\n\n" + text.lstrip("\n")


def build_messages(
    *,
    system_prompt: str,
    instruction_prompt: str,
    screenshots: List[str],
    responses: List[str],
    start_step: int,
    total_steps: int,
    folded_prefix_k: int,
    collapse_text: str,
    cli_outputs: Optional[List[str]] = None,
    response_transform: Callable[[str], str] = lambda text: text,
) -> List[Dict]:
    """Render the policy prompt for the current turn.

    Layout per turn (``user`` role throughout — there is no ``role="tool"``):
      * first in-window turn: ``[image, instruction_prompt]``, NO ``<tool_response>`` wrap
      * later turns:          ``<tool_response>`` + [CLI output] + image + ``</tool_response>``
      * collapsed turns:      same, but ``collapse_text`` replaces the image entirely
      * a collapsed FIRST turn drops the placeholder and keeps only ``instruction_prompt``
      * the current (last) turn gets no trailing assistant message

    ``cli_outputs`` is index-aligned with ``screenshots``; ``cli_outputs[i]`` is the CLI
    text produced by step ``i+1``'s action and is rendered with the NEXT step's
    screenshot (that screenshot IS the post-action screen). ``None`` = no CLI channel.
    """
    messages: List[Dict] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
    ]

    for step_num in range(start_step, total_steps + 1):
        is_first_turn = step_num == start_step
        is_collapsed = should_collapse_step(step_num, folded_prefix_k)

        prev_idx = step_num - 2
        prev_cli = ""
        if cli_outputs and not is_first_turn and 0 <= prev_idx < len(cli_outputs) and cli_outputs[prev_idx]:
            prev_cli = "CLI output:\n" + cli_outputs[prev_idx]
        cli_parts = [{"type": "text", "text": prev_cli + "\n"}] if prev_cli else []

        if is_collapsed:
            if is_first_turn:
                user_content = [{"type": "text", "text": instruction_prompt}]
            else:
                user_content = wrap_tool_response(
                    cli_parts + [{"type": "text", "text": collapse_text}]
                )
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


def to_slime_image_parts(messages: List[Dict]) -> List[Dict]:
    """``{"type":"image_url",...}`` -> slime's ``{"type":"image","image":url}``.

    Only the image part's shape changes; text parts (system prompt, ``<tool_response>``
    tags, CLI output, collapse text) stay byte-identical — that is what the train/eval
    token alignment rests on. slime's ``process_vision_info`` expects the ``image`` form.
    """
    out: List[Dict] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        out.append({
            **msg,
            "content": [
                {"type": "image", "image": part.get("image_url", {}).get("url", "")}
                if isinstance(part, dict) and part.get("type") == "image_url"
                else part
                for part in content
            ],
        })
    return out
