"""KimiHybridAgent — Kimi driving a single bash surface with real-pixel coordinates.

Subclasses :class:`mm_agents.kimi.KimiAgent` so the vendored agent stays untouched
(CLAUDE.md: OSWorld internals are read-only). Reused from the parent as-is:

  * ``call_llm`` — the httpx transport, its 20-attempt retry and ``finish_reason``
    check, and the ``KIMI_BASE_URL`` / ``KIMI_API_KEY`` env contract,
  * the multi-image history window (``max_image_history_length``) and its collapse of
    older turns into one text block,
  * the ``## Thought / ## Action: / ## Code:`` response contract and the
    ``computer.*`` pseudo-function style.

Three things are overridden:

  * ``predict`` — forked to return **typed action dicts** (``kind="bash"|"control"``)
    instead of pyautogui strings, to take the extra ``step_exec_results`` argument the
    loop threads in, and to splice CLI output into the history.
  * ``_parse_response`` — the code block is a SHELL command, not python. Only
    ``computer.*`` control calls are interpreted; everything else is passed through
    verbatim as a bash command.
  * the system prompt — see ``prompts.py``.

**Kimi's coordinate projection is deliberately NOT used.** The parent runs every
pyautogui call through ``project_coordinate_to_absolute_scale``, which multiplies any
coordinate <= 1.0 by the screen size — correct for Kimi's native 0-1.0 relative output,
catastrophic here. The screenshot is sent at full resolution and the model reports real
screen pixels, so the command reaches the VM byte-for-byte as written. This is the one
substantive behavioural change versus stock Kimi and the most likely source of a score
delta; the prompt states the pixel contract twice.

The episode loop is ``mm_agents.c_gui_pixel.run_single_example_c_gui_pixel``, re-exported
from this package's ``__init__`` as ``run_single_example_kimi_hybrid``. It dispatches on
the action dict's ``kind`` and never inspects this class.
"""
from __future__ import annotations

import ast
import logging
import re
from typing import Dict, List, Optional, Tuple

from mm_agents.c_gui_pixel.actions import bash_action, control_action, describe
from mm_agents.c_gui_pixel.selftest import selftest as run_selftest
from mm_agents.kimi.kimi_agent import KimiAgent, STEP_TEMPLATE, encode_image

from .prompts import build_kimi_hybrid_system_prompt

logger = logging.getLogger("desktopenv.kimi_hybrid")

#: The per-step user prompt. Mirrors ``kimi_agent.INSTRUCTION_TEMPLATE`` but names the
#: terminal, since the model's only actuator is a shell command.
INSTRUCTION_TEMPLATE = (
    "# Task Instruction:\n{instruction}\n\n"
    "Please generate the next move according to the screenshot, task instruction, "
    "previous steps and the previous command's output (if provided).\n"
)

#: Any fenced code block. Wider than the parent's ``(?:code|python)?`` because the shell
#: blocks are tagged ```bash / ```sh / ```shell.
_CODE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+-]*[ \t]*\r?\n?(.*?)```", re.DOTALL)

#: A ``computer.<name>(...)`` control call occupying the whole code block.
_CONTROL_CALL_RE = re.compile(r"^\s*computer\.(\w+)\s*\((.*)\)\s*$", re.DOTALL)


def _parse_call_args(arg_text: str) -> Dict:
    """Parse a pseudo-function call's arguments into a dict.

    Parsed with ``ast`` rather than a regex so quoting, escapes and nested brackets
    behave. Unparseable arguments yield ``{}``: the ACTION still stands (a bare
    ``computer.terminate()`` is meaningful), only its parameters are lost.
    """
    try:
        call = ast.parse(f"f({arg_text})", mode="eval").body
    except SyntaxError:
        return {}
    args: Dict = {}
    for keyword in getattr(call, "keywords", []):
        if keyword.arg is None:
            continue
        try:
            args[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, SyntaxError):
            pass
    # A single positional arg is common shorthand (computer.wait(3), answer("42")).
    positional = getattr(call, "args", [])
    if positional:
        try:
            args["_positional"] = ast.literal_eval(positional[0])
        except (ValueError, SyntaxError):
            pass
    return args


class KimiHybridAgent(KimiAgent):
    """Kimi with one bash surface. ``predict`` returns typed action dicts."""

    def __init__(self, *args, max_output_chars: int = 2000, **kwargs):
        super().__init__(*args, **kwargs)
        #: Per-step cap on command output entering the prompt. Output is never reclaimed
        #: (the image window drops only images), so uncapped it accumulates until the
        #: gateway rejects the request on context length.
        self.max_output_chars = max_output_chars
        #: Index-aligned with ``self.actions``: the output produced BY step i.
        self.cli_outputs: List[str] = []
        # Replace the parent's prompt (its code block is python on OSWorld's own
        # channel; ours is a shell command). self.history_template stays as the parent
        # set it — the thinking/non-thinking skeletons are unchanged.
        self.system_prompt = build_kimi_hybrid_system_prompt(
            password=self.password,
            thinking=self.thinking,
            screen_width=self.screen_size[0],
            screen_height=self.screen_size[1],
        )

    def reset(self, _logger=None):
        super().reset(_logger)
        global logger
        if _logger is not None:
            logger = _logger
        self.cli_outputs = []

    # -- history -------------------------------------------------------------
    def _truncate(self, text: str) -> str:
        """Head-keep ``max_output_chars``, marking the cut so a clipped listing is not
        read as complete."""
        if len(text) <= self.max_output_chars:
            return text
        return text[: self.max_output_chars] + "\n...(truncated)..."

    def _build_messages(self, instruction: str, obs: Dict) -> List[Dict]:
        """Forked from ``KimiAgent.predict``'s message assembly, plus CLI output turns.

        The parent's window logic is preserved exactly: the most recent
        ``max_image_history_length`` steps carry their screenshot as a user turn, and
        everything older collapses into a single assistant text block. The addition is
        that each step's command output follows its assistant turn as a user message —
        that output IS the feedback for the action just described, and without it the
        model cannot read stdout at all.
        """
        messages: List[Dict] = [{"role": "system", "content": self.system_prompt}]
        history_step_texts: List[str] = []
        n = len(self.actions)

        for i in range(n):
            history_content = STEP_TEMPLATE.format(step_num=i + 1) + self.history_template.format(
                thought=self.cots[i].get("thought"),
                action=self.cots[i].get("action"),
            )
            output = self._truncate(self.cli_outputs[i]) if i < len(self.cli_outputs) else ""

            if i > n - self.max_image_history_length:
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                                   + encode_image(self.observations[i]["screenshot"])
                        },
                    }],
                })
                messages.append({"role": "assistant", "content": history_content})
                if output:
                    messages.append({"role": "user", "content": f"Command output:\n{output}"})
            else:
                # Collapsed (image-free) tail: fold the output into the same text block
                # so it survives the collapse instead of vanishing with the screenshot.
                if output:
                    history_content += f"\n## Command output:\n{output}\n"
                history_step_texts.append(history_content)
                if i == n - self.max_image_history_length:
                    messages.append({
                        "role": "assistant",
                        "content": "\n".join(history_step_texts),
                    })

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + encode_image(obs["screenshot"])
                    },
                },
                {"type": "text", "text": INSTRUCTION_TEMPLATE.format(instruction=instruction)},
            ],
        })
        return messages

    # -- predict -------------------------------------------------------------
    def predict(  # type: ignore[override]
        self,
        instruction: str,
        obs: Dict,
        step_exec_results: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Tuple[str, List[Dict]]:
        """Return ``(raw_response, actions)`` for the current screen.

        ``step_exec_results`` is the PREVIOUS turn's command output: it is the feedback
        for the action already recorded, not for the one being chosen now, so it is
        attached to the previous step's slot.
        """
        if step_exec_results and self.cli_outputs:
            previous = "\n".join(r.get("text", "") for r in step_exec_results if r.get("text"))
            if previous:
                separator = "\n" if self.cli_outputs[-1] else ""
                self.cli_outputs[-1] = self.cli_outputs[-1] + separator + previous

        logger.info("========= %s step %d =========", self.model, len(self.actions) + 1)
        messages = self._build_messages(instruction, obs)

        max_retry = 5
        response = None
        low_level_instruction = ""
        actions: List[Dict] = []
        sections: Dict = {}

        for attempt in range(max_retry):
            try:
                response = self.call_llm({
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "top_p": self.top_p,
                    # Mirror the parent: a retry nudges temperature off 0 so a
                    # deterministic bad parse is not simply reproduced.
                    "temperature": self.temperature if attempt == 0 else max(0.2, self.temperature),
                }, self.model)
                if not response:
                    raise ValueError("empty response from the Kimi gateway")

                logger.info("KimiHybrid output: %s", response)
                low_level_instruction, actions, sections = self._parse_response(response)
                if not actions:
                    raise ValueError(f"no action parsed from response: {low_level_instruction}")
                break
            except Exception as exc:  # noqa: BLE001 - one bad turn must not kill the episode
                logger.error("KimiHybrid predict attempt %d/%d failed: %s",
                             attempt + 1, max_retry, exc)
                if attempt == max_retry - 1:
                    logger.error("Maximum retries reached; failing the task.")
                    self._record_turn(obs, "Fail the task: no parsable action.",
                                      {"thought": "", "action": "parse failure"})
                    return str(exc), [control_action("FAIL")]

        self._record_turn(obs, low_level_instruction, sections)

        # Parent behaviour: force termination at the step ceiling unless the model
        # already chose to end the episode.
        if len(self.actions) >= self.max_steps and not any(
            a["kind"] == "control" and a["control"] in ("DONE", "FAIL") for a in actions
        ):
            logger.warning("Reached maximum steps %d. Forcing termination.", self.max_steps)
            self.actions[-1] = "Fail the task because reaching the maximum step limit."
            actions = [control_action("FAIL")]

        logger.info("Action: %s", low_level_instruction)
        logger.info("Parsed: %s", describe(actions))
        return response, actions

    def _record_turn(self, obs: Dict, low_level_instruction: str, sections: Dict) -> None:
        """Append this turn to the four index-aligned history lists."""
        self.observations.append(obs)
        self.actions.append(low_level_instruction)
        self.cots.append(sections)
        self.cli_outputs.append("")  # filled by the NEXT predict from step_exec_results

    # -- response parsing ----------------------------------------------------
    def _parse_response(self, response: Dict) -> Tuple[str, List[Dict], Dict]:
        """Kimi's response format -> ``(action_line, [action dicts], sections)``.

        The code block is a SHELL command unless it is a ``computer.*`` control call.
        Anything unrecognized is passed through as bash verbatim rather than rejected:
        a shell command we failed to anticipate is still a valid shell command.
        """
        content = (response.get("content") or "").lstrip()
        sections: Dict = {}

        if self.thinking:
            sections["thought"] = (response.get("reasoning_content") or "").strip()
            # Drop any preamble before "## Action" so the action regex cannot match
            # inside the model's own narration (parent does the same).
            match = re.search(r"^##\s*Action\b", content, flags=re.MULTILINE)
            if match:
                content = content[match.start():]
        else:
            thought = re.search(r"^##\s*Thought\s*:?[\n\r]+(.*?)(?=^##\s*Action:|^##|\Z)",
                                content, re.DOTALL | re.MULTILINE)
            sections["thought"] = thought.group(1).strip() if thought else ""

        action_match = re.search(r"^\s*##\s*Action\s*:?\s*[\n\r]+(.*?)(?=^\s*##|\Z)",
                                 content, re.DOTALL | re.MULTILINE)
        sections["action"] = action_match.group(1).strip() if action_match else ""

        blocks = _CODE_BLOCK_RE.findall(content)
        if not blocks:
            return f"<Error>: no code block found in: {content[:400]}", [], sections

        code = blocks[-1].strip()
        sections["original_code"] = code

        control = self._parse_control(code)
        if control is not None:
            sections["code"] = control["control"]
            return sections["action"] or code, [control], sections

        # Not a control call -> a shell command, passed through untouched. NO coordinate
        # projection: the model already reports real screen pixels.
        sections["code"] = code
        return sections["action"] or "run a shell command", [bash_action(code)], sections

    @staticmethod
    def _parse_control(code: str) -> Optional[Dict]:
        """Recognize ``computer.wait`` / ``computer.terminate`` / ``computer.answer``.

        Returns None when the block is not a control call, i.e. when it is shell.
        """
        match = _CONTROL_CALL_RE.match(code)
        if not match:
            return None
        name = match.group(1).lower()
        args = _parse_call_args(match.group(2))

        if name == "wait":
            return control_action("WAIT")
        if name == "terminate":
            status = str(args.get("status") or args.get("_positional") or "").strip().lower()
            # An explicit answer on a successful terminate is worth keeping: some
            # question-type tasks are graded on it.
            answer = args.get("answer")
            if status in ("failure", "fail"):
                return control_action("FAIL")
            return control_action("DONE", answer=answer if answer is not None else None)
        if name == "answer":
            text = args.get("text", args.get("_positional", ""))
            return control_action("DONE", answer=str(text))
        return None

    # -- wiring check --------------------------------------------------------
    def selftest(self) -> bool:
        """Verify the gateway answers and that images actually reach the model."""
        def call(text: str, image: Optional[bytes]) -> str:
            content: List[Dict] = []
            if image is not None:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + encode_image(image)
                    },
                })
            content.append({"type": "text", "text": text})
            try:
                response = self.call_llm({
                    "model": self.model,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 512,
                    "top_p": self.top_p,
                    "temperature": self.temperature,
                }, self.model)
            except Exception as exc:  # noqa: BLE001 - the check reports, never raises
                print(f"  transport error: {type(exc).__name__}: {str(exc)[:200]}")
                return ""
            return (response or {}).get("content") or ""

        return run_selftest(call, label="kimi-hybrid selftest")
