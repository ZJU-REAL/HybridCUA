"""ClaudeHybridAgent — Claude driving a single bash surface with real-pixel coordinates.

Subclasses :class:`mm_agents.anthropic.AnthropicAgent` so the vendored agent stays
untouched (CLAUDE.md: OSWorld internals are read-only). Reused from it, read-only:

  * ``get_claude_runtime_profile`` — the 4.5/4.6 legacy-thinking vs 4.7 adaptive-effort
    split, so an older model keeps running exactly as it did,
  * ``get_model_name`` / ``APIProvider`` — provider model-id mapping,
  * ``_response_to_params`` (thinking-block signatures included), ``_inject_prompt_caching``,
    ``_maybe_filter_to_n_most_recent_images``.

``predict`` is a FORK rather than an override chain, for two reasons that cannot be
reached incrementally: the parent hard-codes the native ``computer`` tool in ``tools=``,
and it answers every ``tool_use`` with the constant string ``"Success"`` plus a
screenshot. Here a bash command's real stdout/stderr IS the feedback. The fork keeps the
parent's error handling verbatim in spirit — long retry loop, 413/25MB payload errors
halving the image budget, backup key — because those are load-bearing on long runs.

Kept in sync with the parent by hand. If ``mm_agents/anthropic/main.py`` changes its
retry ladder or its message bookkeeping, this fork does not track it automatically.

Two things this agent does NOT do, both consequences of the single surface:

  * no native computer tool, hence no computer-use beta flag,
  * no 1280x720 resize and no ``resize_factor`` coordinate rescaling. The screenshot goes
    out at its own resolution and the model reports real screen pixels. NOTE: Anthropic
    downscales images whose long edge exceeds 1568px, so at 1920x1080 the model sees a
    ~1568x882 rendering while reasoning in 1920x1080 coordinates. The prompt states the
    true screen size; the ``--screen_width/--screen_height`` values must match the VM.

The episode loop is ``mm_agents.c_gui_pixel.run_single_example_c_gui_pixel``, re-exported
as ``run_single_example_claude_hybrid``. It dispatches on the action dict's ``kind`` and
knows nothing about this class being stateful.
"""
from __future__ import annotations

import base64
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from mm_agents.anthropic.main import API_RETRY_INTERVAL, API_RETRY_TIMES, AnthropicAgent
from mm_agents.anthropic.utils import (
    CLAUDE_47_PROFILE,
    APIProvider,
    PROMPT_CACHING_BETA_FLAG,
    _inject_prompt_caching,
    _maybe_filter_to_n_most_recent_images,
    _response_to_params,
    get_claude_runtime_profile,
    get_model_name,
)
from mm_agents.c_gui_pixel.actions import describe, typed_action
from mm_agents.c_gui_pixel.selftest import selftest as run_selftest

from .prompts import (
    TOOL_NAME,
    build_claude_hybrid_system_prompt,
    build_claude_hybrid_tool_def,
)

logger = logging.getLogger("desktopenv.claude_hybrid")

#: One id per process, sent as ``x-claude-code-session-id`` on the gateway path only
#: (see ``_build_client``). The tclaude gateway 400s without it.
SESSION_ID = os.environ.get(
    "OSWORLD_SESSION_ID", "osworld-claude-hybrid-" + uuid.uuid4().hex[:16]
)

#: Gateway-hosted NON-Claude models. ``get_claude_runtime_profile`` only knows Claude
#: names, so these fall through to the 4.6 legacy profile and a fixed 2048-token
#: thinking budget. Measured 2026-09-13: they handle ``thinking: adaptive`` fine.
_GATEWAY_ADAPTIVE_MARKERS = ("kimi", "glm", "deepseek", "hy3", "qwen")

#: Payload-size error signatures (413 / 25MB). Same set the parent matches on.
_SIZE_ERROR_MARKERS = (
    "25000000",
    "Member must have length less than or equal to",
    "request_too_large",
    "maximum size",
    "413",
)


def _is_size_error(message: str) -> bool:
    lowered = message.lower()
    return any(
        marker.lower() in lowered if marker != "413" else "413" in message
        for marker in _SIZE_ERROR_MARKERS
    )


#: Model-name markers for generations at or beyond Claude 4.7, i.e. adaptive thinking.
#: ``get_claude_runtime_profile`` only recognizes 4-5 / 4-6 / 4-7 explicitly and falls
#: back to the 4.6 legacy profile for anything else — which silently downgrades a newer
#: model to a fixed 2048-token thinking budget. The runner's own default model is
#: ``claude-opus-4-8``, so that fallback would hit the common case.
_ADAPTIVE_MARKERS = ("4-8", "4.8", "4-9", "4.9", "-5", "opus-5", "sonnet-5", "haiku-5")


def _resolve_runtime_profile(model_name: str):
    """The parent's profile lookup, extended forward to 4.8+ and the Claude 5 family."""
    profile = get_claude_runtime_profile(model_name)
    if profile.thinking_mode == "adaptive":
        return profile
    normalized = (model_name or "").lower().replace("_", "-")
    # An explicitly recognized older generation wins: 4-5/4-6 must keep running as they
    # did, and "claude-3-5-sonnet" must not be read as a Claude 5.
    for legacy in ("4-5", "4.5", "4-6", "4.6", "3-5", "3.5", "3-7", "3.7"):
        if legacy in normalized:
            return profile
    if any(marker in normalized for marker in _ADAPTIVE_MARKERS):
        return CLAUDE_47_PROFILE
    # Gateway-hosted non-Claude models (kimi/glm/deepseek/hy3). Checked AFTER the legacy
    # whitelist so a hypothetical "claude-glm-4-5" still reads as legacy.
    if any(marker in normalized for marker in _GATEWAY_ADAPTIVE_MARKERS):
        return CLAUDE_47_PROFILE
    return profile


class ClaudeHybridAgent(AnthropicAgent):
    """Claude with one bash surface. ``predict`` returns typed action dicts."""

    def __init__(
        self,
        *args,
        base_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        password: str = "osworld-public-evaluation",
        max_output_chars: int = 4000,
        tool_name: str = TOOL_NAME,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # The parent hard-codes Anthropic(api_key=...); gateways need a base_url and a
        # Bearer token, which the runner already passes.
        self.base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")
        self.auth_token = auth_token or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        self.password = password
        #: Per-command cap on output entering the prompt. Text is never reclaimed (image
        #: filtering drops only images), so uncapped it grows until the context limit.
        self.max_output_chars = max_output_chars
        self.tool_name = tool_name
        #: ``tool_use`` blocks from the last assistant turn awaiting a ``tool_result``,
        #: as ``(id, is_bash)`` in emission order. Bash results carry real output; control
        #: results are acknowledgements.
        self._pending: List[Tuple[str, bool]] = []
        # No native computer tool -> no 1280x720 contract -> no coordinate rescaling.
        self.resize_factor = None
        self.runtime_profile = _resolve_runtime_profile(self.model_name)
        logger.info("Runtime profile for %s: %s", self.model_name, self.runtime_profile.label)

    def reset(self, _logger=None, *args, **kwargs):
        super().reset(_logger, *args, **kwargs)
        global logger
        if _logger is not None:
            logger = _logger
        self._pending = []

    # -- transport -----------------------------------------------------------
    def _build_client(self, api_key: Optional[str] = None):
        """Build the SDK client, preferring Bearer auth over ``x-api-key``.

        The parent only ever constructs ``Anthropic(api_key=...)`` against the public
        API. Bedrock/Vertex are delegated to the parent's own client construction.
        """
        from anthropic import Anthropic, AnthropicBedrock, AnthropicVertex

        if self.provider == APIProvider.VERTEX:
            return AnthropicVertex(), False
        if self.provider == APIProvider.BEDROCK:
            return AnthropicBedrock(
                aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                aws_region=os.getenv("AWS_DEFAULT_REGION"),
            ), False

        kwargs: Dict[str, Any] = {"max_retries": 4}
        if self.base_url:
            kwargs["base_url"] = self.base_url
            # A local gateway 400s without this header; only sent on the gateway path.
            kwargs["default_headers"] = {"x-claude-code-session-id": SESSION_ID}
        token = api_key if api_key is not None else self.auth_token
        if token and api_key is None:
            kwargs["auth_token"] = token
        else:
            kwargs["api_key"] = api_key or self.api_key
        # Prompt caching is only claimed for the first-party API; a gateway may not
        # honour the beta, and a rejected beta flag costs the whole run.
        caching = not self.base_url
        return Anthropic(**kwargs), caching

    # -- tool_result bookkeeping --------------------------------------------
    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        return text[: self.max_output_chars] + "\n...(truncated)..."

    def _answer_pending(
        self, step_exec_results: Optional[List[Dict]], screenshot: Optional[bytes]
    ) -> None:
        """Emit one ``tool_result`` per pending ``tool_use``, in order.

        Claude requires every ``tool_use`` to be answered, by id, before the next
        assistant turn — an unanswered id is a hard 400. ``step_exec_results`` holds one
        entry per BASH action only (control actions produce no output), so the two are
        zipped by walking the pending list and consuming a result for each bash entry.

        The final result also carries the turn's end-of-turn screenshot: that is the
        post-command screen, and attaching it to the last block is what keeps exactly one
        image per turn.
        """
        if not self._pending:
            return
        results = list(step_exec_results or [])
        content: List[Dict[str, Any]] = []

        for index, (tool_id, is_bash) in enumerate(self._pending):
            if is_bash and results:
                text = self._truncate(results.pop(0).get("text", "") or "(no output)")
            elif is_bash:
                # No result for a bash call: the loop stopped early (episode ended
                # mid-turn). Say so rather than claiming success.
                text = "(no output captured: the episode ended before this command reported)"
            else:
                text = "Success"

            block: Dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": [{"type": "text", "text": text}],
            }
            if screenshot is not None and index == len(self._pending) - 1:
                block["content"].append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(screenshot).decode("utf-8"),
                    },
                })
            content.append(block)

        self.messages.append({"role": "user", "content": content})
        self._pending = []

        if self.runtime_profile.use_claude_47_prompt:
            self.messages[-1]["content"].append({
                "type": "text",
                "text": f"[Current step: {self.current_step}/{self.max_steps}]",
            })

    # -- predict -------------------------------------------------------------
    def predict(  # type: ignore[override]
        self,
        instruction: str,
        obs: Optional[Dict] = None,
        step_exec_results: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Tuple[str, List[Dict]]:
        """Return ``(raw_response_text, actions)`` for the current screen."""
        self.current_step += 1
        screenshot = (obs or {}).get("screenshot")

        system_text = build_claude_hybrid_system_prompt(
            password=self.password,
            screen_width=self.screen_size[0],
            screen_height=self.screen_size[1],
            max_steps=self.max_steps,
            tool_name=self.tool_name,
            suffix=self.system_prompt_suffix,
        )
        system: Dict[str, Any] = {"type": "text", "text": system_text}

        if not self.messages:
            # Opening turn: the screen plus the task. No resize — full resolution.
            content: List[Dict[str, Any]] = []
            if screenshot is not None:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(screenshot).decode("utf-8"),
                    },
                })
            content.append({"type": "text", "text": instruction})
            self.messages.append({"role": "user", "content": content})
        else:
            self._answer_pending(step_exec_results, screenshot)

        client, enable_caching = self._build_client()
        betas: List[str] = []
        image_truncation_threshold = 10
        if enable_caching:
            betas.append(PROMPT_CACHING_BETA_FLAG)
            _inject_prompt_caching(self.messages)
            image_truncation_threshold = 20
            system["cache_control"] = {"type": "ephemeral"}

        if self.only_n_most_recent_images:
            _maybe_filter_to_n_most_recent_images(
                self.messages,
                self.only_n_most_recent_images,
                min_removal_threshold=image_truncation_threshold,
            )

        tools = [build_claude_hybrid_tool_def(
            self.screen_size[0], self.screen_size[1], self.tool_name
        )]
        extra_body, max_tokens = self._thinking_config()

        response = self._call_with_retry(
            client, system, tools, betas, extra_body, max_tokens, image_truncation_threshold
        )
        if response is None:
            logger.error("No response from the Anthropic API; failing the step.")
            return "", []

        response_params = _response_to_params(response)
        raw_response = self._extract_raw_response_string(response)
        logger.info("ClaudeHybrid output: %s", raw_response)
        self.messages.append({"role": "assistant", "content": response_params})

        actions, self._pending = self._parse(response_params)

        # The infeasible escape hatch OSWorld's grading depends on. Checked before the
        # empty-action fallback so a text-only refusal is scored as intended.
        if raw_response and "[INFEASIBLE]" in raw_response:
            logger.info("Detected [INFEASIBLE]; ending the episode with FAIL.")
            self._pending = []
            return raw_response, [{"kind": "control", "control": "FAIL"}]

        if not actions:
            # A text-only reply with no tool call means Claude thinks it is finished
            # (or has nothing to do). Parent behaviour: treat it as DONE.
            logger.info("No tool call in the response; treating the episode as DONE.")
            self._pending = []
            return raw_response, [{"kind": "control", "control": "DONE"}]

        logger.info("Parsed: %s", describe(actions))
        return raw_response, actions

    def _thinking_config(self) -> Tuple[Dict[str, Any], int]:
        """Thinking parameters + effective max_tokens for this model's profile.

        Mirrors the parent's branch: 4.7+ uses adaptive thinking with an effort level,
        4.5/4.6 use a fixed extended-thinking budget (and require max_tokens above it).
        """
        if self.runtime_profile.thinking_mode == "adaptive":
            logger.info("Runtime profile: %s; thinking: ADAPTIVE; effort: %s",
                        self.runtime_profile.label, self.effort)
            return (
                {"thinking": {"type": "adaptive"}, "output_config": {"effort": self.effort}},
                max(self.max_tokens, self.runtime_profile.default_max_tokens),
            )
        if self.no_thinking:
            logger.info("Runtime profile: %s; thinking: DISABLED", self.runtime_profile.label)
            return {}, self.max_tokens

        budget_tokens = 2048
        max_tokens = self.max_tokens
        if max_tokens <= budget_tokens:
            max_tokens = budget_tokens + 500
            logger.warning("Regular thinking requires max_tokens > budget_tokens; raising "
                           "max_tokens from %d to %d", self.max_tokens, max_tokens)
        logger.info("Runtime profile: %s; thinking: %s", self.runtime_profile.label,
                    "INTERLEAVED (ISP)" if self.use_isp else "REGULAR")
        return {"thinking": {"type": "enabled", "budget_tokens": budget_tokens}}, max_tokens

    def _create(self, client, request: Dict[str, Any]):
        """Issue one request, streaming when pointed at a local gateway.

        The gateway always answers SSE, and the SDK's non-streaming path mis-parses it:
        ``beta.messages.create`` raises ``AttributeError: 'str' object has no attribute
        'content'``. ``stream()`` rebuilds the same Message, so everything downstream is
        untouched. The public API keeps the non-streaming call.
        """
        if self.base_url:
            with client.beta.messages.stream(**request) as stream:
                return stream.get_final_message()
        return client.beta.messages.create(**request)

    def _call_with_retry(
        self, client, system, tools, betas, extra_body, max_tokens, image_truncation_threshold
    ):
        """The parent's retry ladder: long retry, shrink images on 413, then backup key."""
        from anthropic import (
            Anthropic,
            APIError,
            APIResponseValidationError,
            APIStatusError,
        )

        request = dict(
            max_tokens=max_tokens,
            messages=self.messages,
            model=get_model_name(self.provider, self.model_name),
            system=[system],
            tools=tools,
            extra_body=extra_body,
            **self._get_sampling_params(),
        )
        if betas:
            request["betas"] = betas

        for attempt in range(API_RETRY_TIMES):
            try:
                return self._create(client, request)
            except (APIError, APIStatusError, APIResponseValidationError) as exc:
                detail = str(exc)
                logger.warning("Anthropic API error (attempt %d/%d): %s",
                               attempt + 1, API_RETRY_TIMES, detail[:400])
                if _is_size_error(detail):
                    before = self.only_n_most_recent_images or 2
                    self.only_n_most_recent_images = max(1, before // 2)
                    _maybe_filter_to_n_most_recent_images(
                        self.messages, self.only_n_most_recent_images,
                        min_removal_threshold=image_truncation_threshold,
                    )
                    logger.info("Payload too large; image budget %d -> %d",
                                before, self.only_n_most_recent_images)
                if attempt < API_RETRY_TIMES - 1:
                    time.sleep(API_RETRY_INTERVAL)
                    continue

                backup_key = os.environ.get("ANTHROPIC_API_KEY_BACKUP")
                if not backup_key:
                    logger.error("Retries exhausted and no ANTHROPIC_API_KEY_BACKUP set.")
                    return None
                try:
                    logger.warning("Retrying with the backup API key...")
                    backup_client, _ = self._build_client(api_key=backup_key)
                    return self._create(backup_client, request)
                except Exception as backup_exc:  # noqa: BLE001
                    logger.exception("Backup API call also failed: %s", backup_exc)
                    return None
            except Exception as exc:  # noqa: BLE001 - transport must not kill the episode
                logger.warning("Anthropic transport error (attempt %d/%d): %s: %s",
                               attempt + 1, API_RETRY_TIMES, type(exc).__name__, str(exc)[:300])
                if attempt < API_RETRY_TIMES - 1:
                    time.sleep(API_RETRY_INTERVAL)
                    continue
                return None
        return None

    # -- response parsing ----------------------------------------------------
    def _parse(self, response_params: List[Dict]) -> Tuple[List[Dict], List[Tuple[str, bool]]]:
        """``tool_use`` blocks -> ``(action dicts, pending (id, is_bash) pairs)``.

        Every ``tool_use`` id is recorded as pending even when its input is malformed:
        Claude requires all of them to be answered, and a dropped id is a hard 400 on the
        next turn. A malformed call simply contributes no action.
        """
        actions: List[Dict] = []
        pending: List[Tuple[str, bool]] = []

        for block in response_params:
            if block.get("type") != "tool_use":
                continue
            tool_id = block.get("id")
            params = block.get("input") or {}
            action = typed_action(params.get("action"), params)
            if action is None:
                logger.warning("Unusable tool_use input, skipping the action: %s", params)
                if tool_id:
                    pending.append((tool_id, False))
                continue
            actions.append(action)
            if tool_id:
                pending.append((tool_id, action["kind"] == "bash"))
        return actions, pending

    # -- wiring check --------------------------------------------------------
    def selftest(self) -> bool:
        """Verify the API/gateway answers, that images reach the model, and that the
        custom tool schema is accepted (a rejected schema is a run-ending 400)."""
        client, _ = self._build_client()
        extra_body, max_tokens = self._thinking_config()
        tools = [build_claude_hybrid_tool_def(
            self.screen_size[0], self.screen_size[1], self.tool_name
        )]

        def call(text: str, image: Optional[bytes]) -> str:
            content: List[Dict[str, Any]] = []
            if image is not None:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(image).decode("utf-8"),
                    },
                })
            content.append({"type": "text", "text": text})
            try:
                # Tools are sent on every probe: a gateway that rejects the schema must
                # fail here, not on the first real step of a 300-task run.
                message = self._create(client, dict(
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": content}],
                    model=get_model_name(self.provider, self.model_name),
                    system=[{"type": "text", "text": "Answer briefly and directly."}],
                    tools=tools,
                    extra_body=extra_body,
                    **self._get_sampling_params(),
                ))
            except Exception as exc:  # noqa: BLE001 - the check reports, never raises
                print(f"  API error: {type(exc).__name__}: {str(exc)[:300]}")
                return ""
            return "".join(
                block.text for block in (message.content or [])
                if getattr(block, "type", None) == "text"
            )

        return run_selftest(call, label="claude-hybrid selftest")
