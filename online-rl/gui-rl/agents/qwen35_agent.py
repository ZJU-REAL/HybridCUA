import asyncio
import base64
import logging
import os
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

import config
from agents.utils.qwen_actions import iter_tool_call_params, parse_internal_response
from agents.utils.qwen_history import (
    build_messages,
    ensure_empty_think_prefix,
    previous_actions_text,
    to_slime_image_parts,
    update_folding_state,
)
from agents.utils.qwen_prompts import (
    build_instruction_prompt,
    build_internal_system_prompt,
    build_internal_tools_def,
)
from agents.utils.qwen_vl_utils import smart_resize
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.processing_utils import process_vision_info as slime_process_vision_info

logger = None


# ===== [weight-update abort retry] (aligned with qwen3vl_agent) =====
# update_weights aborts in-flight sglang requests with HTTP 200 +
# finish_reason.type="abort" (no status_code). Resending is safe: the step's
# action hasn't executed yet, so env state is unchanged; sglang queues the
# resend until continue_generation, then generates with fresh weights. Without
# this, aborted trajectories get silently dropped during every weight update.
_ABORT_MESSAGES = frozenset({"Aborted", "Abort in waiting queue"})


def _is_weight_update_abort(finish_reason: dict) -> bool:
    """True iff this abort was triggered by a weight update (not a real error)."""
    if finish_reason.get("type") != "abort":
        return False
    if finish_reason.get("status_code") is not None:
        return False
    return (finish_reason.get("message") or "") in _ABORT_MESSAGES


def encode_image(image_content: bytes) -> str:
    return base64.b64encode(image_content).decode("utf-8")


# Per-image pixel budgets, aligned with the SFT data (sft_c_hybrid_v5): the CURRENT
# (latest) frame is kept large while HISTORY frames are downscaled, which shrinks the
# multimodal prompt (and thus training-forward activations). Overridable via env.
CURRENT_IMAGE_MAX_PIXELS = int(os.getenv("GUI_CURRENT_IMAGE_MAX_PIXELS", 2088960))
HISTORY_IMAGE_MAX_PIXELS = int(os.getenv("GUI_HISTORY_IMAGE_MAX_PIXELS", 548800))
IMAGE_MIN_PIXELS = int(os.getenv("GUI_IMAGE_MIN_PIXELS", 40768))


def process_image(
    image_bytes: bytes,
    max_pixels: int = CURRENT_IMAGE_MAX_PIXELS,
    min_pixels: int = IMAGE_MIN_PIXELS,
) -> str:
    """Resize + re-encode screenshot and return base64 PNG.

    Defaults to the CURRENT-frame budget; pass ``max_pixels=HISTORY_IMAGE_MAX_PIXELS``
    for history frames.
    """
    image = Image.open(BytesIO(image_bytes))
    width, height = image.size

    resized_height, resized_width = smart_resize(
        height=height,
        width=width,
        factor=32,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    image = image.resize((resized_width, resized_height))

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    processed_bytes = buffer.getvalue()

    return base64.b64encode(processed_bytes).decode("utf-8")


class Qwen35VLAgentLocal:
    """
    Lightweight Qwen3.5-VL agent (local sglang variant).

    Advertises the INTERNAL ``computer_use`` surface (19 actions) and organizes context
    exactly like the eval-side ``mm_agents.qwen.QwenAgent`` — the agent that
    ``scripts/bash/osworld/run_qwen_qwen35_{9b,27b}_sharded.sh`` scores these checkpoints
    with. Prompt, tool schema, folding and message layout are byte-identical to it; see
    ``agents/utils/qwen_{prompts,history,actions}.py`` and
    ``tests/test_gui_only_eval_parity.py``.

    Characteristics:
    - XML tool-call output format.
    - Turn-window truncation by `history_n`.
    - Old screenshot folding to `collapse_text` by `image_max` / `fold_size`.
    """

    COLLAPSED_SCREENSHOT_TEXT = "This screenshot has been collapsed."

    def __init__(
        self,
        platform: str = "ubuntu",
        model: str = "qwen35-vl",
        max_steps: int = 100,
        max_image_history_length: int = 3,
        max_tokens: int = 32768,
        top_p: float = 0.9,
        temperature: float = 0.0,
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        coordinate_type: str = "relative",
        example_result_dir: Optional[str] = None,
        add_thought_prefix: bool = False,
        # Eval runs history_n=50 / image_max=5 / fold_size=1; training uses a tighter
        # image budget (3 live frames) to bound the multimodal sequence, and a 30-turn
        # text window. Qwen35HybridCuaAgentLocal always passes these explicitly, so it
        # is unaffected by these defaults.
        history_n: int = 30,
        fold_size: int = 1,
        collapse_text: Optional[str] = None,
        **_unused_kwargs: Any,
    ):
        self.platform = platform
        self.model = model
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.image_max = max(1, int(max_image_history_length))
        self.coordinate_type = coordinate_type
        self.example_result_dir = example_result_dir or os.getcwd()
        self.history_n = int(history_n)
        self.fold_size = max(1, int(fold_size))
        self.collapse_text = collapse_text or self.COLLAPSED_SCREENSHOT_TEXT

        assert action_space == "pyautogui", "qwen35vl_agent only supports pyautogui action space"
        assert observation_type == "screenshot", "qwen35vl_agent only supports screenshot observations"

        self.actions: List[str] = []
        self.responses: List[str] = []
        self.screenshots: List[str] = []
        self.folded_prefix_k = 0

    def reset(self, _logger=None):
        global logger
        logger = _logger if _logger is not None else logging.getLogger("desktopenv.qwen35_agent_local")

        self.actions = []
        self.responses = []
        self.screenshots = []
        self.folded_prefix_k = 0

    def get_tool_spec(
        self,
        processed_width: Optional[int] = None,
        processed_height: Optional[int] = None,
    ) -> Dict[str, Any]:
        # `processed_*` only matter for coordinate_type="absolute"; under the default
        # "relative" the schema advertises a fixed 1000x1000 space.
        return build_internal_tools_def(processed_width, processed_height, self.coordinate_type)

    def get_system_prompt(
        self,
        processed_width: Optional[int] = None,
        processed_height: Optional[int] = None,
    ) -> str:
        tools_def = self.get_tool_spec(processed_width=processed_width, processed_height=processed_height)
        return build_internal_system_prompt(tools_def, self.collapse_text)

    def build_train_system_message(self) -> Dict[str, Any]:
        return {"role": "system", "content": self.get_system_prompt()}

    def build_instruction_prompt(self, instruction: str, previous_actions_str: str) -> str:
        # Takes the ALREADY-JOINED text (as upstream does) so previous_actions_text stays
        # the single source of truth for the "Step N: ..." / "None" formatting.
        return build_instruction_prompt(instruction, previous_actions_str)

    @staticmethod
    def _extract_multimodal(messages: List[Dict[str, Any]], processor: Any) -> Dict[str, Any]:
        if not processor:
            return {}
        return slime_process_vision_info(messages, processor) or {}

    def build_policy_messages(self, instruction: str, obs: Dict) -> Dict[str, Any]:
        """Render this turn's prompt, mirroring the eval-side QwenAgent.predict.

        Two independent budgets, both from upstream:
          * ``history_n`` bounds how many TURNS survive; the ones dropped are summarized
            as one-liners in ``Previous actions``.
          * ``image_max`` / ``fold_size`` bound how many of those turns keep a real
            SCREENSHOT. ``folded_prefix_k`` is a monotone ratchet, so older turns keep
            their text and assistant reply but show ``collapse_text`` instead of an image.
        """
        step_index = len(self.actions)
        screenshot_bytes: bytes = obs["screenshot"]

        img0 = Image.open(BytesIO(screenshot_bytes))
        original_width, original_height = img0.size

        processed_image_b64 = process_image(screenshot_bytes)
        processed_img = Image.open(BytesIO(base64.b64decode(processed_image_b64)))
        processed_width, processed_height = processed_img.size

        all_screenshots = list(self.screenshots) + [processed_image_b64]
        total_steps = len(all_screenshots)

        # Persists across turns (init/reset to 0) — folding never rewinds.
        self.folded_prefix_k = update_folding_state(
            total_steps, self.folded_prefix_k, self.image_max, self.fold_size
        )
        start_step = max(1, total_steps - self.history_n)

        system_prompt = self.get_system_prompt(
            processed_width=processed_width, processed_height=processed_height
        )
        tool_spec = self.get_tool_spec(
            processed_width=processed_width, processed_height=processed_height
        )
        instruction_prompt = self.build_instruction_prompt(
            instruction, previous_actions_text(self.actions, start_step)
        )

        messages = to_slime_image_parts(build_messages(
            system_prompt=system_prompt,
            instruction_prompt=instruction_prompt,
            screenshots=all_screenshots,
            responses=self.responses,
            start_step=start_step,
            total_steps=total_steps,
            folded_prefix_k=self.folded_prefix_k,
            collapse_text=self.collapse_text,
            # Eval applies this to every replayed assistant turn unconditionally
            # (main.py:262) — it is NOT gated on enable_thinking. Only the prompt is
            # affected: trajectory.py:618 masks historical assistant turns out of the loss.
            response_transform=ensure_empty_think_prefix,
        ))

        return {
            "messages": messages,
            "step_index": step_index,
            "processed_image_b64": processed_image_b64,
            "original_width": original_width,
            "original_height": original_height,
            "processed_width": processed_width,
            "processed_height": processed_height,
            "system_prompt": system_prompt,
            "tool_spec": tool_spec,
        }

    async def generate_with_sglang(
        self,
        *,
        args: Any,
        state: GenerateState,
        messages: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
        sampling_seed: int | None = None,
        tool_spec: Dict[str, Any] | None = None,
        timings: Any = None,  # accepted for API parity; not yet sub-profiled here
    ) -> Tuple[str, str, str | None]:
        tokenizer = state.tokenizer
        processor = state.processor
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=[tool_spec or self.get_tool_spec()],
        )
        current_sampling_params = dict(sampling_params)
        if sampling_seed is not None:
            current_sampling_params["sampling_seed"] = int(sampling_seed)

        payload: Dict[str, Any] = {"sampling_params": current_sampling_params, "return_logprob": True}
        image_data: List[str] = []
        if processor:
            multimodal_inputs = self._extract_multimodal(messages, processor)
            images = multimodal_inputs.get("images") or []
            if images:
                image_data = [encode_image_for_rollout_engine(img) for img in images]

        if image_data:
            payload["text"] = prompt_text
            payload["image_data"] = image_data
        else:
            payload["input_ids"] = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        output = await self._post_with_abort_retry(url, payload, args=args)
        mi = output["meta_info"]
        finish_type = mi["finish_reason"]["type"]
        if "output_token_logprobs" in mi:
            output_tokens = [item[1] for item in mi["output_token_logprobs"]]
            response = tokenizer.decode(output_tokens).removesuffix(tokenizer.eos_token or "")
        else:
            response = output.get("text", "")
        # weight_version: keep the (response, finish_type, weight_version) contract in
        # sync with Qwen3VLAgentLocal so the trajectory loop unpacks uniformly.
        return response, finish_type, mi.get("weight_version")

    async def _post_with_abort_retry(self, url: str, payload: Dict[str, Any], *, args: Any) -> Dict[str, Any]:
        """POST /generate, retrying on weight-update abort (aligned with qwen3vl_agent).

        A resend is safe: at abort time the step's action has not executed, so env
        state is unchanged; sglang queues the resend until continue_generation, then
        generates with fresh weights. Real failures (status_code 4xx/5xx) are NOT
        retried and fall through to the caller.
        """
        max_retry = getattr(args, "gui_abort_retry_max", config.abort_retry_max())
        backoff = getattr(args, "gui_abort_retry_backoff", config.abort_retry_backoff())
        output: Dict[str, Any] = {}
        for attempt in range(max_retry + 1):
            output = await post(url, payload)
            finish_reason = output.get("meta_info", {}).get("finish_reason", {})
            if finish_reason.get("type") != "abort":
                return output
            if not _is_weight_update_abort(finish_reason) or attempt >= max_retry:
                return output
            await asyncio.sleep(backoff)
            if logger:
                logger.info("sglang abort (weight update), retry attempt=%d", attempt + 1)
        return output

    @staticmethod
    def _align_loss_mask_multimodal(
        tokenizer, input_ids: List[int], loss_mask_text: List[int]
    ) -> List[int]:
        """Align a text-only loss mask to multimodal input_ids by skipping vision tokens.

        Aligned with qwen3vl_agent: the previous approach (垫 diff 个 0 在开头,
        get_loss_mask_with_multimodal_alignment) drifts on multi-turn mixed
        text+image turns because image tokens are NOT all at the front. Here we
        walk input_ids: vision tokens -> mask 0; every other token consumes the
        next text-mask entry in order.
        """
        vision_token_ids = set()
        for tok in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>"):
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != tokenizer.unk_token_id:
                vision_token_ids.add(tid)

        loss_mask = []
        text_idx = 0
        for token_id in input_ids:
            if token_id in vision_token_ids:
                loss_mask.append(0)
            else:
                loss_mask.append(loss_mask_text[text_idx] if text_idx < len(loss_mask_text) else 0)
                text_idx += 1
        return loss_mask

    def build_train_data(
        self,
        *,
        args: Any,
        state: GenerateState,
        train_messages: List[Dict[str, Any]],
        tool_spec: Dict[str, Any] | None = None,
    ) -> Tuple[List[int], List[int], Dict[str, Any] | None]:
        tokenizer = state.tokenizer
        processor = state.processor

        # Qwen3.5's chat template unconditionally requires at least one
        # non-tool_response user message (enforced at template line 79).
        # In abort/fallback scenarios the rollout entrypoint passes system-only
        # messages. Pad with dummy turns whose step_loss_mask=0 so they
        # never contribute to training loss.
        if not any(m.get("role") == "user" for m in train_messages):
            train_messages = list(train_messages) + [
                {"role": "user", "content": "N/A"},
                {"role": "assistant", "content": "N/A", "step_loss_mask": 0},
            ]

        text_prompt = tokenizer.apply_chat_template(
            train_messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=[tool_spec or self.get_tool_spec()],
        )
        if processor:
            multimodal_inputs = self._extract_multimodal(train_messages, processor)
            # mm_token_type_ids 是 per-token [1, seq_len]，而 mm_train 每个 key 都会被
            # data.py 沿 dim0 拼接（只对 per-image 张量成立）-> thd 多样本 microbatch 会炸。
            # 模型未使用它（Qwen3VLModel.forward 里直接 del）。
            kwargs: Dict[str, Any] = {
                "text": [text_prompt],
                "return_tensors": "pt",
                "return_mm_token_type_ids": False,
                **multimodal_inputs,
            }
            proc_out = processor(**kwargs)
            input_ids = proc_out["input_ids"][0].tolist()

            mm_train = {
                k: (v.cpu().numpy() if hasattr(v, "cpu") else v)
                for k, v in proc_out.items()
                if k not in ["input_ids", "attention_mask", "mm_token_type_ids"]
            } or None
        else:
            input_ids = tokenizer(text_prompt, add_special_tokens=False)["input_ids"]
            mm_train = None

        mask_generator = MultiTurnLossMaskGenerator(
            tokenizer, tokenizer_type=getattr(args, "loss_mask_type", "qwen3")
        )
        if processor:
            # Multimodal: strip images -> text-only mask, then align to the
            # vision-expanded input_ids (aligned with qwen3vl_agent). The old
            # get_loss_mask_with_multimodal_alignment (pad diff zeros at front)
            # drifts on multi-turn mixed text+image turns.
            text_messages = []
            for msg in train_messages:
                if isinstance(msg.get("content"), list):
                    text_parts = [
                        item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
                        else item if isinstance(item, str) else ""
                        for item in msg["content"]
                    ]
                    text_messages.append({**msg, "content": " ".join(p for p in text_parts if p)})
                else:
                    text_messages.append(msg)
            _, loss_mask_text = mask_generator.get_loss_mask(text_messages, tools=[tool_spec or self.get_tool_spec()])
            loss_mask = self._align_loss_mask_multimodal(tokenizer, input_ids, loss_mask_text)
        else:
            _, loss_mask = mask_generator.get_loss_mask_with_multimodal_alignment(
                train_messages, input_ids, tools=[tool_spec or self.get_tool_spec()]
            )
        return input_ids, loss_mask, mm_train

    def record_policy_turn(self, *, action_text: str, response: str, screenshot_bytes: bytes) -> None:
        self.actions.append(action_text)
        self.responses.append(response)
        # This frame is now history -> store at the smaller history budget. The current
        # frame is (re)processed at the full budget in build_policy_messages.
        self.screenshots.append(process_image(screenshot_bytes, max_pixels=HISTORY_IMAGE_MAX_PIXELS))

    def parse_response(
        self,
        response: str,
        original_width: int,
        original_height: int,
        processed_width: Optional[int] = None,
        processed_height: Optional[int] = None,
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """XML tool call -> (action text, pyautogui/control actions, diagnostics).

        Thin adapter over the verbatim eval-side mapping so the two cannot drift; the
        only addition is the ``other`` diagnostics the rollout records to traj.jsonl.

        Two control-token behaviours are inherited deliberately (both are what eval does):
        ``terminate status=failure`` and an infeasible-sounding ``call_user`` yield
        ``"FAIL"`` (-> Sample.Status.FAILED at trajectory.py:489), while a response with
        no tool call at all falls back to ``"DONE"``/``"FAIL"`` rather than an empty list.
        """
        natural_action, actions = parse_internal_response(
            response or "",
            coordinate_type=self.coordinate_type,
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
        )
        return natural_action, actions, {
            "raw_response": response,
            # Mirrors the old semantics: only calls that carried an `action` were recorded.
            "tool_calls": [p for p in iter_tool_call_params(response or "") if p.get("action")],
            "action": natural_action,
            "code": actions,
        }
