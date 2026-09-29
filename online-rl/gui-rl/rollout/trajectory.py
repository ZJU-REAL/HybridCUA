"""One GUI rollout trajectory: env lifecycle + turn loop + per-step sample build.

Two responsibilities, previously split across episode.py + trajectory.py, now
unified here:

1. :class:`Trajectory` drives one rollout's remote-env lifecycle (acquire ->
   reset -> per-step act -> evaluate -> close) and the multi-turn policy loop
   (query policy, parse action, step env), producing a :class:`TrajectoryResult`.
2. :func:`build_dynamic_history_samples` turns the episode's per-step snapshots
   into training :class:`Sample` objects (dynamic-history GRPO): each snapshot
   becomes one sample whose loss mask covers only that step's response suffix.

PRM (process reward) is delegated to :class:`reward.prm_hook.PrmHook`.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import logging
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

import config
from env_client import GuiEnvClient
from reward.prm_hook import PrmHook
from reward.reward_func import cli_aware_enabled
from utils.timing import Timings
from utils.utils import save_image

# ===== DEBUG: real env occupancy =====
# All worker processes share ONE json file {pid: [leases]}; each process updates
# only its own pid key. Concurrent read-modify-write is serialized with an
# fcntl.flock (exclusive), so no lost updates / corruption. Sum all leases across
# pids = true number of envs in use. See count_env.sh.
import fcntl as _fcntl
import os as _os
import time as _time

_HELD_LEASES: set[str] = set()
_INFLIGHT_DIR: Path | None = None


def _dump_inflight(action: str = "", task: str = "") -> None:
    """Track in-flight envs across all workers, guarded by one exclusive flock:
    - inflight.json  {pid:[leases]}  : current snapshot (overwritten).
    - counter.txt    timeline log    : APPENDED one line per change:
        "<ts>  total=<N>  <action> <task>  pid=<pid>"
      where <task> = "<example_id>#<sample_index>" (the task that grabbed/released an env).
    """
    global _INFLIGHT_DIR
    try:
        if _INFLIGHT_DIR is None:
            _INFLIGHT_DIR = Path(_os.getenv("GUI_RESULT_DIR", "/tmp")) / "_env_inflight"
            _INFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
        pid = str(_os.getpid())
        with open(_INFLIGHT_DIR / "inflight.json", "a+", encoding="utf-8") as fh:
            _fcntl.flock(fh, _fcntl.LOCK_EX)
            try:
                fh.seek(0)
                raw = fh.read()
                data = json.loads(raw) if raw.strip() else {}
                if _HELD_LEASES:
                    data[pid] = sorted(_HELD_LEASES)
                else:
                    data.pop(pid, None)  # this process holds nothing -> drop its key
                fh.seek(0)
                fh.truncate()
                fh.write(json.dumps(data))
                fh.flush()
                # APPEND a timeline line to counter.txt (total + this change)
                total = sum(len(v) for v in data.values())
                ts = _time.strftime("%H:%M:%S")
                with open(_INFLIGHT_DIR / "counter.txt", "a", encoding="utf-8") as cf:
                    cf.write(f"{ts}  total={total}  {action} {task}  pid={pid}\n")
            finally:
                _fcntl.flock(fh, _fcntl.LOCK_UN)
    except Exception:
        pass  # debug-only, never break rollout

logger = logging.getLogger(__name__)


def _response_suffix_mask(token_ids: list[int], loss_mask: list[int]) -> tuple[int, int, list[int]] | None:
    """Convert a full-sequence loss mask to a response-suffix mask.

    Training expects the mask to start at the first trainable (response) token.
    Returns ``(response_start, response_length, suffix_mask)`` or ``None`` when
    the snapshot has no trainable tokens (skip it).
    """
    active_positions = [i for i in range(len(loss_mask)) if i < len(token_ids) and int(loss_mask[i]) == 1]
    if not active_positions:
        return None
    response_start = active_positions[0]
    response_length = len(token_ids) - response_start
    if response_length <= 0:
        return None
    suffix_mask = [int(loss_mask[i]) if i < len(loss_mask) else 0 for i in range(response_start, len(token_ids))]
    return response_start, response_length, suffix_mask


#: In-VM execution-layer failures. run_python_script *returns* these rather than
#: raising, so the step still reaches the gradient -- hence the exemption below.
INFRA_ERROR_MARKERS = ("retry limit reached", "failed to execute command")


def step_kind(action: Any) -> str | None:
    """'gui' (pyautogui heredoc) / 'cli' (plain shell) / None (control token).

    Both share one action=bash surface, so only command text tells them apart.
    """
    if not (isinstance(action, dict) and action.get("action_type") in ("bash", "cli", "shell")):
        return None
    return "gui" if "pyautogui" in str(action.get("command") or "") else "cli"


def exec_failed(exec_result: dict[str, Any]) -> bool:
    """Shell-level execution failure -> e_t = -1 (eq. 5). See docs for rationale.

    1. In-VM infra failure -> exempt (not the model's fault).
    2. No return_code but status=error -> timeout, penalized.
    3. rc != 0 -> failure, except rc==1 with empty stderr (grep/diff/test
       return 1 as a semantic result, not an error).
    """
    status = str(exec_result.get("status") or "").lower()
    # strip() is load-bearing: the wrapper always prints stderr, so "no stderr"
    # arrives as "\n" and bool("\n") would defeat layer 3.
    stderr = str(exec_result.get("error") or "").strip()

    if any(marker in stderr.lower() for marker in INFRA_ERROR_MARKERS):
        return False

    rc = exec_result.get("return_code")
    if rc is not None:
        with contextlib.suppress(TypeError, ValueError):
            rc = int(rc)
            return False if rc == 0 else (rc != 1 or bool(stderr))
    return status == "error"


def _build_child_metadata(
    base_sample: Sample,
    step_idx: int,
    outcome_reward: float,
    prm_score_by_step: dict[int, float] | None,
    exec_penalty_by_step: dict[int, float] | None = None,
) -> dict[str, Any]:
    """Minimal, explicit metadata for one dynamic-history child sample."""
    meta = copy.deepcopy(base_sample.metadata or {})
    meta["dynamic_step_index"] = step_idx
    meta["dynamic_outcome_reward"] = float(outcome_reward)
    # This step's own e_t. The child's loss mask covers exactly this action's
    # response tokens, so a scalar here == eq. (7)'s per-token term.
    meta["gui_exec_penalty"] = (exec_penalty_by_step or {}).get(step_idx, 0.0)
    if prm_score_by_step is not None:
        # rollout infers the span from loss_mask, so one score/index per child.
        meta["step_wise"] = {
            "step_scores": [float(prm_score_by_step.get(step_idx, 0.0))],
            "step_indices": [int(step_idx)],
        }
    return meta


def build_dynamic_history_samples(
    args: Any,
    state: GenerateState,
    agent: Any,
    base_sample: Sample,
    step_snapshots: list[dict[str, Any]],
    outcome_reward: float,
    prm_score_by_step: dict[int, float] | None = None,
    exec_penalty_by_step: dict[int, float] | None = None,
) -> list[Sample]:
    """Turn each step snapshot into one training Sample (dynamic-history GRPO).

    Each child carries the full token sequence for that step's context+response,
    a response-suffix loss mask, and the (outcome or PRM) reward. Snapshots with
    no trainable response tokens are skipped.
    """
    reward_key = getattr(args, "reward_key", None) or "score"
    prm_enabled = getattr(args, "prm_enable", False)
    dynamic_samples: list[Sample] = []

    for snapshot in step_snapshots:
        step_idx = int(snapshot["step_idx"])
        messages = snapshot["train_messages"]
        response_text = snapshot["response_text"]
        tool_spec = snapshot.get("tool_spec")

        token_ids, loss_mask, mm_train = agent.build_train_data(
            args=args,
            state=state,
            train_messages=messages,
            tool_spec=tool_spec,
        )

        suffix = _response_suffix_mask(token_ids, loss_mask)
        if suffix is None:
            continue
        _response_start, response_length, child_loss_mask = suffix

        child_reward = {"score": float(outcome_reward), reward_key: float(outcome_reward)}
        # When PRM or a CLI-aware term is on, reward is composed in reward_func;
        # leave it unset so generate_and_rm dispatches to the RM.
        child_reward_for_sample = None if (prm_enabled or cli_aware_enabled(args)) else child_reward

        child = Sample(
            group_index=base_sample.group_index,
            index=base_sample.index,
            prompt=base_sample.prompt,
            tokens=token_ids,
            multimodal_inputs=base_sample.multimodal_inputs,
            multimodal_train_inputs=mm_train,
            response=response_text,
            response_length=response_length,
            label=base_sample.label,
            reward=child_reward_for_sample,
            loss_mask=child_loss_mask,
            weight_versions=[v] if (v := snapshot.get("weight_version")) else [],
            rollout_log_probs=None,
            rollout_routed_experts=None,
            remove_sample=base_sample.remove_sample,
            status=base_sample.status,
            metadata=_build_child_metadata(
                base_sample, step_idx, outcome_reward, prm_score_by_step, exec_penalty_by_step
            ),
            generate_function_path=base_sample.generate_function_path,
            train_metadata=base_sample.train_metadata,
            non_generation_time=base_sample.non_generation_time,
            spec_info=base_sample.spec_info,
            prefix_cache_info=base_sample.prefix_cache_info,
        )
        dynamic_samples.append(child)

    return dynamic_samples


@dataclass
class TrajectoryResult:
    """Everything ``generate`` needs to build training samples from one trajectory."""

    status: Sample.Status
    eval_score: float
    # Per-step snapshots consumed by build_dynamic_history_samples.
    step_snapshots: list[dict[str, Any]] = field(default_factory=list)
    assistant_responses: list[str] = field(default_factory=list)
    # Messages/tool_spec to build the single-sample (non-dynamic) loss target.
    train_messages_for_loss: list[dict[str, Any]] = field(default_factory=list)
    tool_spec_for_loss: dict[str, Any] | None = None
    prm: PrmHook = field(default_factory=PrmHook.disabled)
    result_dir: Path | None = None
    # Set when the trajectory failed; recorded onto the sample metadata by caller.
    error_stage: str | None = None
    error_message: str | None = None
    # CLI-aware reward signals (paper eq. 6/7).
    used_cli: bool = False
    exec_penalty_by_step: dict[int, float] = field(default_factory=dict)


class Trajectory:
    """Drive one rollout: hold env/agent/state and execute the turn loop."""

    def __init__(
        self,
        *,
        args: Any,
        agent: Any,
        env_client: GuiEnvClient,
        state: GenerateState,
        ep_cfg: Any,
        sampling_params: dict[str, Any],
        sample: Sample,
        instruction: str,
        task_config: dict[str, Any] | None,
        result_dir: Path,
        evaluation: bool = False,
    ) -> None:
        self.args = args
        self.agent = agent
        self.env_client = env_client
        self.state = state
        self.cfg = ep_cfg
        self.sampling_params = sampling_params
        self.sample = sample
        self.instruction = instruction
        self.task_config = task_config
        self.result_dir = result_dir
        self.evaluation = evaluation
        self.traj_path = result_dir / "traj.jsonl"
        self.domain = str((sample.metadata or {}).get("domain", ""))
        self.example_id = str((sample.metadata or {}).get("example_id", ""))

    # --- env lifecycle ------------------------------------------------------------

    async def _acquire(self) -> str:
        """Acquire a lease with bounded retry (env capacity is the bottleneck).

        Returns the lease_id. ``task_config`` is applied later at reset().
        """
        last_error: Exception | None = None
        for attempt in range(self.cfg.allocate_retries):
            try:
                episode_id = f"{self.domain}:{self.example_id}:{uuid.uuid4().hex[:8]}"
                # task_type buckets the session on the env-server dashboard. Derive
                # it from this rollout's evaluation flag — NOT config.env_mode(),
                # which returns "train"/"eval", not the server's "training"/
                # "evaluation" vocabulary.
                lease = await self.env_client.allocate(
                    episode_id=episode_id,
                    user_id=config.user_id(),
                    task_type="evaluation" if self.evaluation else "training",
                    job_id=config.job_id() or None,
                    runtime=(self.sample.metadata or {}).get("runtime"),
                )
                _HELD_LEASES.add(lease["lease_id"])  # DEBUG: env acquired
                _dump_inflight("acquire", f"{self.example_id}#{self.sample.index}")
                return lease["lease_id"]
            except Exception as e:  # external service call
                last_error = e
                if attempt < self.cfg.allocate_retries - 1:
                    logger.warning(
                        "GUI acquire failed (%d/%d), retry in %.1fs: %s",
                        attempt + 1,
                        self.cfg.allocate_retries,
                        self.cfg.allocate_backoff_seconds,
                        e,
                    )
                    await asyncio.sleep(self.cfg.allocate_backoff_seconds)
        raise RuntimeError(
            f"GUI env acquire failed after {self.cfg.allocate_retries} retries: {last_error}"
        )

    # --- run ----------------------------------------------------------------------

    async def run(self) -> TrajectoryResult:
        res = TrajectoryResult(status=Sample.Status.COMPLETED, eval_score=0.0, result_dir=self.result_dir)
        res.train_messages_for_loss = [self.agent.build_train_system_message()]
        self.timings = Timings(config.gui_profile())
        trace_records: list[dict[str, Any]] = []
        error_stage = "init"
        lease_id: str | None = None

        try:
            error_stage = "acquire"
            logger.info(
                "GUI rollout start sample=%s group=%s domain=%s example=%s",
                self.sample.index, self.sample.group_index, self.domain, self.example_id,
            )
            async with self.timings.aspan("acquire"):
                lease_id = await self._acquire()

            error_stage = "reset"
            async with self.timings.aspan("reset"):
                obs = await self.env_client.reset(lease_id=lease_id, task_config=self.task_config)
                if self.cfg.wait_after_reset > 0:
                    await asyncio.sleep(self.cfg.wait_after_reset)
                    obs = await self.env_client.get_obs(lease_id)
            save_image(obs["screenshot"], self.result_dir / "step_0.png")

            res.prm = PrmHook.create(self.args, self.state, str(self.result_dir))

            async with self.timings.aspan("turn_loop"):
                res.status, obs = await self._turn_loop(lease_id, obs, res, trace_records)

            async with self.timings.aspan("prm_collect"):
                await res.prm.collect()

            if res.status != Sample.Status.COMPLETED:
                try:
                    await self.env_client.step(lease_id=lease_id, action="FAIL", sleep_after_execution=0)
                except Exception:
                    logger.debug("Failed to send terminal FAIL action", exc_info=True)

            error_stage = "evaluate"
            async with self.timings.aspan("evaluate"):
                res.eval_score = await self.env_client.evaluate(lease_id=lease_id)
            logger.info(
                "GUI rollout end sample=%s status=%s score=%.4f",
                self.sample.index, res.status.value, res.eval_score,
            )
            self._write_results(res, trace_records)
        except Exception as e:
            res.status = Sample.Status.ABORTED
            res.error_stage = error_stage
            res.error_message = str(e)[:500]
            tb = traceback.format_exc()
            try:
                with open(self.traj_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"Error": str(e), "Traceback": tb}, ensure_ascii=False) + "\n")
            except Exception:
                pass
            print(
                f"[GUI_ROLLOUT_ERROR] sample_index={self.sample.index} "
                f"domain={self.domain} example_id={self.example_id}\n{tb}",
                file=sys.stderr, flush=True,
            )
            logger.exception("GUI rollout failed for sample %s", self.sample.index)
        finally:
            if lease_id is not None:
                try:
                    async with self.timings.aspan("close"):
                        await self.env_client.close(lease_id=lease_id)
                        _HELD_LEASES.discard(lease_id)  # DEBUG: env released
                        _dump_inflight("release", f"{self.example_id}#{self.sample.index}")
                except Exception:
                    logger.exception("Failed to close env lease: %s", lease_id)

        if self.timings.enabled:
            try:
                with open(self.result_dir / "timings.json", "w", encoding="utf-8") as f:
                    json.dump(self.timings.summary(), f, ensure_ascii=False, indent=2)
            except Exception:
                logger.exception("Failed to write timings.json")

        return res

    # --- turn loop ----------------------------------------------------------------

    async def _turn_loop(
        self,
        lease_id: str,
        obs: dict[str, Any],
        res: TrajectoryResult,
        trace_records: list[dict[str, Any]],
    ) -> tuple[Sample.Status, dict[str, Any]]:
        """Run up to ``max_steps`` policy turns. Returns (final_status, last_obs)."""
        for step_idx in range(self.cfg.max_steps):
            # Per-step phase breakdown (C1-C5). `measure`/`ameasure` also fold
            # each delta into self.timings totals; no-op when profiling is off.
            st: dict[str, float] = {}

            # NOTE: heartbeat disabled — profiling showed it cost ~11.6s/step
            # (~23% of turn_loop), an unexpectedly heavy cost for a keepalive
            # ping. Skipping it; re-enable if the env server starts expiring
            # leases mid-trajectory.
            # async with self.timings.ameasure("heartbeat", st):
            #     await self.env_client.heartbeat(lease_id)
            st["heartbeat"] = 0.0
            with self.timings.measure("build_policy_messages", st):
                parse_ctx = self.agent.build_policy_messages(instruction=self.instruction, obs=obs)
            policy_messages = parse_ctx["messages"]
            tool_spec = parse_ctx.get("tool_spec")
            # Track latest context as the fallback single-sample loss target.
            res.train_messages_for_loss = policy_messages
            res.tool_spec_for_loss = tool_spec

            async with self.timings.ameasure("sglang_generate", st):
                response, finish_type, step_version = await self.agent.generate_with_sglang(
                    args=self.args,
                    state=self.state,
                    messages=policy_messages,
                    sampling_params=self.sampling_params,
                    sampling_seed=((int(self.sample.index or 0) + 1) * 1000003 + step_idx * 9973),
                    tool_spec=tool_spec,
                    timings=self.timings,
                )
            self._log_step(step_idx, finish_type, response)

            if finish_type == "abort":
                return Sample.Status.ABORTED, obs  # aborted step never snapshots -> never stamped

            self._record_snapshot(res, step_idx, policy_messages, tool_spec, response, step_version)

            with self.timings.measure("parse_response", st):
                natural_action, actions, info_dict = self.agent.parse_response(
                    response=response,
                    original_width=int(parse_ctx["original_width"]),
                    original_height=int(parse_ctx["original_height"]),
                    processed_width=int(parse_ctx["processed_width"]),
                    processed_height=int(parse_ctx["processed_height"]),
                )
            self.agent.record_policy_turn(
                action_text=natural_action or "Execute action",
                response=response,
                screenshot_bytes=obs["screenshot"],
            )

            if not actions or actions[0] == "":
                return Sample.Status.FAILED, obs
            if str(actions[0]).upper() == "FAIL":
                return Sample.Status.FAILED, obs

            async with self.timings.ameasure("env_step", st):
                obs, done, step_executed, cli_text = await self._execute_actions(
                    lease_id, step_idx, actions, response, info_dict, res
                )

            # Thread this turn's command stdout/stderr back to the agent so the next
            # prompt renders it (bash-surface / c_gui hybrid agent). Guarded by getattr:
            # agents without a CLI channel (qwen3vl/qwen35) are unaffected.
            if cli_text:
                record_exec = getattr(self.agent, "record_step_exec_results", None)
                if record_exec is not None:
                    record_exec(cli_text)

            if self.timings.enabled:
                step_record = {
                    "sample_index": int(self.sample.index) if self.sample.index is not None else -1,
                    "group_index": int(self.sample.group_index) if self.sample.group_index is not None else -1,
                    "step_idx": step_idx,
                    "heartbeat": st["heartbeat"],
                    "build_msg": st["build_policy_messages"],
                    "sglang": st["sglang_generate"],
                    "parse": st["parse_response"],
                    "env_step": st["env_step"],
                    "step_time": round(sum(st.values()), 4),
                }
                self.timings.steps.append(step_record)
                with open(self.result_dir / "step_times.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(step_record, ensure_ascii=False) + "\n")

            trace_records.append({
                "messages": policy_messages,
                "response": response,
                "actions": actions,
                "step_idx": step_idx,
                "step_executed": step_executed,
            })

            # Dispatch PRM judging right after the action executed (non-blocking).
            if step_executed:
                res.prm.submit_step(
                    self.args,
                    instruction=self.instruction,
                    actions_history=list(self.agent.actions),
                    policy_response=response,
                    step_index=step_idx,
                )

            if done:
                return Sample.Status.COMPLETED, obs

        return Sample.Status.TRUNCATED, obs

    async def _execute_actions(
        self,
        lease_id: str,
        step_idx: int,
        actions: list[Any],
        response: str,
        info_dict: dict[str, Any],
        res: TrajectoryResult,
    ) -> tuple[dict[str, Any], bool, bool, str]:
        """Execute each action of one turn; returns (last_obs, done, step_executed, cli_text).

        ``cli_text`` is the concatenated stdout/stderr of any bash (TOOL/cli) actions this
        turn (empty for pure GUI/control turns), used to feed the next prompt's CLI output.
        """
        obs: dict[str, Any] = {}
        done = False
        step_executed = False
        cli_chunks: list[str] = []
        for action in actions:
            obs, reward, done, info = await self.env_client.step(
                lease_id=lease_id, action=action, sleep_after_execution=self.cfg.sleep_after_execution
            )
            step_executed = True
            exec_result = (info or {}).get("exec_result") or {}
            # CLI-aware signals. GUI (pyautogui) faults are left to the task
            # reward rather than double-counted here.
            kind = step_kind(action)
            if kind == "cli":
                res.used_cli = True
                if exec_failed(exec_result):
                    res.exec_penalty_by_step[step_idx] = -1.0
            chunk = "\n".join(
                str(x) for x in (exec_result.get("output"), exec_result.get("error")) if x
            )
            if chunk:
                cli_chunks.append(chunk)
            step_image_path = self.result_dir / f"step_{step_idx + 1}.png"
            save_image(obs["screenshot"], step_image_path)
            with open(self.traj_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action": action,
                    "natural_language_action": info_dict.get("action"),
                    "response": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    # Diagnostics for calibrating exec_failed's layer-3 heuristic.
                    "cli_diag": {
                        "kind": kind,
                        "rc": exec_result.get("return_code"),
                        "stderr_empty": not str(exec_result.get("error") or "").strip(),
                        "cmd_head": str(action.get("command", ""))[:40] if isinstance(action, dict) else "",
                    },
                    "screenshot_file": step_image_path.name,
                }, ensure_ascii=False))
                f.write("\n")
            if done:
                break
        return obs, done, step_executed, "\n".join(cli_chunks)

    # --- helpers ------------------------------------------------------------------

    def _record_snapshot(
        self,
        res: TrajectoryResult,
        step_idx: int,
        policy_messages: list[dict[str, Any]],
        tool_spec: dict[str, Any] | None,
        response: str,
        weight_version: str | None,
    ) -> None:
        """Record one trainable step: history assistant turns masked off, current on."""
        step_train_messages = copy.deepcopy(policy_messages)
        for msg in step_train_messages:
            if msg.get("role") == "assistant":
                msg["step_loss_mask"] = 0
        step_train_messages.append({"role": "assistant", "content": response, "step_loss_mask": 1})

        res.train_messages_for_loss = step_train_messages
        res.tool_spec_for_loss = tool_spec
        res.assistant_responses.append(response)
        res.step_snapshots.append({
            "step_idx": step_idx,
            "train_messages": step_train_messages,
            "response_text": response,
            "tool_spec": tool_spec,
            "weight_version": weight_version,
        })

    def _log_step(self, step_idx: int, finish_type: str, response: str) -> None:
        preview = response
        n = self.cfg.response_preview_chars
        if n > 0 and len(response) > n:
            preview = response[:n] + "...(truncated)"
        logger.info(
            "step sample=%s step=%s finish=%s response=%s",
            self.sample.index, step_idx, finish_type, preview,
        )

    def _write_results(self, res: TrajectoryResult, trace_records: list[dict[str, Any]]) -> None:
        with open(self.result_dir / "result.txt", "w", encoding="utf-8") as f:
            f.write(f"{res.eval_score}\n")
        with open(self.result_dir / "trajectory.json", "w", encoding="utf-8") as f:
            json.dump({
                "meta": {"result": res.eval_score},
                "trajectory": trace_records,
                "reward_trajectory": res.prm.reward_trajectory,
            }, f, ensure_ascii=False, indent=2)
