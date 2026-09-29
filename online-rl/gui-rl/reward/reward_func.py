"""Reward computation for GUI rollouts (slime ``--custom-rm-path`` entry).

``reward_func`` is the entry slime calls after a rollout to score samples. It
composes the task outcome reward (success -> +1, else -1) with optional PRM
(process reward model) per-step scores. ``mark_aborted_samples`` keeps the
training pipeline stable for broken trajectories.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from slime.utils.types import Sample


class Signal(NamedTuple):
    coef_attr: str
    meta_key: str
    scope: str  # "traj" = inside sigma_G (eq. 6); "step" = outside it (eq. 7)


#: CLI-aware signals. Adding one is a single line here.
#: Note a coefficient of 0 does NOT mean "observe only": with every coefficient
#: at 0 the gate in trajectory_runner prefills sample.reward, the RM is skipped
#: entirely, and none of these keys ever appear. Read traj.jsonl's cli_diag to
#: observe signals before enabling a term.
SIGNALS = {
    "cli": Signal("lambda_cli", "gui_cli_reward", "traj"),
    "exec": Signal("lambda_exec", "gui_exec_penalty", "step"),
}


def coef(args, attr) -> float:
    return float(getattr(args, attr, 0.0) or 0.0)


def cli_aware_enabled(args) -> bool:
    """True when any CLI-aware term is active; False restores pre-change behavior."""
    return any(coef(args, s.coef_attr) for s in SIGNALS.values())


def mark_aborted_samples(samples: list[Sample]) -> None:
    """Give ABORTED samples a default reward and exclude them from training.

    The rollout pipeline skips reward_func for lists containing any ABORTED
    sample, which leaves reward=None and crashes downstream metrics. Setting a
    default reward + remove_sample=True keeps the pipeline stable while ensuring
    these broken trajectories never contribute to the gradient.
    """
    for s in samples:
        if s.status == Sample.Status.ABORTED:
            if s.reward is None:
                s.reward = {"score": 0.0, "acc": 0.0}
            s.remove_sample = True


def single_reward(sample: Sample, args: Any = None) -> dict[str, float]:
    """Outcome reward plus CLI-aware terms: ``score`` for GRPO, ``acc`` for logging."""
    outcome_score = 0.0
    raw_acc = 0.0
    if isinstance(sample.reward, dict):
        outcome_score = float(sample.reward.get("score", 0.0))
        raw_acc = float(sample.reward.get("acc", 0.0))
    elif isinstance(sample.metadata, dict):
        raw_acc = float(sample.metadata.get("gui_score", 0.0))
        # outcome_score = 1.0 if raw_acc == 1.0 else -1.0
        outcome_score = raw_acc

    result = {"score": outcome_score, "acc": raw_acc, "base_score": outcome_score}
    meta = sample.metadata if isinstance(sample.metadata, dict) else {}
    for name, sig in SIGNALS.items():
        result[name] = signal = float(meta.get(sig.meta_key, 0.0))
        if sig.scope == "traj":
            result["score"] += coef(args, sig.coef_attr) * signal
    return result


def _compose_with_prm(args: Any, s: Sample) -> dict[str, float]:
    """Combine outcome reward with PRM step scores (no-op when PRM is off)."""
    result = single_reward(s, args)
    if not getattr(args, "prm_enable", False):
        return result

    prm_metadata = s.metadata.get("prm", {}) if isinstance(s.metadata, dict) else {}
    if not isinstance(prm_metadata, dict):
        prm_metadata = {}

    prm_step_mean = float(prm_metadata.get("step_mean_score", 0.0))
    outcome_reward = float(result.get("score", 0.0))
    final_score = outcome_reward + float(getattr(args, "prm_step_coef", 1.0)) * prm_step_mean
    result["base_score"] = outcome_reward
    result["prm_step_score"] = prm_step_mean
    result["score"] = final_score

    # Expose one concrete PRM raw output for quick sanity-checking (like retool).
    prm_example_eval = ""
    step_details = prm_metadata.get("step_details", [])
    if isinstance(step_details, list) and step_details:
        first_step = step_details[0] if isinstance(step_details[0], dict) else {}
        votes = first_step.get("votes", []) if isinstance(first_step, dict) else []
        if isinstance(votes, list) and votes:
            first_vote = votes[0] if isinstance(votes[0], dict) else {}
            raw_text = first_vote.get("raw_text", "") if isinstance(first_vote, dict) else ""
            if isinstance(raw_text, str):
                prm_example_eval = raw_text
    result["prm_example_eval"] = prm_example_eval

    # Populate step_wise composed rewards for step_wise advantage.
    if isinstance(s.metadata, dict):
        step_wise_meta = s.metadata.get("step_wise", {})
        if not isinstance(step_wise_meta, dict):
            step_wise_meta = {}
        step_wise_meta["outcome_reward"] = outcome_reward
        raw_step_scores = step_wise_meta.get("step_scores", [])
        if isinstance(raw_step_scores, list):
            step_wise_meta["step_scores_with_outcome"] = [
                float(step_score) + outcome_reward for step_score in raw_step_scores
            ]
        else:
            step_wise_meta["step_scores_with_outcome"] = []
        s.metadata["step_wise"] = step_wise_meta
    return result


async def reward_func(args, sample: Sample | list[Sample], **kwargs):
    """slime reward entry: score one sample or a list of samples."""
    if isinstance(sample, list):
        return [_compose_with_prm(args, s) for s in sample]
    return _compose_with_prm(args, sample)
