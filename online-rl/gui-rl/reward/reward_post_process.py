"""GRPO reward post-processing (slime ``--custom-reward-post-process-path``).

Implements eq. (7) under ``dynamic_history``:

    A(t,j) = (R(tau) - mu_G) / (sigma_G + eps) + lambda_exec * e_t

e_t is per-step and sits outside the normalization, so it must be added after
it. slime's default path cannot express that: it dedupes step samples by
``(group_index, index)`` -- which every child of one trajectory shares -- so a
per-step term folded into ``score`` survives only for step 0.

The dedup+normalize block below is copied from
``slime/ray/rollout.py:664-694`` (``_post_process_rewards``, dynamic_history
branch). Re-compare it after a slime upgrade.
"""

from __future__ import annotations

import logging

import torch

from reward.reward_func import SIGNALS, coef

logger = logging.getLogger(__name__)


def traj_key(sample, fallback: int) -> tuple[int, int]:
    """Trajectory identity, shared by every step sample of one trajectory."""
    group_idx = int(sample.group_index) if sample.group_index is not None else -1
    traj_idx = int(sample.index) if sample.index is not None else fallback
    return group_idx, traj_idx


def step_bonuses(args, samples) -> list[float]:
    """Per-sample sum of step-scoped terms; all zeros when none are enabled."""
    active = [
        (coef(args, s.coef_attr), s.meta_key)
        for s in SIGNALS.values()
        if s.scope == "step" and coef(args, s.coef_attr)
    ]
    if not active:
        return [0.0] * len(samples)

    out = []
    for sample in samples:
        meta = sample.metadata if isinstance(sample.metadata, dict) else {}
        total = 0.0
        for c, key in active:
            try:
                total += c * float(meta.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                pass
        out.append(total)
    return out


def normalize(vals: torch.Tensor, std_norm: bool) -> torch.Tensor:
    vals = vals - vals.mean()
    if std_norm:
        return vals / (vals.std() + 1e-6) if len(vals) > 1 else torch.zeros_like(vals)
    return vals


def reward_post_process(args, samples):
    """Returns ``(raw_rewards, rewards)``; ``rewards`` feeds the advantage."""
    if not getattr(args, "dynamic_history", False):
        raise ValueError(
            "reward_post_process requires dynamic_history=true; drop "
            "--custom-reward-post-process-path for the non-dynamic path"
        )

    raw_rewards = [float(s.get_reward_value(args)) for s in samples]

    if not (
        args.advantage_estimator in ("grpo", "gspo", "reinforce_plus_plus_baseline")
        and args.rewards_normalization
    ):
        return raw_rewards, raw_rewards

    std_norm = args.advantage_estimator in ("grpo", "gspo") and args.grpo_std_normalization

    # Dedupe so a long trajectory does not weigh more in mu_G / sigma_G.
    traj_reward: dict[tuple[int, int], float] = {}
    group_keys: dict[int, list[tuple[int, int]]] = {}
    keys = []
    for i, sample in enumerate(samples):
        key = traj_key(sample, i)
        keys.append(key)
        if key not in traj_reward:
            traj_reward[key] = raw_rewards[i]
            group_keys.setdefault(key[0], []).append(key)

    normalized: dict[tuple[int, int], float] = {}
    for group in group_keys.values():
        vals = normalize(torch.tensor([traj_reward[k] for k in group], dtype=torch.float32), std_norm)
        normalized.update({k: float(vals[j]) for j, k in enumerate(group)})

    bonuses = step_bonuses(args, samples)
    rewards = [normalized[k] + bonuses[i] for i, k in enumerate(keys)]

    nonzero = sum(1 for b in bonuses if b)
    if nonzero:
        logger.info("reward_post_process: %d/%d step samples carry a step term", nonzero, len(samples))
    return raw_rewards, rewards
