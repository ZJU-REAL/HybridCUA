"""CUA-Gym world adapter.

CUA-Gym desktop tasks run on the SAME OSWorld VM image and use the SAME task /
setup schema as OSWorld, with one difference: the reward is a self-contained
Python script (evaluator ``{type: python}``) that prints ``REWARD: <float>`` as
its last line, instead of an OSWorld built-in ``metrics`` function.

So this is a thin subclass of :class:`OSWorldWorldAdapter`: DesktopEnv
construction, adopt/reclaim, ``step`` / ``observe`` / ``liveness`` / ``close`` are
all inherited unchanged (they use ``cls(...)``, so they yield CuaGym adapters).
Only the two ends of an episode are overridden:

- ``reset``    : stash the reward script and the evaluator's ``postconfig``, then
                 swap CUA-Gym's ``{type:python}`` evaluator for a sentinel OSWorld
                 can parse. ``DesktopEnv.reset`` runs ``getattr(metrics,
                 evaluator["func"])`` at reset time and would ``KeyError`` on
                 CUA-Gym's schema; ``infeasible`` is a real 0-arg metric, and our
                 ``evaluate`` never actually calls it.
- ``evaluate`` : first run the evaluator's ``postconfig`` (e.g. Ctrl+S to flush
                 the edited file to disk), exactly as OSWorld's own ``evaluate``
                 does, then run the reward script in the VM and parse
                 ``REWARD: X.X`` from its stdout.

Reward execution reuses OSWorld's controller verbatim:
``controller.run_python_script(code)`` POSTs ``/run_python`` to the in-VM server,
which writes the code to a temp file and runs it as ``python3 <file>`` (a real
subprocess, so ``__name__ == "__main__"`` holds) and returns stdout in ``output``.
The reward script reads the task artifacts under ``/home/user`` itself to score.

No OSWorld or platform-core code is touched.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from cluster.schemas import EvaluationResult, Observation
from cluster.worlds.osworld.adapter import OSWorldWorldAdapter

logger = logging.getLogger("cluster.worlds.cua_gym")

# ``def infeasible(): pass`` is a real 0-arg metric in OSWorld's metrics/__init__.
# It only has to satisfy DesktopEnv.reset()'s ``getattr(metrics, func)`` parse —
# evaluate() is overridden and never calls it.
_SENTINEL_EVALUATOR: Dict[str, Any] = {"func": "infeasible"}
_REWARD_RE = re.compile(r"REWARD:\s*([0-9]*\.?[0-9]+)")


def _parse_reward(output: str) -> float:
    """The last ``REWARD: X.X`` the reward script printed, clamped to [0, 1]."""
    matches = _REWARD_RE.findall(output or "")
    return max(0.0, min(1.0, float(matches[-1]))) if matches else 0.0


class CuaGymWorldAdapter(OSWorldWorldAdapter):
    """OSWorld desktop VM + CUA-Gym python-script reward."""

    # Class defaults so evaluate() is safe even before reset() and on adopted
    # slots (whose __init__ is OSWorldWorldAdapter's, not ours).
    _reward_code: Optional[str] = None
    _eval_postconfig: List[Dict[str, Any]] = []

    def reset(self, task_payload: Dict[str, Any]) -> Observation:
        payload = dict(task_payload or {})
        # Skip the release-time reset({}) call (pool.py:531) — it passes an empty dict
        # with no id/instruction/config, which would KeyError in _set_task_info.
        # The next acquire's real reset(task_config) will revert the snapshot and
        # set up the new task properly, so this cleanup reset is unnecessary.
        if not payload.get("id"):
            self._reward_code = None
            self._eval_postconfig = []
            return None  # pool.py ignores the return value on release
        evaluator = dict(payload.get("evaluator") or {})
        # Reward script is inlined by the loader (load_tasks.py) as ``reward_code``;
        # fall back to an evaluator["code"] field if a caller inlines it there.
        self._reward_code = payload.pop("reward_code", None) or evaluator.get("code")
        # postconfig (e.g. Ctrl+S) must run before scoring — stash it before the
        # evaluator is replaced by the sentinel below.
        self._eval_postconfig = list(evaluator.get("postconfig") or [])
        if not self._reward_code:
            logger.warning("CUA-Gym task %r has no inline reward code", payload.get("id"))
        payload["evaluator"] = _SENTINEL_EVALUATOR
        return super().reset(payload)

    def evaluate(self) -> EvaluationResult:
        # 1) Flush edited docs to disk (postconfig), mirroring DesktopEnv.evaluate().
        if self._eval_postconfig:
            try:
                self._env.setup_controller.setup(self._eval_postconfig, self._env.enable_proxy)
            except Exception:  # a broken postconfig must not crash scoring
                logger.warning("postconfig failed during reward eval", exc_info=True)
        # 2) Run the reward script in the VM; parse REWARD: X.X from its stdout.
        if not self._reward_code:
            return EvaluationResult(score=0.0, success=False, reason="no reward code")
        res = self._env.controller.run_python_script(self._reward_code) or {}
        output = res.get("output") or ""
        score = _parse_reward(output)
        return EvaluationResult(
            score=score,
            success=score >= 1.0,
            metrics={"raw_score": score, "exec_status": res.get("status")},
            reason="cua_gym python reward",
        )

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info["world_id"] = "cua_gym"
        return info
