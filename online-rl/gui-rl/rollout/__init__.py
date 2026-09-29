"""rollout: GUI trajectory rollout for slime (train + eval).

Each trajectory runs on its own long-lived Ray worker actor
(:class:`rollout.ray_actor_pool.RayActorPool`), so per-trajectory synchronous CPU
work never serializes on the RolloutManager event loop.

slime entry points:
- ``rollout.partial_async_gui_rollout.generate``               -> ``--custom-generate-function-path``
- ``rollout.partial_async_gui_rollout.fast_eval_rollout``      -> ``--eval-function-path`` (semi-async)
- ``rollout.fully_async_rollout.generate_rollout_fully_async`` -> ``--rollout-function-path`` (fully-async)
- ``rollout.fully_async_rollout.eval_rollout_fully_async``     -> ``--eval-function-path`` (fully-async, opt-in)

All paths converge on :func:`rollout.trajectory_runner.run_trajectory`, branching
only on the ``evaluation`` flag.
"""
