# online-rl — GRPO RL training for GUI agents

## Project Overview

Online RL (GRPO) training of multimodal GUI agents against live desktop/mobile
environments. The trainer is [slime](https://github.com/THUDM/slime) (Megatron-LM
backend for the actor, SGLang for rollout); `gui-rl/` is our own package that
plugs GUI trajectory rollout, reward, and data sourcing into slime's extension
points — **slime and Megatron-LM stay unpatched**.

Environments are served remotely by the sibling `env_infra` platform over the
`/v1/sessions` protocol (OSWorld / MobileWorld / CUA-Gym). This repo never
launches VMs or containers itself; it only speaks HTTP to an env server.

### Architecture

```
gui-rl/scripts/*.sh                 launcher: ray start --head -> ray job submit
  └── train_fully_async.py          main loop (or slime/train_async.py)
        ├── slime RolloutManager
        │     └── --rollout-function-path  rollout/fully_async_rollout.py
        │           (fixed in-flight pool across rollout boundaries + staleness filter)
        │             └── --custom-generate-function-path  rollout/partial_async_gui_rollout.py
        │                   └── rollout/ray_actor_pool.py -> rollout/trajectory_runner.py
        │                         └── rollout/trajectory.py   one episode:
        │                               acquire -> reset -> N x (policy -> action -> step) -> evaluate -> close
        │                                 ├── agents/*        prompt build + action parse
        │                                 ├── clients/*       HTTP to env_infra /v1/sessions
        │                                 └── reward/prm_hook per-step PRM (optional)
        ├── --custom-rm-path               reward/reward_func.py
        └── --data-source-path             data/gui_data_source.py
```

### Two layers of asynchrony (they are orthogonal — do not conflate)

| Layer | Owner | What it hides |
|---|---|---|
| **Worker pool** | `rollout/ray_actor_pool.py` | Per-trajectory *synchronous CPU* work (tokenize, image base64, `build_train_data`) would serialize on the RolloutManager event loop. Each trajectory runs on its own long-lived Ray actor; results return via plasma. |
| **In-flight pool** | `rollout/fully_async_rollout.py` | The *long tail*: a background worker keeps a fixed number of trajectories in flight across rollout boundaries, so step N+1 never waits on the slowest episode of step N. |

`*_fully_async.sh` scripts stack both. The plain scripts (`HybridCUA-9B_8gpu.sh`,
`gui_qwen3vl_8b.sh`, `gui_qwen3vl_16gpu.sh`, `gui_qwen3.5_9B_16gpu.sh`) use only
the worker pool ("partial/semi-async").

**Align the three concurrency knobs.** The real ceiling is `min()` of them, so
keeping them equal removes hidden bottlenecks:

```
in-flight pool = SGLANG_SERVER_CONCURRENCY x num_engines   (TARGET_IN_FLIGHT, default 64)
worker pool    = GUI_FAST_ROLLOUT_PROCS                    (default 64)
env sessions   = GUI_TRAJECTORY_CONCURRENCY                (default 64; protects env_infra)
```

Scripts derive `SGLANG_SERVER_CONCURRENCY = ceil(TARGET_IN_FLIGHT / num_engines)`.
Never leave it at sglang's default 512 — that puts 512 × engines in flight and
floods both the worker queue and the env cluster.

### Key directories

- `gui-rl/rollout/` — trajectory execution. `trajectory.py` (one episode + per-step
  sample build for dynamic-history GRPO), `trajectory_runner.py` (single-trajectory
  entry, shared by train and eval), `partial_async_gui_rollout.py` (slime entry +
  pool dispatch), `fully_async_rollout.py` (in-flight pool + staleness),
  `ray_actor_pool.py` (the worker pool: N actors, plasma result return).
- `gui-rl/agents/` — one class per model family: `Qwen3VLAgentLocal`,
  `Qwen3VLMobileAgentLocal`, `Qwen35VLAgentLocal`, `Qwen35HybridCuaAgentLocal`.
  Selected at runtime by `GUI_AGENT_CLASS_PATH`. `utils/` holds prompt/history/
  action-parsing helpers.
- `gui-rl/clients/` — env clients. `SessionGuiEnvClient` speaks `/v1/sessions`;
  `env_client.py` (repo root of gui-rl) is the legacy lease-HTTP `/allocate` client.
  Chosen by `GUI_ENV_CLIENT=session|legacy`.
- `gui-rl/reward/` — `reward_func.py` (slime `--custom-rm-path`; outcome ±1 composed
  with optional CLI-aware signals), `reward_post_process.py` (GRPO eq. 7 under
  dynamic-history), `prm_hook.py` + `{local,external}_reward_agent.py` (process RM).
- `gui-rl/data/` — `gui_data_source.py`: `GuiMetaDataSource`, `MobileWorldDataSource`,
  `CuaGymDataSource`, and `MultiPlatformDataSource` (mix / interleaved / single).
- `gui-rl/config.py` — **single source of truth for every `GUI_*` env var**: names,
  defaults, type coercion. Read at call time, not import time.
- `gui-rl/scripts/` — launchers, YAML custom-configs, node topologies.
- `gui-rl/evaluation_examples/` — OSWorld/MobileWorld task metadata and splits.
- `slime/`, `Megatron-LM/` — vendored upstream. **Read-only.**
- `install_env.sh` — builds the py3.12 venv (torch 2.11+cu129). Must run on a node
  with GPU + nvcc: flash-attn / apex / TE compile in place.

### Configuration: three channels, and they are not interchangeable

1. **CLI flags** — anything slime's argparse registers. Passed by the launch script.
2. **`--custom-config-path <yaml>`** (`scripts/gui_partial_async*.yaml`) — keys slime
   does *not* register but our code reads via `getattr(args, ...)`. slime `setattr`s
   each key onto `args`. `dynamic_history`, `gui_max_steps`, `gui_eval_*`,
   `prm_enable` live here. Passing these on the CLI instead is **silently dropped**
   by Megatron's `ignore_unknown_args` — a classic source of "my flag did nothing".
3. **`GUI_*` env vars** — read by `config.py` inside rollout worker processes.
   Launch scripts must also list them in `RUNTIME_ENV_JSON` so Ray actors inherit them.

`config.EpisodeConfig.resolve` prefers (2) and falls back to (3).

## Principles (MUST FOLLOW)

1. **Do not patch `slime/` or `Megatron-LM/`.** Every hook we need already has an
   extension point (`--custom-*-path`, `--rollout-function-path`,
   `--eval-function-path`, `--custom-config-path`). If something seems to need a
   patch, it almost certainly belongs in `gui-rl/`.
2. **No secrets, no internal hostnames, no absolute private paths in code.** This
   tree was scrubbed deliberately. `WANDB_API_KEY` and `HF_CKPT` use bash `:?`
   validation so a missing value fails loudly instead of defaulting. Proxy support
   is off by default (`USE_STAR_PROXY=0`) and requires an explicit `STAR_PROXY_URL`.
   Do not reintroduce a default.
3. **Env vars go through `config.py`.** No inline `os.getenv` for `GUI_*` in new
   code — the full configuration surface must stay discoverable in one file.
4. **New shared shell boilerplate goes in `scripts/_common.sh`.** It provides
   `setup_background_exec`, `kill_stale_python`, `cleanup_ray`, `setup_proxy`,
   `ensure_libnuma`. Most existing scripts still inline their own copies; prefer
   sourcing `_common.sh` when touching one.
5. **`kill_stale_python`, never a blanket `pkill -9 python`.** The env server often
   runs on the same node; killing it breaks the next run's `/healthz` gate. The
   spared pattern is `GUI_ENV_SERVER_PROC_PATTERN` (default `cluster.master.server`).
6. **Eval and train share `run_trajectory`**, branching only on `evaluation`.
   Keep it that way — divergence there is how eval/train parity silently rots.
   Note `dynamic_history` fan-out fires only when `evaluation=False`.
7. **In-training eval is off under fully-async.** It contends with the in-flight
   pool for the same worker/env slots, skewing both eval numbers and throughput.
   All `*_fully_async.sh` default `GUI_EVAL_INTERVAL=0` (no `--eval-interval`, so
   the hook never fires) and `eval_rollout_fully_async` raises unless
   `GUI_FULLY_ASYNC_ALLOW_EVAL=1`. Eval saved ckpts offline instead.

## Common Patterns

### Launching a run

```bash
# Head node (has the actor GPUs):
WANDB_API_KEY=... HF_CKPT=/path/to/Qwen3.5-9B \
  GUI_ENV_SERVER_URL=http://<env-node>:19000 \
  bash gui-rl/scripts/HybridCUA-9B_16gpu_fully_async.sh

# Each worker node:
RAY_HEAD_ADDR=<head_ip> WORKER_NUM_GPUS=8 bash gui-rl/scripts/gpu_worker_join_ray.sh
```

Scripts self-detach (`setsid`) and log to `gui-rl/logs/<prefix>_<ts>.log`; set
`BACKGROUND=0` to stay in the foreground. The head waits `EXPECTED_NODES` for 600s
and then **submits anyway** — a late worker silently yields wrong actor/rollout
placement, so coordinate the bring-up.

`scripts/run_rl_24gpu.sh` wraps clean/env/train/worker/status against a topology
file (`scripts/topologies/{2node,3node}.env`, selected with `TOPOLOGY=`).

### Script matrix

| Script | GPUs | Model | Async |
|---|---|---|---|
| `HybridCUA-9B_{8,16,24,32,40}gpu_fully_async.sh` | 8–40 | Qwen3.5-9B dense VLM | both layers |
| `HybridCUA-9B_8gpu.sh` | 8 | Qwen3.5-9B | worker pool only |
| `gui_qwen3.5_9B_{16,24}gpu*.sh` | 16/24 | Qwen3.5-9B | per suffix |
| `gui_qwen3.5_35B_A3B_16gpu_fully_async.sh` | 16 | Qwen3.5-35B-A3B (MoE) | both |
| `gui_qwen3vl_{8b,16gpu}*.sh` | 8/16 | Qwen3-VL-8B | per suffix |

All default to `ROLLOUT_GPUS=8` as 1-GPU sglang engines, actor `TRAIN_TP=4`.

### Adding a new model

1. Add `slime/scripts/models/<name>.sh` if upstream lacks it (it is `source`d for
   `MODEL_ARGS`) — this is the one allowed touch under `slime/`, it is config not code.
2. Add an agent class under `agents/` (subclass the nearest existing one).
3. Copy the closest launch script; set `GUI_AGENT_CLASS_PATH`, the model script
   `source`, and TP/PP. Verify the flags in `RUNTIME_ENV_JSON` too.

### Adding a new environment/benchmark

1. Add a `DataSource` subclass in `data/gui_data_source.py` (see `CuaGymDataSource`).
2. Point `GUI_ENV_RUNTIME` at the `env_infra` world name; the session client is
   generic, so usually no client change is needed.
3. Set `GUI_DATA_SOURCE_PATH` in the launch script.

### Tests

```bash
PYTHONPATH=gui-rl:slime python3 gui-rl/tests/test_cli_aware_reward.py
PYTHONPATH=gui-rl:slime python3 gui-rl/tests/test_gui_only_eval_parity.py
```

Plain scripts, not pytest. `test_cli_aware_reward.py` also asserts against
`slime/slime/ray/rollout.py` to catch upstream drift.

## Known state

**One worker-pool backend only.** `process_pool.py` (`FastRolloutPool`) was
removed; `rollout/ray_actor_pool.py` is the sole backend and absorbed the four
helpers that used to live in `process_pool` (`_get_env_client`, `_worker_init`,
`_run_one`, plus pool sizing, now `config.rollout_pool_size`). The
`GUI_ROLLOUT_BACKEND` switch is gone — do not reintroduce it. `GUI_FAST_ROLLOUT_PROCS`
keeps its name (it is the pool size) even though workers are actors, not processes.

`sglang` is **not** vendored here — clone it at the commit pinned in
`install_env.sh` during environment setup.

## Environment Notes

- Venv: `<project-root>/venvs/online-rl` (py3.12, torch 2.11.0+cu129). py3.13 is
  ruled out twice over: Megatron's `numpy 1.x` assert (no cp313 wheel) and
  Megatron-Bridge's `requires-python <3.13`.
- Env server: `env_infra` master on `:19000` (`/v1/sessions`, use
  `GUI_ENV_CLIENT=session`). The legacy `/allocate` lease server is `:18000`
  (`GUI_ENV_CLIENT=legacy`). Scripts gate on `GET /healthz` for 120s before submitting.
- Ray plasma: `OBJECT_STORE_GB` (default 600) **must match on head and every
  worker**. The default (~30% RAM) gets pegged by dynamic-history image fan-out,
  triggering multi-TB spill that stalls training until NCCL times out.
- `libnuma.so.1`: `sgl_kernel` needs it and worker nodes often lack it. Scripts look
  for a shared-disk copy at `<gui-rl>/../vendor_libs/libnuma.so.1` first, then
  apt/yum. The resolved dir is injected into the actors' `LD_LIBRARY_PATH`.
- cuDNN ABI: the system ships an older cuDNN 9 than the venv's `nvidia-cudnn-cu12`.
  Scripts force the venv's cuDNN dir to the front of the actors'
  `LD_LIBRARY_PATH`, or every actor dies at `transformer_engine` import with
  `undefined symbol ... libcudnn_graph.so.9`.
- Qwen3.5-9B specifics: dense (no MoE/EP flags), Gated Delta Net
  (`--use-gated-attention`, thd+dynamic-batch enabled via the GDN varlen backport,
  see `gui-rl/docs/thd_gdn_backport.md`), and `GUI_DISABLE_MTP=1` is required
  (`mtp_num_hidden_layers=1` otherwise breaks bridge mapping). `TRAIN_TP=8` trips
  the Megatron GQA output-gate patch since `num_query_groups=4 < world_size`.
- Wandb: `WANDB_BASE_URL` defaults to a self-hosted instance in several scripts;
  unset it (or point it at `https://api.wandb.ai`) for cloud wandb.
