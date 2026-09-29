# Scripts Directory

This directory contains OSWorld evaluation, cluster, model-serving, and cleanup
entry points. Run scripts from the repository root unless a script explicitly
states otherwise.

## Related READMEs

- `cluster/README.md`: master/node architecture, Cluster API, dashboard, leases.
- `desktop_env/providers/docker_server/README.md`: `DockerServerProvider` and
  `OSWorldRemoteClient -> docker_server -> Cluster` data path.
- `scripts/README.md`: this file, focused on script entry points and common
  operational flows.

## Layout

```text
scripts/
├── bash/       # Shell wrappers for repeatable local/remote runs
├── python/     # Python runners used by the shell wrappers
└── README.md
```

The bash wrappers set environment defaults, logs, background mode, and common
arguments. The Python runners contain the multiprocessing evaluation logic.

## Main Evaluation Paths

| Workflow | Bash entry | Python entry | Environment |
|----------|------------|--------------|-------------|
| Generic local eval | custom / `run_multienv.py` | `python/run_multienv.py` | `DesktopEnv(provider_name=docker)` |
| Qwen3VL local | `bash/run_qwen3vl.sh` | `python/run_multienv_qwen3vl.py` | local OSWorld env |
| Qwen3VL remote cluster | `bash/run_qwen3vl_remote.sh` | `python/run_multienv_qwen3vl_remote.py` | `OSWorldRemoteClient(provider_name=docker_server)` |
| Qwen3VL sharded remote | `bash/run_qwen3vl_8gpu_sharded.sh` | `python/run_multienv_qwen3vl_8gpu_sharded.py` | remote cluster + multiple OpenAI-compatible endpoints |
| Qwen3VL 32B thinking sharded | `bash/run_qwen3vl_32b_thinking_8gpu_sharded.sh` | `python/run_multienv_qwen3vl_8gpu_sharded.py` | remote cluster + 32B thinking vLLM shards |
| Claude local/original | `bash/run_claude.sh` | `python/run_multienv_claude.py` | original Claude path |
| Claude remote cluster | `bash/run_claude_remote.sh` | `python/run_multienv_claude_remote.py` | remote cluster + Anthropic-compatible API |
| Manual task check | `bash/run_manual_examine.sh` | `python/manual_examine.py` | manual OSWorld task inspection |
| Result cleanup | `bash/cleanup_results.sh` | `python/cleanup_results.py` | result directory maintenance |

Remote runners explicitly import:

```python
from cluster.client import OSWorldRemoteClient as DesktopEnv
```

Original OSWorld runners that import `desktop_env.desktop_env.DesktopEnv` keep
their original local behavior.

## Cluster Startup

Start the master:

```bash
bash scripts/bash/start_cluster_master.sh
```

Start each node:

```bash
NODE_MASTER_URL=http://master-ip:18000 \
bash scripts/bash/start_cluster_node.sh
```

Node startup defaults to `GUI_PROVIDER_NAME=docker_fast` when configured by the
wrapper. The client-side remote evaluation provider remains `docker_server`.

Useful health checks:

```bash
curl -sS http://master-ip:18000/healthz
curl -sS http://master-ip:18000/nodes
```

## Qwen3VL Remote

Single OpenAI-compatible endpoint:

```bash
GUI_ENV_SERVER_URL=http://master-ip:18000 \
OPENAI_BASE_URL=http://llm-ip:9000/v1 \
NUM_ENVS=48 \
bash scripts/bash/run_qwen3vl_remote.sh
```

Sharded endpoints:

```bash
GUI_ENV_SERVER_URL=http://master-ip:18000 \
OPENAI_BASE_URLS=http://llm-ip:8000/v1,http://llm-ip:8001/v1,http://llm-ip:8002/v1,http://llm-ip:8003/v1 \
NUM_ENVS=64 \
bash scripts/bash/run_qwen3vl_8gpu_sharded.sh
```

32B thinking sharded evaluation:

```bash
GUI_ENV_SERVER_URL=http://master-ip:18000 \
NUM_ENVS=128 \
bash scripts/bash/run_qwen3vl_32b_thinking_8gpu_sharded.sh
```

Before large sharded runs, verify every model endpoint:

```bash
for port in 8000 8001 8002 8003; do
  curl -sS -o /dev/null -w "$port %{http_code}\n" "http://llm-ip:$port/v1/models"
done
```

## vLLM Startup

Common wrappers:

- `bash/start_vllm_qwen3vl.sh`
- `bash/start_vllm_qwen3vl_8gpu.sh`
- `bash/start_vllm_qwen3vl_32b_thinking_8gpu.sh`

Example:

```bash
bash scripts/bash/start_vllm_qwen3vl_32b_thinking_8gpu.sh
```

Check logs under `logs/` and verify `/v1/models` before starting evaluation.

## Claude Remote

Claude remote uses `OSWorldRemoteClient` for the GUI environment and reuses the
existing Anthropic agent logic.

```bash
GUI_ENV_SERVER_URL=http://master-ip:18000 \
ANTHROPIC_BASE_URL=http://anthropic-compatible-host:8010 \
ANTHROPIC_API_KEY=... \
NUM_ENVS=8 \
bash scripts/bash/run_claude_remote.sh
```

Do not commit real API keys. Prefer passing secrets through the environment.

## Result Cleanup

`cleanup_results.sh` is dry-run by default.

Show missing-result tasks:

```bash
ROOT=results_qwen3vl_remote MODE=missing \
bash scripts/bash/cleanup_results.sh
```

Delete task directories without `result.txt`:

```bash
APPLY=1 ROOT=results_qwen3vl_remote MODE=missing \
bash scripts/bash/cleanup_results.sh
```

Delete task directories whose score/result is not `1` and rewrite
`summary/results.json` by removing the corresponding entries:

```bash
APPLY=1 ROOT=results_qwen3vl_remote MODE=failed \
bash scripts/bash/cleanup_results.sh
```

The cleanup tool backs up metadata unless `NO_BACKUP=1` is set.

## Logging And Background Mode

Most bash wrappers:

- write logs under `logs/`
- print background PIDs without creating `.pid` files
- default to `BACKGROUND=1`
- support `BACKGROUND=0` for foreground debugging

Example foreground smoke:

```bash
BACKGROUND=0 NUM_ENVS=1 DOMAIN=chrome MAX_STEPS=1 \
bash scripts/bash/run_qwen3vl_remote.sh
```

## Adding A New Runner

Prefer this shape:

1. Add or reuse a Python runner in `scripts/python/`.
2. Add a thin bash wrapper in `scripts/bash/`.
3. Keep local and remote entry points separate.
4. For remote cluster mode, import `OSWorldRemoteClient` explicitly.
5. Keep secrets in environment variables, not in committed scripts.
6. Add logging and `BACKGROUND` behavior if the run is long-lived.
