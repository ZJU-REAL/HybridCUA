# DockerServerProvider

A remote environment provider that connects to OSWorld's cluster (or standalone node) via HTTP. Used by evaluation scripts to allocate and control desktop environments without running Docker locally.

## How It Works

```
Evaluation Script
    │
    │  RemoteDesktopEnv(provider_name="docker_server", server_url="http://master:18000")
    │
    ▼
DockerServerProvider (this module)
    │
    │  HTTP: /allocate, /reset, /step, /evaluate, /close
    │
    ▼
Master (or standalone Node server)
    │
    ▼
Node → Docker Container → QEMU VM → Ubuntu Desktop
```

The provider abstracts away the cluster for the explicit remote client. Use
`RemoteDesktopEnv` for cluster mode; the original OSWorld `DesktopEnv` does not
accept `server_url` and should remain on local/cloud providers unless the core
OSWorld factory is intentionally extended.

## Usage

### In evaluation scripts

```python
from cluster.client import RemoteDesktopEnv

env = RemoteDesktopEnv(
    provider_name="docker_server",
    server_url="http://master-ip:18000",  # or standalone node URL
)

obs = env.reset(task_config=config)
obs, reward, done, info = env.step(action)
result = env.evaluate()
env.close()  # releases lease back to pool
```

Existing OSWorld scripts that import:

```python
from desktop_env.desktop_env import DesktopEnv
```

continue to use the original `DesktopEnv`. To use the cluster, switch the
evaluation entry point to import `RemoteDesktopEnv` explicitly, for example
`scripts/python/run_multienv_qwen3vl_remote.py`.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GUI_ENV_SERVER_URL` | `http://127.0.0.1:18080` | Server URL (master or standalone node) |
| `OSWORLD_USER_ID` | `anonymous` | User ID sent with allocation requests |
| `OSWORLD_TASK_TYPE` | `evaluation` | Task type (`evaluation` or `training`) — used for quota enforcement |

### Constructor parameters

```python
DockerServerProvider(server_url="http://master:18000")
```

The `server_url` parameter takes precedence over `GUI_ENV_SERVER_URL`.

## Allocation Flow

```
1. _allocate()
   POST /allocate {user_id, task_type}
   ← {lease_id, env_id, node_url}
   
   Remote operations are keyed by lease_id. Direct container ports may be present
   for diagnostics/VNC, but reset/step/evaluate should go through the server API.

2. reset(task_config)
   POST /reset {lease_id, task_config}
   ← observation

3. step(action) / get_obs() / evaluate()
   POST /step|get_obs|evaluate {lease_id, ...}
   ← result

4. _release()
   POST /close {lease_id}
   ← env returns to pool
```

## Retry Behavior

Allocation retries **10 times** with **3-second intervals** (30 seconds total). If the cluster is at capacity, the request fails after exhausting retries.

All data-path calls (`/reset`, `/step`, `/evaluate`) do NOT retry — they fail immediately on error.

## Cluster vs Standalone

| Mode | `server_url` points to | Behavior |
|------|----------------------|----------|
| **Cluster** | Master (`http://master:18000`) | Master schedules across nodes |
| **Standalone** | Node directly (`http://node:18080`) | Single node, no master needed |

In cluster mode, the provider initially talks to the master for `/allocate`. After allocation, data-path calls (`/reset`, `/step`, etc.) are routed through the master which forwards them to the correct node.

## Relationship to Other Providers

```
desktop_env/providers/
├── base.py              ← Abstract VMProvider interface
├── docker/provider.py   ← LOCAL Docker: creates containers directly
├── docker_server/       ← REMOTE: talks to cluster via HTTP (this module)
├── aws/                 ← AWS EC2 instances
├── azure/               ← Azure VMs
└── ...
```

`DockerServerProvider` implements the provider side of remote operations, but the
supported client boundary is `RemoteDesktopEnv`. This keeps cluster behavior
separate from the original OSWorld `DesktopEnv` and avoids monkeypatching.
