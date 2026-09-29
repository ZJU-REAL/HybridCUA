# OSWorld Cluster

A lightweight master-node cluster system for managing pools of Docker-based Ubuntu desktop environments (QEMU VMs inside Docker containers). Designed for parallel evaluation of GUI agents at scale.

## Architecture

```
                        ┌────────────────────┐
                        │   Master :18000    │  Scheduling, lease tracking,
                        │ master/server.py   │  dashboard, health monitoring
                        └────────┬───────────┘
                   heartbeat/    │    \allocate
                  register/      │     \forward
                 ┌───────────────┼───────────────┐
                 │               │               │
          ┌──────┴─────┐  ┌─────┴──────┐  ┌─────┴──────┐
          │ Node :18080 │  │ Node :18080 │  │ Node :18080 │
          │ server.py   │  │ server.py   │  │ server.py   │
          └──────┬──────┘  └─────┬──────┘  └─────┬──────┘
                 │               │               │
          ┌──────┴──────┐       ...             ...
          │Docker + QEMU│ × N (up to max_envs)
          │Ubuntu 桌面   │
          └─────────────┘
```

## Files

| File | Role |
|------|------|
| `master/server.py` | Master process — scheduling, lease management, dashboard, health reaper |
| `node/server.py` | Node process entry — registers with master, sends heartbeats, wraps `server.py` |
| `master/models.py` | Shared dataclasses: `NodeInfo`, `LeaseRecord`, `UserQuota` |
| `master/scheduler.py` | Pluggable scheduling strategies: `LeastLoadedScheduler`, `RoundRobinScheduler` |
| `frontend/dashboard.html` | Web UI for cluster status (served by master) |

## Quick Start

### Start Master

```bash
# Minimal:
python -m cluster.master.server --port 18000

# Via script (backgrounds automatically):
bash scripts/bash/start_cluster_master.sh
```

### Start Node

```bash
# Minimal:
NODE_MASTER_URL=http://master-ip:18000 \
NODE_URL=http://this-node-ip:18080 \
NODE_ID=my-node-01 \
python -m cluster.node.server \
  --port 18080 \
  --path-to-vm docker_vm_data/Ubuntu.qcow2 \
  --max-envs 64 \
  --prewarm-envs 32

# Via script:
bash scripts/bash/start_cluster_node.sh
```

### Connect Client (evaluation script)

```bash
export GUI_ENV_SERVER_URL=http://master-ip:18000
python -m cluster.client.run_multienv_remote --server-url "$GUI_ENV_SERVER_URL" --task-config path/to/task.json
```

For custom clients, use the explicit remote environment class:

```python
from cluster.client import OSWorldRemoteClient

env = OSWorldRemoteClient(provider_name="docker_server")
obs = env.reset(task_config)
obs, reward, done, info = env.step(action)
score = env.evaluate()
env.close()
```

## Configuration

### Master

| CLI Arg | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `--host` | — | `0.0.0.0` | Listen address |
| `--port` | — | `18000` | Listen port |
| `--scheduler` | — | `least-loaded` | Node selection strategy (`least-loaded` / `round-robin`) |
| `--node-secret` | `MASTER_NODE_SECRET` | *(empty)* | Shared secret for node authentication |
| `--unhealthy-timeout` | — | `30` | Seconds without heartbeat before marking node unhealthy |
| `--dead-timeout` | — | `60` | Seconds without heartbeat before marking node dead |
| — | `LEASE_EXPIRE_SECONDS` | `7200` | Auto-release leases idle longer than this (seconds) |

### Node

| CLI Arg | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `--max-envs` | `GUI_POOL_MAX_ENVS` | `16` | Maximum concurrent environments on this node |
| `--prewarm-envs` | `GUI_PREWARM_ENVS` | `0` | Environments to pre-create at startup |
| `--prewarm-concurrency` | `GUI_PREWARM_CONCURRENCY` | `2` | Parallel workers for prewarm/scaling |
| `--scale-buffer` | `GUI_SCALE_BUFFER` | `4` | Minimum idle envs maintained by background scaler |
| `--scale-interval` | `GUI_SCALE_INTERVAL` | `5` | Scaler check interval (seconds) |
| `--idle-ttl-seconds` | `GUI_POOL_IDLE_TTL_SECONDS` | `600` | Idle time before env is reaped |
| `--reset-on-close` | `GUI_RESET_ON_CLOSE` | `1` | Reset env state when lease is released |
| — | `PATH_TO_VM` | world.yaml default | Path to Ubuntu qcow2 image (folded into `NODE_WORLD_CONFIG`) |
| — | `NODE_MASTER_URL` | — | Master URL for registration |
| — | `NODE_URL` | — | This node's externally reachable URL |
| — | `NODE_ID` | auto-generated | Unique node identifier |
| — | `NODE_HEARTBEAT_INTERVAL` | `10` | Heartbeat interval (seconds) |

## Key Concepts

### Lease

A lease represents a client's exclusive hold on one environment. Flow:

```
POST /allocate → lease_id
  ... use environment (reset, step, evaluate) ...
POST /close {lease_id} → lease released, env returns to pool
```

Leases auto-expire after `LEASE_EXPIRE_SECONDS` of inactivity. Orphan leases (master restarted or lost tracking) are cleaned up via the heartbeat protocol — master tells each node which leases are valid, node releases the rest.

### Environment Pool Lifecycle

```
Startup: prewarm min_envs environments in background
Runtime: scaler maintains idle >= scale_buffer (creates on-demand, up to max_envs)
Idle:    reaper removes envs idle > idle_ttl_seconds (down to min_envs floor)
```

### Health Monitoring

- Nodes send heartbeats every `NODE_HEARTBEAT_INTERVAL` seconds
- Master marks nodes `unhealthy` / `dead` based on timeout thresholds
- Scheduler only allocates to `healthy` nodes

## Dashboard

Access at `http://master-ip:18000/`. Shows:
- Node status (healthy/unhealthy/dead), capacity, busy/idle counts
- Active leases with user, type, age
- Environment list with container details
- Quota management
- Job execution panel

## API Endpoints

### Master (client-facing)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/allocate` | Get an environment (returns lease_id) |
| POST | `/reset` | Reset env to a task |
| POST | `/step` | Execute an action |
| POST | `/get_obs` | Get current observation |
| POST | `/evaluate` | Run evaluation |
| POST | `/close` | Release lease |
| POST | `/heartbeat` | Client heartbeat |
| GET | `/` | Dashboard data (JSON) |

### Node (internal, called by master)

Same as above, plus:

| Method | Path | Description |
|--------|------|-------------|
| POST | `/allocate` | Allocate from local pool |
| GET | `/emulators` | List all local environments |
| GET | `/vnc/<env_id>/` | VNC proxy (HTTP only) |
