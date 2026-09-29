# env_infra — Multi-World Evaluation Platform

## Project Overview

A platform for managing and evaluating agents across three currently registered worlds: OSWorld, MobileWorld, and CUA-Gym. Worlds are declared by `cluster/worlds/*/world.yaml`.

### Architecture

```
Master (`cluster/master/server.py`)
  ├── Dashboard and job management
  ├── Session allocation, release, and heartbeats
  └── Data-plane forwarding fallback and live-view proxy

Node (`cluster/node/world_server.py`)
  ├── RuntimePool per hosted world
  ├── World adapters (OSWorld / MobileWorld / CUA-Gym)
  └── Environment and container lifecycle

Client (evaluation scripts)
  ├── ClusterSessionClient (generic control/data-plane client)
  ├── OSWorldSessionClient / CuaGymSessionClient
  └── MobileWorldSessionClient
```

### Key Directories

- `cluster/master/` — Master server, scheduler, and dashboard endpoints.
- `cluster/node/` — Node entry point, session manager, pools, and diagnostics.
- `cluster/client/base/` — Generic session client and benchmark-neutral `EvalRunner` / `EvalTaskSource`.
- `cluster/client/{osworld,mobileworld,cua_gym}/` — World-specific session clients and evaluation task sources.
- `cluster/worlds/{osworld,mobileworld,cua_gym}/` — World manifests and adapters; shared interfaces live in `cluster/worlds/base/`.
- `cluster/frontend/` — Dashboard HTML.
- `OSWorld/`, `MobileWorld/` — Benchmark implementations and their own tooling. Prefer the platform adapters and runners for cluster evaluation; these directories are not registered as top-level git submodules.
- `scripts/bash/{osworld,mobileworld,cua_gym}/` — World-specific evaluation launchers.
- `scripts/python/{osworld,mobileworld,cua_gym}/` — World-specific Python evaluation entry points; shared tools live under `scripts/python/`.

### Unified Session Protocol

All benchmarks communicate via the same API:
```
POST   /v1/sessions              → acquire session
POST   /v1/sessions/{id}/reset   → initialize task
POST   /v1/sessions/{id}/step    → execute action
POST   /v1/sessions/{id}/observe → get observation
POST   /v1/sessions/{id}/evaluate → get score
DELETE /v1/sessions/{id}         → release session
```

### RuntimeDriver Protocol

Each world adapter implements `make_driver(config) → RuntimeDriver`:
```python
RuntimeDriver(create=..., create_batch=..., adopt=...)
  - create(): build one adapter (required)
  - create_batch(n): batch create N adapters (optional)
  - adopt(): discover existing running containers (optional)
```

## Principles (MUST FOLLOW)

1. **Generality / Reusability / Extensibility** — #1 priority. No hardcoding. Every design must consider how new benchmarks plug in without touching platform core.

2. **Code Simplicity / Elegance** — #2 priority. Clean interfaces, no redundancy, no dead code.

3. **DO NOT modify OSWorld or MobileWorld internal code** — Only touch the adapter layer (`cluster/worlds/`). Benchmark source code is read-only.

4. **Adapter-declared metadata** — Use `world.yaml` config_schema and `adapter.get_info()` for world-specific configuration, not platform-side detection or hardcoding.

5. **Async/cached for slow operations** — Docker ps, psutil scans, and any IO-heavy operations must be cached with background refresh (never block `/slots` or API responses).

6. **Session lifecycle** — `release(close=False)` keeps slot alive for reuse. `reset_on_release=True` cleans environment between tasks. Never destroy containers on session release.

7. **Health check on acquire** — Pool verifies adapter health before assigning a slot. Dead slots are auto-removed and rebuilt.

## Common Patterns

### Adding a new benchmark

1. Create `cluster/worlds/<name>/world.yaml` (config_schema, capabilities)
2. Create `cluster/worlds/<name>/adapter.py` (implement WorldAdapter + `make_driver`)
3. Create `cluster/client/<name>_remote.py` (inherit ClusterSessionClient, implement benchmark's client interface)
4. Create evaluation scripts in `scripts/python/` and `scripts/bash/<name>/`

### Evaluation scripts

- OSWorld: `EvalRunner` + `OSWorldRemoteClient` (multi-process, task from JSON files)
- MobileWorld: `run_agent_with_evaluation` + `MobileWorldRemoteClient` (joblib parallel, task from registry)
- Bash scripts set env vars and call Python entry points
- Scripts must handle `OSWORLD_JOB_ID` for frontend job execution (skip `cd` when set)

### Live View

- MobileWorld: VNC via noVNC (started post-ready via `docker exec`)
- OSWorld: VNC via container's built-in noVNC (port 8006)
- Frontend uses `live_view: {protocol: "vnc"|"http", port: N}` from `get_info()`
- Master proxies `/view/<slot_id>/` to node, node proxies to container

## Environment Notes

- Master: this host, port `19000`
- Node: this host, port `18080`
- MobileWorld image: `ghcr.io/tongyi-mai/mobile_world:v1.4`
- OSWorld image: `happysixd/osworld-docker`
- Proxy: `http://star-proxy.oa.com:3128` (TLS issues with pypi/Google; use `no_proxy` for pypi.org)
- MobileWorld `.env` at `MobileWorld/.env` (mounted into containers)
- Node venv: `../venvs/env_infra/` (python 3.12, `python -m venv`; built by `install_env.sh`)
