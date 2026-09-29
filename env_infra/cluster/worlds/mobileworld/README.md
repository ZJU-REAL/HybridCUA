# MobileWorld plugin

The mobile surface's reference world — wraps the official
[Tongyi-MAI/MobileWorld](https://github.com/Tongyi-MAI/MobileWorld) benchmark,
proving the platform's second surface axis (Surface=mobile × Driver=docker)
without changing master/node core.

## Layout
- `world.yaml` — `capabilities.environment: mobile`, `os: android`, the official
  action vocabulary, `driver.kind: docker`.
- `adapter.py` — `MobileWorldAdapter(MobileWorld)`. Lazily imports the official
  `mobile_world.runtime.client.AndroidEnvClient` and translates generic actions
  to its native `JSONAction` API. No Android/emulator deps to load the module.
- `sidecar.py` — entrypoint serving the EnvSession REST protocol.

## How it maps to the official API
The official `AndroidEnvClient` talks HTTP to a MobileWorld backend that drives
the emulator. The adapter maps the platform contract onto it:

| Platform contract | Official MobileWorld API |
|---|---|
| `reset({"task_name": ...})` | `client.initialize_task(task_name) -> Observation` |
| `step(Action(kind="device", type="click", payload={x,y}))` | `client.execute_action(JSONAction(action_type="click", x=.., y=..))` |
| `step(Action.done())` / `Action.fail())` / `Action.wait()` | `JSONAction(action_type="finished" / "error_env" / "wait")` |
| `observe()` | `client.get_observation()` (PIL screenshot → base64 PNG) |
| `evaluate()` | `client.get_task_score(task) -> (score, reason)` |
| `close()` | `client.tear_down_task(task)` + `client.close()` |
| `health_check()` | `client.health()` |

Episode terminates when the native `action_type` is `finished`, `error_env`, or
`unknown`. Device action types accepted (from `mobile_world` `_ACTION_TYPES`):
`click, double_tap, long_press, scroll, swipe, drag, input_text, keyboard_enter,
navigate_home, navigate_back, open_app, answer, ask_user, status, mcp`.

## Wiring the real MobileWorld
The official repo is vendored at the top level as `MobileWorld/` (peer of
`OSWorld/`). To run for real:
1. Install it: `pip install -e MobileWorld` (brings `mobile_world` onto the path;
   needs its deps — loguru, etc.). The adapter imports it lazily, so the platform
   and conformance tests run without it.
2. Start the MobileWorld backend server (drives the Android emulator) — see the
   upstream README; note its docker image `ghcr.io/tongyi-mai/mobile_world:latest`.
3. Point the sidecar/adapter at it via `--server-url` / `MOBILEWORLD_SERVER_URL`.

> The adapter only depends on the official **public API** (`AndroidEnvClient`,
> `JSONAction`, `Observation`); the `MobileWorld/` repo itself is never modified.

## Run a node hosting it
```bash
python -m cluster.node.world_server --node-worlds mobileworld \
  --master-url http://master:19000
```
The node advertises `runtimes=[mobileworld]` + mobile capabilities; the master's
capability scheduler routes mobile session requests only to mobile-capable nodes.

## Smoke session
```bash
curl -X POST http://master:19000/v1/sessions \
  -d '{"runtime":"mobileworld","task_payload":{"task_name":"calendar.add_event"},
       "capability_requirements":{"environment":"mobile","action":["click"]}}'
# then reset / step (click) / evaluate / close
```

## Env vars
- `MOBILEWORLD_SERVER_URL` — MobileWorld backend URL (default `http://localhost:8000`).
- `MOBILEWORLD_DEVICE` — adb device (default `emulator-5554`).
- `MOBILEWORLD_STEP_WAIT` — per-step settle time (default 1.0s).
- `MOBILEWORLD_TASK` — default task name when reset omits one.
- `MOBILEWORLD_CONFIG` — extra JSON config merged into the client.
