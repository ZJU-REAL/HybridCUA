"""MobileWorld sidecar: ``python -m cluster.worlds.mobileworld.sidecar``.

Builds the AndroidEnvClient config from CLI/env and serves the standard
EnvSession REST protocol. MobileWorld-specific config lives here (not in the
platform core).
"""

from __future__ import annotations

import argparse
import json
import os

from cluster.worlds.base.sidecar import create_sidecar_app
from cluster.worlds.mobileworld.adapter import MobileWorldAdapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MobileWorld world sidecar")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18092)
    p.add_argument("--server-url", default=os.getenv("MOBILEWORLD_SERVER_URL", "http://localhost:8000"),
                   help="URL of the MobileWorld backend server that drives the emulator")
    p.add_argument("--device", default=os.getenv("MOBILEWORLD_DEVICE", "emulator-5554"))
    p.add_argument("--step-wait-time", type=float, default=float(os.getenv("MOBILEWORLD_STEP_WAIT", "1.0")))
    p.add_argument("--enable-mcp", action="store_true", default=False)
    p.add_argument("--task-name", default=os.getenv("MOBILEWORLD_TASK", None),
                   help="default task to initialize when reset omits task_name")
    p.add_argument("--config", default=os.getenv("MOBILEWORLD_CONFIG", ""),
                   help="extra JSON config dict merged into the AndroidEnvClient config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config) if args.config else {}
    config.setdefault("url", args.server_url)
    config.setdefault("device", args.device)
    config.setdefault("step_wait_time", args.step_wait_time)
    config.setdefault("enable_mcp", args.enable_mcp)
    if args.task_name:
        config.setdefault("task_name", args.task_name)

    adapter = MobileWorldAdapter.make_driver(config)()
    app = create_sidecar_app(adapter, name="mobileworld-sidecar")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
