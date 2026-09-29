"""OSWorld sidecar: ``python -m cluster.worlds.osworld.sidecar``.

Builds DesktopEnv kwargs from CLI/env and serves the standard EnvSession REST
protocol via the shared sidecar app. OSWorld-specific configuration lives here
(not in the platform core), keeping the master/node world-neutral.
"""

from __future__ import annotations

import argparse
import os

from cluster.worlds.base.sidecar import create_sidecar_app
from cluster.worlds.osworld.adapter import OSWorldWorldAdapter


def build_env_kwargs(args: argparse.Namespace) -> dict:
    require_a11y_tree = args.observation_type in {"a11y_tree", "screenshot_a11y_tree", "som"}
    require_terminal = args.observation_type == "terminal"
    return {
        "path_to_vm": args.path_to_vm,
        "snapshot_name": args.snapshot_name,
        "action_space": args.action_space,
        "provider_name": os.getenv("GUI_PROVIDER_NAME", args.provider_name),
        "region": None,
        "cache_dir": args.cache_dir,
        "screen_size": (args.screen_width, args.screen_height),
        "headless": args.headless,
        "os_type": args.os_type,
        "require_a11y_tree": require_a11y_tree,
        "require_terminal": require_terminal,
        "enable_proxy": bool(args.enable_proxy),
        "client_password": args.client_password,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OSWorld world sidecar")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18091)
    p.add_argument("--provider_name", default="docker")
    p.add_argument("--path_to_vm", default=None)
    p.add_argument("--snapshot_name", default="init_state")
    p.add_argument("--action_space", default="pyautogui",
                   choices=["pyautogui", "computer_13", "claude_computer_use"])
    p.add_argument("--observation_type", default="screenshot",
                   choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som", "terminal"])
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--screen_width", type=int, default=int(os.environ.get("SCREEN_WIDTH", 1920)))
    p.add_argument("--screen_height", type=int, default=int(os.environ.get("SCREEN_HEIGHT", 1080)))
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--os_type", default="Ubuntu")
    p.add_argument("--enable_proxy", action="store_true", default=False)
    p.add_argument("--client_password", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env_kwargs = build_env_kwargs(args)
    # make_driver builds the DesktopEnv lazily; call the factory once for this sidecar.
    adapter = OSWorldWorldAdapter.make_driver(env_kwargs, action_space=args.action_space)()
    app = create_sidecar_app(adapter, name="osworld-sidecar")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
