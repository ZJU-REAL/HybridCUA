from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if TYPE_CHECKING:
    from desktop_env.desktop_env import DesktopEnv


TASK_ID = "bb5e4c0d-f964-439c-97b6-bdb9747de3f4"
TASK_DOMAIN = "chrome"
TASK_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation_examples",
    "examples",
    TASK_DOMAIN,
    f"{TASK_ID}.json",
)

SCROLL_DOWN_ACTION = {"action_type": "SCROLL", "dx": 0, "dy": -5}

FIXED_ACTIONS: List[Dict[str, Any]] = [
    {
        "label": "1. CLICK(1343, 94){button: LEFT, num_clicks: 1}",
        "action": {"action_type": "CLICK", "x": 1343, "y": 94, "button": "left", "num_clicks": 1},
    },
    {
        "label": "2. MOVE_TO(1016, 560)",
        "action": {"action_type": "MOVE_TO", "x": 1016, "y": 560},
    },
    {
        "label": "3. SCROLL_DOWN",
        "action": copy.deepcopy(SCROLL_DOWN_ACTION),
    },
    {
        "label": "4. CLICK(989, 662){button: LEFT, num_clicks: 1}",
        "action": {"action_type": "CLICK", "x": 989, "y": 662, "button": "left", "num_clicks": 1},
    },
    {
        "label": "5. CLICK(182, 396){button: LEFT, num_clicks: 1}",
        "action": {"action_type": "CLICK", "x": 182, "y": 396, "button": "left", "num_clicks": 1},
    },
    {
        "label": "6. MOVE_TO(1093, 304)",
        "action": {"action_type": "MOVE_TO", "x": 1093, "y": 304},
    },
    {
        "label": "7. DRAG_TO(1091, 307)",
        "action": {"action_type": "DRAG_TO", "x": 1091, "y": 307},
    },
    {
        "label": "8. MOVE_TO(666, 663)",
        "action": {"action_type": "MOVE_TO", "x": 666, "y": 663},
    },
    {
        "label": "9. SCROLL_DOWN",
        "action": copy.deepcopy(SCROLL_DOWN_ACTION),
    },
    {
        "label": "10. MOVE_TO(665, 665)",
        "action": {"action_type": "MOVE_TO", "x": 665, "y": 665},
    },
    {
        "label": "11. SCROLL_DOWN",
        "action": copy.deepcopy(SCROLL_DOWN_ACTION),
    },
    {
        "label": "12. MOVE_TO(1066, 575)",
        "action": {"action_type": "MOVE_TO", "x": 1066, "y": 575},
    },
    {
        "label": "13. DRAG_TO(1066, 576)",
        "action": {"action_type": "DRAG_TO", "x": 1066, "y": 576},
    },
    {
        "label": "14. CLICK(1018, 608){button: LEFT, num_clicks: 1}",
        "action": {"action_type": "CLICK", "x": 1018, "y": 608, "button": "left", "num_clicks": 1},
    },
    {
        "label": "15. MOVE_TO(80, 550)",
        "action": {"action_type": "MOVE_TO", "x": 80, "y": 550},
    },
    {
        "label": "16. SCROLL_DOWN",
        "action": copy.deepcopy(SCROLL_DOWN_ACTION),
    },
    {
        "label": "17. MOVE_TO(80, 552)",
        "action": {"action_type": "MOVE_TO", "x": 80, "y": 552},
    },
    {
        "label": "18. SCROLL_DOWN",
        "action": copy.deepcopy(SCROLL_DOWN_ACTION),
    },
    {
        "label": "19. MOVE_TO(75, 562)",
        "action": {"action_type": "MOVE_TO", "x": 75, "y": 562},
    },
    {
        "label": "20. SCROLL_DOWN",
        "action": copy.deepcopy(SCROLL_DOWN_ACTION),
    },
    {
        "label": "21. MOVE_TO(74, 565)",
        "action": {"action_type": "MOVE_TO", "x": 74, "y": 565},
    },
    {
        "label": "22. SCROLL_DOWN",
        "action": copy.deepcopy(SCROLL_DOWN_ACTION),
    },
    {
        "label": "23. CLICK(49, 662){button: LEFT, num_clicks: 1}",
        "action": {"action_type": "CLICK", "x": 49, "y": 662, "button": "left", "num_clicks": 1},
    },
    {
        "label": "24. CLICK(58, 531){button: LEFT, num_clicks: 1}",
        "action": {"action_type": "CLICK", "x": 58, "y": 531, "button": "left", "num_clicks": 1},
    },
    {
        "label": "25. MOVE_TO(168, 102)",
        "action": {"action_type": "MOVE_TO", "x": 168, "y": 102},
    },
    {
        "label": "26. MOUSE_DOWN{button: left}",
        "action": {"action_type": "MOUSE_DOWN", "button": "left"},
    },
]


logger = logging.getLogger("desktopenv.fixed_task_multienv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel fixed-task stress test using DesktopEnv directly, without the HTTP server."
    )
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel workers/environments.")
    parser.add_argument("--provider-name", type=str, default="docker")
    parser.add_argument("--os-type", type=str, default="Ubuntu")
    parser.add_argument("--client-password", type=str, default="")
    parser.add_argument("--path-to-vm", type=str, default=None)
    parser.add_argument("--snapshot-name", type=str, default="init_state")
    parser.add_argument("--screen-width", type=int, default=1920)
    parser.add_argument("--screen-height", type=int, default=1080)
    parser.add_argument("--pause", type=float, default=1.0, help="Pause passed to env.step(..., pause=...).")
    parser.add_argument("--settle-after-reset", type=float, default=8.0)
    parser.add_argument("--settle-before-evaluate", type=float, default=2.0)
    parser.add_argument(
        "--result-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "results_fixed_task_direct"),
    )
    parser.add_argument("--name-prefix", type=str, default="fixed-task-direct")
    parser.add_argument("--keep-environments", action="store_true")
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(processName)s %(name)s: %(message)s",
    )


def load_task_config(task_path: str) -> Dict[str, Any]:
    with open(task_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_bytes(path: str, content: bytes) -> None:
    with open(path, "wb") as f:
        f.write(content)


def get_screenshots_dir(worker_dir: str) -> str:
    screenshots_dir = os.path.join(worker_dir, "screenshots")
    ensure_dir(screenshots_dir)
    return screenshots_dir


def build_env_kwargs(args: argparse.Namespace, worker_idx: int, run_dir: str) -> Dict[str, Any]:
    env_kwargs: Dict[str, Any] = {
        "provider_name": args.provider_name,
        "action_space": "computer_13",
        "os_type": args.os_type,
        "headless": args.headless,
        "screen_size": (args.screen_width, args.screen_height),
        "require_a11y_tree": False,
        "require_terminal": False,
        "snapshot_name": args.snapshot_name,
        "cache_dir": os.path.join(run_dir, "cache", f"worker_{worker_idx:03d}"),
    }
    if args.client_password:
        env_kwargs["client_password"] = args.client_password
    if args.path_to_vm:
        env_kwargs["path_to_vm"] = args.path_to_vm
    return env_kwargs


def extract_env_metadata(env: "DesktopEnv") -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "vm_ip": getattr(env, "vm_ip", None),
        "server_port": getattr(env, "server_port", None),
        "chromium_port": getattr(env, "chromium_port", None),
        "vnc_port": getattr(env, "vnc_port", None),
        "vlc_port": getattr(env, "vlc_port", None),
        "path_to_vm": getattr(env, "path_to_vm", None),
    }
    provider = getattr(env, "provider", None)
    container = getattr(provider, "container", None)
    if container is not None:
        try:
            container.reload()
        except Exception:
            pass
        metadata["docker"] = {
            "container_id": getattr(container, "id", "")[:12] or None,
            "status": getattr(container, "status", None),
            "name": getattr(container, "name", None),
        }
    return metadata


def init_worker_record(worker_idx: int, args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "worker_index": worker_idx,
        "name": f"{args.name_prefix}-{worker_idx:03d}",
        "task_id": TASK_ID,
        "task_domain": TASK_DOMAIN,
        "created_at": dt.datetime.now().isoformat(),
        "provider_name": args.provider_name,
        "status": "running",
        "timings": {},
        "environment": {},
        "action_trace": [],
        "artifacts": {
            "screenshots_dir": "screenshots",
            "initial_screenshot": None,
            "final_screenshot": None,
            "recording_file": None,
        },
        "evaluate_result": None,
        "error": None,
    }


def save_observation_screenshot(
    observation: Optional[Dict[str, Any]],
    worker_dir: str,
    filename: str,
) -> Optional[str]:
    if not isinstance(observation, dict):
        return None
    screenshot = observation.get("screenshot")
    if not screenshot:
        return None
    relative_path = os.path.join("screenshots", filename)
    write_bytes(os.path.join(worker_dir, relative_path), screenshot)
    return relative_path


def capture_current_screenshot(
    env: Optional["DesktopEnv"],
    worker_dir: str,
    filename: str,
) -> Optional[str]:
    if env is None or not hasattr(env, "controller") or env.controller is None:
        return None
    try:
        screenshot = env.controller.get_screenshot()
    except Exception:
        logger.exception("Worker current screenshot capture failed.")
        return None
    if not screenshot:
        return None
    relative_path = os.path.join("screenshots", filename)
    write_bytes(os.path.join(worker_dir, relative_path), screenshot)
    return relative_path


def worker_run(worker_idx: int, args_dict: Dict[str, Any], task_config: Dict[str, Any], run_dir: str) -> None:
    args = argparse.Namespace(**args_dict)
    configure_logging(args.log_level)

    worker_dir = os.path.join(run_dir, f"worker_{worker_idx:03d}")
    ensure_dir(worker_dir)
    get_screenshots_dir(worker_dir)

    record = init_worker_record(worker_idx, args)
    env: Optional[DesktopEnv] = None
    recording_started = False

    try:
        from desktop_env.desktop_env import DesktopEnv

        logger.info("Worker %03d creating DesktopEnv...", worker_idx)
        create_started = time.perf_counter()
        env = DesktopEnv(**build_env_kwargs(args, worker_idx, run_dir))
        record["timings"]["create_seconds"] = round(time.perf_counter() - create_started, 4)
        record["environment"] = extract_env_metadata(env)

        logger.info("Worker %03d resetting task %s...", worker_idx, TASK_ID)
        reset_started = time.perf_counter()
        reset_obs = env.reset(task_config=copy.deepcopy(task_config))
        record["timings"]["reset_seconds"] = round(time.perf_counter() - reset_started, 4)

        if args.settle_after_reset > 0:
            time.sleep(args.settle_after_reset)
        record["artifacts"]["initial_screenshot"] = (
            capture_current_screenshot(env, worker_dir, "initial.png")
            or save_observation_screenshot(reset_obs, worker_dir, "initial.png")
        )

        if hasattr(env, "controller") and env.controller is not None:
            try:
                env.controller.start_recording()
                recording_started = True
            except Exception:
                logger.exception("Worker %03d failed to start recording.", worker_idx)
                record["recording_error"] = "start_recording failed"

        for action_idx, item in enumerate(FIXED_ACTIONS, start=1):
            logger.info(
                "Worker %03d action %02d/%02d: %s",
                worker_idx,
                action_idx,
                len(FIXED_ACTIONS),
                item["label"],
            )
            step_started = time.perf_counter()
            obs, reward, done, info = env.step(copy.deepcopy(item["action"]), pause=args.pause)
            screenshot_file = save_observation_screenshot(
                obs,
                worker_dir,
                f"step_{action_idx:02d}.png",
            )
            record["action_trace"].append(
                {
                    "index": action_idx,
                    "label": item["label"],
                    "action": copy.deepcopy(item["action"]),
                    "elapsed_seconds": round(time.perf_counter() - step_started, 4),
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": screenshot_file,
                }
            )

        if args.settle_before_evaluate > 0:
            time.sleep(args.settle_before_evaluate)

        record["artifacts"]["final_screenshot"] = capture_current_screenshot(
            env,
            worker_dir,
            "final.png",
        )

        logger.info("Worker %03d evaluating task result...", worker_idx)
        evaluate_started = time.perf_counter()
        record["evaluate_result"] = float(env.evaluate())
        record["timings"]["evaluate_seconds"] = round(time.perf_counter() - evaluate_started, 4)
        record["status"] = "ok"
    except Exception as exc:
        logger.exception("Worker %03d failed.", worker_idx)
        record["status"] = "error"
        record["error"] = str(exc)
        record["traceback"] = traceback.format_exc()
        if record["artifacts"]["final_screenshot"] is None:
            record["artifacts"]["final_screenshot"] = capture_current_screenshot(
                env,
                worker_dir,
                "final.png",
            )
    finally:
        if recording_started and env is not None and hasattr(env, "controller") and env.controller is not None:
            try:
                env.controller.end_recording(os.path.join(worker_dir, "recording.mp4"))
                record["artifacts"]["recording_file"] = "recording.mp4"
            except Exception:
                logger.exception("Worker %03d failed to save recording.", worker_idx)
                record["recording_error"] = "end_recording failed"

        if env is not None:
            if args.keep_environments:
                record["kept_environment"] = True
                record["environment"] = extract_env_metadata(env)
            else:
                try:
                    close_started = time.perf_counter()
                    env.close()
                    record["timings"]["close_seconds"] = round(time.perf_counter() - close_started, 4)
                except Exception as exc:
                    logger.exception("Worker %03d failed during env.close().", worker_idx)
                    record["close_error"] = str(exc)

        write_json(os.path.join(worker_dir, "result.json"), record)


def collect_worker_results(num_envs: int, run_dir: str, exitcodes: Dict[int, Optional[int]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for worker_idx in range(1, num_envs + 1):
        result_path = os.path.join(run_dir, f"worker_{worker_idx:03d}", "result.json")
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                results.append(json.load(f))
            continue

        results.append(
            {
                "worker_index": worker_idx,
                "name": f"worker_{worker_idx:03d}",
                "task_id": TASK_ID,
                "task_domain": TASK_DOMAIN,
                "status": "error",
                "timings": {},
                "environment": {},
                "action_trace": [],
                "artifacts": {
                    "screenshots_dir": "screenshots",
                    "initial_screenshot": None,
                    "final_screenshot": None,
                    "recording_file": None,
                },
                "evaluate_result": None,
                "error": f"Worker exited without writing result.json (exitcode={exitcodes.get(worker_idx)})",
            }
        )

    results.sort(key=lambda item: item["worker_index"])
    return results


def summarize_results(results: List[Dict[str, Any]], args: argparse.Namespace, run_dir: str) -> Dict[str, Any]:
    ok_results = [item for item in results if item["status"] == "ok"]
    eval_scores = [float(item["evaluate_result"]) for item in ok_results if item.get("evaluate_result") is not None]
    create_times = [item["timings"].get("create_seconds") for item in results if item["timings"].get("create_seconds") is not None]
    reset_times = [item["timings"].get("reset_seconds") for item in results if item["timings"].get("reset_seconds") is not None]
    eval_times = [item["timings"].get("evaluate_seconds") for item in results if item["timings"].get("evaluate_seconds") is not None]
    close_times = [item["timings"].get("close_seconds") for item in results if item["timings"].get("close_seconds") is not None]
    action_times = [
        trace["elapsed_seconds"]
        for item in results
        for trace in item.get("action_trace", [])
        if trace.get("elapsed_seconds") is not None
    ]

    def _avg(values: List[float]) -> Optional[float]:
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    return {
        "task_id": TASK_ID,
        "task_domain": TASK_DOMAIN,
        "num_envs": args.num_envs,
        "provider_name": args.provider_name,
        "os_type": args.os_type,
        "headless": args.headless,
        "pause": args.pause,
        "settle_after_reset": args.settle_after_reset,
        "settle_before_evaluate": args.settle_before_evaluate,
        "keep_environments": args.keep_environments,
        "worker_count": len(results),
        "ok_count": len(ok_results),
        "error_count": len(results) - len(ok_results),
        "avg_evaluate_result": _avg(eval_scores),
        "avg_create_seconds": _avg(create_times),
        "avg_reset_seconds": _avg(reset_times),
        "avg_action_seconds": _avg(action_times),
        "avg_evaluate_seconds": _avg(eval_times),
        "avg_close_seconds": _avg(close_times),
        "run_dir": run_dir,
        "workers": results,
    }


def terminate_processes(processes: List[mp.Process]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5)
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    if args.num_envs < 1:
        raise ValueError("--num-envs must be >= 1")

    task_config = load_task_config(TASK_PATH)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.result_dir, TASK_ID, timestamp)
    ensure_dir(run_dir)

    with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    logger.info(
        "Starting direct fixed-task run: task=%s num_envs=%d provider=%s",
        TASK_ID,
        args.num_envs,
        args.provider_name,
    )

    ctx = mp.get_context("spawn")
    processes: List[mp.Process] = []
    exitcodes: Dict[int, Optional[int]] = {}
    args_dict = vars(args).copy()

    try:
        for worker_idx in range(1, args.num_envs + 1):
            process = ctx.Process(
                target=worker_run,
                args=(worker_idx, args_dict, copy.deepcopy(task_config), run_dir),
                name=f"FixedTaskWorker-{worker_idx:03d}",
            )
            process.start()
            processes.append(process)

        for worker_idx, process in enumerate(processes, start=1):
            process.join()
            exitcodes[worker_idx] = process.exitcode
    except KeyboardInterrupt:
        logger.warning("Interrupted by user, terminating workers...")
        terminate_processes(processes)
        for worker_idx, process in enumerate(processes, start=1):
            exitcodes[worker_idx] = process.exitcode
    finally:
        for worker_idx, process in enumerate(processes, start=1):
            exitcodes.setdefault(worker_idx, process.exitcode)

    results = collect_worker_results(args.num_envs, run_dir, exitcodes)
    summary = summarize_results(results, args, run_dir)
    write_json(os.path.join(run_dir, "summary.json"), summary)

    logger.info(
        "Completed direct fixed-task run. ok=%d error=%d avg_result=%s summary=%s",
        summary["ok_count"],
        summary["error_count"],
        summary["avg_evaluate_result"],
        os.path.join(run_dir, "summary.json"),
    )

    if summary["error_count"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
