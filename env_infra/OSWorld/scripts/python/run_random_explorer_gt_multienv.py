from __future__ import annotations

import argparse
import collections
import copy
import datetime as dt
import html
import json
import logging
import multiprocessing as mp
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if TYPE_CHECKING:
    from desktop_env.desktop_env import DesktopEnv


DEFAULT_RECORDINGS_URL = "https://os-world.github.io/static/data/test_small_recording.json"
DEFAULT_ANNOTATIONS_URL = "https://os-world.github.io/static/data/test_small_annotations_2.json"
DEFAULT_MANIFEST_CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache", "osworld_explorer")
DEFAULT_RESULT_DIR = os.path.join(PROJECT_ROOT, "results_explorer_random_gt")
EXAMPLES_ROOT = os.path.join(PROJECT_ROOT, "evaluation_examples", "examples")
EXAMPLES_WINDOWS_ROOT = os.path.join(PROJECT_ROOT, "evaluation_examples", "examples_windows")
SCROLL_DOWN_ACTION = {"action_type": "SCROLL", "dx": 0, "dy": -5}
SCROLL_UP_ACTION = {"action_type": "SCROLL", "dx": 0, "dy": 5}
CONTROL_CHAR_TO_LETTER = {chr(index): chr(ord("a") + index - 1) for index in range(1, 27)}
DISPLAY_EMOJI_TOKENS = ["\ud83d\udd79\ufe0f", "\ud83d\udd3d", "\ud83d\udd3c", "\u2328\ufe0f", "\ud83d\udcbb"]
WINDOWS_ONLY_DOMAINS = {"Windows-Workflow", "Excel", "Word", "PowerPoint"}


logger = logging.getLogger("desktopenv.explorer_gt_multienv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel random-task replay using OSWorld explorer tasks and their displayed ground-truth actions."
    )
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel workers/environments.")
    parser.add_argument("--provider-name", type=str, default="docker")
    parser.add_argument("--os-type", type=str, default="Ubuntu", choices=["Ubuntu", "Windows"])
    parser.add_argument("--client-password", type=str, default="")
    parser.add_argument("--path-to-vm", type=str, default=None)
    parser.add_argument("--snapshot-name", type=str, default="init_state")
    parser.add_argument("--screen-width", type=int, default=1920)
    parser.add_argument("--screen-height", type=int, default=1080)
    parser.add_argument("--pause", type=float, default=1.0, help="Pause passed to env.step(..., pause=...).")
    parser.add_argument("--settle-after-reset", type=float, default=8.0)
    parser.add_argument("--settle-before-evaluate", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for task sampling.")
    parser.add_argument("--domains", nargs="*", default=None, help="Optional explorer domains to sample from.")
    parser.add_argument(
        "--sample-with-replacement",
        action="store_true",
        help="Sample tasks with replacement. Enabled automatically if the pool is smaller than --num-envs.",
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=None,
        help="Optional cap on the number of parsed GT actions per task after normalization.",
    )
    parser.add_argument(
        "--recordings-url",
        type=str,
        default=DEFAULT_RECORDINGS_URL,
        help="Explorer manifest URL that lists the displayed task subset.",
    )
    parser.add_argument(
        "--annotations-url",
        type=str,
        default=DEFAULT_ANNOTATIONS_URL,
        help="Explorer annotation URL that contains instruction/video/action strings.",
    )
    parser.add_argument(
        "--recordings-path",
        type=str,
        default=None,
        help="Optional local path to a downloaded copy of the explorer recording manifest.",
    )
    parser.add_argument(
        "--annotations-path",
        type=str,
        default=None,
        help="Optional local path to a downloaded copy of the explorer annotations.",
    )
    parser.add_argument(
        "--manifest-cache-dir",
        type=str,
        default=DEFAULT_MANIFEST_CACHE_DIR,
        help="Directory used to cache explorer manifests fetched from the website.",
    )
    parser.add_argument("--result-dir", type=str, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--name-prefix", type=str, default="explorer-gt")
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


def download_text(url: str, timeout_seconds: int = 30) -> str:
    request = Request(url, headers={"User-Agent": "OSWorldExplorerGTRunner/1.0"})
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read().decode("utf-8")


def load_json_resource(
    local_path: Optional[str],
    url: str,
    cache_path: str,
) -> Any:
    if local_path:
        with open(local_path, "r", encoding="utf-8-sig") as f:
            return json.load(f)

    ensure_dir(os.path.dirname(cache_path))
    try:
        content = download_text(url)
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(content)
        return json.loads(content)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        if os.path.exists(cache_path):
            logger.warning("Falling back to cached explorer manifest %s because fetch failed: %s", cache_path, exc)
            with open(cache_path, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        raise


def build_local_task_index() -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    for root in (EXAMPLES_ROOT, EXAMPLES_WINDOWS_ROOT):
        if not os.path.isdir(root):
            continue
        for path in Path(root).rglob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                logger.exception("Failed to read task config while indexing: %s", path)
                continue

            task_id = data.get("id")
            if not task_id:
                continue
            index.setdefault(task_id, []).append(str(path))
    return index


def infer_task_os_type(task_config_path: str) -> str:
    normalized = os.path.normpath(task_config_path)
    if os.path.normpath(EXAMPLES_WINDOWS_ROOT) in normalized:
        return "Windows"
    return "Ubuntu"


def normalize_html_action_text(actions_html: str) -> List[str]:
    text = actions_html.replace("\\>", ">")
    text = text.replace("<br />", "\n").replace("<br/>", "\n").replace("<br>", "\n")
    text = re.sub(r"<img[^>]*>", "", text)
    text = re.sub(r"</?font[^>]*>", "", text)
    text = html.unescape(text)
    for token in DISPLAY_EMOJI_TOKENS:
        text = text.replace(token, "")
    lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        line = re.sub(r"^\d+\.\s*", "", line)
        line = "".join(ch for ch in line if ord(ch) < 128)
        if line:
            lines.append(line)
    return lines


def parse_brace_parameters(block: Optional[str]) -> Dict[str, Any]:
    if not block:
        return {}
    parameters: Dict[str, Any] = {}
    for part in block.split(","):
        item = part.strip()
        if not item or ":" not in item:
            continue
        key, value = item.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "button":
            parameters[key] = value.lower()
        elif key == "num_clicks":
            parameters[key] = int(value)
        else:
            parameters[key] = value
    return parameters


def normalize_key_name(key: str) -> str:
    if key == " ":
        return key
    stripped = key.strip()
    if len(stripped) == 1 and stripped.isalpha():
        return stripped.lower()
    return stripped.lower()


def build_ctrl_hotkey(letter: str) -> Dict[str, Any]:
    return {"action_type": "HOTKEY", "keys": ["ctrl", letter]}


def parse_press_token(token: str) -> Dict[str, Any]:
    if len(token) == 1 and token in CONTROL_CHAR_TO_LETTER:
        return build_ctrl_hotkey(CONTROL_CHAR_TO_LETTER[token])
    return {"action_type": "PRESS", "key": normalize_key_name(token)}


def expand_typing_text(text: str) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    buffer: List[str] = []
    index = 0
    while index < len(text):
        if text.startswith("ctrl_l", index):
            if buffer:
                expanded.append({"action_type": "TYPING", "text": "".join(buffer)})
                buffer = []
            expanded.append(build_ctrl_hotkey("l"))
            index += len("ctrl_l")
            continue

        current = text[index]
        if current in CONTROL_CHAR_TO_LETTER:
            if buffer:
                expanded.append({"action_type": "TYPING", "text": "".join(buffer)})
                buffer = []
            expanded.append(build_ctrl_hotkey(CONTROL_CHAR_TO_LETTER[current]))
            index += 1
            continue

        buffer.append(current)
        index += 1

    if buffer:
        expanded.append({"action_type": "TYPING", "text": "".join(buffer)})

    return expanded or [{"action_type": "TYPING", "text": text}]


def parse_action_line(line: str) -> List[Dict[str, Any]]:
    if line == "SCROLL_DOWN":
        return [copy.deepcopy(SCROLL_DOWN_ACTION)]
    if line == "SCROLL_UP":
        return [copy.deepcopy(SCROLL_UP_ACTION)]

    click_match = re.fullmatch(r"CLICK\(([-\d.]+),\s*([-\d.]+)\)(?:\{([^}]*)\})?", line)
    if click_match:
        action: Dict[str, Any] = {
            "action_type": "CLICK",
            "x": int(float(click_match.group(1))),
            "y": int(float(click_match.group(2))),
        }
        action.update(parse_brace_parameters(click_match.group(3)))
        return [action]

    move_match = re.fullmatch(r"MOVE_TO\(([-\d.]+),\s*([-\d.]+)\)", line)
    if move_match:
        return [{
            "action_type": "MOVE_TO",
            "x": int(float(move_match.group(1))),
            "y": int(float(move_match.group(2))),
        }]

    drag_match = re.fullmatch(r"DRAG_TO\(([-\d.]+),\s*([-\d.]+)\)", line)
    if drag_match:
        return [{
            "action_type": "DRAG_TO",
            "x": int(float(drag_match.group(1))),
            "y": int(float(drag_match.group(2))),
        }]

    mouse_down_match = re.fullmatch(r"MOUSE_DOWN(?:\{([^}]*)\})?", line)
    if mouse_down_match:
        action = {"action_type": "MOUSE_DOWN"}
        action.update(parse_brace_parameters(mouse_down_match.group(1)))
        return [action]

    mouse_up_match = re.fullmatch(r"MOUSE_UP(?:\{([^}]*)\})?", line)
    if mouse_up_match:
        action = {"action_type": "MOUSE_UP"}
        action.update(parse_brace_parameters(mouse_up_match.group(1)))
        return [action]

    press_match = re.fullmatch(r"PRESS\((.*)\)", line)
    if press_match:
        return [parse_press_token(press_match.group(1))]

    typing_match = re.fullmatch(r'TYPING\("(.*)"\)', line)
    if typing_match:
        return expand_typing_text(typing_match.group(1))

    raise ValueError(f"Unsupported explorer action line: {line}")


def parse_actions_html(actions_html: str, max_actions: Optional[int] = None) -> Dict[str, Any]:
    cleaned_lines = normalize_html_action_text(actions_html)
    parsed_items: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for line_index, line in enumerate(cleaned_lines, start=1):
        actions = parse_action_line(line)
        if len(actions) > 1:
            warnings.append(f"Line {line_index} expanded into {len(actions)} actions: {line}")
        for action_index, action in enumerate(actions, start=1):
            label = line if len(actions) == 1 else f"{line} [expanded {action_index}/{len(actions)}]"
            parsed_items.append(
                {
                    "index": len(parsed_items) + 1,
                    "source_line_index": line_index,
                    "label": label,
                    "action": action,
                }
            )

    if max_actions is not None:
        parsed_items = parsed_items[:max_actions]

    return {
        "display_lines": cleaned_lines,
        "actions": parsed_items,
        "warnings": warnings,
        "display_line_count": len(cleaned_lines),
        "parsed_action_count": len(parsed_items),
    }


def load_task_config(task_path: str, explorer_instruction: Optional[str] = None) -> Dict[str, Any]:
    with open(task_path, "r", encoding="utf-8") as f:
        task_config = json.load(f)
    if explorer_instruction:
        task_config["instruction"] = explorer_instruction
    return task_config


def build_explorer_task_pool(args: argparse.Namespace) -> List[Dict[str, Any]]:
    recordings = load_json_resource(
        local_path=args.recordings_path,
        url=args.recordings_url,
        cache_path=os.path.join(args.manifest_cache_dir, "test_small_recording.json"),
    )
    annotations = load_json_resource(
        local_path=args.annotations_path,
        url=args.annotations_url,
        cache_path=os.path.join(args.manifest_cache_dir, "test_small_annotations_2.json"),
    )

    explorer_ids: List[str] = []
    for task_ids in recordings.values():
        explorer_ids.extend(task_ids)
    allowed_ids = set(explorer_ids)
    annotation_by_id = {item["id"]: item for item in annotations if item.get("id") in allowed_ids}
    local_task_index = build_local_task_index()

    domain_filter = {domain.casefold() for domain in args.domains} if args.domains else None
    pool: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for task_id in sorted(allowed_ids):
        annotation = annotation_by_id.get(task_id)
        if annotation is None:
            skipped.append(f"{task_id}: missing from annotations")
            continue

        explorer_domain = annotation.get("domain")
        if args.os_type == "Ubuntu" and explorer_domain in WINDOWS_ONLY_DOMAINS:
            continue
        if args.os_type == "Windows" and explorer_domain not in WINDOWS_ONLY_DOMAINS:
            continue

        candidate_paths = local_task_index.get(task_id, [])
        matching_paths = [path for path in candidate_paths if infer_task_os_type(path) == args.os_type]
        if not matching_paths:
            skipped.append(f"{task_id}: missing local task config")
            continue

        config_path = sorted(matching_paths)[0]
        task_os_type = infer_task_os_type(config_path)

        if domain_filter and annotation.get("domain", "").casefold() not in domain_filter:
            continue

        try:
            parsed = parse_actions_html(annotation.get("actions", ""), max_actions=args.max_actions)
        except Exception as exc:
            skipped.append(f"{task_id}: failed to parse explorer actions ({exc})")
            continue

        if not parsed["actions"]:
            skipped.append(f"{task_id}: no parsed actions")
            continue

        pool.append(
            {
                "task_id": task_id,
                "domain": annotation.get("domain"),
                "instruction": annotation.get("instruction"),
                "video_url": annotation.get("video"),
                "task_config_path": config_path,
                "task_os_type": task_os_type,
                "display_line_count": parsed["display_line_count"],
                "parsed_action_count": parsed["parsed_action_count"],
                "actions": parsed["actions"],
                "parse_warnings": parsed["warnings"],
                "display_lines": parsed["display_lines"],
            }
        )

    if skipped:
        logger.info("Skipped %d explorer tasks while building the pool.", len(skipped))

    return pool


def choose_random_tasks(pool: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    if not pool:
        raise RuntimeError("No explorer tasks are available after filtering.")

    should_replace = args.sample_with_replacement or args.num_envs > len(pool)
    rng = random.Random(args.seed)

    if should_replace:
        return [copy.deepcopy(rng.choice(pool)) for _ in range(args.num_envs)]
    return [copy.deepcopy(item) for item in rng.sample(pool, k=args.num_envs)]


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


def init_worker_record(worker_idx: int, args: argparse.Namespace, task_spec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "worker_index": worker_idx,
        "name": f"{args.name_prefix}-{worker_idx:03d}",
        "task_id": task_spec["task_id"],
        "task_domain": task_spec.get("domain"),
        "task_instruction": task_spec.get("instruction"),
        "task_config_path": task_spec.get("task_config_path"),
        "task_video_url": task_spec.get("video_url"),
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
        "task_source": "OSWorld explorer",
        "gt_display_line_count": task_spec.get("display_line_count"),
        "gt_parsed_action_count": task_spec.get("parsed_action_count"),
        "parse_warnings": task_spec.get("parse_warnings", []),
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


def worker_run(worker_idx: int, args_dict: Dict[str, Any], task_spec: Dict[str, Any], run_dir: str) -> None:
    args = argparse.Namespace(**args_dict)
    configure_logging(args.log_level)

    worker_dir = os.path.join(run_dir, f"worker_{worker_idx:03d}")
    ensure_dir(worker_dir)
    get_screenshots_dir(worker_dir)

    record = init_worker_record(worker_idx, args, task_spec)
    env: Optional[DesktopEnv] = None
    recording_started = False

    try:
        from desktop_env.desktop_env import DesktopEnv

        task_config = load_task_config(task_spec["task_config_path"], explorer_instruction=task_spec.get("instruction"))

        logger.info("Worker %03d creating DesktopEnv for task %s...", worker_idx, task_spec["task_id"])
        create_started = time.perf_counter()
        env = DesktopEnv(**build_env_kwargs(args, worker_idx, run_dir))
        record["timings"]["create_seconds"] = round(time.perf_counter() - create_started, 4)
        record["environment"] = extract_env_metadata(env)

        logger.info("Worker %03d resetting explorer task %s...", worker_idx, task_spec["task_id"])
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

        for action_item in task_spec["actions"]:
            logger.info(
                "Worker %03d task %s action %03d/%03d: %s",
                worker_idx,
                task_spec["task_id"],
                action_item["index"],
                len(task_spec["actions"]),
                action_item["label"],
            )
            step_started = time.perf_counter()
            obs, reward, done, info = env.step(copy.deepcopy(action_item["action"]), pause=args.pause)
            screenshot_file = save_observation_screenshot(
                obs,
                worker_dir,
                f"step_{action_item['index']:03d}.png",
            )
            record["action_trace"].append(
                {
                    "index": action_item["index"],
                    "source_line_index": action_item["source_line_index"],
                    "label": action_item["label"],
                    "action": copy.deepcopy(action_item["action"]),
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

        logger.info("Worker %03d evaluating explorer task %s...", worker_idx, task_spec["task_id"])
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
            recording_path = os.path.join(worker_dir, "recording.mp4")
            try:
                env.controller.end_recording(recording_path)
                if os.path.exists(recording_path) and os.path.getsize(recording_path) > 0:
                    record["artifacts"]["recording_file"] = "recording.mp4"
                else:
                    record["recording_error"] = "recording file missing or empty after end_recording"
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


def average(values: Iterable[float]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return round(sum(filtered) / len(filtered), 4)


def summarize_results(
    results: List[Dict[str, Any]],
    selected_tasks: List[Dict[str, Any]],
    args: argparse.Namespace,
    run_dir: str,
    seed_used: int,
) -> Dict[str, Any]:
    ok_results = [item for item in results if item["status"] == "ok"]
    eval_scores = [item.get("evaluate_result") for item in ok_results if item.get("evaluate_result") is not None]
    create_times = [item["timings"].get("create_seconds") for item in results]
    reset_times = [item["timings"].get("reset_seconds") for item in results]
    eval_times = [item["timings"].get("evaluate_seconds") for item in results]
    close_times = [item["timings"].get("close_seconds") for item in results]
    action_times = [
        trace.get("elapsed_seconds")
        for item in results
        for trace in item.get("action_trace", [])
        if trace.get("elapsed_seconds") is not None
    ]
    domain_counts = collections.Counter(task.get("domain") for task in selected_tasks)

    return {
        "num_envs": args.num_envs,
        "provider_name": args.provider_name,
        "os_type": args.os_type,
        "headless": args.headless,
        "pause": args.pause,
        "settle_after_reset": args.settle_after_reset,
        "settle_before_evaluate": args.settle_before_evaluate,
        "keep_environments": args.keep_environments,
        "seed": seed_used,
        "domains_filter": args.domains,
        "sample_with_replacement": args.sample_with_replacement or args.num_envs > len(set(task["task_id"] for task in selected_tasks)),
        "worker_count": len(results),
        "ok_count": len(ok_results),
        "error_count": len(results) - len(ok_results),
        "avg_evaluate_result": average(eval_scores),
        "avg_create_seconds": average(create_times),
        "avg_reset_seconds": average(reset_times),
        "avg_action_seconds": average(action_times),
        "avg_evaluate_seconds": average(eval_times),
        "avg_close_seconds": average(close_times),
        "selected_task_ids": [task["task_id"] for task in selected_tasks],
        "selected_domains": dict(domain_counts),
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

    seed_used = args.seed if args.seed is not None else int(time.time())
    args.seed = seed_used

    task_pool = build_explorer_task_pool(args)
    if not task_pool:
        raise RuntimeError(
            f"No explorer tasks available for os_type={args.os_type} and domains={args.domains or 'ALL'}."
        )

    selected_tasks = choose_random_tasks(task_pool, args)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.result_dir, args.os_type.lower(), timestamp)
    ensure_dir(run_dir)

    write_json(os.path.join(run_dir, "args.json"), vars(args))
    write_json(
        os.path.join(run_dir, "selected_tasks.json"),
        {
            "seed": seed_used,
            "count": len(selected_tasks),
            "tasks": [
                {
                    "task_id": task["task_id"],
                    "domain": task["domain"],
                    "instruction": task["instruction"],
                    "video_url": task["video_url"],
                    "task_config_path": task["task_config_path"],
                    "display_line_count": task["display_line_count"],
                    "parsed_action_count": task["parsed_action_count"],
                    "parse_warnings": task["parse_warnings"],
                }
                for task in selected_tasks
            ],
        },
    )

    logger.info(
        "Starting explorer GT random run: os_type=%s num_envs=%d pool=%d seed=%d",
        args.os_type,
        args.num_envs,
        len(task_pool),
        seed_used,
    )

    ctx = mp.get_context("spawn")
    processes: List[mp.Process] = []
    exitcodes: Dict[int, Optional[int]] = {}
    args_dict = vars(args).copy()

    try:
        for worker_idx, task_spec in enumerate(selected_tasks, start=1):
            process = ctx.Process(
                target=worker_run,
                args=(worker_idx, args_dict, copy.deepcopy(task_spec), run_dir),
                name=f"ExplorerGTWorker-{worker_idx:03d}",
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
    summary = summarize_results(results, selected_tasks, args, run_dir, seed_used)
    write_json(os.path.join(run_dir, "summary.json"), summary)

    logger.info(
        "Completed explorer GT random run. ok=%d error=%d avg_result=%s summary=%s",
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
