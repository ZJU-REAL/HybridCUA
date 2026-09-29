from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import shutil
import signal
import sys
import time
from multiprocessing import Manager, Process, current_process
from typing import Dict, List, Sequence

# Add project root to path for imports.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
# Also add the repo root (parent of OSWorld/) so `cluster.*` resolves even
# without an editable install of the project.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import lib_run_single
from cluster.client import OSWorldRemoteClient as DesktopEnv
from mm_agents.qwen35vl_agent import Qwen35VLAgent


active_environments = []
processes = []
is_terminating = False
logger = logging.getLogger("desktopenv.experiment")


if os.path.exists(".env"):
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OSWorld Qwen3.5-VL evaluation through OSWorldRemoteClient and the cluster master."
    )

    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument("--headless", action="store_true", help="Run in headless machine")
    parser.add_argument("--action_space", type=str, default="pyautogui", help="Action type")
    parser.add_argument(
        "--observation_type",
        choices=["screenshot"],
        default="screenshot",
        help="Observation type. Qwen35VLAgent currently supports screenshot only.",
    )
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=50)

    parser.add_argument("--test_config_base_dir", type=str, default="evaluation_examples")
    parser.add_argument("--examples_subdir", type=str, default="examples")
    parser.add_argument("--history_n", type=int, default=100)
    parser.add_argument(
        "--coord",
        type=str,
        choices=["absolute", "relative"],
        default="relative",
        help="Coordinate system for agent outputs",
    )
    parser.add_argument("--add_thought_prefix", action="store_true")
    parser.add_argument("--image_max", type=int, default=20)
    parser.add_argument("--fold_size", type=int, default=10)

    parser.add_argument("--model", type=str, default="qwen35-vl")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument(
        "--openai_base_urls",
        type=str,
        default=None,
        help=(
            "Comma-separated OpenAI-compatible base URLs. "
            "Falls back to --base_url, then OPENAI_BASE_URLS, then OPENAI_BASE_URL."
        ),
    )

    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument("--test_all_meta_path", type=str, default="evaluation_examples/test_nogdrive.json")

    parser.add_argument("--result_dir", type=str, default="./results_qwen35vl_remote")
    parser.add_argument("--simple_path", action="store_true")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run in parallel")
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )

    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region for the VM")
    parser.add_argument(
        "--provider_name",
        type=str,
        default="docker_server",
        choices=["aws", "virtualbox", "vmware", "docker", "docker_server", "azure", "aliyun"],
        help="Provider name. Use docker_server for cluster remote mode.",
    )
    parser.add_argument("--client_password", type=str, default="", help="Client password")
    parser.add_argument("--screen_width", type=int, default=1920, help="Screen width")
    parser.add_argument("--screen_height", type=int, default=1080, help="Screen height")
    parser.add_argument(
        "--cluster_url",
        type=str,
        default=os.environ.get("GUI_ENV_SERVER_URL", "http://127.0.0.1:18000"),
        help="Cluster master URL, for example http://master-ip:18000",
    )

    return parser.parse_args()


def setup_logging(log_level_name: str) -> None:
    global logger

    root_logger = logging.getLogger()
    log_level = getattr(logging, log_level_name.upper())
    root_logger.setLevel(log_level)
    if getattr(root_logger, "_qwen35vl_remote_logging_configured", False):
        logger = logging.getLogger("desktopenv.experiment")
        return

    os.makedirs("logs", exist_ok=True)
    datetime_str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

    file_handler = logging.FileHandler(
        os.path.join("logs", f"qwen35vl-remote-normal-{datetime_str}.log"),
        encoding="utf-8",
    )
    debug_handler = logging.FileHandler(
        os.path.join("logs", f"qwen35vl-remote-debug-{datetime_str}.log"),
        encoding="utf-8",
    )
    stdout_handler = logging.StreamHandler(sys.stdout)

    file_handler.setLevel(logging.INFO)
    debug_handler.setLevel(logging.DEBUG)
    stdout_handler.setLevel(log_level)

    formatter = logging.Formatter(
        fmt=(
            "\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s "
            "\x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] "
            "\x1b[0m%(message)s"
        )
    )
    file_handler.setFormatter(formatter)
    debug_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)

    stdout_handler.addFilter(logging.Filter("desktopenv"))

    root_logger.addHandler(file_handler)
    root_logger.addHandler(debug_handler)
    root_logger.addHandler(stdout_handler)
    root_logger._qwen35vl_remote_logging_configured = True

    logger = logging.getLogger("desktopenv.experiment")


def distribute_tasks(test_all_meta: Dict[str, List[str]]) -> List[tuple]:
    all_tasks = []
    for domain, examples in test_all_meta.items():
        for example_id in examples:
            all_tasks.append((domain, example_id))
    return all_tasks


def build_config_file_path(args: argparse.Namespace, domain: str, example_id: str) -> str:
    return os.path.join(
        args.test_config_base_dir,
        args.examples_subdir,
        domain,
        f"{example_id}.json",
    )


def build_example_result_dir(args: argparse.Namespace, domain: str, example_id: str) -> str:
    if args.simple_path:
        return os.path.join(args.result_dir, domain, example_id)
    return os.path.join(
        args.result_dir,
        args.action_space,
        args.observation_type,
        args.model,
        domain,
        example_id,
    )


def build_desktop_env(args: argparse.Namespace) -> DesktopEnv:
    region = args.region
    screen_size = (args.screen_width, args.screen_height)
    snapshot_name = "init_state"
    if args.provider_name == "aws":
        from desktop_env.providers.aws.manager import IMAGE_ID_MAP

        snapshot_name = IMAGE_ID_MAP[region].get(screen_size, IMAGE_ID_MAP[region][(1920, 1080)])

    cluster_url = args.cluster_url or os.environ.get("GUI_ENV_SERVER_URL") or "http://127.0.0.1:18000"
    os.environ["GUI_ENV_SERVER_URL"] = cluster_url
    return DesktopEnv(
        path_to_vm=args.path_to_vm,
        action_space=args.action_space,
        provider_name=args.provider_name,
        region=region,
        snapshot_name=snapshot_name,
        screen_size=screen_size,
        headless=args.headless,
        os_type="Ubuntu",
        require_a11y_tree=False,
        enable_proxy=True,
        client_password=args.client_password,
        server_url=cluster_url,
    )


def build_agent(args: argparse.Namespace) -> Qwen35VLAgent:
    return Qwen35VLAgent(
        model=args.model,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        action_space=args.action_space,
        observation_type=args.observation_type,
        coordinate_type=args.coord,
        add_thought_prefix=args.add_thought_prefix,
        history_n=args.history_n,
        image_max=args.image_max,
        fold_size=args.fold_size,
        api_backend="openai",
    )


def parse_openai_base_urls(args: argparse.Namespace) -> List[str]:
    raw_urls = args.openai_base_urls
    if raw_urls:
        return [url.strip() for url in raw_urls.split(",") if url.strip()]

    if args.base_url:
        return [args.base_url.strip()]

    env_urls = os.environ.get("OPENAI_BASE_URLS")
    if env_urls:
        return [url.strip() for url in env_urls.split(",") if url.strip()]

    fallback_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    return [fallback_url] if fallback_url else []


def choose_openai_base_url(worker_idx: int, base_urls: Sequence[str]) -> str:
    if not base_urls:
        return ""
    return base_urls[worker_idx % len(base_urls)]


def run_env_tasks(
    worker_idx: int,
    task_queue,
    args: argparse.Namespace,
    shared_scores: list,
    openai_base_urls: Sequence[str],
) -> None:
    setup_logging(args.log_level)
    selected_base_url = choose_openai_base_url(worker_idx, openai_base_urls)
    if selected_base_url:
        os.environ["OPENAI_BASE_URL"] = selected_base_url
    elif args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    worker_environments = []
    env = None

    try:
        logger.info(
            "%s using OPENAI_BASE_URL=%s",
            current_process().name,
            os.environ.get("OPENAI_BASE_URL", ""),
        )
        env = build_desktop_env(args)
        worker_environments.append(env)
        agent = build_agent(args)
        logger.info("Process %s started with remote Qwen35VLAgent.", current_process().name)

        while True:
            try:
                item = task_queue.get(timeout=5)
            except Exception:
                break

            domain, example_id = item
            try:
                config_file = build_config_file_path(args, domain, example_id)
                with open(config_file, "r", encoding="utf-8") as file_obj:
                    example = json.load(file_obj)

                logger.info("[%s][Domain]: %s", current_process().name, domain)
                logger.info("[%s][Example ID]: %s", current_process().name, example_id)
                logger.info("[%s][Instruction]: %s", current_process().name, example["instruction"])

                example_result_dir = build_example_result_dir(args, domain, example_id)
                os.makedirs(example_result_dir, exist_ok=True)

                try:
                    lib_run_single.run_single_example(
                        agent,
                        env,
                        example,
                        args.max_steps,
                        example["instruction"],
                        args,
                        example_result_dir,
                        shared_scores,
                    )
                except Exception as exc:
                    import traceback

                    logger.error("Exception in %s %s/%s: %s", current_process().name, domain, example_id, exc)
                    logger.error(traceback.format_exc())
                    try:
                        env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))
                    except Exception:
                        pass
                    with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as file_obj:
                        file_obj.write(json.dumps({"Error": f"{domain}/{example_id} - {exc}"}, ensure_ascii=False))
                        file_obj.write("\n")
            except Exception as exc:
                import traceback

                logger.error("Task-level error in %s: %s", current_process().name, exc)
                logger.error(traceback.format_exc())
    except Exception as exc:
        import traceback

        logger.error("Process-level error in %s: %s", current_process().name, exc)
        logger.error(traceback.format_exc())
    finally:
        logger.info("%s cleaning up environment...", current_process().name)
        for active_env in worker_environments:
            try:
                active_env.close()
                logger.info("%s environment closed successfully", current_process().name)
            except Exception as exc:
                logger.error("%s error during environment cleanup: %s", current_process().name, exc)


def signal_handler(signum, frame) -> None:
    global is_terminating, active_environments, processes
    if is_terminating:
        return
    is_terminating = True
    logger.info("Received signal %s. Gracefully shutting down...", signum)

    for env in active_environments:
        try:
            env.close()
        except Exception:
            pass

    for process in processes:
        if process.is_alive():
            try:
                process.terminate()
            except Exception:
                pass

    time.sleep(1)
    logger.info("Shutdown complete. Exiting.")
    sys.exit(0)


def test(
    args: argparse.Namespace,
    test_all_meta: Dict[str, List[str]],
    openai_base_urls: Sequence[str],
) -> None:
    global processes

    logger.info("Args: %s", args)
    if openai_base_urls:
        logger.info("Configured OPENAI_BASE_URL shards: %s", ", ".join(openai_base_urls))
    all_tasks = distribute_tasks(test_all_meta)
    logger.info("Total tasks: %d", len(all_tasks))
    if not all_tasks:
        logger.info("No tasks to run.")
        return

    with Manager() as manager:
        shared_scores = manager.list()
        task_queue = manager.Queue()
        for item in all_tasks:
            task_queue.put(item)

        processes = []
        for idx in range(args.num_envs):
            selected_url = choose_openai_base_url(idx, openai_base_urls)
            process = Process(
                target=run_env_tasks,
                args=(idx, task_queue, args, shared_scores, openai_base_urls),
                name=f"EnvProcess-{idx + 1}",
            )
            process.daemon = True
            process.start()
            processes.append(process)
            logger.info(
                "Started process %s with PID %s using OPENAI_BASE_URL=%s",
                process.name,
                process.pid,
                selected_url or os.environ.get("OPENAI_BASE_URL", ""),
            )

        try:
            while True:
                alive_count = 0
                for idx, process in enumerate(processes):
                    if not process.is_alive():
                        if process.exitcode == 0:
                            logger.info("Process %s finished normally", process.name)
                        elif not task_queue.empty():
                            logger.warning(
                                "Process %s died (exit=%s), restarting...",
                                process.name,
                                process.exitcode,
                            )
                            new_process = Process(
                                target=run_env_tasks,
                                args=(idx, task_queue, args, shared_scores, openai_base_urls),
                                name=f"EnvProcess-Restart-{idx + 1}",
                            )
                            new_process.daemon = True
                            new_process.start()
                            processes[idx] = new_process
                            selected_url = choose_openai_base_url(idx, openai_base_urls)
                            logger.info(
                                "Restarted process %s with PID %s using OPENAI_BASE_URL=%s",
                                new_process.name,
                                new_process.pid,
                                selected_url or os.environ.get("OPENAI_BASE_URL", ""),
                            )
                    else:
                        alive_count += 1

                if task_queue.empty():
                    logger.info("All tasks finished.")
                    break
                if alive_count == 0:
                    logger.error("All processes died, exiting.")
                    break
                time.sleep(5)

            for process in processes:
                process.join()
        except KeyboardInterrupt:
            logger.info("Main process received KeyboardInterrupt. Initiating graceful shutdown...")
            raise

        scores = list(shared_scores)
    logger.info("Average score: %s", (sum(scores) / len(scores)) if scores else 0)


def get_unfinished(
    action_space: str,
    use_model: str,
    observation_type: str,
    result_dir: str,
    total_file_json: Dict[str, List[str]],
    simple_path: bool = False,
) -> Dict[str, List[str]]:
    if simple_path:
        target_dir = result_dir
    else:
        target_dir = os.path.join(result_dir, action_space, observation_type, use_model)

    if not os.path.exists(target_dir):
        return total_file_json

    finished: Dict[str, List[str]] = {}
    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if not os.path.isdir(domain_path):
            continue

        finished[domain] = []
        for example_id in os.listdir(domain_path):
            example_path = os.path.join(domain_path, example_id)
            if not os.path.isdir(example_path):
                continue
            if os.path.exists(os.path.join(example_path, "result.txt")):
                finished[domain].append(example_id)
                continue

            for item in os.listdir(example_path):
                item_path = os.path.join(example_path, item)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                except Exception:
                    pass

    if not finished:
        return total_file_json

    remaining = {}
    for domain, examples in total_file_json.items():
        done_ids = set(finished.get(domain, []))
        left = [example_id for example_id in examples if example_id not in done_ids]
        if left:
            remaining[domain] = left
    return remaining


def get_result(
    action_space: str,
    use_model: str,
    observation_type: str,
    result_dir: str,
    simple_path: bool = False,
):
    if simple_path:
        target_dir = result_dir
    else:
        target_dir = os.path.join(result_dir, action_space, observation_type, use_model)

    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None

    all_result = []
    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if not os.path.isdir(domain_path):
            continue
        for example_id in os.listdir(domain_path):
            example_path = os.path.join(domain_path, example_id)
            if not os.path.isdir(example_path):
                continue
            result_file = os.path.join(example_path, "result.txt")
            if not os.path.exists(result_file):
                continue
            try:
                with open(result_file, "r", encoding="utf-8") as file_obj:
                    all_result.append(float(file_obj.read()))
            except Exception:
                all_result.append(0.0)

    if not all_result:
        print("New experiment, no result yet.")
        return None

    print("Current Success Rate:", sum(all_result) / len(all_result) * 100, "%")
    return all_result


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args = config()
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    openai_base_urls = parse_openai_base_urls(args)
    if openai_base_urls:
        os.environ["OPENAI_BASE_URL"] = openai_base_urls[0]

    setup_logging(args.log_level)

    try:
        if args.simple_path:
            path_to_args = os.path.join(args.result_dir, "args.json")
        else:
            path_to_args = os.path.join(
                args.result_dir,
                args.action_space,
                args.observation_type,
                args.model,
                "args.json",
            )

        os.makedirs(os.path.dirname(path_to_args), exist_ok=True)
        args_dict = vars(args).copy()
        args_dict["resolved_openai_base_urls"] = list(openai_base_urls)
        with open(path_to_args, "w", encoding="utf-8") as file_obj:
            json.dump(args_dict, file_obj, indent=2, ensure_ascii=False)

        with open(args.test_all_meta_path, "r", encoding="utf-8") as file_obj:
            test_all_meta = json.load(file_obj)

        if args.domain != "all":
            test_all_meta = {args.domain: test_all_meta[args.domain]}

        test_file_list = get_unfinished(
            args.action_space,
            args.model,
            args.observation_type,
            args.result_dir,
            test_all_meta,
            simple_path=args.simple_path,
        )

        left_info = ""
        for domain, examples in test_file_list.items():
            left_info += f"{domain}: {len(examples)}\n"
        logger.info("Left tasks:\n%s", left_info)

        get_result(
            args.action_space,
            args.model,
            args.observation_type,
            args.result_dir,
            simple_path=args.simple_path,
        )
        test(args, test_file_list, openai_base_urls)
    except KeyboardInterrupt:
        logger.info("Main process received KeyboardInterrupt.")
    except Exception as exc:
        logger.error("Unexpected error in main process: %s", exc, exc_info=True)
        signal_handler(signal.SIGTERM, None)
    finally:
        logger.info("Main process final cleanup...")
        for env in active_environments:
            if env is not None:
                try:
                    env.close()
                except Exception as exc:
                    logger.error("Error during final environment cleanup: %s", exc)
        for process in processes:
            if process is not None and process.is_alive():
                try:
                    process.terminate()
                except Exception as exc:
                    logger.error("Error terminating process: %s", exc)


if __name__ == "__main__":
    main()
