from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import signal
import sys
import time
from multiprocessing import Manager, Process, current_process
from typing import List

# Add project root to path for imports.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
# Also add the repo root (parent of OSWorld/) so `cluster.*` resolves even
# without an editable install of the project.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import lib_run_single
from cluster.client import OSWorldRemoteClient as DesktopEnv
from lib_results_logger import log_task_error
from mm_agents.anthropic import AnthropicAgent
from mm_agents.anthropic.utils import APIProvider, validate_model_support


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


def _default_api_provider() -> str:
    return os.environ.get(
        "ANTHROPIC_API_PROVIDER",
        os.environ.get("API_PROVIDER", "anthropic"),
    )


def _api_provider(value: str) -> APIProvider:
    return APIProvider(value.lower().strip())


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OSWorld Claude evaluation through OSWorldRemoteClient and the cluster master."
    )

    # environment config
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument("--headless", action="store_true", help="Run in headless machine")
    parser.add_argument("--action_space", type=str, default="claude_computer_use", help="Action type")
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="screenshot",
        help="Observation type",
    )
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=15)

    # agent config
    parser.add_argument("--max_trajectory_length", type=int, default=3)
    parser.add_argument("--test_config_base_dir", type=str, default="evaluation_examples")

    # lm config
    parser.add_argument("--model", type=str, default="claude-opus-4-6-20260205")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max_tokens", type=int, default=3000)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--effort",
        type=str,
        default="max",
        choices=["low", "medium", "high", "max"],
        help="Effort level for Claude adaptive-thinking profiles",
    )
    parser.add_argument(
        "--api_provider",
        type=str,
        default=_default_api_provider(),
        choices=[provider.value for provider in APIProvider],
        help="Claude API provider. Remote cluster defaults to direct Anthropic-compatible API.",
    )
    parser.add_argument(
        "--anthropic_base_url",
        type=str,
        default=os.environ.get("ANTHROPIC_BASE_URL"),
        help="Optional base URL for direct Anthropic-compatible API.",
    )

    # example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--test_all_meta_path", type=str, default="evaluation_examples/test_nogdrive.json"
    )
    parser.add_argument(
        "--specific_task_id",
        type=str,
        default=None,
        help="Run only a specific task ID (overrides domain filtering)",
    )

    # logging related
    parser.add_argument("--result_dir", type=str, default="./results_claude_remote")
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of environments to run in parallel"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )

    # provider config
    parser.add_argument("--region", type=str, default="local", help="Provider region")
    parser.add_argument(
        "--provider_name",
        type=str,
        default="docker_server",
        choices=[
            "aws",
            "virtualbox",
            "vmware",
            "docker",
            "docker_fast",
            "docker_server",
            "azure",
            "aliyun",
            "fastvm",
        ],
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
    if getattr(root_logger, "_claude_remote_logging_configured", False):
        logger = logging.getLogger("desktopenv.experiment")
        return

    os.makedirs("logs", exist_ok=True)
    datetime_str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

    file_handler = logging.FileHandler(
        os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8"
    )
    debug_handler = logging.FileHandler(
        os.path.join("logs", "debug-{:}.log".format(datetime_str)), encoding="utf-8"
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
    root_logger._claude_remote_logging_configured = True

    logger = logging.getLogger("desktopenv.experiment")


def distribute_tasks(test_all_meta: dict) -> List[tuple]:
    all_tasks = []
    for domain, examples in test_all_meta.items():
        for example_id in examples:
            all_tasks.append((domain, example_id))
    return all_tasks


def build_desktop_env(args: argparse.Namespace) -> DesktopEnv:
    region = args.region
    screen_size = (args.screen_width, args.screen_height)
    snapshot_name = "init_state"
    if args.provider_name == "aws":
        from desktop_env.providers.aws.manager import IMAGE_ID_MAP

        ami_id = IMAGE_ID_MAP[region].get(screen_size, IMAGE_ID_MAP[region][(1920, 1080)])
        snapshot_name = ami_id

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
        require_a11y_tree=args.observation_type
        in [
            "a11y_tree",
            "screenshot_a11y_tree",
            "som",
        ],
        enable_proxy=True,
        client_password=args.client_password,
        server_url=cluster_url,
    )


def build_agent(args: argparse.Namespace, env: DesktopEnv) -> AnthropicAgent:
    api_provider = _api_provider(args.api_provider)
    api_key = os.environ.get("ANTHROPIC_API_KEY") if api_provider == APIProvider.ANTHROPIC else None
    return AnthropicAgent(
        env=env,
        model=args.model,
        provider=api_provider,
        api_key=api_key,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        action_space=args.action_space,
        observation_type=args.observation_type,
        max_trajectory_length=args.max_trajectory_length,
        provider_name=args.provider_name,
        screen_size=(args.screen_width, args.screen_height),
        effort=args.effort,
        max_steps=args.max_steps,
    )


def run_env_tasks(task_queue, args: argparse.Namespace, shared_scores: list):
    setup_logging(args.log_level)
    worker_environments = []
    env = None
    try:
        env = build_desktop_env(args)
        worker_environments.append(env)
        agent = build_agent(args, env)
        logger.info("Process %s started.", current_process().name)

        while True:
            try:
                item = task_queue.get(timeout=5)
            except Exception:
                break
            domain, example_id = item
            try:
                config_file = os.path.join(
                    args.test_config_base_dir, f"examples/{domain}/{example_id}.json"
                )
                with open(config_file, "r", encoding="utf-8") as f:
                    example = json.load(f)
                logger.info("[%s][Domain]: %s", current_process().name, domain)
                logger.info("[%s][Example ID]: %s", current_process().name, example_id)
                logger.info(
                    "[%s][Instruction]: %s",
                    current_process().name,
                    example["instruction"],
                )
                example_result_dir = os.path.join(
                    args.result_dir,
                    args.action_space,
                    args.observation_type,
                    args.model,
                    domain,
                    example_id,
                )
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
                except Exception as e:
                    import traceback

                    logger.error(
                        "Exception in %s %s/%s: %s",
                        current_process().name,
                        domain,
                        example_id,
                        e,
                    )
                    logger.error(traceback.format_exc())
                    try:
                        log_task_error({"id": example_id}, str(e), example_result_dir, args)
                    except Exception as log_e:
                        logger.error("Failed to log error to results.json: %s", log_e)
                    try:
                        env.controller.end_recording(
                            os.path.join(example_result_dir, "recording.mp4")
                        )
                    except Exception as rec_e:
                        logger.error("Failed to end recording: %s", rec_e)
                    with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                        f.write(json.dumps({"Error": f"{domain}/{example_id} - {e}"}))
                        f.write("\n")
            except Exception as e:
                import traceback

                logger.error("Task-level error in %s: %s", current_process().name, e)
                logger.error(traceback.format_exc())
    except Exception as e:
        import traceback

        logger.error("Process-level error in %s: %s", current_process().name, e)
        logger.error(traceback.format_exc())
    finally:
        logger.info("%s cleaning up environment...", current_process().name)
        for active_env in worker_environments:
            try:
                active_env.close()
                logger.info("%s environment closed successfully", current_process().name)
            except Exception as e:
                logger.error("%s error during environment cleanup: %s", current_process().name, e)


def signal_handler(signum, frame):
    global is_terminating, active_environments, processes
    if is_terminating:
        return
    is_terminating = True
    logger.info("Received signal %s. Gracefully shutting down...", signum)
    for env in active_environments:
        try:
            logger.info("Closing environment...")
            env.close()
            logger.info("Environment closed successfully")
        except Exception as e:
            logger.error("Error closing environment: %s", e)
    for p in processes:
        if p.is_alive():
            try:
                logger.info("Sending termination signal to process %s...", p.name)
                p.terminate()
            except Exception as e:
                logger.error("Error sending termination signal to process: %s", e)
    time.sleep(1)
    for p in processes:
        if p.is_alive():
            try:
                logger.info("Forcefully terminating process %s...", p.name)
                os.kill(p.pid, signal.SIGKILL)
            except Exception as e:
                logger.error("Error forcefully terminating process: %s", e)
    logger.info("Shutdown complete. Exiting.")
    sys.exit(0)


def test(args: argparse.Namespace, test_all_meta: dict) -> None:
    global processes
    logger.info("Args: %s", args)
    all_tasks = distribute_tasks(test_all_meta)
    logger.info("Total tasks: %d", len(all_tasks))
    with Manager() as manager:
        shared_scores = manager.list()
        task_queue = manager.Queue()
        for item in all_tasks:
            task_queue.put(item)
        processes = []
        for i in range(args.num_envs):
            p = Process(
                target=run_env_tasks,
                args=(task_queue, args, shared_scores),
                name=f"EnvProcess-{i + 1}",
            )
            p.daemon = True
            p.start()
            processes.append(p)
            logger.info("Started process %s with PID %s", p.name, p.pid)
        try:
            while True:
                alive_count = 0
                for idx, p in enumerate(processes):
                    if not p.is_alive():
                        if p.exitcode == 0:
                            logger.info("Process %s finished normally", p.name)
                        elif not task_queue.empty():
                            logger.warning(
                                "Process %s died (exit=%s), restarting...",
                                p.name,
                                p.exitcode,
                            )
                            new_p = Process(
                                target=run_env_tasks,
                                args=(task_queue, args, shared_scores),
                                name=f"EnvProcess-Restart-{idx + 1}",
                            )
                            new_p.daemon = True
                            new_p.start()
                            processes[idx] = new_p
                            logger.info("Restarted process %s with PID %s", new_p.name, new_p.pid)
                    else:
                        alive_count += 1
                if task_queue.empty():
                    logger.info("All tasks finished.")
                    break
                if alive_count == 0:
                    logger.error("All processes died, exiting.")
                    break
                time.sleep(5)
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            logger.info("Main process received KeyboardInterrupt. Initiating graceful shutdown...")
            raise
        except Exception as e:
            logger.error("Unexpected error while waiting for processes: %s", e, exc_info=True)
            for p in processes:
                if p.is_alive():
                    try:
                        logger.info("Terminating process %s due to error...", p.name)
                        p.terminate()
                    except Exception as term_e:
                        logger.error("Error terminating process %s: %s", p.name, term_e)
            raise
        scores = list(shared_scores)
    logger.info("Average score: %s", sum(scores) / len(scores) if scores else 0)


def get_unfinished(
    action_space, use_model, observation_type, result_dir, total_file_json
):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)

    if not os.path.exists(target_dir):
        return total_file_json

    finished = {}
    for domain in os.listdir(target_dir):
        finished[domain] = []
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                if example_id == "onboard":
                    continue
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" not in os.listdir(example_path):
                        for file in os.listdir(example_path):
                            os.remove(os.path.join(example_path, file))
                    else:
                        finished[domain].append(example_id)

    if not finished:
        return total_file_json

    for domain, examples in finished.items():
        if domain in total_file_json:
            total_file_json[domain] = [
                x for x in total_file_json[domain] if x not in examples
            ]

    return total_file_json


def get_result(action_space, use_model, observation_type, result_dir, total_file_json):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None

    all_result = []

    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" in os.listdir(example_path):
                        try:
                            value_str = open(
                                os.path.join(example_path, "result.txt"), "r"
                            ).read()
                            all_result.append(float(value_str))
                        except Exception:
                            all_result.append(0.0)

    if not all_result:
        print("New experiment, no result yet.")
        return None
    print("Current Success Rate:", sum(all_result) / len(all_result) * 100, "%")
    return all_result


def filter_tasks(args: argparse.Namespace, test_all_meta: dict) -> dict:
    if args.specific_task_id:
        logger.info("Filtering for specific task ID: %s", args.specific_task_id)
        for domain, task_ids in test_all_meta.items():
            if args.specific_task_id in task_ids:
                logger.info("Found task %s in domain: %s", args.specific_task_id, domain)
                return {domain: [args.specific_task_id]}
        logger.error("Task ID %s not found in test file!", args.specific_task_id)
        sys.exit(1)

    if args.domain != "all":
        return {args.domain: test_all_meta[args.domain]}
    return test_all_meta


def validate_args(args: argparse.Namespace) -> None:
    if not args.model or args.model.strip() == "":
        print("ERROR: Model must be specified. Use --model <model_name>")
        sys.exit(1)

    api_provider = _api_provider(args.api_provider)
    validation_kwargs = {
        "provider": api_provider,
        "effort": args.effort,
    }
    if args.temperature is not None:
        validation_kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        validation_kwargs["top_p"] = args.top_p
    if not validate_model_support(args.model, **validation_kwargs):
        print(f"\nModel '{args.model}' api sample failed")
        sys.exit(1)


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args = config()
    if args.anthropic_base_url:
        os.environ["ANTHROPIC_BASE_URL"] = args.anthropic_base_url
    setup_logging(args.log_level)
    validate_args(args)

    try:
        path_to_args = os.path.join(
            args.result_dir,
            args.action_space,
            args.observation_type,
            args.model,
            "args.json",
        )
        os.makedirs(os.path.dirname(path_to_args), exist_ok=True)
        with open(path_to_args, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=4)

        with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
            test_all_meta = json.load(f)

        test_file_list = filter_tasks(args, test_all_meta)
        test_file_list = get_unfinished(
            args.action_space,
            args.model,
            args.observation_type,
            args.result_dir,
            test_file_list,
        )
        left_info = ""
        for domain in test_file_list:
            left_info += f"{domain}: {len(test_file_list[domain])}\n"
        logger.info("Left tasks:\n%s", left_info)

        get_result(
            args.action_space,
            args.model,
            args.observation_type,
            args.result_dir,
            test_all_meta,
        )
        test(args, test_file_list)
    except KeyboardInterrupt:
        logger.info("Main process received KeyboardInterrupt.")
    except Exception as e:
        logger.error("Unexpected error in main process: %s", e, exc_info=True)
        signal_handler(signal.SIGTERM, None)
    finally:
        logger.info("Main process final cleanup...")
        for env in active_environments:
            if env is not None:
                try:
                    logger.info("Closing environment in final cleanup...")
                    env.close()
                    logger.info("Environment closed successfully in final cleanup")
                except Exception as e:
                    logger.error("Error during final environment cleanup: %s", e)
        for p in processes:
            if p is not None and p.is_alive():
                try:
                    logger.info("Terminating process %s...", p.name)
                    p.terminate()
                except Exception as e:
                    logger.error("Error terminating process: %s", e)
        time.sleep(1)
        for p in processes:
            if p is not None and p.is_alive():
                try:
                    logger.info("Force killing process %s...", p.name)
                    os.kill(p.pid, signal.SIGKILL)
                    logger.info("Process %s force killed", p.name)
                except Exception as e:
                    logger.error("Error force killing process: %s", e)


if __name__ == "__main__":
    main()
