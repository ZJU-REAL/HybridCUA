"""Base eval layer: benchmark-neutral process orchestration + task-source ABC.

Mirrors ``cluster/worlds/base/adapter.py``: just as a world plugs into the
platform by subclassing :class:`WorldAdapter`, a benchmark plugs its evaluation
into :class:`EvalRunner` by subclassing :class:`EvalTaskSource`. The runner
itself is benchmark-neutral — it owns only the proven process-per-env
orchestration (one env per worker, a shared ``Manager.Queue`` for work-stealing,
restart-on-death, graceful signal teardown) and drives a benchmark purely
through the task-source contract.

Architecture (the gui-env model, preserved):

    main: Manager.Queue(all pending tasks)
       └─ N worker processes, each:
            env   = build_env(args)            # built once, reused across tasks
            agent = agent_factory(args, env)    # env passed in (some agents need env.tools)
            while item = task_queue.get():
                example, result_dir = task_source.prepare_example(args, item)
                task_source.run_episode(agent, env, example, args, result_dir, scores)
            finally: env.close()                # releases the cluster session

Adding a new benchmark = write a ``EvalTaskSource`` subclass; EvalRunner never
changes. ``agent_factory`` / ``build_env`` stay on the runner (not the task
source) because agent is a ``(benchmark × agent)`` concern orthogonal to the
task model: OSWorld has many agents sharing one task model.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from abc import ABC, abstractmethod
from multiprocessing import Manager, Process, current_process
from typing import Any, Callable, Optional, Protocol

logger = logging.getLogger("desktopenv.experiment")


# ---------------------------------------------------------------------------
# Agent protocol — any agent that implements predict() works.
# ---------------------------------------------------------------------------
class Agent(Protocol):
    def predict(self, instruction: str, observation: dict) -> Any: ...
    def reset(self) -> None: ...


# Unified agent factory signature: ``(args, env) -> Agent``. env is passed in
# because some benchmarks' agents read state off the env at construction time
# (e.g. MobileWorld's create_agent reads env.tools). Agents that don't need it
# accept and ignore it (``def build_agent(args, env=None)``).
AgentFactory = Callable[[argparse.Namespace, Any], Any]
# build_env builds one env (cluster session) per worker; reused across tasks.
EnvFactory = Callable[[argparse.Namespace], Any]


# ---------------------------------------------------------------------------
# Common argparse arguments shared by all evaluation scripts.
# ---------------------------------------------------------------------------
def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the standard set of arguments every eval script needs."""
    # environment
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument("--headless", action="store_true", help="Run in headless machine")
    parser.add_argument("--action_space", type=str, default="pyautogui")
    parser.add_argument("--observation_type", choices=["screenshot"], default="screenshot")
    parser.add_argument("--sleep_after_execution", type=float, default=5.0)
    parser.add_argument("--max_steps", type=int, default=100)

    # evaluation
    parser.add_argument("--test_config_base_dir", type=str, default="evaluation_examples")
    parser.add_argument("--examples_subdir", type=str, default="examples")
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument("--test_all_meta_path", type=str, default="evaluation_examples/test_nogdrive.json")

    # result
    parser.add_argument("--result_dir", type=str, default="./results_remote")
    parser.add_argument("--simple_path", action="store_true")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run in parallel")

    # logging
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    # cluster
    parser.add_argument("--client_password", type=str, default="")
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--cluster_url", type=str,
                        default=os.environ.get("GUI_ENV_SERVER_URL", "http://127.0.0.1:19000"))
    return parser


# ---------------------------------------------------------------------------
# Logging setup (per-process safe).
# ---------------------------------------------------------------------------
def setup_logging(log_level_name: str, prefix: str = "eval-remote") -> None:
    import datetime

    root_logger = logging.getLogger()
    log_level = getattr(logging, log_level_name.upper())
    root_logger.setLevel(log_level)
    if getattr(root_logger, "_eval_logging_configured", False):
        return

    log_dir = os.environ.get("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    datetime_str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"{prefix}-normal-{datetime_str}.log"), encoding="utf-8")
    debug_handler = logging.FileHandler(
        os.path.join(log_dir, f"{prefix}-debug-{datetime_str}.log"), encoding="utf-8")
    stdout_handler = logging.StreamHandler(sys.stdout)

    file_handler.setLevel(logging.INFO)
    debug_handler.setLevel(logging.DEBUG)
    stdout_handler.setLevel(log_level)

    formatter = logging.Formatter(
        "[%(asctime)s %(levelname)s %(module)s/%(lineno)d-%(processName)s] %(message)s")
    file_handler.setFormatter(formatter)
    debug_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(logging.Filter("desktopenv"))

    root_logger.addHandler(file_handler)
    root_logger.addHandler(debug_handler)
    root_logger.addHandler(stdout_handler)
    root_logger._eval_logging_configured = True


# ---------------------------------------------------------------------------
# EvalTaskSource — the benchmark-specific task model (mirrors WorldAdapter).
# ---------------------------------------------------------------------------
class EvalTaskSource(ABC):
    """Benchmark-specific task model for evaluation: list, resume, prep, execute.

    EvalRunner treats ``task_item`` as opaque — it is only ever passed back to
    :meth:`prepare_example` / :meth:`item_label`, never unpacked by the runner.
    A benchmark subclasses this (one per benchmark); the runner never changes.

    Subclasses implement the 3 abstract methods (``load_tasks``,
    ``prepare_example``, ``run_episode``) and override the defaulted ones as
    needed (``pending_tasks`` for resume, ``save_args``/``summarize``/``error_dump``
    for result layout).
    """

    @abstractmethod
    def load_tasks(self, args: argparse.Namespace) -> list:
        """Full task list for this run (before resume filtering)."""

    def pending_tasks(self, args: argparse.Namespace, all_tasks: list) -> list:
        """Resume: filter to tasks not already finished on disk.

        Default = no resume (return all). Override to skip tasks that already
        have a result on disk.
        """
        return all_tasks

    @abstractmethod
    def prepare_example(self, args: argparse.Namespace, task_item) -> tuple:
        """Load/parse the task's example payload and make its result dir.

        Returns ``(example, result_dir)``. OSWorld reads a JSON config file and
        builds an action_space/observation_type/model/domain/example_id dir;
        MobileWorld synthesizes ``{"id": task_name}`` and returns the
        log_file_root (TrajLogger makes the subdir).
        """

    @abstractmethod
    def run_episode(self, agent, env, example, args: argparse.Namespace, result_dir: str, shared_scores: list) -> None:
        """Execute one task's episode. Append the score to ``shared_scores``."""

    def item_label(self, task_item) -> str:
        """Stable human label for logs/errors. Default: ``str(task_item)``."""
        return str(task_item)

    def error_dump(self, args: argparse.Namespace, task_item, result_dir: str, exc: BaseException) -> None:
        """Best-effort error record on episode failure. Default: append
        ``{"Error": ...}`` to ``result_dir/traj.jsonl``. Specs override for
        their own layout."""
        try:
            os.makedirs(result_dir, exist_ok=True)
            with open(os.path.join(result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"Error": f"{self.item_label(task_item)} - {exc}"}, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def save_args(self, args: argparse.Namespace, model_name: str = "") -> None:
        """Save args.json before the run. Default: ``result_dir/args.json``.
        OSWorld overrides with its action_space/observation_type/model layout."""
        result_dir = getattr(args, "result_dir", None)
        if not result_dir:
            return
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

    def summarize(self, args: argparse.Namespace, model_name: str = "") -> None:
        """Optional: print success rate from disk. Default: no-op."""
        pass


# ---------------------------------------------------------------------------
# EvalRunner — benchmark-neutral process orchestrator.
# ---------------------------------------------------------------------------
class EvalRunner:
    """Agent-agnostic multi-environment parallel evaluation runner.

    Args:
        args: Parsed argparse namespace (must include common args from add_common_args).
        agent_factory: ``Callable(args, env) -> Agent``. env is passed so agents
            that need it (e.g. MobileWorld create_agent reads env.tools) get it;
            agents that don't accept ``env=None``.
        task_source: :class:`EvalTaskSource` subclass instance — owns the
            benchmark's task list, resume, example prep, and episode loop.
        build_env: Optional override for environment (cluster session)
            construction. Default: OSWorldSessionClient.
        model_name: Model identifier for result directory structure.
        log_prefix: Prefix for log filenames.
        env_setup: Optional ``Callable(args)`` run once before agent/env creation
            in each worker (e.g. pin an OPENAI_BASE_URL env var per worker).
    """

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        agent_factory: AgentFactory,
        task_source: EvalTaskSource,
        build_env: EnvFactory | None = None,
        model_name: str = "",
        log_prefix: str = "eval-remote",
        env_setup: Callable[[argparse.Namespace], None] | None = None,
    ) -> None:
        self.args = args
        self.agent_factory = agent_factory
        self.task_source = task_source
        self.build_env = build_env or self._default_build_env
        self.model_name = model_name or getattr(args, "model", "")
        self.log_prefix = log_prefix
        self.env_setup = env_setup
        self._processes: list[Process] = []
        self._is_terminating = False
        # PID of the process that owns self._processes. Workers fork and inherit
        # this signal handler, but their self._processes entries belong to the
        # parent — only the parent may call p.is_alive()/p.terminate() on them.
        self._main_pid = os.getpid()

    @staticmethod
    def _default_build_env(args: argparse.Namespace):
        from cluster.client.osworld.session_client import OSWorldSessionClient
        cluster_url = args.cluster_url or os.environ.get("GUI_ENV_SERVER_URL", "http://127.0.0.1:19000")
        os.environ["GUI_ENV_SERVER_URL"] = cluster_url
        enable_proxy = os.environ.get("ENABLE_PROXY", "1") == "1"
        return OSWorldSessionClient(
            cluster_url=cluster_url,
            action_space=args.action_space,
            screen_size=(args.screen_width, args.screen_height),
            headless=args.headless,
            enable_proxy=enable_proxy,
            client_password=args.client_password,
        )

    def _worker(self, task_queue, shared_scores: list) -> None:
        setup_logging(self.args.log_level, prefix=self.log_prefix)
        if self.env_setup:
            self.env_setup(self.args)

        env = None
        try:
            env = self.build_env(self.args)
            agent = self.agent_factory(self.args, env)
            logger.info("Process %s started.", current_process().name)

            while True:
                try:
                    item = task_queue.get(timeout=5)
                except Exception:
                    break

                label = self.task_source.item_label(item)
                result_dir = ""
                try:
                    example, result_dir = self.task_source.prepare_example(self.args, item)
                    logger.info("[%s] %s", current_process().name, label)
                    self.task_source.run_episode(agent, env, example, self.args, result_dir, shared_scores)
                except Exception as exc:
                    logger.error("Task error in %s %s: %s", current_process().name, label, exc, exc_info=True)
                    try:
                        self.task_source.error_dump(self.args, item, result_dir, exc)
                    except Exception:
                        pass
        except Exception as exc:
            logger.error("Process-level error in %s: %s", current_process().name, exc, exc_info=True)
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception as exc:
                    logger.error("Error closing env in %s: %s", current_process().name, exc)

    def _signal_handler(self, signum, frame) -> None:
        # A worker inherited this handler via fork. It must NOT touch
        # self._processes (those Process objects belong to the parent;
        # p.is_alive() would raise "can only test a child process"). Let the
        # worker raise KeyboardInterrupt so its own `finally: env.close()`
        # runs and the session is released instead of leaking.
        if os.getpid() != self._main_pid:
            raise KeyboardInterrupt
        if self._is_terminating:
            return
        self._is_terminating = True
        logger.info("Received signal %s. Shutting down...", signum)
        for p in self._processes:
            if p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    pass
        time.sleep(1)
        sys.exit(0)

    def run(self) -> None:
        """Load tasks, spawn workers, wait for completion."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        setup_logging(self.args.log_level, prefix=self.log_prefix)

        # --- delegated to the task source (benchmark-specific) ---
        self.task_source.save_args(self.args, self.model_name)

        all_tasks = self.task_source.load_tasks(self.args)
        tasks = self.task_source.pending_tasks(self.args, all_tasks)

        logger.info("Remaining tasks: %d", len(tasks))
        self.task_source.summarize(self.args, self.model_name)

        if not tasks:
            logger.info("No tasks to run.")
            return

        logger.info("Total tasks: %d, parallel envs: %d", len(tasks), self.args.num_envs)

        # --- benchmark-neutral process orchestration (unchanged) ---
        with Manager() as manager:
            shared_scores = manager.list()
            task_queue = manager.Queue()
            for item in tasks:
                task_queue.put(item)

            self._processes = []
            for idx in range(self.args.num_envs):
                p = Process(target=self._worker, args=(task_queue, shared_scores), name=f"Env-{idx+1}")
                p.daemon = True
                p.start()
                self._processes.append(p)
                logger.info("Started %s (PID %s)", p.name, p.pid)

            try:
                while True:
                    alive = 0
                    for idx, p in enumerate(self._processes):
                        if not p.is_alive():
                            if p.exitcode != 0 and not task_queue.empty():
                                logger.warning("Process %s died (exit=%s), restarting...", p.name, p.exitcode)
                                new_p = Process(target=self._worker, args=(task_queue, shared_scores),
                                                name=f"Env-Restart-{idx+1}")
                                new_p.daemon = True
                                new_p.start()
                                self._processes[idx] = new_p
                                alive += 1
                            # else: finished cleanly (exitcode 0) or queue empty —
                            # don't restart, don't count as alive.
                        else:
                            alive += 1

                    if task_queue.empty():
                        logger.info("All tasks distributed. Waiting for workers to finish...")
                        break
                    if alive == 0:
                        logger.error("All workers died.")
                        break
                    time.sleep(5)

                for p in self._processes:
                    p.join(timeout=300)
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received.")

            scores = list(shared_scores)

        if scores:
            logger.info("Final average score: %.4f (%d tasks)", sum(scores) / len(scores), len(scores))
        else:
            logger.info("No scores collected.")
