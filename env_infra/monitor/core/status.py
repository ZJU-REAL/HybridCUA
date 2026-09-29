"""Per-experiment task status, computed off the request path.

Scanning an experiment on network storage is slow — ~15s to enumerate 361 task
directories, plus a read of each `traj.jsonl` — so doing it inside a request
handler means a page that hangs for minutes on first load. Instead a background
thread does the work and requests read whatever is ready, reporting progress so
the UI can poll until it settles.

Terminal statuses (`Done*`, `Error`) are cached indefinitely: a finished task
never changes. Everything else gets a short TTL so a live run keeps updating,
but the TTL only makes it *eligible* for a re-read — the status is recomputed
only when the task's `traj.jsonl` has actually changed size or mtime. That
matters because an abandoned run leaves tasks parked in `Running` forever, and
re-reading each of their trajectories every TTL cost ~170s per pass on network
storage for no new information.

The directory listing gets its own, much longer TTL — enumerating a large
experiment costs ~15s, and re-walking it on every request is what used to make
an already-loaded experiment feel as slow as a cold one.
"""

import os
import threading
import time
from datetime import datetime

from .discover import scan_task_dirs
from .traj import count_steps, last_step, read_result

# How long a non-terminal status may be trusted without even checking the file.
LIVE_TTL = 10.0

# How long the directory listing is trusted before being re-walked. Task
# directories only appear when a run starts one, so this can be far longer than
# LIVE_TTL; the point is that a settled experiment stops paying for the walk.
SCAN_TTL = 120.0


def _fingerprint(task_dir):
    """Cheap change-detector for a task: `traj.jsonl`'s size and mtime.

    One stat instead of reading the whole trajectory. Returns None when the file
    is missing, which is itself a distinguishable state (the task is Preparing).
    """
    try:
        st = os.stat(os.path.join(task_dir, "traj.jsonl"))
    except OSError:
        return None
    return (st.st_mtime, st.st_size)


def _format_timestamp(value):
    try:
        return datetime.strptime(value, "%Y%m%d@%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return "None"


def _mtime_string(path):
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        return None


def _tail_text(path, limit=2048):
    """Last `limit` bytes of a text file, for exit-condition sniffing."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            start = max(0, f.tell() - limit)
            f.seek(start)
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def is_terminal(status):
    return status.startswith("Done") or status == "Error"


def compute_status(task_dir, max_steps):
    """Status of one task from its output files.

    Mirrors the runner's own exit conditions: an explicit `done` flag or `Error`
    in the final step, a message/thought exit recorded in the log, or simply
    running out of steps.
    """
    traj_file = os.path.join(task_dir, "traj.jsonl")
    if not os.path.exists(traj_file):
        # Directory exists but the first step hasn't landed yet.
        return {"status": "Preparing", "progress": 0, "max_steps": max_steps,
                "last_update": _mtime_string(task_dir), "result": None}

    step_count = count_steps(traj_file)
    if step_count == 0:
        return {"status": "Initializing", "progress": 0, "max_steps": max_steps,
                "last_update": _mtime_string(traj_file), "result": None}

    final = last_step(traj_file) or {}
    if final.get("done"):
        status = "Done"
    elif final.get("Error"):
        status = "Error"
    else:
        status = "Running"
        log_file = os.path.join(task_dir, "runtime.log")
        if os.path.exists(log_file):
            tail = _tail_text(log_file)
            if "message_exit: True" in tail:
                status = "Done (Message Exit)"
            elif "thought_exit: True" in tail:
                status = "Done (Thought Exit)"

    if status == "Running" and max_steps and step_count >= max_steps:
        status = "Done (Max Steps)"

    return {
        "status": status,
        "progress": step_count,
        "max_steps": max_steps,
        "last_update": _format_timestamp(final.get("action_timestamp")),
        "result": read_result(task_dir) if status.startswith("Done") else None,
    }


class ExperimentIndex:
    """Background scan + status cache for one experiment directory."""

    def __init__(self, experiment_path, max_steps):
        self.experiment_path = experiment_path
        self.max_steps = max_steps
        self._lock = threading.Lock()
        self._tasks = []          # [(task_type, task_id, path)]
        self._statuses = {}       # {(task_type, task_id): (status, timestamp, fingerprint)}
        self._scanned = False     # has the directory listing completed at least once
        self._scanned_at = 0.0    # when it completed, for SCAN_TTL
        self._worker = None
        self._generation = 0      # bumped to invalidate an in-flight worker

    # ---------- reading ----------

    def snapshot(self):
        """Current tasks and statuses, plus how much of the list is known.

        Never blocks: a status still being computed comes back as None.

        Progress is "how many statuses do we have", not "how far has the current
        pass got". The two differ once a pass re-verifies an already-loaded
        experiment: a per-pass counter drops back to zero and the UI reads it as
        the scan having restarted, which is the flapping progress line this used
        to show. Counting known statuses only ever moves forward, so `complete`
        stays true while re-verification happens quietly in the background.
        """
        with self._lock:
            tasks = list(self._tasks)
            statuses = {k: v[0] for k, v in self._statuses.items()}
            scanned = self._scanned
        done = sum(1 for task_type, task_id, _ in tasks if (task_type, task_id) in statuses)
        return {
            "tasks": tasks,
            "statuses": statuses,
            "scan": {
                "scanned": scanned,
                "done": done,
                "total": len(tasks),
                "complete": scanned and done >= len(tasks),
            },
        }

    def status_of(self, task_type, task_id, task_dir):
        """Status for a single task, recomputing it synchronously on a cache miss.

        Used by the detail page, where one directory read is cheap and the
        caller genuinely needs an answer now.
        """
        key = (task_type, task_id)
        with self._lock:
            cached = self._statuses.get(key)
        if self._is_fresh(cached, task_dir):
            return cached[0]
        status = compute_status(task_dir, self.max_steps)
        self._store(key, status, _fingerprint(task_dir))
        return status

    # ---------- writing ----------

    def _still_valid(self, entry):
        """Whether a cached entry is inside its TTL. Falsy entry = a miss.

        A terminal status is valid forever; a live one only until LIVE_TTL, after
        which `_is_fresh` decides whether anything actually needs re-reading.
        """
        if not entry:
            return False
        status, stamp = entry[0], entry[1]
        return is_terminal(status["status"]) or (time.time() - stamp) < LIVE_TTL

    def _is_fresh(self, entry, task_dir):
        """Whether `entry` can be served without recomputing the status.

        Past the TTL, a live task still counts as fresh when its trajectory is
        byte-for-byte unchanged — one stat instead of a full read. Tasks parked
        in `Running` by an abandoned run therefore cost almost nothing.
        """
        if self._still_valid(entry):
            return True
        if not entry or len(entry) < 3:
            return False
        return entry[2] is not None and entry[2] == _fingerprint(task_dir)

    def _store(self, key, status, fingerprint):
        with self._lock:
            self._statuses[key] = (status, time.time(), fingerprint)

    def clear(self):
        """Drop all cached state and force the next refresh to redo everything."""
        with self._lock:
            self._statuses.clear()
            self._tasks = []
            self._scanned = False
            self._scanned_at = 0.0
            self._generation += 1

    def _work_pending(self):
        """Whether a pass would actually recompute anything.

        Without this check every request started a worker that re-walked the
        directory and re-read every live task, so an experiment that had already
        settled kept paying full price and the progress line flapped back to
        partial. Called under `self._lock`.

        Only TTLs are consulted here, never the filesystem — this runs inside the
        lock on the request path, so it must not stat 361 directories. A pass that
        starts and then finds every fingerprint unchanged is cheap and silent.
        """
        if not self._scanned or (time.time() - self._scanned_at) >= SCAN_TTL:
            return True
        return any(not self._still_valid(self._statuses.get((t, i)))
                   for t, i, _ in self._tasks)

    def refresh(self):
        """Start a background pass if one isn't already running and one is due."""
        with self._lock:
            if self._worker and self._worker.is_alive():
                return
            if not self._work_pending():
                return
            self._generation += 1
            generation = self._generation
            self._worker = threading.Thread(
                target=self._run, args=(generation,),
                name=f"monitor-index-{os.path.basename(self.experiment_path)}",
                daemon=True,
            )
            worker = self._worker
        worker.start()

    def _run(self, generation):
        # Re-walking the tree is the single most expensive thing here, so reuse a
        # listing that is still within SCAN_TTL and only refresh statuses.
        with self._lock:
            fresh_listing = self._scanned and (time.time() - self._scanned_at) < SCAN_TTL
            tasks = list(self._tasks) if fresh_listing else None

        if tasks is None:
            tasks = scan_task_dirs(self.experiment_path)
            scanned_at = time.time()
            with self._lock:
                if generation != self._generation:
                    return
                self._tasks = tasks
                self._scanned = True
                self._scanned_at = scanned_at

        for task_type, task_id, task_dir in tasks:
            key = (task_type, task_id)
            with self._lock:
                if generation != self._generation:
                    return
                cached = self._statuses.get(key)
            if self._is_fresh(cached, task_dir):
                # Unchanged on disk: keep the status but restamp it, so the next
                # pass waits another TTL before even stat-ing this task again.
                if not self._still_valid(cached):
                    with self._lock:
                        if generation != self._generation:
                            return
                        if self._statuses.get(key) is cached:
                            self._statuses[key] = (cached[0], time.time(), cached[2])
            else:
                status = compute_status(task_dir, self.max_steps)
                fingerprint = _fingerprint(task_dir)
                with self._lock:
                    if generation != self._generation:
                        return
                    self._statuses[key] = (status, time.time(), fingerprint)


class IndexRegistry:
    """One ExperimentIndex per experiment path, created on demand."""

    def __init__(self):
        self._lock = threading.Lock()
        self._indexes = {}

    def get(self, experiment_path, max_steps):
        with self._lock:
            index = self._indexes.get(experiment_path)
            if index is None:
                index = ExperimentIndex(experiment_path, max_steps)
                self._indexes[experiment_path] = index
            elif index.max_steps != max_steps:
                # max_steps feeds the "Done (Max Steps)" verdict, so a change
                # invalidates every cached status.
                index.max_steps = max_steps
                index.clear()
        return index

    def drop(self, experiment_path):
        with self._lock:
            index = self._indexes.get(experiment_path)
        if index:
            index.clear()
        return index is not None
