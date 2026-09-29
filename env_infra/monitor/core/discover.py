"""Find task directories by scanning the filesystem.

The dashboard used to list tasks from a config JSON, which meant a wrong path
produced an empty page with no explanation. Scanning instead means a single
experiment path always yields whatever is actually on disk; the config is only
consulted afterwards, to *add* tasks that were never started.

A directory is a task if it holds any runner output. Checking for markers
rather than globbing `traj.jsonl` matters: a task that was set up but died
before its first step has a directory and a `runtime.log` but no `traj.jsonl`,
and it should still show up (as "Preparing") rather than vanish.
"""

import os

# Any one of these marks a directory as a task's output.
_MARKER_FILES = ("traj.jsonl", "result.txt", "runtime.log")
_MARKER_PREFIX = "step_"

# Task dirs sit 1-2 levels below the experiment root in every layout the runner
# produces, but a user may aim the dashboard at an ancestor (e.g. the directory
# holding several models' results), so the scan is allowed to go deeper. It
# still stops at the first level where tasks appear, so a correctly-aimed path
# costs no extra traversal.
MAX_DEPTH = 6

ROOT_TYPE = "(root)"


def is_task_dir(entries):
    """True when a directory listing looks like runner output."""
    for name in entries:
        if name in _MARKER_FILES:
            return True
        if name.startswith(_MARKER_PREFIX) and name.endswith(".png"):
            return True
    return False


def scan_task_dirs(experiment_path, max_depth=MAX_DEPTH):
    """Yield (task_type, task_id, abs_path) for every task dir under the root.

    `task_type` is the path between the experiment root and the task dir, so
    both the nested layout (`<exp>/<domain>/<id>`) and the flat `--simple_path`
    one (`<exp>/<id>`, type `"(root)"`) work without special-casing. Recursion
    stops at a task dir, so screenshots and nested artifacts are never walked.
    """
    if not experiment_path or not os.path.isdir(experiment_path):
        return []

    root = os.path.abspath(experiment_path)
    found = []

    def walk(directory, depth):
        try:
            with os.scandir(directory) as it:
                entries = list(it)
        except OSError:
            return

        if is_task_dir(e.name for e in entries):
            relative = os.path.relpath(directory, root)
            if relative == os.curdir:
                # The experiment path is itself a single task dir.
                found.append((ROOT_TYPE, os.path.basename(root), directory))
            else:
                parts = relative.split(os.sep)
                task_type = "/".join(parts[:-1]) or ROOT_TYPE
                found.append((task_type, parts[-1], directory))
            return

        if depth >= max_depth:
            return
        for entry in entries:
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    walk(entry.path, depth + 1)
            except OSError:
                continue

    walk(root, 0)
    found.sort()
    return found
