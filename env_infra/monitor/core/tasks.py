"""Task definitions: the config listing tasks, and their instructions.

These live beside the results rather than inside them — the runner records where
in `derive.py`. Both readers degrade to "nothing found" instead of raising, so a
config that has moved costs you instruction text, never the dashboard itself.
"""

import json
import os
import threading

# Shown when a task's definition could not be read.
NO_INFO = "No task info available"

_task_list_cache = {}   # {task_config_path: (mtime, data)}


def load_task_list(task_config_path):
    """Load the `{task_type: [task_id, ...]}` config, or {} if unavailable.

    Cached on mtime — the dashboard polls while a scan is running, and this file
    sits on the same network storage as everything else.
    """
    if not task_config_path or not os.path.exists(task_config_path):
        return {}
    try:
        mtime = os.path.getmtime(task_config_path)
    except OSError:
        return {}
    cached = _task_list_cache.get(task_config_path)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        with open(task_config_path, "r") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    _task_list_cache[task_config_path] = (mtime, data)
    return data


def forget_task_list(task_config_path):
    """Drop the cached config for `task_config_path`.

    Returns whether anything was actually cached, so callers can report what
    they cleared without overstating it. A falsy path clears nothing — the
    caller simply had no config to forget.
    """
    if not task_config_path:
        return False
    return _task_list_cache.pop(task_config_path, None) is not None


def load_task_info(examples_dir, task_type, task_id):
    """One task's definition, from `<examples_dir>/<task_type>/<task_id>.json`."""
    if not examples_dir:
        return None
    task_file = os.path.join(examples_dir, task_type, f"{task_id}.json")
    try:
        with open(task_file, "r") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class InstructionIndex:
    """Every task instruction under an examples directory, loaded in one pass.

    Reading one JSON per task inside a request handler is what made the
    dashboard's first load take minutes on network storage. The whole directory
    is walked once on a background thread instead; until it finishes, `get`
    returns None and callers show a placeholder.
    """

    def __init__(self, examples_dir):
        self.examples_dir = examples_dir
        self._lock = threading.Lock()
        self._instructions = {}   # {(task_type, task_id): instruction}
        self._loaded = False
        self._worker = None

    def ready(self):
        with self._lock:
            return self._loaded

    def get(self, task_type, task_id):
        with self._lock:
            return self._instructions.get((task_type, task_id))

    def instruction_or_placeholder(self, task_type, task_id):
        """The instruction, `None` while still loading, or a placeholder if absent."""
        found = self.get(task_type, task_id)
        if found is not None:
            return found
        return NO_INFO if self.ready() else None

    def refresh(self):
        """Start the background load if it hasn't run yet."""
        with self._lock:
            if self._loaded or (self._worker and self._worker.is_alive()):
                return
            self._worker = threading.Thread(target=self._run, daemon=True,
                                            name="monitor-instructions")
            worker = self._worker
        worker.start()

    def _run(self):
        found = {}
        try:
            with os.scandir(self.examples_dir) as types:
                for type_entry in types:
                    if not type_entry.is_dir() or type_entry.name.startswith("."):
                        continue
                    with os.scandir(type_entry.path) as files:
                        for file_entry in files:
                            if not file_entry.name.endswith(".json"):
                                continue
                            task_id = file_entry.name[:-len(".json")]
                            try:
                                with open(file_entry.path, "r") as f:
                                    info = json.load(f)
                            except (OSError, ValueError):
                                continue
                            found[(type_entry.name, task_id)] = info.get(
                                "instruction", "No instruction provided")
        except OSError:
            pass
        with self._lock:
            self._instructions = found
            self._loaded = True


class InstructionRegistry:
    """One InstructionIndex per examples directory, started on first request."""

    def __init__(self):
        self._lock = threading.Lock()
        self._indexes = {}

    def get(self, examples_dir):
        """The index for `examples_dir`, or None when there is no such directory."""
        if not examples_dir:
            return None
        with self._lock:
            index = self._indexes.get(examples_dir)
            if index is None:
                index = InstructionIndex(examples_dir)
                self._indexes[examples_dir] = index
        index.refresh()
        return index

    def drop(self, examples_dir):
        """Forget an examples directory so the next request reloads it.

        Returns whether anything was cached. Unlike the status cache this is
        keyed by examples directory, which several experiments may share.
        """
        if not examples_dir:
            return False
        with self._lock:
            return self._indexes.pop(examples_dir, None) is not None
