"""Reading trajectory data out of a task/rollout directory.

A task directory holds `traj.jsonl` (one JSON object per step), `result.txt`
(the score), `runtime.log`, and `step_<n>_<timestamp>.png` screenshots.

The dashboard only needs two facts about `traj.jsonl` for its grid — the step
count and the last step — so those get dedicated readers that avoid parsing
the whole file. Both are pure-Python: an earlier version shelled out to
`wc -l` and `tail`, which cost two forks per task and dominated cold-load time
on experiments with hundreds of tasks.
"""

import json
import os

_CHUNK = 1 << 16


def count_steps(traj_file):
    """Number of lines in `traj_file`, or 0 if it is missing/unreadable."""
    try:
        count = 0
        tail = b""
        with open(traj_file, "rb") as f:
            while True:
                chunk = f.read(_CHUNK)
                if not chunk:
                    break
                count += chunk.count(b"\n")
                tail = chunk[-1:]
        # A final line with no trailing newline still counts.
        if tail and tail != b"\n":
            count += 1
        return count
    except OSError:
        return 0


def last_step(traj_file):
    """Parse the last non-empty line of `traj_file`; None if unavailable.

    Reads backwards from EOF in chunks so file size doesn't matter.
    """
    try:
        with open(traj_file, "rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            buf = b""
            while end > 0:
                size = min(_CHUNK, end)
                end -= size
                f.seek(end)
                buf = f.read(size) + buf
                # Drop trailing newlines, then look for the newline that starts
                # the final line. Finding one proves that line is whole; without
                # it the line may still extend past the left edge of `buf`, so
                # keep reading backwards.
                stripped = buf.rstrip(b"\n")
                cut = stripped.rfind(b"\n")
                if cut != -1:
                    return json.loads(stripped[cut + 1:])
                if end == 0:
                    return json.loads(stripped) if stripped.strip() else None
            return None
    except (OSError, ValueError):
        return None


def load_steps(traj_file):
    """Every step in `traj_file` as a list of dicts; malformed lines are skipped."""
    if not os.path.exists(traj_file):
        return []
    steps = []
    try:
        with open(traj_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    steps.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return steps


def read_result(task_dir):
    """Contents of `result.txt` (the score), or None if absent/unreadable."""
    try:
        with open(os.path.join(task_dir, "result.txt"), "r") as f:
            return f.read().strip()
    except OSError:
        return None


def find_screenshot(task_dir, step_n):
    """Locate `step_<n>_*.png`, falling back to an exact `step_<n>.png`."""
    if not task_dir or not os.path.isdir(task_dir):
        return None
    prefix = f"step_{step_n}_"
    try:
        for name in sorted(os.listdir(task_dir)):
            if name.startswith(prefix) and name.endswith(".png"):
                return name
    except OSError:
        return None
    exact = f"step_{step_n}.png"
    if os.path.exists(os.path.join(task_dir, exact)):
        return exact
    return None


def assign_step_images(steps, task_dir):
    """Attach `image_file` to each step — the screenshot visible WHEN it ran.

    Each step's own `screenshot_file` is its *post*-action capture, so step i
    displays step i-1's. The first step has no predecessor and falls back to
    `step_0.png` (or `step_1.png`).
    """
    if not steps:
        return
    initial = find_screenshot(task_dir, 0) or find_screenshot(task_dir, 1)
    for i, step in enumerate(steps):
        step["image_file"] = initial if i == 0 else steps[i - 1].get("screenshot_file")
