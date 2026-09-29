#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flask backend for the OSWorld task dashboard.

Point it at an experiment directory and everything else is inferred: the task
list comes from scanning that directory, and `args.json` inside it supplies the
task config, the examples directory, and the step ceiling. Whatever cannot be
inferred is reported in `warnings` rather than silently omitted — an earlier
version listed tasks from a hand-entered config path and rendered a blank page
when it was wrong.

Scanning network storage is slow, so status computation happens on a background
thread (see `core/status.py`) and requests return whatever is ready along with
progress, letting the front-end poll until it settles.
"""

import json
import os
import shutil
from urllib.parse import urlencode

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file

from core import (
    DEFAULT_MAX_STEPS,
    NO_INFO,
    IndexRegistry,
    InstructionRegistry,
    ROOT_TYPE,
    assign_step_images,
    compute_status,
    derive_experiment,
    forget_task_list,
    is_contained,
    load_steps,
    load_task_info,
    load_task_list,
    safe_path,
)

load_dotenv()

app = Flask(__name__)

# Optional seeds for the path inputs; users normally type or bookmark their own.
SEED_EXPERIMENT_PATH = os.getenv("MONITOR_EXPERIMENT_PATH", "")
SEED_COMPARE_PATH = os.getenv("MONITOR_COMPARE_PATH", "")
FALLBACK_MAX_STEPS = int(os.getenv("MAX_STEPS", str(DEFAULT_MAX_STEPS)))

_indexes = IndexRegistry()
_instructions = InstructionRegistry()
_derivation_cache = {}   # {experiment_path: (args_mtime, derivation)}


@app.template_filter("pretty_json")
def _pretty_json(value):
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


# ---------- request plumbing ----------

def experiment_path():
    """The experiment directory for this request.

    `results_base_path` is the pre-auto-discovery parameter name, still accepted
    so existing bookmarks keep working.
    """
    raw = request.args.get("path") or request.args.get("results_base_path") or ""
    return os.path.abspath(raw) if raw else ""


def _back_query(path):
    """Query string that returns to the dashboard the user came from.

    A compare view links to *one* side's trajectory, so `path` here is whichever
    side was clicked. Sending that back as the dashboard's only parameter would
    reopen compare mode with the clicked side on both halves — B compared against
    itself. Carrying `compare_with` and `mode` through the detail page keeps the
    original pair, and `side` records which half to restore it into.
    """
    mode = request.args.get("mode")
    other = request.args.get("compare_with") or ""
    if mode != "compare" or not other:
        return urlencode({"path": path})
    # `side` is which half the clicked run occupied, so A/B don't swap on return.
    return urlencode({
        "path": other if request.args.get("side") == "b" else path,
        "path_b": path if request.args.get("side") == "b" else other,
        "mode": "compare",
    })


def derivation_for(path):
    """Derived settings for `path`, recomputed when its args.json changes."""
    if not path:
        return derive_experiment("", FALLBACK_MAX_STEPS)
    try:
        mtime = os.path.getmtime(os.path.join(path, "args.json"))
    except OSError:
        mtime = None
    cached = _derivation_cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    derived = derive_experiment(path, FALLBACK_MAX_STEPS)
    _derivation_cache[path] = (mtime, derived)
    return derived


def index_for(path, derived):
    return _indexes.get(path, derived["max_steps"])


# ---------- task list ----------

NOT_STARTED = "Not Started"


def _not_started(max_steps):
    return {"status": NOT_STARTED, "progress": 0, "max_steps": max_steps,
            "last_update": None, "result": None}


def build_task_list(path, derived):
    """Tasks on disk, plus never-started ones from the config if it resolved.

    The filesystem is authoritative — the config only *adds* entries, so a
    missing or unreadable config degrades the view instead of emptying it.
    """
    index = index_for(path, derived)
    index.refresh()
    snapshot = index.snapshot()

    instructions = _instructions.get(derived["examples_dir"])

    def instruction_for(task_type, task_id):
        if instructions is None:
            return NO_INFO
        return instructions.instruction_or_placeholder(task_type, task_id)

    grouped = {}
    seen = set()
    for task_type, task_id, _ in snapshot["tasks"]:
        seen.add((task_type, task_id))
        grouped.setdefault(task_type, []).append({
            "id": task_id,
            "instruction": instruction_for(task_type, task_id),
            "status": snapshot["statuses"].get((task_type, task_id)),
        })

    pending = 0
    for task_type, task_ids in load_task_list(derived["task_config_path"]).items():
        for task_id in task_ids:
            if (task_type, task_id) in seen:
                continue
            pending += 1
            grouped.setdefault(task_type, []).append({
                "id": task_id,
                "instruction": instruction_for(task_type, task_id),
                "status": _not_started(derived["max_steps"]),
            })

    for entries in grouped.values():
        entries.sort(key=lambda e: e["id"])

    scan = dict(snapshot["scan"])
    scan["not_started"] = pending
    scan["instructions_ready"] = instructions.ready() if instructions else True
    scan["complete"] = scan["complete"] and scan["instructions_ready"]
    return {"tasks": grouped, "scan": scan}


def find_task_dir(path, task_type, task_id):
    """Absolute path of a task's directory, or None if it isn't on disk."""
    if not path:
        return None
    relative = task_id if task_type == ROOT_TYPE else os.path.join(task_type, task_id)
    candidate = os.path.join(path, relative)
    return candidate if os.path.isdir(candidate) else None


def _load_run(task_type, task_id, path):
    """One run's view of a task: its status, steps, and screenshots.

    None when the path is empty or holds no directory for this task, which is how
    a comparison renders a task only one side ran.
    """
    if not path:
        return None
    derived = derivation_for(path)
    task_dir = find_task_dir(path, task_type, task_id)
    if not task_dir:
        return None
    status = compute_status(task_dir, derived["max_steps"])
    status["steps"] = load_steps(os.path.join(task_dir, "traj.jsonl"))
    assign_step_images(status["steps"], task_dir)
    return {"path": path, "status": status,
            "actions": [st.get("action") for st in status["steps"]]}


# ---------- pages ----------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/task/<path:task_type>/<task_id>")
def task_detail(task_type, task_id):
    """One task, from one run or from both side by side.

    Arriving from a comparison (`compare_with` set) renders both runs' steps in
    parallel columns: comparing two runs means comparing their screenshots, and
    a link to the other side would make you hold one of them in your head.
    """
    path = experiment_path()
    derived = derivation_for(path)
    info = load_task_info(derived["examples_dir"], task_type, task_id)

    other = request.args.get("compare_with") or ""
    other = os.path.abspath(other) if other else ""
    # `side` says which half the clicked run occupied, so A stays A on this page.
    flipped = request.args.get("side") == "b"
    paths = (other, path) if flipped else (path, other)

    runs = [
        {"label": label, "run": _load_run(task_type, task_id, p)}
        for label, p in zip(("A", "B"), paths)
    ]
    present = [r for r in runs if r["run"]]
    if not present and not info:
        return "Task not found", 404
    if not present:
        runs = [{"label": "", "run": {"path": path, "actions": [],
                                     "status": dict(_not_started(derived["max_steps"]), steps=[])}}]
    elif not other:
        # Single-run visit: no A/B labelling, since there is nothing to contrast.
        runs = [{"label": "", "run": present[0]["run"]}]

    return render_template(
        "task_detail.html",
        task_id=task_id, task_type=task_type,
        task_info=info or {"instruction": NO_INFO},
        runs=runs,
        compare=bool(other),
        results_base_path=path,
        back_query=_back_query(path),
    )


# ---------- api ----------

@app.route("/api/settings")
def api_settings():
    """Seed values for the front-end form."""
    return jsonify({"path": SEED_EXPERIMENT_PATH, "path_b": SEED_COMPARE_PATH})


@app.route("/api/experiment")
def api_experiment():
    """Everything inferred from the experiment path, with provenance."""
    path = experiment_path()
    derived = dict(derivation_for(path))
    derived["exists"] = bool(safe_path(path))
    if path and not derived["exists"]:
        derived["warnings"] = [f"{path} is not an existing directory"] + derived["warnings"]
    return jsonify(derived)


@app.route("/api/tasks")
def api_tasks():
    path = experiment_path()
    if not safe_path(path):
        return jsonify({"tasks": {}, "scan": {"scanned": False, "done": 0, "total": 0,
                                              "complete": True, "not_started": 0}})
    return jsonify(build_task_list(path, derivation_for(path)))


# Superseded by /api/tasks; kept so old front-end bundles and scripts keep working.
@app.route("/api/tasks/brief")
def api_tasks_brief():
    payload = api_tasks().get_json()
    return jsonify(payload["tasks"])


@app.route("/api/task/<path:task_type>/<task_id>")
def api_task_detail(task_type, task_id):
    path = experiment_path()
    derived = derivation_for(path)
    task_dir = find_task_dir(path, task_type, task_id)
    info = load_task_info(derived["examples_dir"], task_type, task_id)
    if not task_dir and not info:
        return jsonify({"error": "Task does not exist"}), 404
    if task_dir:
        status = compute_status(task_dir, derived["max_steps"])
        status["steps"] = load_steps(os.path.join(task_dir, "traj.jsonl"))
        assign_step_images(status["steps"], task_dir)
    else:
        status = _not_started(derived["max_steps"])
        status["steps"] = []
    return jsonify({"info": info, "status": status})


def _send_task_file(task_type, task_id, filename, mimetype, extra_headers=None):
    """Serve a file from inside a task directory, never from outside it."""
    path = experiment_path()
    base = safe_path(path)
    if not base:
        return "Invalid experiment path", 400
    task_dir = find_task_dir(base, task_type, task_id)
    if not task_dir:
        return "Task directory does not exist", 404
    target = os.path.join(task_dir, filename)
    if not is_contained(base, target):
        return "Path escapes the experiment directory", 400
    if not os.path.exists(target):
        return "File does not exist", 404
    response = send_file(target, mimetype=mimetype)
    for key, value in (extra_headers or {}).items():
        response.headers[key] = value
    return response


@app.route("/task/<path:task_type>/<task_id>/screenshot/<path:filename>")
def task_screenshot(task_type, task_id, filename):
    return _send_task_file(task_type, task_id, filename, "image/png")


@app.route("/task/<path:task_type>/<task_id>/recording")
def task_recording(task_type, task_id):
    return _send_task_file(task_type, task_id, "recording.mp4", "video/mp4", {
        "Accept-Ranges": "bytes",
        "Cache-Control": "public, max-age=3600",
        "X-Content-Type-Options": "nosniff",
    })


@app.route("/api/clear-cache", methods=["POST"])
def api_clear_cache():
    """Forget everything cached for an experiment and re-read it from disk.

    Four caches back a single view, and a partial clear is worse than none — it
    leaves the page showing a mix of fresh and stale data. So the derivation is
    resolved *before* being dropped, to learn which examples directory and task
    config this experiment pointed at.
    """
    path = experiment_path()
    # Read the derivation before dropping it: it names the examples directory and
    # task config to clear. Note whether it was already cached first, since
    # resolving it here would otherwise make args.json always look like a hit.
    had_derivation = path in _derivation_cache
    derived = derivation_for(path)

    cleared = []
    if _indexes.drop(path):
        cleared.append("task status")
    if _instructions.drop(derived["examples_dir"]):
        cleared.append("instructions")
    if forget_task_list(derived["task_config_path"]):
        cleared.append("task config")
    _derivation_cache.pop(path, None)
    if had_derivation:
        cleared.append("args.json")

    if not cleared:
        return jsonify({"message": f"Nothing cached for: {path or '(empty)'}",
                        "cleared": []})
    return jsonify({"message": f"Cleared {', '.join(cleared)} for: {path}",
                    "cleared": cleared})


# ---------- destructive: cleanup / reset ----------

def _remove_task_dir(base, task_type, task_id):
    """Delete one task's result directory. Returns (ok, message_or_path)."""
    if not task_type or not task_id:
        return False, "Missing task_type or task_id"
    task_dir = find_task_dir(base, task_type, task_id)
    if not task_dir:
        return False, f"Not a directory: {task_type}/{task_id}"
    if not is_contained(base, task_dir):
        return False, f"Unsafe path: {task_type}/{task_id}"
    try:
        shutil.rmtree(task_dir)
    except OSError as exc:
        return False, f"{task_type}/{task_id}: {exc}"
    return True, task_dir


@app.route("/api/cleanup/preview")
def api_cleanup_preview():
    """Task directories whose status is not a finished `Done*` state."""
    path = experiment_path()
    base = safe_path(path)
    if not base:
        return jsonify({"items": []})

    derived = derivation_for(base)
    index = index_for(base, derived)
    index.refresh()
    snapshot = index.snapshot()

    items = []
    for task_type, task_id, task_dir in snapshot["tasks"]:
        status = snapshot["statuses"].get((task_type, task_id))
        if status is None:
            # Still being computed in the background; ask directly rather than
            # letting an unknown task slip into a delete list.
            status = index.status_of(task_type, task_id, task_dir)
        if status["status"].startswith("Done"):
            continue
        items.append({
            "task_type": task_type,
            "task_id": task_id,
            "path": task_dir,
            "status": status["status"],
            "progress": status.get("progress") or 0,
            "max_steps": status.get("max_steps") or 0,
            "last_update": status.get("last_update"),
        })
    return jsonify({"items": items})


@app.route("/api/cleanup/execute", methods=["POST"])
def api_cleanup_execute():
    """Delete the caller-confirmed subset. Body: `{items: [{task_type, task_id}, ...]}`."""
    base = safe_path(experiment_path())
    if not base:
        return jsonify({"deleted": [], "errors": ["No experiment path configured"]}), 400

    items = (request.get_json(silent=True) or {}).get("items") or []
    deleted, errors = [], []
    for item in items:
        ok, message = _remove_task_dir(base, item.get("task_type", ""), item.get("task_id", ""))
        (deleted if ok else errors).append(message)
    if deleted:
        _indexes.drop(base)
    return jsonify({"deleted": deleted, "errors": errors})


@app.route("/api/task/<path:task_type>/<task_id>/reset", methods=["POST"])
def api_task_reset(task_type, task_id):
    """Delete a single task's result directory so it can be re-run."""
    base = safe_path(experiment_path())
    if not base:
        return jsonify({"error": "No experiment path configured"}), 400
    ok, message = _remove_task_dir(base, task_type, task_id)
    if not ok:
        return jsonify({"error": message}), 400
    _indexes.drop(base)
    return jsonify({"ok": True, "path": message})


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "8080"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug, threaded=True)
