"""Infer a monitor configuration from an experiment directory alone.

The runner writes `args.json` (a dump of its argparse namespace) into the
experiment directory, which records everything the dashboard needs: which task
config it ran, where the task definitions live, and the step ceiling. Those
paths are relative to the runner's cwd (the repo root), which isn't stored —
but it *is* recoverable, because the runner also built the experiment path
itself out of args we can read back. See `_strip_result_suffix`.

Nothing here raises. A directory with no `args.json`, a moved repo, or an
unrecognized layout all yield a usable result with `warnings` explaining what
could not be determined — the dashboard stays functional on filesystem scanning
alone, and the UI surfaces the gaps instead of silently rendering nothing.
"""

import json
import os

# How far up to look for the repo root when suffix-stripping doesn't apply.
_MAX_ASCENT = 8

DEFAULT_MAX_STEPS = 100


def _read_args(experiment_path):
    """Load `args.json` from the experiment dir. Returns {} when unavailable."""
    args_file = os.path.join(experiment_path, "args.json")
    if not os.path.exists(args_file):
        return {}
    try:
        with open(args_file, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _strip_result_suffix(experiment_path, args):
    """Recover the runner's cwd by removing the path it appended to it.

    `build_result_dir()` in cluster/client/osworld/eval.py composes
    `<result_dir>/<action_space>/<observation_type>/<model>/...` (or just
    `<result_dir>/...` under `--simple_path`), so the experiment path ends with
    that suffix and everything before it is the repo root.

    Returns (repo_root, error_message). Only one is ever non-None.
    """
    result_dir = args.get("result_dir")
    if not result_dir:
        return None, "args.json has no `result_dir`"

    if args.get("simple_path"):
        suffix = os.path.normpath(result_dir)
    else:
        parts = [result_dir, args.get("action_space"),
                 args.get("observation_type"), args.get("model")]
        if not all(parts):
            return None, ("args.json is missing one of result_dir/action_space/"
                          "observation_type/model, so the repo root cannot be derived")
        suffix = os.path.normpath(os.path.join(*parts))

    # normpath keeps a leading "./" out but can leave "..", which we can't invert.
    if suffix.startswith(".."):
        return None, f"`result_dir` escapes upward ({result_dir!r}); repo root is ambiguous"

    depth = len(suffix.split(os.sep))
    segments = experiment_path.rstrip(os.sep).split(os.sep)
    if depth >= len(segments):
        return None, (f"experiment path is only {len(segments)} segments deep but "
                      f"args.json implies a {depth}-segment suffix")

    actual = os.sep.join(segments[-depth:])
    if actual != suffix:
        return None, (f"experiment path ends with {actual!r} but args.json implies "
                      f"{suffix!r} — the results may have been moved or renamed")

    return os.sep.join(segments[:-depth]) or os.sep, None


def _ascend_for(experiment_path, relative_dir):
    """Walk up from the experiment path for an ancestor containing `relative_dir`.

    Fallback for when suffix-stripping fails (moved results, hand-made layout).
    """
    if not relative_dir:
        return None
    current = experiment_path
    for _ in range(_MAX_ASCENT):
        parent = os.path.dirname(current)
        if not parent or parent == current:
            break
        current = parent
        if os.path.isdir(os.path.join(current, relative_dir)):
            return current
    return None


def _coerce_int(value, fallback):
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def derive_experiment(experiment_path, default_max_steps=DEFAULT_MAX_STEPS):
    """Infer monitor settings from `experiment_path`.

    Every derived field is accompanied by an entry in `sources` naming where it
    came from, so callers can distinguish a value read from `args.json` from one
    that was guessed or defaulted:

      args.json    read straight out of the file
      derived      computed from args.json (repo root, resolved paths)
      scan         found by walking up the filesystem
      default      nothing to go on; fell back to a built-in
      unavailable  could not be determined at all

    `warnings` holds human-readable explanations for anything unavailable.
    """
    experiment_path = os.path.abspath(experiment_path) if experiment_path else ""
    args = _read_args(experiment_path)
    sources = {}
    warnings = []

    def note(field, source):
        sources[field] = source

    action_space = args.get("action_space") or ""
    observation_type = args.get("observation_type") or ""
    model = args.get("model") or ""
    note("action_space", "args.json" if action_space else "unavailable")
    note("observation_type", "args.json" if observation_type else "unavailable")
    note("model", "args.json" if model else "unavailable")

    if args.get("max_steps") is not None:
        max_steps = _coerce_int(args.get("max_steps"), default_max_steps)
        note("max_steps", "args.json")
    else:
        max_steps = default_max_steps
        note("max_steps", "default")

    if not args:
        warnings.append(
            f"No args.json in {experiment_path or '(empty path)'} — task config and "
            "task instructions are unavailable. Tasks are listed from the directory "
            "contents alone."
        )
        for field in ("repo_root", "task_config_path", "examples_dir"):
            note(field, "unavailable")
        return {
            "experiment_path": experiment_path,
            "repo_root": None,
            "task_config_path": None,
            "examples_dir": None,
            "max_steps": max_steps,
            "model": model,
            "action_space": action_space,
            "observation_type": observation_type,
            "model_args": args,
            "sources": sources,
            "warnings": warnings,
        }

    repo_root, error = _strip_result_suffix(experiment_path, args)
    if repo_root:
        note("repo_root", "derived")
    else:
        # Suffix-stripping failed; try to find the repo by its config directory.
        repo_root = _ascend_for(experiment_path, args.get("test_config_base_dir"))
        if repo_root:
            note("repo_root", "scan")
            warnings.append(f"{error}. Found a likely repo root by scanning upward: {repo_root}")
        else:
            note("repo_root", "unavailable")
            warnings.append(error)

    def resolve(relative, field, label):
        """Join `relative` onto the repo root and confirm it exists."""
        if not relative:
            note(field, "unavailable")
            warnings.append(f"args.json does not record {label}")
            return None
        if not repo_root:
            note(field, "unavailable")
            warnings.append(
                f"{label} is {relative!r} in args.json but is relative to the repo "
                "root, which could not be determined"
            )
            return None
        resolved = os.path.normpath(os.path.join(repo_root, relative))
        if not os.path.exists(resolved):
            note(field, "unavailable")
            warnings.append(f"{label} resolved to {resolved} but that does not exist")
            return None
        note(field, sources.get("repo_root", "derived"))
        return resolved

    task_config_path = resolve(args.get("test_all_meta_path"),
                               "task_config_path", "the task config")

    examples_relative = None
    if args.get("test_config_base_dir"):
        examples_relative = os.path.join(args["test_config_base_dir"],
                                         args.get("examples_subdir") or "examples")
    examples_dir = resolve(examples_relative, "examples_dir", "the examples directory")

    return {
        "experiment_path": experiment_path,
        "repo_root": repo_root,
        "task_config_path": task_config_path,
        "examples_dir": examples_dir,
        "max_steps": max_steps,
        "model": model,
        "action_space": action_space,
        "observation_type": observation_type,
        "model_args": args,
        "sources": sources,
        "warnings": warnings,
    }
