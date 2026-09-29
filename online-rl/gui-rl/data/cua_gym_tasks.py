"""Load CUA-Gym task bundles into ``env.reset(task_config=...)`` payloads.

Self-contained copy of ``cluster.client.cua_gym.tasks`` from env_infra (same
reason ``clients/`` vendors the session clients: the training node must not
import the whole ``cluster`` package). Pure stdlib — keep it that way so the
two copies stay trivially diffable.

Dropped from the original, both dead: ``_PIP_INSTALL_STEP`` (CUA-Gym deps ship
pre-installed in Ubuntu-cua-gym.qcow2, so nothing referenced it) and
``load_tasks`` (the bulk loader — :class:`CuaGymDataSource` loads lazily, one
bundle per sample, so a whole-tree eager load has no caller here).

Each bundle is::

    <uuid>/
      task.json          # {id, instruction, app_type, config[], evaluator{type:python,url:"./reward.py"}, ...}
      initial_setup.*    # .py / .sh (script) or .pptx / .docx / .xlsx (data file)
      reward.py          # prints "REWARD: <float>" as its last line

Two rewrites, both confirmed across all 10910 upstream bundles:

  * Every config ``download`` url is a bundle-relative ``"./file"`` (0 are
    http/oss). OSWorld's ``_download_setup`` does ``requests.get()``, which cannot
    fetch a relative path, so we rewrite ``download`` -> ``upload_file`` sourced
    from the local bundle. The VM-side ``path`` is preserved verbatim so any later
    ``open`` / ``execute`` step and the reward script still resolve the same file.
  * 8383 ``execute`` steps carry a *string* command; OSWorld's ``_execute_setup``
    defaults ``shell=False``, which breaks a bare ``"python3 x.py"`` string, so we
    add ``shell=True``. (1734 ``execute`` steps use a *list* command — passed
    through untouched, as are ``sleep`` / ``open`` / ``launch``.)

``reward.py`` is inlined as ``reward_code`` so evaluation is self-contained (no
bundle needed at eval time; ``CuaGymWorldAdapter`` pops it at reset, runs it in
the VM at evaluate, and parses ``REWARD: X.X``). ``ground_truth`` / ``difficulty``
/ ``app_type`` are ignored by DesktopEnv and pass through harmlessly.

The ``pip.conf`` setup step only points pip at a reachable mirror for any extra
deps an ``initial_setup.sh`` installs itself (those use ``set -e``, so a pip
failure aborts the whole setup); the snapshot rollback at reset wipes it, hence
re-applying every episode.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union


def _localize_setup(steps: List[Dict[str, Any]], bundle: Path) -> List[Dict[str, Any]]:
    """Make setup steps node-local and OSWorld-executable (see module docstring)."""
    out: List[Dict[str, Any]] = []
    for step in steps:
        stype = step.get("type")
        params = dict(step.get("parameters", {}))
        if stype == "download":
            uploads: List[Dict[str, str]] = []
            keep: List[Dict[str, str]] = []
            for f in step.get("parameters", {}).get("files", []):
                url, vm_path = f.get("url", ""), f.get("path", "")
                if url.startswith(("http://", "https://")):
                    keep.append(f)  # genuinely remote & reachable: leave as a download
                else:
                    # "./file" (also covers legacy oss:// / placeholders): ship the
                    # bundle's own copy. Keep vm_path so later steps still match.
                    uploads.append(
                        {"local_path": str(bundle / os.path.basename(url)), "path": vm_path}
                    )
            if uploads:
                out.append({"type": "upload_file", "parameters": {"files": uploads}})
            if keep:
                out.append({"type": "download", "parameters": {"files": keep}})
        elif stype == "execute" and isinstance(params.get("command"), str):
            params.setdefault("shell", True)  # a string command must run under a shell
            out.append({**step, "parameters": params})
        else:
            out.append(step)  # sleep / open / launch / list-form execute: verbatim
    return out


_PIP_CONFIG_STEP: Dict[str, Any] = {
    "type": "execute",
    "parameters": {
        "command": (
            "mkdir -p ~/.pip && printf '[global]\\n"
            "timeout = 120\\n"
            "index-url = https://pypi.tuna.tsinghua.edu.cn/simple\\n"
            "retries = 3\\n' > ~/.pip/pip.conf"
        ),
        "shell": True,
    },
}


def load_task(bundle_dir: Union[str, Path]) -> Dict[str, Any]:
    """Read one bundle dir into the dict passed to ``env.reset(task_config=...)``."""
    bundle = Path(bundle_dir).resolve()  # absolute: the node's setup_controller reads local_path
    task_file = bundle / "task.json"
    if not task_file.exists():
        task_file = bundle / "config.json"  # tolerate pipeline-native naming
    task = json.loads(task_file.read_text())
    task["config"] = _localize_setup(task.get("config", []), bundle)
    cfg = task.get("config") or []
    if not cfg or cfg[0] != _PIP_CONFIG_STEP:
        task["config"] = [_PIP_CONFIG_STEP] + cfg
    reward = bundle / "reward.py"  # what evaluator.url ("./reward.py") points at
    task["reward_code"] = reward.read_text() if reward.exists() else None
    return task
