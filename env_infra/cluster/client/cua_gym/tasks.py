"""Load CUA-Gym HF task bundles into env.reset() payloads.

Verified against the real HF release (xlangai/CUA-Gym, 10910 bundles). Each
bundle is:

    <uuid>/
      task.json          # {id, instruction, app_type, config[], evaluator{type:python,url:"./reward.py"}, difficulty?, ground_truth?}
      initial_setup.*    # .py / .sh (script) or .pptx / .docx / .xlsx (data file)
      reward.py          # prints "REWARD: <float>" as its last line

Two facts drive the two rewrites below (both confirmed across all 10910 tasks):

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
bundle needed at eval time; CuaGymWorldAdapter runs it in the VM and parses
``REWARD: X.X``). ``ground_truth`` (666 tasks) is dead metadata — empty and never
read by reward.py — and ``difficulty`` / ``app_type`` are ignored by DesktopEnv;
all are passed through harmlessly.
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


# CUA-Gym bundles' initial_setup.py needs python-pptx/docx/openpyxl/PyMuPDF etc.,
# but the OSWorld docker image (happysixd/osworld-docker) doesn't ship them — so
# setup silently fails (ModuleNotFoundError) and reward=0. Inject a pip install as
# the FIRST setup step so reset runs it before initial_setup.py. ~15-30s, cached
# by pip after the first task on a given VM. Verified: replay pass-rate 21%→87%.
#
# TWO steps: (1) configure pip to use the proxy + Tsinghua mirror — VM has no
# direct pypi access and reset wipes any pre-config, so this MUST run before any
# pip install (incl. the pip installs inside initial_setup.sh, which use `set -e`
# and abort the whole setup on pip failure). (2) pre-install the common deps.
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
_PIP_INSTALL_STEP: Dict[str, Any] = {
    "type": "execute",
    "parameters": {
        "command": (
            "for i in 1 2 3; do "
            "pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "
            "--disable-pip-version-check python-pptx python-docx openpyxl PyMuPDF "
            "odfpy SQLAlchemy pandas fpdf2 python-dotenv pydantic piexif psutil "
            "PyPDF2 pypdf pikepdf pdfplumber && "
            "python3 -c 'import pptx,docx,openpyxl,fitz,odf,sqlalchemy,pandas,fpdf,dotenv,pydantic,piexif,psutil,pypdf,pikepdf,pdfplumber' && break; "
            "echo \"pip install attempt $i failed, retrying...\"; sleep 3; done"
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
    # CUA-Gym deps are pre-installed in the custom qcow2 (Ubuntu-cua-gym.qcow2).
    # No pip install needed at reset time — just ensure pip can reach pypi for any
    # extra deps inside initial_setup.sh (configure mirror, no proxy needed).
    cfg = task.get("config") or []
    if not cfg or cfg[0] != _PIP_CONFIG_STEP:
        task["config"] = [_PIP_CONFIG_STEP] + cfg
    reward = bundle / "reward.py"  # what evaluator.url ("./reward.py") points at
    task["reward_code"] = reward.read_text() if reward.exists() else None
    return task


def load_tasks(root: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load every bundle under ``root`` (a directory of ``<uuid>/`` bundles)."""
    root = Path(root)
    return [
        load_task(d)
        for d in sorted(root.iterdir())
        if d.is_dir() and ((d / "task.json").exists() or (d / "config.json").exists())
    ]
