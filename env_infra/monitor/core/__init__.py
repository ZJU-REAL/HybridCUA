"""Reading an experiment directory: discovery, derivation, and status.

Kept separate from `main.py` so the Flask layer stays request plumbing while
these rules stay independently testable. The modules follow the pipeline the
dashboard runs on every experiment:

    derive     args.json  -> task config, examples dir, max_steps (+ provenance)
    discover   filesystem -> which task directories exist
    traj       task dir   -> steps, screenshots, score
    tasks      task config + examples dir -> the task list and its instructions
    status     the above  -> a status label, cached and filled in the background
    paths      guards for every path that arrives over HTTP

Re-exported here is what the Flask layer needs. The lower-level readers
(`traj.count_steps`, `discover.scan_task_dirs`, ...) are used by `status` and by
tests, which import them from their own modules.
"""

from .derive import DEFAULT_MAX_STEPS, derive_experiment
from .discover import ROOT_TYPE
from .paths import is_contained, safe_path
from .status import IndexRegistry, compute_status
from .tasks import (
    NO_INFO,
    InstructionRegistry,
    forget_task_list,
    load_task_info,
    load_task_list,
)
from .traj import assign_step_images, load_steps

__all__ = [
    "DEFAULT_MAX_STEPS",
    "NO_INFO",
    "IndexRegistry",
    "InstructionRegistry",
    "ROOT_TYPE",
    "assign_step_images",
    "compute_status",
    "derive_experiment",
    "forget_task_list",
    "is_contained",
    "load_steps",
    "load_task_info",
    "load_task_list",
    "safe_path",
]
