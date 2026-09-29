"""CUA-Gym eval task source.

CUA-Gym's task model is a flat directory of ``<uuid>/`` bundles (each with
``task.json`` + ``reward.py`` + ``initial_setup.*``), with ``app_type`` acting as
the "domain". Everything else about evaluation is identical to OSWorld — the same
DesktopEnv-compatible env, the same ``lib_run_single.run_single_example`` episode
loop (env.evaluate() routes to CuaGymWorldAdapter, which runs reward.py), the same
result-dir layout, resume, and summary. So this subclasses
:class:`OSWorldEvalSource` and overrides only the two task-model methods:

- :meth:`load_tasks`   — scan ``--tasks_root`` for bundles instead of reading a
  ``test_all_meta`` JSON; key each task by ``(app_type, bundle_dir_name)``.
- :meth:`prepare_example` — build the reset payload with the CUA-Gym loader
  instead of reading ``examples/{domain}/{id}.json``.

``pending_tasks`` (resume), ``run_episode``, ``item_label``, ``error_dump``,
``save_args`` and ``summarize`` are inherited unchanged — they treat the
``(app_type, id)`` pair exactly as OSWorld treats ``(domain, example_id)``.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os

from cluster.client.cua_gym.tasks import load_task
from cluster.client.osworld.eval import OSWorldEvalSource, build_result_dir

logger = logging.getLogger("desktopenv.experiment")


class CuaGymEvalSource(OSWorldEvalSource):
    """CUA-Gym task model: a flat dir of ``<uuid>/`` bundles, ``app_type`` as domain."""

    def load_tasks(self, args: argparse.Namespace) -> list:
        root = args.tasks_root
        if not root or not os.path.isdir(root):
            raise FileNotFoundError(f"--tasks_root not found: {root!r}")
        # A meta JSON (OSWorld's test_all_meta shape: ``{app_type: [uuid, ...]}``)
        # pins an exact task subset; without it we scan every bundle under root.
        meta_path = getattr(args, "tasks_meta", None)
        items = self._tasks_from_meta(root, meta_path) if meta_path else self._tasks_from_scan(root)
        if args.domain != "all":
            selected = {x.strip() for x in args.domain.split(",") if x.strip()}
            unknown = selected - {app for app, _ in items}
            if unknown:
                logger.warning("no tasks for requested app_type(s): %s", sorted(unknown))
            items = [it for it in items if it[0] in selected]
        return items

    def _tasks_from_scan(self, root: str) -> list:
        """Every ``<uuid>/`` bundle under ``root`` (default: run the whole set)."""
        items: list = []
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            if not os.path.isdir(d):
                continue
            task_json = next((p for n in ("task.json", "config.json")
                              if os.path.exists(p := os.path.join(d, n))), None)
            if task_json is None:
                continue
            try:
                app_type = json.load(open(task_json, encoding="utf-8")).get("app_type", "unknown")
            except Exception:
                logger.warning("skipping unreadable %s in %s", os.path.basename(task_json), d)
                continue
            # Key by the on-disk dir name (== id for HF bundles); prepare_example
            # locates the bundle by this name, so it must be the real directory.
            items.append((app_type, os.path.basename(d)))
        return items

    def _tasks_from_meta(self, root: str, meta_path: str) -> list:
        """Only the ``(app_type, uuid)`` pairs listed in an OSWorld-shaped meta JSON.

        ``{app_type: [uuid, ...]}`` — the app_type key is the domain, exactly as
        OSWorld's ``test_all_meta``. A listed bundle whose dir is missing under
        ``root`` is skipped with a warning (the run continues on the rest).
        """
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"--tasks_meta not found: {meta_path!r}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            raise ValueError(
                f"--tasks_meta must be a JSON object {{app_type: [uuid, ...]}}, got {type(meta).__name__}"
            )
        items: list = []
        for app_type, uuids in meta.items():
            if not isinstance(uuids, list):
                continue
            for uuid in uuids:
                if not any(os.path.exists(os.path.join(root, uuid, n))
                           for n in ("task.json", "config.json")):
                    logger.warning("tasks_meta lists %r but no bundle under %s; skipping", uuid, root)
                    continue
                items.append((app_type, uuid))
        return items

    def prepare_example(self, args: argparse.Namespace, task_item) -> tuple:
        app_type, bundle_name = task_item
        example = load_task(os.path.join(args.tasks_root, bundle_name))
        result_dir = build_result_dir(args, app_type, bundle_name, model=self.model_name)
        os.makedirs(result_dir, exist_ok=True)
        return example, result_dir
