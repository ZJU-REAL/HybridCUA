#!/usr/bin/env python3
"""Clean incomplete or failed OSWorld evaluation results.

The script is dry-run by default. Pass --apply to remove task directories and,
for summary-aware cleanup, rewrite summary/results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_ROOT = Path("results")


@dataclass(frozen=True)
class TaskDir:
    application: str
    task_id: str
    path: Path

    @property
    def key(self) -> tuple[str, str]:
        return (self.application, self.task_id)


def is_uuid_name(name: str) -> bool:
    try:
        uuid.UUID(name)
    except ValueError:
        return False
    return True


def iter_task_dirs(root: Path) -> Iterable[TaskDir]:
    """Yield UUID-named task dirs without descending into them."""
    for dirpath, dirnames, _ in os.walk(root):
        dirnames.sort()
        task_names = {dirname for dirname in dirnames if is_uuid_name(dirname)}

        for task_name in sorted(task_names):
            task_path = Path(dirpath) / task_name
            yield TaskDir(
                application=task_path.parent.name,
                task_id=task_name,
                path=task_path,
            )

        dirnames[:] = [dirname for dirname in dirnames if dirname not in task_names]


def result_key(entry: dict) -> tuple[str, str] | None:
    application = entry.get("application")
    task_id = entry.get("task_id")
    if not isinstance(application, str) or not isinstance(task_id, str):
        return None
    return (application, task_id)


def load_results(results_file: Path) -> list[dict]:
    if not results_file.is_file():
        raise ValueError(f"results file does not exist: {results_file}")
    try:
        data = json.loads(results_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {results_file}: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON list in {results_file}")
    return data


def write_results(results_file: Path, entries: list[dict]) -> None:
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def parse_score(value) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if text.lower() == "true":
            return 1.0
        if text.lower() == "false":
            return 0.0
        try:
            return float(text)
        except ValueError:
            return None
    return None


def is_success_score(value) -> bool:
    score = parse_score(value)
    return score is not None and abs(score - 1.0) < 1e-9


def task_result_score(task_dir: TaskDir) -> float | None:
    result_file = task_dir.path / "result.txt"
    if not result_file.is_file():
        return None
    score = parse_score(result_file.read_text(encoding="utf-8", errors="replace"))
    return score if score is not None else 0.0


def classify_failed_dirs(
    task_dirs: list[TaskDir],
    summary_failed_keys: set[tuple[str, str]],
    summary_success_keys: set[tuple[str, str]],
) -> dict[str, list[TaskDir]]:
    classified = {
        "summary_failed": [],
        "result_failed": [],
    }

    for task_dir in task_dirs:
        if task_dir.key in summary_failed_keys and task_dir.key not in summary_success_keys:
            classified["summary_failed"].append(task_dir)
            continue

        score = task_result_score(task_dir)
        if score is None:
            continue

        if not is_success_score(score):
            classified["result_failed"].append(task_dir)

    return classified


def unique_task_dirs(groups: Sequence[list[TaskDir]]) -> list[TaskDir]:
    dirs_to_remove: list[TaskDir] = []
    seen_paths: set[Path] = set()
    for group in groups:
        for task_dir in group:
            if task_dir.path not in seen_paths:
                dirs_to_remove.append(task_dir)
                seen_paths.add(task_dir.path)
    return dirs_to_remove


def print_category(name: str, dirs: list[TaskDir], root: Path, limit: int, action: str) -> None:
    print(f"{name}: {len(dirs)}")
    shown = dirs if limit == 0 else dirs[:limit]
    for task_dir in shown:
        print(f"  {action}: {relative_to_root(task_dir.path, root)}")
    if limit and len(dirs) > limit:
        print(f"  ... {len(dirs) - limit} more omitted")


def remove_task_dirs(task_dirs: list[TaskDir]) -> int:
    removed = 0
    for task_dir in task_dirs:
        if task_dir.path.is_symlink():
            print(f"skip symlink task dir: {task_dir.path}", file=sys.stderr)
            continue
        shutil.rmtree(task_dir.path)
        removed += 1
    return removed


def parse_args(
    argv: Sequence[str] | None = None,
    default_root: Path = DEFAULT_ROOT,
    default_mode: str = "missing",
    default_limit: int = 80,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean OSWorld task result directories.",
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=default_root,
        help=f"results root to clean (default: {default_root})",
    )
    parser.add_argument(
        "--mode",
        choices=["missing", "failed"],
        default=default_mode,
        help=(
            "missing: delete UUID task dirs without result.txt; "
            "failed: delete task dirs whose score is not 1 and rewrite summary/results.json"
        ),
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help="summary results JSON file for --mode failed (default: ROOT/summary/results.json)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually delete task dirs and rewrite results.json when needed; default is dry-run",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="alias for --apply",
    )
    parser.add_argument(
        "--keep-missing-result-txt",
        action="store_true",
        help="deprecated compatibility flag; use --mode missing to handle missing result.txt dirs",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="in --mode failed, do not create results.json backup when --apply rewrites it",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=default_limit,
        help="print at most N directories per category; 0 means print all",
    )
    args = parser.parse_args(argv)
    if getattr(args, "delete", False):
        args.apply = True
    return args


def cleanup_missing(root: Path, args: argparse.Namespace) -> int:
    task_dirs = list(iter_task_dirs(root))
    missing_dirs = [
        task_dir for task_dir in task_dirs if not (task_dir.path / "result.txt").is_file()
    ]

    action = "delete" if args.apply else "would delete"
    print(f"mode: {'apply' if args.apply else 'dry-run'}")
    print(f"cleanup_mode: missing")
    print(f"root: {root}")
    print(f"task_dirs_scanned: {len(task_dirs)}")
    print_category("missing_result_txt_dirs", missing_dirs, root, args.limit, action)

    if not args.apply:
        print("dry-run only; pass --apply to perform cleanup")
        return 0

    removed = remove_task_dirs(missing_dirs)
    print(f"removed_task_dirs: {removed}")
    print("cleanup complete")
    return 0


def cleanup_failed(root: Path, args: argparse.Namespace) -> int:
    results_file = (
        args.results_file.expanduser().resolve()
        if args.results_file
        else root / "summary" / "results.json"
    )

    try:
        entries = load_results(results_file)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    success_entries = [entry for entry in entries if is_success_score(entry.get("score"))]
    failed_entries = [entry for entry in entries if not is_success_score(entry.get("score"))]
    success_keys = {key for entry in success_entries if (key := result_key(entry)) is not None}
    failed_keys = {key for entry in failed_entries if (key := result_key(entry)) is not None}

    task_dirs = list(iter_task_dirs(root))
    classified = classify_failed_dirs(
        task_dirs=task_dirs,
        summary_failed_keys=failed_keys,
        summary_success_keys=success_keys,
    )

    dirs_to_remove = unique_task_dirs(
        [
            classified["summary_failed"],
            classified["result_failed"],
        ]
    )
    result_failed_keys = {task_dir.key for task_dir in classified["result_failed"]}
    rewritten_entries = [
        entry
        for entry in success_entries
        if result_key(entry) not in result_failed_keys
    ]

    action = "delete" if args.apply else "would delete"
    print(f"mode: {'apply' if args.apply else 'dry-run'}")
    print(f"cleanup_mode: failed")
    print(f"root: {root}")
    print(f"results_file: {results_file}")
    print(f"summary_entries: {len(entries)}")
    print(f"summary_score_not_1_entries_to_remove: {len(failed_entries)}")
    print(f"summary_entries_after_cleanup: {len(rewritten_entries)}")
    print(f"task_dirs_scanned: {len(task_dirs)}")
    print_category("summary_score_not_1_task_dirs", classified["summary_failed"], root, args.limit, action)
    print_category("result_txt_score_not_1_task_dirs", classified["result_failed"], root, args.limit, action)

    if not args.apply:
        print("dry-run only; pass --apply to perform cleanup")
        return 0

    backup_path = None
    if results_file.exists() and not args.no_backup:
        backup_path = results_file.with_name(
            f"{results_file.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        shutil.copy2(results_file, backup_path)

    removed = remove_task_dirs(dirs_to_remove)
    write_results(results_file, rewritten_entries)

    if backup_path is not None:
        print(f"backup: {backup_path}")
    print(f"removed_task_dirs: {removed}")
    print("cleanup complete")
    return 0


def main(
    argv: Sequence[str] | None = None,
    default_root: Path = DEFAULT_ROOT,
    default_mode: str = "missing",
    default_limit: int = 80,
) -> int:
    args = parse_args(
        argv=argv,
        default_root=default_root,
        default_mode=default_mode,
        default_limit=default_limit,
    )
    root = args.root.expanduser().resolve()

    if not root.is_dir():
        print(f"error: results root does not exist or is not a directory: {root}", file=sys.stderr)
        return 1

    if args.mode == "missing":
        return cleanup_missing(root, args)
    return cleanup_failed(root, args)


if __name__ == "__main__":
    raise SystemExit(main())
