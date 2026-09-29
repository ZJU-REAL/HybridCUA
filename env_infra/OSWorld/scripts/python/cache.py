#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import requests


HF_FILE_CACHE_PREFIX = (
    "https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/"
)
HISTORY_DB_URL = (
    "https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/"
    "chrome/44ee5668-ecd5-4366-a6ce-c1c9b8d4e938/history_empty.sqlite?download=true"
)


@dataclass(frozen=True)
class CacheTarget:
    task_id: str
    url: str
    cache_path: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prefetch OSWorld task assets into cache/<task_id>/..."
    )
    parser.add_argument(
        "--test-meta",
        default="evaluation_examples/test_all.json",
        help="Path to task meta json.",
    )
    parser.add_argument(
        "--examples-dir",
        default="evaluation_examples/examples",
        help="Path to example json directory.",
    )
    parser.add_argument(
        "--cache-dir",
        default="cache",
        help="Base cache directory used by DesktopEnv.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Concurrent download workers. Keep low if proxy is unstable.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=8,
        help="Retry count for each unique URL.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Max seconds per download attempt.",
    )
    parser.add_argument(
        "--use-curl",
        action="store_true",
        help="Prefer curl over requests when available.",
    )
    parser.add_argument(
        "--manifest-out",
        default="cache/prefetch_manifest.json",
        help="Where to write the resolved prefetch manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only resolve and print manifest summary.",
    )
    return parser.parse_args()


def is_hf_file_cache_url(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(HF_FILE_CACHE_PREFIX)


def maybe_replace_hf_endpoint(url: str) -> str:
    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    if hf_endpoint and "hf-mirror.com" in hf_endpoint:
        return url.replace("huggingface.co", "hf-mirror.com")
    return url


def setup_download_cache_path(cache_dir: str, task_id: str, url: str, vm_path: str) -> str:
    basename = os.path.basename(vm_path)
    filename = f"{uuid.uuid5(uuid.NAMESPACE_URL, url)}_{basename}"
    return os.path.join(cache_dir, task_id, filename)


def history_cache_path(cache_dir: str, task_id: str) -> str:
    return os.path.join(cache_dir, task_id, "history_new.sqlite")


def get_cloud_cache_path(cache_dir: str, task_id: str, dest: str) -> str:
    return os.path.join(cache_dir, task_id, dest)


def iter_remote_path_dest_pairs(node: Any) -> Iterable[tuple[str, str]]:
    if isinstance(node, dict):
        if "path" in node and "dest" in node:
            paths = node["path"]
            dests = node["dest"]
            if isinstance(paths, str) and isinstance(dests, str):
                if is_hf_file_cache_url(paths):
                    yield paths, dests
            elif isinstance(paths, list) and isinstance(dests, list):
                for path_value, dest_value in zip(paths, dests):
                    if is_hf_file_cache_url(path_value) and isinstance(dest_value, str):
                        yield path_value, dest_value
        for value in node.values():
            yield from iter_remote_path_dest_pairs(value)
    elif isinstance(node, list):
        for item in node:
            yield from iter_remote_path_dest_pairs(item)


def build_targets(test_meta_path: str, examples_dir: str, cache_dir: str) -> list[CacheTarget]:
    with open(test_meta_path, "r", encoding="utf-8") as f:
        test_meta = json.load(f)

    targets: dict[tuple[str, str, str], CacheTarget] = {}

    for domain, example_ids in test_meta.items():
        for task_id in example_ids:
            example_path = os.path.join(examples_dir, domain, f"{task_id}.json")
            if not os.path.exists(example_path):
                continue

            with open(example_path, "r", encoding="utf-8") as f:
                example = json.load(f)

            for step in example.get("config", []):
                step_type = step.get("type")
                params = step.get("parameters", {})

                if step_type == "download":
                    for file_info in params.get("files", []):
                        url = file_info.get("url")
                        vm_path = file_info.get("path")
                        if not is_hf_file_cache_url(url) or not isinstance(vm_path, str):
                            continue
                        cache_path = setup_download_cache_path(cache_dir, task_id, url, vm_path)
                        target = CacheTarget(task_id, url, cache_path, "setup_download")
                        targets[(task_id, url, cache_path)] = target

                elif step_type == "update_browse_history":
                    cache_path = history_cache_path(cache_dir, task_id)
                    target = CacheTarget(task_id, HISTORY_DB_URL, cache_path, "history_db")
                    targets[(task_id, HISTORY_DB_URL, cache_path)] = target

            for url, dest in iter_remote_path_dest_pairs(example):
                cache_path = get_cloud_cache_path(cache_dir, task_id, dest)
                target = CacheTarget(task_id, url, cache_path, "evaluator_cloud_file")
                targets[(task_id, url, cache_path)] = target

    return list(targets.values())


def shared_store_path(shared_dir: str, url: str) -> str:
    parsed = urlparse(url)
    basename = os.path.basename(parsed.path) or "download.bin"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return os.path.join(shared_dir, f"{digest}_{basename}")


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def copy_into_cache(src: str, dest: str) -> None:
    ensure_parent(dest)
    if os.path.exists(dest):
        src_size = os.path.getsize(src)
        dest_size = os.path.getsize(dest)
        if src_size == dest_size and src_size > 0:
            return
    shutil.copy2(src, dest)


def download_with_curl(url: str, dest: str, retries: int, timeout: int) -> None:
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is not installed")
    tmp = f"{dest}.part"
    cmd = [
        curl,
        "-L",
        "--fail",
        "--http1.1",
        "--retry",
        str(retries),
        "--retry-delay",
        "2",
        "--retry-all-errors",
        "--connect-timeout",
        "20",
        "--max-time",
        str(timeout),
        "-o",
        tmp,
        url,
    ]
    subprocess.run(cmd, check=True)
    os.replace(tmp, dest)


def download_with_requests(url: str, dest: str, retries: int, timeout: int) -> None:
    last_error: Exception | None = None
    tmp = f"{dest}.part"
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=(20, timeout)) as response:
                response.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            os.replace(tmp, dest)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt == retries:
                raise
    if last_error:
        raise last_error


def download_shared_file(
    url: str,
    shared_dir: str,
    prefer_curl: bool,
    retries: int,
    timeout: int,
) -> str:
    resolved_url = maybe_replace_hf_endpoint(url)
    shared_path = shared_store_path(shared_dir, resolved_url)
    if os.path.exists(shared_path) and os.path.getsize(shared_path) > 0:
        return shared_path

    ensure_parent(shared_path)
    if prefer_curl and shutil.which("curl"):
        download_with_curl(resolved_url, shared_path, retries, timeout)
    else:
        download_with_requests(resolved_url, shared_path, retries, timeout)
    return shared_path


def write_manifest(path: str, targets: list[CacheTarget]) -> None:
    ensure_parent(path)
    payload = [
        {
            "task_id": target.task_id,
            "url": target.url,
            "cache_path": target.cache_path,
            "reason": target.reason,
        }
        for target in targets
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def main() -> int:
    args = parse_args()
    targets = build_targets(args.test_meta, args.examples_dir, args.cache_dir)
    write_manifest(args.manifest_out, targets)

    reasons = Counter(target.reason for target in targets)
    unique_tasks = len({target.task_id for target in targets})
    grouped: dict[str, list[CacheTarget]] = defaultdict(list)
    for target in targets:
        grouped[target.url].append(target)

    print(f"Tasks covered: {unique_tasks}")
    print(f"Cache targets: {len(targets)}")
    print(f"Unique URLs: {len(grouped)}")
    print("By reason:")
    for reason, count in sorted(reasons.items()):
        print(f"  {reason}: {count}")
    print(f"Manifest: {args.manifest_out}")

    if args.dry_run:
        return 0

    shared_dir = os.path.join(args.cache_dir, ".prefetch_shared")
    os.makedirs(shared_dir, exist_ok=True)

    completed = 0
    failures: list[tuple[str, str]] = []

    def worker(url: str, targets_for_url: list[CacheTarget]) -> tuple[str, int]:
        shared_path = download_shared_file(
            url=url,
            shared_dir=shared_dir,
            prefer_curl=args.use_curl,
            retries=args.retries,
            timeout=args.timeout,
        )
        copied = 0
        for target in targets_for_url:
            copy_into_cache(shared_path, target.cache_path)
            copied += 1
        return url, copied

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(worker, url, url_targets): url
            for url, url_targets in grouped.items()
        }
        for future in as_completed(futures):
            url = futures[future]
            try:
                _, copied = future.result()
                completed += copied
                print(f"[ok] {copied:>3} targets <- {url}")
            except Exception as exc:  # noqa: BLE001
                failures.append((url, str(exc)))
                print(f"[fail] {url} :: {exc}", file=sys.stderr)

    print(f"Completed cache writes: {completed}/{len(targets)}")
    if failures:
        print(f"Failed URLs: {len(failures)}", file=sys.stderr)
        for url, message in failures[:20]:
            print(f"  {url} :: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
