from __future__ import annotations

import argparse
import json
from pathlib import Path


def _iter_meta(meta_path: Path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    for domain, example_ids in meta.items():
        for example_id in example_ids:
            yield str(domain), str(example_id)


def build_dataset(base_dir: Path, meta_path: Path, output_path: Path) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for domain, example_id in _iter_meta(meta_path):
            cfg_path = base_dir / "examples" / domain / f"{example_id}.json"
            if not cfg_path.exists():
                continue
            with open(cfg_path, "r", encoding="utf-8") as f:
                task = json.load(f)
            row = {
                "prompt": task.get("instruction", ""),
                "label": "",
                "metadata": {
                    "domain": domain,
                    "example_id": example_id,
                    "instruction": task.get("instruction", ""),
                    "task_config": task,
                },
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GUI RL jsonl from evaluation_examples meta json")
    parser.add_argument("--base-dir", type=str, required=True)
    parser.add_argument("--meta-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    count = build_dataset(
        base_dir=Path(args.base_dir),
        meta_path=Path(args.meta_path),
        output_path=Path(args.output_path),
    )
    print(f"wrote {count} samples to {args.output_path}")


if __name__ == "__main__":
    main()
