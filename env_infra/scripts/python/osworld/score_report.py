#!/usr/bin/env python3
"""Per-domain report for an OSWorld result dir: score / accuracy / avg-step,
GUI-vs-CLI step mix, trajectory composition, and CLI execution errors.

Layout-agnostic: a task dir is any directory holding a `result.txt` or
`traj.jsonl`; its domain is the parent dir name (`.../<domain>/<task_id>/`).
One scan of `traj.jsonl` yields everything:
  * score      — the 0..1 value in `result.txt` (partial credit kept; acc == mean)
  * steps      — lines in `traj.jsonl`
  * GUI / CLI  — a bash action whose command runs pyautogui is a GUI step; any
                 other bash command is a CLI step (control actions are neither)
  * comp       — gui_only / cli_only / hybrid / none, from the GUI & CLI counts
  * errors     — a step errored when its `info.exec_result.error` (stderr) shows
                 a real failure; `status` is always "success" (it only means the
                 command was delivered), so stderr is the only signal

Each record pairs an action with its own result: `action.command`,
`info.exec_result` and `screenshot_file` (named `step_N_post_cli_*`) all belong
to the same step. Note the model only *sees* a step's stderr on the next turn.

Usage:
    python scripts/python/osworld/score_report.py <result_dir> [...] [--errors]

    --errors  add the CLI execution-error section (rate + type breakdown)
    --meta    test-set JSON ({domain: [task_id, ...]}) for the done/total column;
              defaults to OSWorld/evaluation_examples/test_nogdrive.json if present.
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

GUI_MARKER = "pyautogui"  # a bash action running pyautogui is a GUI step; else CLI

# Trailing "ExceptionName: message" line of a Python traceback.
_PY_EXC = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Warning|Interrupt|Exit))\b\s*:?(.*)$")

# stderr text that is a prompt or notice, not a failure.
_NOISE = ("[sudo] password for", "retype new password")

# Substring -> label, for shell-level failures with no Python traceback.
_SHELL_ERRORS = (
    ("invalid syntax", "SyntaxError"),
    ("command not found", "ShellCommandNotFound"),
    ("permission denied", "ShellPermissionDenied"),
    ("no such file", "ShellNoSuchFile"),
    ("no such key", "GsettingsNoSuchKey"),
    ("no such schema", "GsettingsNoSuchKey"),
    ("sorry, try again", "SudoAuthFailed"),
    ("incorrect password", "SudoAuthFailed"),
)


def classify_error(stderr: str) -> str:
    """Bucket a stderr blob into a short error label.

    Returns "" when the text shows no failure (prompt-only or warning-only), so
    callers can treat the step as clean.
    """
    lines = [ln.strip() for ln in stderr.strip().splitlines() if ln.strip()]
    lines = [ln for ln in lines if not any(n in ln.lower() for n in _NOISE)]
    if not lines:
        return ""

    blob = "\n".join(lines)
    if "TimeoutExpired" in blob:  # reported bare, without a traceback
        return "Timeout"
    for ln in reversed(lines):  # last exception line wins
        m = _PY_EXC.match(ln)
        if m:
            return m.group(1)

    low = blob.lower()
    for needle, label in _SHELL_ERRORS:
        if needle in low:
            return label
    if all("warning" in ln.lower() for ln in lines):
        return ""  # warnings only: the command still succeeded
    return "OtherStderr"


def scan(result_dir: Path):
    """One record per task dir (parent of a result.txt or traj.jsonl)."""
    task_dirs = {p.parent for p in result_dir.rglob("result.txt")}
    task_dirs |= {p.parent for p in result_dir.rglob("traj.jsonl")}
    return [_analyze(d) for d in sorted(task_dirs)]


def _analyze(task_dir: Path):
    gui = cli = steps = 0
    errors = Counter()  # label -> count, CLI steps only
    gui_err = 0
    traj = task_dir / "traj.jsonl"
    if traj.exists():
        for line in traj.open():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # half-written line while the run is still live
            steps += 1
            action = rec.get("action")
            if not isinstance(action, dict) or action.get("kind") != "bash":
                continue
            is_gui = GUI_MARKER in (action.get("command") or "")
            exec_result = (rec.get("info") or {}).get("exec_result")
            stderr = exec_result.get("error") or "" if isinstance(exec_result, dict) else ""
            label = classify_error(stderr)
            if is_gui:
                gui += 1
                gui_err += bool(label)
            else:
                cli += 1
                if label:
                    errors[label] += 1

    comp = "hybrid" if gui and cli else "gui_only" if gui else "cli_only" if cli else "none"
    return dict(domain=task_dir.parent.name, score=_read_score(task_dir / "result.txt"),
                steps=steps, gui=gui, cli=cli, comp=comp, errors=errors, gui_err=gui_err)


def _read_score(path: Path):
    try:
        return float(path.read_text().strip())
    except (OSError, ValueError):
        return None  # missing / empty / unfinished


def load_totals(meta_path: Path):
    if meta_path and meta_path.exists():
        return {d: len(ids) for d, ids in json.loads(meta_path.read_text()).items()}
    return {}


def _pct(part, whole):
    return f"{100.0 * part / whole:.1f}%" if whole else "-"


def report(records, totals):
    """One row per domain: score / accuracy / avg-step, GUI-vs-CLI step share,
    and trajectory composition — all in a single table, sorted by accuracy."""
    doms = defaultdict(list)
    for r in records:
        doms[r["domain"]].append(r)

    cols = (f"{'domain':<20}{'done/total':>11}{'score':>7}{'acc%':>7}{'avg_stp':>8}"
            f"{'GUI%':>6}{'CLI%':>6}{'gui_only':>11}{'cli_only':>11}{'hybrid':>11}")
    print(cols + "\n" + "-" * len(cols))

    def row(name, rs, total):
        scored = [r["score"] for r in rs if r["score"] is not None]
        s = sum(scored)
        acc = s / len(scored) * 100 if scored else 0.0
        avg = sum(r["steps"] for r in rs) / len(rs) if rs else 0.0
        g, c = sum(r["gui"] for r in rs), sum(r["cli"] for r in rs)
        bash = g + c or 1
        n = len(rs)
        comp = lambda k: sum(r["comp"] == k for r in rs)
        pct = lambda v: f"{v} ({v / n * 100:.0f}%)" if n else str(v)
        print(f"{name:<20}{f'{len(scored)}/{total}':>11}{s:>7.1f}{acc:>6.1f}%{avg:>8.1f}"
              f"{g / bash * 100:>5.0f}%{c / bash * 100:>5.0f}%"
              f"{pct(comp('gui_only')):>11}{pct(comp('cli_only')):>11}{pct(comp('hybrid')):>11}")

    acc_of = lambda rs: (sum(r["score"] or 0 for r in rs)
                         / (sum(r["score"] is not None for r in rs) or 1))
    for d in sorted(doms, key=lambda d: -acc_of(doms[d])):
        row(d, doms[d], totals.get(d, len(doms[d])))
    grand_total = max(sum(totals.values()), len(records)) if totals else len(records)
    print("-" * len(cols))
    row("TOTAL", records, grand_total)
    print("\ncomposition 余数为 none (无 bash 步的任务, 如 infra error)")


def error_report(records, top=10):
    """CLI execution errors: overall rate, type breakdown, per-domain rate."""
    cli = sum(r["cli"] for r in records)
    gui = sum(r["gui"] for r in records)
    errors = sum((r["errors"] for r in records), Counter())
    n_err = sum(errors.values())
    gui_err = sum(r["gui_err"] for r in records)

    print(f"\nCLI steps : {cli:5d}   errored {n_err:5d}  ({_pct(n_err, cli):>6})")
    print(f"GUI steps : {gui:5d}   errored {gui_err:5d}  ({_pct(gui_err, gui):>6})"
          "   ← pyautogui 极少抛异常; 点错位置不计为 error")
    if not n_err:
        return

    print("\nCLI error breakdown:")
    for label, n in errors.most_common(top):
        print(f"    {label:<28}{n:>6}  ({_pct(n, n_err):>6} of errors, {_pct(n, cli):>6} of CLI steps)")

    print("\nper-domain CLI error rate:")
    doms = defaultdict(list)
    for r in records:
        doms[r["domain"]].append(r)
    stats = [(sum(sum(r["errors"].values()) for r in rs), sum(r["cli"] for r in rs), d)
             for d, rs in doms.items()]
    for errs, tot, d in sorted(stats, key=lambda t: -(t[0] / t[1] if t[1] else 0)):
        print(f"    {d:<22}{errs:>5}/{tot:<6}{_pct(errs, tot):>7}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_dirs", nargs="+", type=Path, help="OSWorld result directory/directories")
    ap.add_argument("--errors", action="store_true", help="add the CLI execution-error section")
    ap.add_argument("--meta", type=Path,
                    default=Path("OSWorld/evaluation_examples/test_nogdrive.json"),
                    help="test-set JSON for done/total totals")
    args = ap.parse_args()

    totals = load_totals(args.meta)
    multi = len(args.result_dirs) > 1
    for d in args.result_dirs:
        if multi:
            print(f"################ {d}")
        if not d.is_dir():
            print(f"  (目录不存在: {d})\n")
            continue
        records = scan(d)
        if not records:
            msg = f"no result.txt / traj.jsonl found under {d}"
            if not multi:
                raise SystemExit(msg)
            print(f"  ({msg})\n")
            continue
        report(records, totals)
        if args.errors:
            error_report(records)
        if multi:
            print()


if __name__ == "__main__":
    main()
