"""Roll up Acc / TIR / ACS across MCP_MODE arms.

Scans result dirs for per-episode `result.txt` (score) and `mcp_usage.json` (tool
usage, written by mcp_run_loop for every mode including off), and prints the
three-arm comparison table the Step-7 go/no-go gate is judged on.

    python3 rollup_mcp_usage.py results-mcp-off results-mcp-bash results-mcp-action
    python3 rollup_mcp_usage.py results-mcp-*            # arm name from the dir
    python3 rollup_mcp_usage.py results-mcp-* --by-domain
    python3 rollup_mcp_usage.py results-mcp-* --csv out.csv

Layout-agnostic: it recurses looking for `result.txt`, so it works with OSWorld's
<action_space>/<obs>/<model>/<domain>/<id>/ nesting and with a flat dir alike.

Metric definitions -- read these before quoting a number
--------------------------------------------------------
Acc   mean of result.txt over episodes that produced one. OSWorld scores are 0/1
      for most tasks but some evaluators return partial credit, so this is a mean,
      not a pass count. `n` is always printed next to it: comparing arms with
      different n is comparing different task sets.

TIR   we report TWO things, because "tool invocation rate" is ambiguous and the
      two answer different questions:
        calls/traj  mean MCP tool calls per episode. This is the number ToolCUA
                    reports (they measured 0.003 for Qwen3-VL-8B). Sensitive to a
                    single episode that loops on one tool.
        traj w/tool fraction of episodes with >=1 call. Robust to that, and the
                    better read on "did the model engage with tools at all".
      Neither is the paper's strict TIR, which asks whether the call was the RIGHT
      one -- that needs per-task ground truth (has_tool.json) we do not have here.

ACS   Avg Completion Steps, averaged over SUCCESSFUL episodes only (score > 0).
      Averaging over failures too would mostly measure how many episodes hit
      max_steps, which is a timeout statistic, not an efficiency one. `n` for ACS
      is therefore the success count, printed as acs_n.

dist  calls to the 26 filesystem_*/git_* distractor tools. These are bait: on a
      desktop task, reaching for one is never the right call. A nonzero rate here
      with flat Acc is the signature of a model that learned "call something".
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

USAGE_KEYS = ("tool_calls", "steps", "distractor_calls", "tools_offered", "mcp_mode")


def read_score(path: str) -> Optional[float]:
    try:
        with open(path, encoding="utf-8") as f:
            return float((f.read() or "").strip())
    except Exception:
        return None


def read_usage(d: str) -> Dict:
    p = os.path.join(d, "mcp_usage.json")
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            u = json.load(f)
        return u if isinstance(u, dict) else {}
    except Exception:
        return {}


def count_steps(d: str) -> Optional[int]:
    """Fallback step count for episodes with no mcp_usage.json (e.g. produced by
    the stock run_c_gui.sh). traj.jsonl has one line per recorded step."""
    p = os.path.join(d, "traj.jsonl")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return None


def collect(root: str) -> List[Dict]:
    """One record per episode directory under `root`."""
    out: List[Dict] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "result.txt" not in filenames:
            continue
        score = read_score(os.path.join(dirpath, "result.txt"))
        if score is None:
            continue
        usage = read_usage(dirpath)
        rel = os.path.relpath(dirpath, root).split(os.sep)
        out.append({
            "example_id": rel[-1] if rel else os.path.basename(dirpath),
            "domain": rel[-2] if len(rel) >= 2 else "?",
            "score": score,
            "has_usage": bool(usage),
            "tool_calls": int(usage.get("tool_calls") or 0),
            "distractor_calls": int(usage.get("distractor_calls") or 0),
            "tools_offered": int(usage.get("tools_offered") or 0),
            "steps": int(usage.get("steps") or 0) or (count_steps(dirpath) or 0),
            "mcp_mode": usage.get("mcp_mode"),
        })
    return out


def aggregate(recs: List[Dict]) -> Dict:
    n = len(recs)
    if not n:
        return {"n": 0}
    wins = [r for r in recs if r["score"] > 0]
    acs_pool = [r["steps"] for r in wins if r["steps"] > 0]
    with_usage = [r for r in recs if r["has_usage"]]
    calls_pool = with_usage or recs
    return {
        "n": n,
        "acc": sum(r["score"] for r in recs) / n,
        "solved": len(wins),
        "calls_per_traj": sum(r["tool_calls"] for r in calls_pool) / max(len(calls_pool), 1),
        "traj_with_tool": sum(1 for r in calls_pool if r["tool_calls"] > 0) / max(len(calls_pool), 1),
        "total_calls": sum(r["tool_calls"] for r in calls_pool),
        "distractor_calls": sum(r["distractor_calls"] for r in calls_pool),
        "traj_with_distractor": sum(1 for r in calls_pool if r["distractor_calls"] > 0),
        "acs": (sum(acs_pool) / len(acs_pool)) if acs_pool else float("nan"),
        "acs_n": len(acs_pool),
        "avg_steps_all": sum(r["steps"] for r in recs) / n,
        "tools_offered": (sum(r["tools_offered"] for r in with_usage) / len(with_usage))
                         if with_usage else 0.0,
        "missing_usage": n - len(with_usage),
    }


def arm_name(path: str) -> str:
    base = os.path.basename(os.path.normpath(path))
    for tag in ("action", "bash", "off"):
        if base.endswith("-" + tag) or base.endswith("_" + tag):
            return tag
    return base


COLUMNS = ("arm", "n", "acc", "solved", "calls/traj", "traj%tool", "dist",
           "ACS", "acs_n", "tools")
HEAD = (f"{COLUMNS[0]:<8}{COLUMNS[1]:>5}{COLUMNS[2]:>8}{COLUMNS[3]:>8}"
        f"{COLUMNS[4]:>12}{COLUMNS[5]:>11}{COLUMNS[6]:>7}{COLUMNS[7]:>8}"
        f"{COLUMNS[8]:>7}{COLUMNS[9]:>7}")


def fmt_row(label: str, a: Dict) -> str:
    if not a.get("n"):
        return f"{label:<8}{0:>5}   (no episodes found)"
    acs = "  n/a" if a["acs"] != a["acs"] else f"{a['acs']:.1f}"
    return (f"{label:<8}{a['n']:>5}{a['acc'] * 100:>7.1f}%{a['solved']:>8}"
            f"{a['calls_per_traj']:>12.3f}{a['traj_with_tool'] * 100:>10.1f}%"
            f"{a['distractor_calls']:>7}{acs:>8}{a['acs_n']:>7}"
            f"{a['tools_offered']:>7.0f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="result dirs, one per arm")
    ap.add_argument("--by-domain", action="store_true", help="also break down per domain")
    ap.add_argument("--csv", metavar="PATH", help="write per-episode rows to CSV")
    ap.add_argument("--paired", action="store_true",
                    help="restrict every arm to the example_ids present in ALL arms")
    a = ap.parse_args()

    arms: Dict[str, List[Dict]] = {}
    for d in a.dirs:
        if not os.path.isdir(d):
            print(f"[!] not a directory, skipping: {d}", file=sys.stderr)
            continue
        recs = collect(d)
        name = arm_name(d)
        if name in arms:
            name = f"{name}:{os.path.basename(os.path.normpath(d))}"
        arms[name] = recs
        print(f"[+] {d} -> arm '{name}': {len(recs)} episode(s)", file=sys.stderr)

    if not arms:
        print("[!] nothing to report", file=sys.stderr)
        return 1

    if a.paired and len(arms) > 1:
        common = set.intersection(*({r["example_id"] for r in v} for v in arms.values()))
        dropped = {k: len(v) - sum(1 for r in v if r["example_id"] in common)
                   for k, v in arms.items()}
        arms = {k: [r for r in v if r["example_id"] in common] for k, v in arms.items()}
        print(f"[+] paired on {len(common)} common example(s); dropped {dropped}",
              file=sys.stderr)

    order = [k for k in ("off", "bash", "action") if k in arms]
    order += [k for k in arms if k not in order]

    print("\n" + HEAD)
    print("-" * len(HEAD))
    aggs = {}
    for k in order:
        aggs[k] = aggregate(arms[k])
        print(fmt_row(k, aggs[k]))

    if "off" in aggs and aggs["off"].get("n"):
        base = aggs["off"]
        print("\ndelta vs off:")
        for k in order:
            if k == "off" or not aggs[k].get("n"):
                continue
            d_acc = (aggs[k]["acc"] - base["acc"]) * 100
            print(f"  {k:<8} acc {d_acc:+.1f}pp   "
                  f"solved {aggs[k]['solved'] - base['solved']:+d}   "
                  f"calls/traj {aggs[k]['calls_per_traj']:.3f}")
            if base["n"] != aggs[k]["n"]:
                print(f"           ^ WARNING: n differs ({base['n']} vs {aggs[k]['n']}); "
                      "not the same task set -- rerun with --paired")

    for k in order:
        miss = aggs[k].get("missing_usage") or 0
        if miss:
            print(f"\n[note] arm '{k}': {miss}/{aggs[k]['n']} episode(s) had no "
                  "mcp_usage.json (stock run_c_gui.sh output?); their tool counts "
                  "are excluded from calls/traj, not counted as zero.")

    print("\n=== Step-7 gate ===")
    for k in order:
        if k == "off" or not aggs[k].get("n"):
            continue
        g = aggs[k]
        tir_ok = g["total_calls"] > 0
        acc_ok = ("off" not in aggs) or (g["acc"] >= aggs["off"]["acc"])
        verdict = "PASS" if (tir_ok and acc_ok) else "FAIL"
        why = []
        if not tir_ok:
            why.append("TIR==0: the model never called a tool -- prompt or capability, "
                       "not scale. Do NOT proceed to the 361-task run.")
        if not acc_ok:
            why.append(f"Acc below off by "
                       f"{(aggs['off']['acc'] - g['acc']) * 100:.1f}pp: tools are a "
                       "net distraction at this model size.")
        print(f"  {k:<8} {verdict}" + ("" if not why else "\n           " +
                                       "\n           ".join(why)))
        if tir_ok and g["distractor_calls"] and g["distractor_calls"] >= g["total_calls"] * 0.3:
            print(f"           NOTE: {g['distractor_calls']}/{g['total_calls']} calls went "
                  "to distractor tools -- calling something, not the right thing.")

    if a.by_domain:
        print("\n=== by domain ===")
        doms: Dict[str, Dict[str, List[Dict]]] = defaultdict(dict)
        for k in order:
            per: Dict[str, List[Dict]] = defaultdict(list)
            for r in arms[k]:
                per[r["domain"]].append(r)
            for dom, rs in per.items():
                doms[dom][k] = rs
        for dom in sorted(doms):
            print(f"\n-- {dom}")
            print(HEAD)
            for k in order:
                if k in doms[dom]:
                    print(fmt_row(k, aggregate(doms[dom][k])))

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["arm", "domain", "example_id", "score", "steps", "tool_calls",
                        "distractor_calls", "tools_offered", "has_usage"])
            for k in order:
                for r in arms[k]:
                    w.writerow([k, r["domain"], r["example_id"], r["score"], r["steps"],
                                r["tool_calls"], r["distractor_calls"],
                                r["tools_offered"], int(r["has_usage"])])
        print(f"\n[+] per-episode rows -> {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
