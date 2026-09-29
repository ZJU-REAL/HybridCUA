# OSWorld Monitor

A web-based monitoring dashboard for OSWorld tasks and executions.

## Overview

Point the monitor at an experiment directory and it works out the rest. It shows:

- Every task found in the experiment directory, grouped by type
- Task status and progress, refreshed as the scan proceeds
- Per-step detail with screenshots and video
- Scores from each task's `result.txt`

> Important! Make sure you run the monitor after the main runner has started executing tasks. Otherwise, it may cause issues when executing tasks.

## Configuration

The only thing you enter is the **Experiment Path** — the directory containing
`args.json` (equivalently `RESULTS_BASE_PATH/ACTION_SPACE/OBSERVATION_TYPE/MODEL`
in the runner's layout). Everything else is derived from `args.json`:

| Derived | Where it comes from |
|---|---|
| Repo root | the experiment path with the suffix `args.json` implies removed |
| Task config | `test_all_meta_path`, resolved against the repo root |
| Examples directory | `test_config_base_dir` + `examples_subdir` |
| Max steps | `max_steps` |
| Model / action space / observation type | recorded as metadata; they no longer affect any path |

Each value is shown in the sidebar with a badge saying where it came from —
`from args.json`, `derived`, `guessed by scanning`, `fallback default`, or
`not found`. Anything that could not be worked out is listed in a yellow warning
block rather than silently omitted.

The task list comes from **scanning the experiment directory**, so a single path
always produces a usable view. When the task config resolves, tasks in it that
have no directory on disk are added as `Not Started`; when it doesn't, you still
see everything that ran.

Because scanning network storage takes a while, the first request returns
immediately and statuses fill in on a background thread — a progress line shows
how far along it is. **Clear Cache** forces a full re-read.

Once an experiment has loaded, revisiting it is cheap:

| Cached | For how long |
|---|---|
| A finished task's status (`Done*`, `Error`) | forever — it cannot change |
| The directory listing | 120s (`SCAN_TTL`), since new task dirs only appear when a run starts one |
| An unfinished task's status | 10s (`LIVE_TTL`), then re-checked by a single `stat` and only re-read if `traj.jsonl` actually changed |

That last row is what keeps a *reloaded* experiment fast. An abandoned run leaves
tasks parked in `Running` forever; re-reading their trajectories on every poll
cost ~170s per pass on network storage and returned nothing new, which made an
already-loaded path feel as slow as a cold one.

## Compare mode

Compare mode puts two experiment directories side by side and scores them
per task. Enter a second path in the sidebar and switch **Single / Compare**.

Each run gets two buttons — **1** and **0** — and you pick one on each side:

| Button | Means |
|---|---|
| **1** | that run scored 1 — it solved the task |
| **0** | that run did not: scored 0, got partial credit, or errored |

Choosing **A = 1** and **B = 0** gives the tasks A solved and B did not, which is
the usual reason to open this mode; swapping them gives the improvements. Picking
one side only ("A = 0") filters on that side alone, and clicking a lit button
again clears it. Each button carries the count you would get by pressing it, so
the size of a set is visible before committing to it.

The split is binary, so the two counts always sum to the total: an error, a
crash, or a run that never finished all count as **0**. A score is read from
`result.txt` and must parse to a number in `[0,1]`; anything else is a 0.

Each card shows both scores as `1 → 0`, and each half opens that task **with both
runs side by side**: two scrolling columns, A's steps beside B's, so you can see
where the two runs diverged instead of holding one in your head. Each column
draws its own click overlays from its own action list. **Back** returns to the
same pairing rather than to a single run.

The banner totals cover every task over the same denominator on both sides, with
an unfinished run contributing 0 — the same rule the 0 button uses. Tasks present
in only one run are kept rather than dropped, since a missing task is itself
information.

Deleting is single-mode only: with two experiments open, "which side?" has no
safe default, so **Clean Unfinished** and the per-card **Reset** button are
hidden. **Clear Cache** clears both paths.

Compare views are shareable — the mode and both paths live in the URL:
`?path=<A>&path_b=<B>&mode=compare`. A link without `mode=compare` always opens
in single mode, so older bookmarks are unaffected.

### Flask server settings (`.env`)

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_PORT` | Port for the web server | `8080` |
| `FLASK_HOST` | Host address for the web server | `0.0.0.0` |
| `FLASK_DEBUG` | Enable debug mode (`true`/`false`) | `false` |
| `MONITOR_EXPERIMENT_PATH` | Optional: pre-fill the Experiment Path input | *empty* |
| `MONITOR_COMPARE_PATH` | Optional: pre-fill the compare-mode second path | *empty* |
| `MAX_STEPS` | Fallback step ceiling, used only when `args.json` has none | `100` |

## Running

1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the monitor:
   ```bash
   python main.py
   ```

3. Open `http://{your-ip-address}:{FLASK_PORT}` and enter an Experiment Path.

The path can also be passed in the URL, which makes views shareable:
`http://host:port/?path=/abs/path/to/experiment`

Run it directly on the host — do not containerize it. The monitor resolves
experiment directories by absolute path, and `args.json` records paths relative
to the repo root, so any path remapping breaks auto-discovery.

Keep `FLASK_DEBUG=false` on network storage: the reloader stats the whole
directory tree and the server ends up accepting connections without answering
them.

## Layout

```
monitor/
  main.py      Flask routes and request plumbing
  core/        reading an experiment directory
    derive.py    args.json  -> task config, examples dir, max_steps (+ provenance)
    discover.py  filesystem -> which task directories exist
    traj.py      task dir   -> steps, screenshots, score
    tasks.py     task config + examples dir -> the task list and instructions
    status.py    the above  -> status labels, cached and filled in the background
    paths.py     guards for every path that arrives over HTTP
  static/, templates/
    index.js     single mode + the shell both modes share
    compare.js   compare mode: joins two task lists and scores them
```

`core/` holds the rules about what an experiment directory means, kept out of
`main.py` so the Flask layer stays request plumbing and the rules stay testable
on their own. It is an ordinary subpackage — `import core` works because
`main.py` runs from this directory; there is no packaging step and no
`sys.path` manipulation.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/experiment?path=` | Derived settings, provenance, and warnings |
| `GET /api/tasks?path=` | Task list plus background-scan progress |
| `GET /api/task/<type>/<id>?path=` | One task's full detail |
| `POST /api/clear-cache?path=` | Drop cached state for an experiment |
| `GET /api/cleanup/preview?path=` | Unfinished task directories |
| `POST /api/cleanup/execute?path=` | **Deletes** the confirmed subset |
| `POST /api/task/<type>/<id>/reset?path=` | **Deletes** one task's directory |

`results_base_path` is still accepted in place of `path`, so older bookmarks keep
working.

Compare mode adds no endpoints. Every API above is keyed by `?path=`, and the
status cache is per path, so the two experiments scan independently and the
front-end simply fetches twice and joins on `(task_type, task_id)`.

## Troubleshooting

1. **Blank dashboard**: check the warning block in the sidebar — it names what
   could not be derived and what was tried. If tasks are listed but every
   instruction reads "No task info available", the examples directory did not
   resolve.
2. **Slow first load**: expected on network storage. Watch the progress line;
   subsequent loads are served from cache. If a *reloaded* experiment is still
   slow, check that the progress line settles and stays settled — it flapping
   back to a partial count means something is invalidating the cache.
3. **Compare mode counts a task as 0 that you expect to be 1**: the score is read
   from `result.txt` and must parse to a number in `[0,1]`. An unfinished or
   errored run also counts as 0. Check that task's `result.txt` directly.
4. **Compare mode's totals look too small**: they deliberately cover only the
   tasks *both* sides finished, so the two numbers share a denominator. The count
   excluded is shown next to the delta.
5. **Wrong status counts**: `max_steps` comes from `args.json`, and it decides
   whether a task counts as `Done (Max Steps)`.
6. Check that the port is not already in use, and that security group rules
   allow access to it.
