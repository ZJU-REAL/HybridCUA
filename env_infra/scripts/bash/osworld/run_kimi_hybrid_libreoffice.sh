#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------------
# Kimi hybrid (GUI+CLI) OSWorld evaluation — LibreOffice ONLY.
#
# Thin wrapper over run_kimi_hybrid.sh: everything (model, gateway, agent, loop,
# proxy handling, background/nohup) is identical; this only points the task meta at
# test_libreoffice.json (libreoffice_calc + _impress + _writer, 117 tasks) and writes
# to a separate result dir so it won't mix with the full run.
#
#   KIMI_API_KEY=sk-... NUM_ENVS=16 bash scripts/bash/osworld/run_kimi_hybrid_libreoffice.sh
#
# Any override still works (they pass straight through to run_kimi_hybrid.sh), e.g.
#   DOMAIN=libreoffice_calc bash scripts/bash/osworld/run_kimi_hybrid_libreoffice.sh   # just calc
#   MAX_STEPS=40 NUM_ENVS=8 bash scripts/bash/osworld/run_kimi_hybrid_libreoffice.sh
# ----------------------------------------------------------------------------

# Locate the canonical runner robustly. $0-based dirname breaks when this script is
# invoked via a pipe or copied to /tmp (then $(dirname $0) is /tmp and the sibling isn't
# there — the "/tmp/run_kimi_hybrid.sh: No such file" error). Try, in order:
#   1) sibling of THIS file resolved via BASH_SOURCE (works for `bash path/to/script.sh`)
#   2) a repo root found by walking up from cwd until OSWorld/ exists
#   3) $OSWORLD_REPO_ROOT if the caller set it
_find_runner() {
  local d
  # (1) sibling via BASH_SOURCE (more reliable than $0)
  d="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd)"
  [[ -f "${d}/run_kimi_hybrid.sh" ]] && { echo "${d}/run_kimi_hybrid.sh"; return; }
  # (2) walk up from cwd to a repo root (dir containing OSWorld/)
  d="$(pwd)"
  while [[ "${d}" != "/" ]]; do
    [[ -f "${d}/scripts/bash/osworld/run_kimi_hybrid.sh" ]] && { echo "${d}/scripts/bash/osworld/run_kimi_hybrid.sh"; return; }
    d="$(dirname "${d}")"
  done
  # (3) explicit override
  [[ -n "${OSWORLD_REPO_ROOT:-}" && -f "${OSWORLD_REPO_ROOT}/scripts/bash/osworld/run_kimi_hybrid.sh" ]] \
    && { echo "${OSWORLD_REPO_ROOT}/scripts/bash/osworld/run_kimi_hybrid.sh"; return; }
  return 1
}

_runner="$(_find_runner)" || {
  echo "ERROR: could not locate run_kimi_hybrid.sh (run from the repo root, or set OSWORLD_REPO_ROOT)." >&2
  exit 1
}

# LibreOffice-only defaults (overridable). run_kimi_hybrid.sh reads these via env.
export TEST_META_PATH="${TEST_META_PATH:-OSWorld/evaluation_examples/test_libreoffice.json}"
export RESULT_DIR="${RESULT_DIR:-./results_kimi_hybrid_libreoffice}"
export NUM_ENVS="${NUM_ENVS:-32}"   # match results_kimi_hybrid_2; 117 tasks -> ~4 waves

# Hand off to the canonical runner (its own cd / background / proxy logic applies).
exec bash "${_runner}" "$@"
