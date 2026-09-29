#!/usr/bin/env bash
set -euo pipefail

_script_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "${_script_dir}/../../server.py" ]]; then
  cd "${_script_dir}/../.."
fi

# ----------------------------------------------------------------------------
# Thin wrapper for scripts/python/cleanup_results.py.
#
# Dry-run by default:
#   bash scripts/bash/cleanup_results.sh
#
# Delete task dirs without result.txt:
#   APPLY=1 ROOT=results_qwen3vl_remote MODE=missing \
#     bash scripts/bash/cleanup_results.sh
#
# Delete task dirs whose score/result.txt is not 1 and rewrite summary/results.json:
#   APPLY=1 ROOT=results_qwen3vl_remote MODE=failed \
#     bash scripts/bash/cleanup_results.sh
# ----------------------------------------------------------------------------

PYTHON_BIN="${PYTHON_BIN:-python}"
ROOT="${ROOT:-${RESULT_ROOT:-results}}"
MODE="${MODE:-missing}"
LIMIT="${LIMIT:-80}"

CMD=(
  "${PYTHON_BIN}" scripts/python/cleanup_results.py
  "${ROOT}"
  --mode "${MODE}"
  --limit "${LIMIT}"
)

if [[ "${APPLY:-0}" == "1" || "${DELETE:-0}" == "1" ]]; then
  CMD+=(--apply)
fi

if [[ -n "${RESULTS_FILE:-}" ]]; then
  CMD+=(--results-file "${RESULTS_FILE}")
fi

if [[ "${KEEP_MISSING_RESULT_TXT:-0}" == "1" ]]; then
  CMD+=(--keep-missing-result-txt)
fi

if [[ "${NO_BACKUP:-0}" == "1" ]]; then
  CMD+=(--no-backup)
fi

echo "Running: ${CMD[*]} $*"
exec "${CMD[@]}" "$@"
