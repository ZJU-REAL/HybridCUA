#!/usr/bin/env bash
# CUA-Gym remote evaluation (Kimi agent) against a node hosting the cua_gym world.
#
# Prereqs:
#   - A node started with the cua_gym world hosted (NODE_WORLDS=cua_gym or all).
#   - CUA-Gym bundles downloaded + extracted (see README): a dir of <uuid>/ bundles.
#   - KIMI_API_KEY / KIMI_BASE_URL exported (or passed via --api_key/--base_url).
#
# Usage:
#   # run every bundle (optionally narrowed by app_type):
#   TASKS_ROOT=/data/cua_gym_bundles DOMAIN=libreoffice_calc bash scripts/bash/cua_gym/run_kimi_remote.sh
#   # run only the tasks listed in a meta JSON ({app_type: [uuid, ...]}, OSWorld-style):
#   TASKS_META=./my_tasks.json bash scripts/bash/cua_gym/run_kimi_remote.sh
set -euo pipefail

# Frontend job execution sets OSWORLD_JOB_ID and runs from the repo root already.
if [ -z "${OSWORLD_JOB_ID:-}" ]; then
  cd "$(dirname "$0")/../../.."
fi

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}" 2>/dev/null || LOG_DIR="/tmp/env_infra_logs" && mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run_kimi_remote_${TIMESTAMP}.log}"

export KIMI_API_KEY="${KIMI_API_KEY:-}"
export KIMI_BASE_URL="${KIMI_BASE_URL:-}"
if [[ -z "$KIMI_BASE_URL" ]]; then echo "Set KIMI_BASE_URL before running" >&2; exit 2; fi
export GUI_ENV_SERVER_URL="${GUI_ENV_SERVER_URL:-http://127.0.0.1:19000}"
export ENABLE_PROXY="${ENABLE_PROXY:-1}"

if [ -z "${KIMI_API_KEY}" ]; then
  echo "ERROR: KIMI_API_KEY is not set. Export it before running this script." >&2
  exit 1
fi

# Kimi API gateway is reached DIRECTLY (no proxy); cluster master + private ranges also
# stay off-proxy. Clear inherited proxy env and set no_proxy accordingly.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true
_gui_host="${GUI_ENV_SERVER_URL#*://}"; _gui_host="${_gui_host%%[:/]*}"
_gw_host="${KIMI_BASE_URL#*://}"; _gw_host="${_gw_host%%[:/]*}"
_no_proxy_add="localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,${_gui_host},${_gw_host}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${_no_proxy_add}"
export no_proxy="${no_proxy:+${no_proxy},}${_no_proxy_add}"

# Detach to the background (BACKGROUND=0 to stay in the foreground). The re-exec sets
# a sentinel so the child runs the real work instead of forking again.
if [[ "${BACKGROUND:-1}" == "1" && -z "${_KIMI_REMOTE_DETACHED:-}" ]]; then
  export _KIMI_REMOTE_DETACHED=1
  nohup bash "$0" "$@" > "${LOG_FILE}" 2>&1 < /dev/null &
  CHILD_PID=$!
  echo "Started in background, PID=${CHILD_PID}"
  echo "Log:  ${LOG_FILE}"
  echo "Tail: tail -f ${LOG_FILE}"
  echo "Stop: kill ${CHILD_PID}"
  exit 0
fi

echo "Logging to ${LOG_FILE}"
echo "GUI_ENV_SERVER_URL=${GUI_ENV_SERVER_URL}"
echo "MODEL=${MODEL:-moonshot/kimi-k2.6}  KIMI_BASE_URL=${KIMI_BASE_URL}"

# When detached the child's stdout is already redirected to LOG_FILE by nohup, so tee
# would duplicate; in the foreground tee mirrors to the log while echoing to the terminal.
if [[ -n "${_KIMI_REMOTE_DETACHED:-}" ]]; then
  TEE_CMD=(cat)
else
  TEE_CMD=(tee "${LOG_FILE}")
fi

# --tasks_meta selects the exact task subset ({app_type: [uuid, ...]}, OSWorld-style).
# Defaults to the 200-task cua_gym_eval set; set TASKS_META= (empty) to run every bundle.
meta_arg=()
TASKS_META="${TASKS_META-./cua_gym_data/cua_gym_eval.json}"
if [ -n "${TASKS_META}" ]; then
  meta_arg=(--tasks_meta "${TASKS_META}")
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
"${PYTHON_BIN}" scripts/python/cua_gym/run_kimi_remote.py \
  --tasks_root "${TASKS_ROOT:-./cua_gym_data/bundles}" \
  "${meta_arg[@]}" \
  --domain "${DOMAIN:-all}" \
  --num_envs "${NUM_ENVS:-4}" \
  --max_steps "${MAX_STEPS:-50}" \
  --result_dir "${RESULT_DIR:-./results_cua_gym}" \
  --model "${MODEL:-moonshot/kimi-k2.6}" \
  --base_url "${KIMI_BASE_URL}" \
  --api_key "${KIMI_API_KEY}" \
  --headless 2>&1 | "${TEE_CMD[@]}"
