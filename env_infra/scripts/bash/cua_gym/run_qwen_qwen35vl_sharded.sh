#!/usr/bin/env bash
# CUA-Gym remote evaluation (Qwen3.5-VL agent) sharded across multiple vLLM endpoints.
#
# Combines scripts/bash/osworld/run_qwen_qwen35vl_sharded.sh (endpoint sharding +
# QwenAgent params) with scripts/bash/cua_gym/run_kimi_remote.sh (cua_gym task
# source + cluster). Each worker pins one vLLM endpoint round-robin by worker index.
#
# Prereqs:
#   - A node started with the cua_gym world hosted (NODE_WORLDS=cua_gym or all).
#   - CUA-Gym bundles downloaded + extracted: a dir of <uuid>/ bundles.
#   - vLLM servers serving Qwen3.5-VL (see scripts/bash/start_vllm_qwen35_*_8gpu.sh).
#
# Usage:
#   OPENAI_BASE_URLS=http://host:8000/v1,...,http://host:8007/v1 \
#     NUM_ENVS=16 bash scripts/bash/cua_gym/run_qwen_qwen35vl_sharded.sh
set -euo pipefail

# Frontend job execution sets OSWORLD_JOB_ID and runs from the repo root already.
if [ -z "${OSWORLD_JOB_ID:-}" ]; then
  cd "$(dirname "$0")/../../.."
fi

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}" 2>/dev/null || LOG_DIR="/tmp/env_infra_logs" && mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run_qwen_qwen35vl_sharded_cua_gym_${TIMESTAMP}.log}"

export GUI_ENV_SERVER_URL="${GUI_ENV_SERVER_URL:-http://127.0.0.1:19000}"
export ENABLE_PROXY="${ENABLE_PROXY:-1}"

# Build default 8-endpoint shard list if OPENAI_BASE_URLS not set.
if [[ -z "${OPENAI_BASE_URLS:-}" ]]; then
  OPENAI_HOST="${OPENAI_HOST:-127.0.0.1}"
  OPENAI_PORT_START="${OPENAI_PORT_START:-8000}"
  OPENAI_NUM_SHARDS="${OPENAI_NUM_SHARDS:-8}"
  _urls=()
  for ((i = 0; i < OPENAI_NUM_SHARDS; i++)); do
    _urls+=("http://${OPENAI_HOST}:$((OPENAI_PORT_START + i))/v1")
  done
  OPENAI_BASE_URLS="$(IFS=,; echo "${_urls[*]}")"
fi
export OPENAI_BASE_URLS
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-local}"

# Host-side proxy for task asset downloads. Cluster master + vLLM endpoints stay
# off the proxy (no_proxy) so session and model traffic go direct.
HOST_PROXY_URL="${HOST_PROXY_URL-http://127.0.0.1:3128}"
if [[ -n "${HOST_PROXY_URL}" ]]; then
  export http_proxy="${HOST_PROXY_URL}"
  export https_proxy="${HOST_PROXY_URL}"
  export HTTP_PROXY="${HOST_PROXY_URL}"
  export HTTPS_PROXY="${HOST_PROXY_URL}"
fi

_no_proxy_add="localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,28.0.0.0/8"
IFS=',' read -r -a _base_url_array <<< "${OPENAI_BASE_URLS}"
for _url in "${_base_url_array[@]}"; do
  _host="${_url#*://}"; _host="${_host%%[:/]*}"
  [[ -n "${_host}" ]] && _no_proxy_add="${_no_proxy_add},${_host}"
done
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${_no_proxy_add}"
export no_proxy="${no_proxy:+${no_proxy},}${_no_proxy_add}"

# Detach to the background (BACKGROUND=0 to stay in the foreground). The re-exec
# sets a sentinel so the child runs the real work instead of forking again.
if [[ "${BACKGROUND:-1}" == "1" && -z "${_QWEN_CUAGYM_SHARDED_DETACHED:-}" ]]; then
  export _QWEN_CUAGYM_SHARDED_DETACHED=1
  nohup bash "$0" "$@" > "${LOG_FILE}" 2>&1 < /dev/null &
  CHILD_PID=$!
  echo "Started in background, PID=${CHILD_PID}"
  echo "Log:  ${LOG_FILE}"
  echo "Tail: tail -f ${LOG_FILE}"
  echo "Stop: kill ${CHILD_PID}"
  exit 0
fi

echo "Logging to ${LOG_FILE}"
echo "OPENAI_BASE_URLS=${OPENAI_BASE_URLS}"
echo "GUI_ENV_SERVER_URL=${GUI_ENV_SERVER_URL}"
echo "MODEL=${MODEL:-qwen35-vl}"

# When detached the child's stdout is already redirected to LOG_FILE by nohup, so
# tee would duplicate; in the foreground tee mirrors to the log and the terminal.
if [[ -n "${_QWEN_CUAGYM_SHARDED_DETACHED:-}" ]]; then
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
CMD=(
  "${PYTHON_BIN}" scripts/python/cua_gym/run_qwen_qwen35vl_sharded.py
  --cluster_url "${GUI_ENV_SERVER_URL}"
  --tasks_root "${TASKS_ROOT:-./cua_gym_data/bundles}"
  "${meta_arg[@]}"
  --domain "${DOMAIN:-all}"
  --action_space "${ACTION_SPACE:-pyautogui}"
  --observation_type "${OBSERVATION_TYPE:-screenshot}"
  --model "${MODEL:-qwen35-vl}"
  --coord "${COORD:-relative}"
  --max_tokens "${MAX_TOKENS:-2048}"
  --temperature "${TEMPERATURE:-0}"
  --top_p "${TOP_P:-0.9}"
  --history_n "${HISTORY_N:-100}"
  --image_max "${IMAGE_MAX:-20}"
  --fold_size "${FOLD_SIZE:-10}"
  --max_steps "${MAX_STEPS:-50}"
  --num_envs "${NUM_ENVS:-8}"
  --result_dir "${RESULT_DIR:-./results_cua_gym_qwen35vl}"
  --client_password "${CLIENT_PASSWORD:-password}"
  --screen_width "${SCREEN_WIDTH:-1920}"
  --screen_height "${SCREEN_HEIGHT:-1080}"
  --sleep_after_execution "${SLEEP_AFTER_EXECUTION:-3}"
  --log_level "${LOG_LEVEL:-INFO}"
  --openai_base_urls "${OPENAI_BASE_URLS}"
  --api_key "${OPENAI_API_KEY}"
  "$@"
)

if [[ "${HEADLESS:-1}" == "1" ]]; then CMD+=(--headless); fi
if [[ "${ADD_THOUGHT_PREFIX:-0}" == "1" ]]; then CMD+=(--add_thought_prefix); fi
if [[ "${ENABLE_THINKING:-0}" == "1" ]]; then CMD+=(--enable_thinking); fi

"${CMD[@]}" 2>&1 | "${TEE_CMD[@]}"
