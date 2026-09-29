#!/usr/bin/env bash
set -euo pipefail

# When run via cluster master, cwd is already repo root.
# When run manually, cd to repo root relative to script location.
if [[ -z "${OSWORLD_JOB_ID:-}" ]]; then
  _script_dir="$(cd "$(dirname "$0")" && pwd)"
  cd "${_script_dir}/../../.."
fi

# ----------------------------------------------------------------------------
# Run OSWorld evaluation with Qwen3.7-Plus via the official mm_agents.qwen.QwenAgent
# against a single hosted Qwen API endpoint (one base_url + one api_key shared by
# all workers — no local vLLM sharding).
#
#   OPENAI_BASE_URL=<your-model-endpoint> \
#   OPENAI_API_KEY=sk-... \
#   NUM_ENVS=8 \
#   bash scripts/bash/osworld/run_qwen_qwen37plus.sh
# ----------------------------------------------------------------------------

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}" 2>/dev/null || LOG_DIR="/tmp/env_infra_logs" && mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run_qwen_qwen37plus_${TIMESTAMP}.log}"
export PROXY_CONFIG_FILE="${PROXY_CONFIG_FILE:-$(pwd)/OSWorld/evaluation_examples/settings/proxy/star_proxy.json}"

export GUI_ENV_SERVER_URL="${GUI_ENV_SERVER_URL:-http://127.0.0.1:19000/}"
export OSWORLD_USER_ID="${OSWORLD_USER_ID:-${USER:-anonymous}}"
export OSWORLD_TASK_TYPE="${OSWORLD_TASK_TYPE:-evaluation}"

# Single hosted Qwen API endpoint.
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
if [[ -z "$OPENAI_BASE_URL" || -z "$OPENAI_API_KEY" ]]; then echo "Set OPENAI_BASE_URL and OPENAI_API_KEY before running" >&2; exit 2; fi

# Host-side proxy for task asset downloads only.
HOST_PROXY_URL="${HOST_PROXY_URL-http://127.0.0.1:3128}"
if [[ -n "${HOST_PROXY_URL}" ]]; then
  export http_proxy="${HOST_PROXY_URL}"
  export https_proxy="${HOST_PROXY_URL}"
  export HTTP_PROXY="${HOST_PROXY_URL}"
  export HTTPS_PROXY="${HOST_PROXY_URL}"
fi

# Keep cluster and the Qwen API endpoint off the proxy path. The API host is an
# internal address (resolves to 10.x) that is directly reachable; routing it
# through the proxy breaks the TLS handshake. no_proxy matches request hostnames
# textually (it does NOT resolve to IP then match CIDR), so the API host must be
# listed by name even though it falls inside 10.0.0.0/8.
_api_host="${OPENAI_BASE_URL#*://}"; _api_host="${_api_host%%[:/]*}"
_no_proxy_add="localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,28.0.0.0/8"
[[ -n "${_api_host}" ]] && _no_proxy_add="${_no_proxy_add},${_api_host}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${_no_proxy_add}"
export no_proxy="${no_proxy:+${no_proxy},}${_no_proxy_add}"

# Background mode.
if [[ "${BACKGROUND:-1}" == "1" && -z "${_QWEN37PLUS_DETACHED:-}" ]]; then
  export _QWEN37PLUS_DETACHED=1
  nohup bash "$0" "$@" > "${LOG_FILE}" 2>&1 < /dev/null &
  CHILD_PID=$!
  echo "Started in background, PID=${CHILD_PID}"
  echo "Log:  ${LOG_FILE}"
  echo "Tail: tail -f ${LOG_FILE}"
  echo "Stop: kill ${CHILD_PID}"
  exit 0
fi

echo "Logging to ${LOG_FILE}"
echo "OPENAI_BASE_URL=${OPENAI_BASE_URL}"
echo "GUI_ENV_SERVER_URL=${GUI_ENV_SERVER_URL}"

if [[ -n "${_QWEN37PLUS_DETACHED:-}" ]]; then
  TEE_CMD=(cat)
else
  TEE_CMD=(tee "${LOG_FILE}")
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CMD=(
  "${PYTHON_BIN}" scripts/python/osworld/run_qwen_qwen37plus.py
  --cluster_url "${GUI_ENV_SERVER_URL}"
  --action_space "${ACTION_SPACE:-pyautogui}"
  --observation_type "${OBSERVATION_TYPE:-screenshot}"
  --model "${MODEL:-tongyi/qwen3.7-plus}"
  --coord "${COORD:-relative}"
  --max_tokens "${MAX_TOKENS:-2048}"
  --temperature "${TEMPERATURE:-0}"
  --top_p "${TOP_P:-0.9}"
  --history_n "${HISTORY_N:-100}"
  --image_max "${IMAGE_MAX:-20}"
  --fold_size "${FOLD_SIZE:-10}"
  --max_steps "${MAX_STEPS:-50}"
  --test_config_base_dir "${TEST_CONFIG_BASE_DIR:-OSWorld/evaluation_examples}"
  --num_envs "${NUM_ENVS:-8}"
  --result_dir "${RESULT_DIR:-./results_qwen_qwen37plus}"
  --test_all_meta_path "${TEST_META_PATH:-OSWorld/evaluation_examples/test_nogdrive.json}"
  --domain "${DOMAIN:-all}"
  --client_password "${CLIENT_PASSWORD:-password}"
  --screen_width "${SCREEN_WIDTH:-1920}"
  --screen_height "${SCREEN_HEIGHT:-1080}"
  --sleep_after_execution "${SLEEP_AFTER_EXECUTION:-3}"
  --log_level "${LOG_LEVEL:-INFO}"
  --base_url "${OPENAI_BASE_URL}"
  --api_key "${OPENAI_API_KEY}"
  "$@"
)

if [[ "${HEADLESS:-1}" == "1" ]]; then CMD+=(--headless); fi
if [[ "${ADD_THOUGHT_PREFIX:-0}" == "1" ]]; then CMD+=(--add_thought_prefix); fi
if [[ "${ENABLE_THINKING:-0}" == "1" ]]; then CMD+=(--enable_thinking); fi

"${CMD[@]}" 2>&1 | "${TEE_CMD[@]}"
