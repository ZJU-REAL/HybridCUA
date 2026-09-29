#!/usr/bin/env bash
set -euo pipefail

_script_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "${_script_dir}/../../server.py" ]]; then
  cd "${_script_dir}/../.."
fi

# ----------------------------------------------------------------------------
# Run OSWorld evaluation with Kimi K2.5/K2.6 through multi-env parallel runner.
#
# Usage:
#   GUI_ENV_SERVER_URL=http://127.0.0.1:19000/ \
#   KIMI_API_KEY=your_key \
#   NUM_ENVS=48 \
#   bash scripts/bash/run_kimi_remote.sh
#
# Thinking mode is enabled by default. To disable it:
#   KIMI_API_KEY=your_key \
#   MODEL=moonshot/kimi-k2.6 \
#   THINKING=0 \
#   NUM_ENVS=48 \
#   bash scripts/bash/run_kimi_remote.sh
# ----------------------------------------------------------------------------

mkdir -p logs

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-logs/run_kimi_remote_${TIMESTAMP}.log}"

export KIMI_API_KEY="${KIMI_API_KEY:-}"
export KIMI_BASE_URL="${KIMI_BASE_URL:-}"
if [[ -z "$KIMI_BASE_URL" ]]; then echo "Set KIMI_BASE_URL before running" >&2; exit 2; fi
export GUI_ENV_SERVER_URL="${GUI_ENV_SERVER_URL:-http://127.0.0.1:19000/}"
export OSWORLD_USER_ID="${OSWORLD_USER_ID:-${USER:-chentongbo}}"
export OSWORLD_TASK_TYPE="${OSWORLD_TASK_TYPE:-evaluation}"

export PROXY_CONFIG_FILE="${PROXY_CONFIG_FILE:-$(pwd)/evaluation_examples/settings/proxy/local.json}"

if [[ -z "${KIMI_API_KEY}" ]]; then
  echo "ERROR: KIMI_API_KEY is not set. Export it before running this script."
  exit 1
fi

_gui_host="${GUI_ENV_SERVER_URL#*://}"
_gui_host="${_gui_host%%[:/]*}"
_no_proxy_add="localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,${_gui_host}"
export no_proxy="${no_proxy:+${no_proxy},}${_no_proxy_add}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${_no_proxy_add}"

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
echo "MODEL=${MODEL:-moonshot/kimi-k2.6}"
echo "KIMI_BASE_URL=${KIMI_BASE_URL}"
echo "KIMI_API_KEY is configured"
echo "PROVIDER_NAME=${PROVIDER_NAME:-docker_server}"
echo "NUM_ENVS=${NUM_ENVS:-8}"
echo "THINKING=${THINKING:-1}"
echo "OSWORLD_USER_ID=${OSWORLD_USER_ID}"
echo "OSWORLD_TASK_TYPE=${OSWORLD_TASK_TYPE}"

if [[ -n "${_KIMI_REMOTE_DETACHED:-}" ]]; then
  TEE_CMD=(cat)
else
  TEE_CMD=(tee "${LOG_FILE}")
fi

HEADLESS_ARGS=()
if [[ "${HEADLESS:-1}" == "1" ]]; then
  HEADLESS_ARGS=(--headless)
fi

PATH_TO_VM_ARGS=()
if [[ -n "${PATH_TO_VM:-}" ]]; then
  PATH_TO_VM_ARGS=(--path_to_vm "${PATH_TO_VM}")
fi

THINKING_ARGS=()
if [[ "${THINKING:-1}" == "1" ]]; then
  THINKING_ARGS=(--thinking)
fi

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CMD=(
  "${PYTHON_BIN}" scripts/python/run_multienv_kimi_k25_remote.py
  --cluster_url "${GUI_ENV_SERVER_URL}"
  "${HEADLESS_ARGS[@]}"
  "${PATH_TO_VM_ARGS[@]}"
  --action_space "${ACTION_SPACE:-pyautogui}"
  --observation_type "${OBSERVATION_TYPE:-screenshot}"
  --model "${MODEL:-moonshot/kimi-k2.6}"
  --coordinate_type "${COORD:-relative}"
  --max_tokens "${MAX_TOKENS:-2048}"
  --temperature "${TEMPERATURE:-1}"
  --top_p "${TOP_P:-0.95}"
  --max_steps "${MAX_STEPS:-50}"
  --max_image_history_length "${MAX_TRAJ_LEN:-3}"
  --test_config_base_dir "${TEST_CONFIG_BASE_DIR:-evaluation_examples}"
  --num_envs "${NUM_ENVS:-8}"
  --result_dir "${RESULT_DIR:-./results_kimi_remote}"
  --test_all_meta_path "${TEST_META_PATH:-evaluation_examples/test_nogdrive.json}"
  --domain "${DOMAIN:-all}"
  --provider_name "${PROVIDER_NAME:-docker_server}"
  --client_password "${CLIENT_PASSWORD:-${OSWORLD_CLIENT_PASSWORD:-password}}"
  --password "${PASSWORD:-osworld-public-evaluation}"
  --region "${REGION:-${AWS_REGION:-us-east-1}}"
  --screen_width "${SCREEN_WIDTH:-1920}"
  --screen_height "${SCREEN_HEIGHT:-1080}"
  --sleep_after_execution "${SLEEP_AFTER_EXECUTION:-3}"
  --base_url "${KIMI_BASE_URL}"
  --api_key "${KIMI_API_KEY}"
  --log_level "${LOG_LEVEL:-INFO}"
  "${THINKING_ARGS[@]}"
  "$@"
)

"${CMD[@]}" 2>&1 | "${TEE_CMD[@]}"
