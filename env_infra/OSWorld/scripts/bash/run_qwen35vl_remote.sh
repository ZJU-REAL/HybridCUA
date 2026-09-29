#!/usr/bin/env bash
set -euo pipefail

_script_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "${_script_dir}/../../server.py" ]]; then
  cd "${_script_dir}/../.."
fi

# ----------------------------------------------------------------------------
# Run OSWorld evaluation with Qwen3.5-VL through OSWorldRemoteClient and Cluster.
#
# Start the cluster first:
#   bash scripts/bash/start_cluster_master.sh
#   NODE_MASTER_URL=http://master-ip:19000 bash scripts/bash/start_cluster_node.sh
#
# Run evaluation with Qwen3.5-VL:
#   GUI_ENV_SERVER_URL=http://127.0.0.1:19000/ \
#   OPENAI_API_KEY=sk-... \
#   MODEL=tongyi/qwen3.5-plus \
#   NUM_ENVS=48 \
#   bash scripts/bash/run_qwen35vl_remote.sh
#
# Run evaluation with another OpenAI-compatible Qwen endpoint:
#   GUI_ENV_SERVER_URL=http://127.0.0.1:19000/ \
#   OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
#   OPENAI_API_KEY=sk-... \
#   MODEL=tongyi/qwen3.5-397b-a17b \
#   NUM_ENVS=48 \
#   bash scripts/bash/run_qwen35vl_remote.sh
# ----------------------------------------------------------------------------

# User-facing defaults. Env vars passed at launch still take priority.
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
if [[ -z "$OPENAI_BASE_URL" ]]; then echo "Set OPENAI_BASE_URL before running" >&2; exit 2; fi

mkdir -p logs

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-logs/run_qwen35vl_remote_${TIMESTAMP}.log}"

export GUI_ENV_SERVER_URL="${GUI_ENV_SERVER_URL:-http://127.0.0.1:19000/}"
export OSWORLD_USER_ID="${OSWORLD_USER_ID:-${USER:-anonymous}}"
export OSWORLD_TASK_TYPE="${OSWORLD_TASK_TYPE:-evaluation}"

if [[ -z "${MODEL:-}" ]]; then
  export MODEL="${QWEN_VL_MODEL:-tongyi/qwen3.5-plus}"
fi
export PROXY_CONFIG_FILE="${PROXY_CONFIG_FILE:-$(pwd)/evaluation_examples/settings/proxy/local.json}"

if [[ -z "${OPENAI_API_KEY}" ]]; then
  echo "ERROR: OPENAI_API_KEY is empty. Set OPENAI_API_KEY." >&2
  exit 2
fi

_gui_host="${GUI_ENV_SERVER_URL#*://}"
_gui_host="${_gui_host%%[:/]*}"
_no_proxy_add="localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,${_gui_host}"
export no_proxy="${no_proxy:+${no_proxy},}${_no_proxy_add}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${_no_proxy_add}"

if [[ "${BACKGROUND:-1}" == "1" && -z "${_QWEN35VL_REMOTE_DETACHED:-}" ]]; then
  export _QWEN35VL_REMOTE_DETACHED=1
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
echo "OPENAI_BASE_URL=${OPENAI_BASE_URL}"
echo "MODEL=${MODEL}"
echo "PROXY_CONFIG_FILE=${PROXY_CONFIG_FILE}"
echo "OSWORLD_USER_ID=${OSWORLD_USER_ID}"
echo "OSWORLD_TASK_TYPE=${OSWORLD_TASK_TYPE}"

if [[ -n "${_QWEN35VL_REMOTE_DETACHED:-}" ]]; then
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

ADD_THOUGHT_PREFIX_ARGS=()
if [[ "${ADD_THOUGHT_PREFIX:-0}" == "1" ]]; then
  ADD_THOUGHT_PREFIX_ARGS=(--add_thought_prefix)
fi

SIMPLE_PATH_ARGS=()
if [[ "${SIMPLE_PATH:-0}" == "1" ]]; then
  SIMPLE_PATH_ARGS=(--simple_path)
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CMD=(
  "${PYTHON_BIN}" scripts/python/run_multienv_qwen35vl_remote.py
  --cluster_url "${GUI_ENV_SERVER_URL}"
  --provider_name "${PROVIDER_NAME:-docker_server}"
  "${HEADLESS_ARGS[@]}"
  "${PATH_TO_VM_ARGS[@]}"
  --action_space "${ACTION_SPACE:-pyautogui}"
  --observation_type "${OBSERVATION_TYPE:-screenshot}"
  --model "${MODEL}"
  --coord "${COORD:-relative}"
  --max_tokens "${MAX_TOKENS:-2048}"
  --temperature "${TEMPERATURE:-0}"
  --top_p "${TOP_P:-0.9}"
  --max_steps "${MAX_STEPS:-50}"
  --history_n "${HISTORY_N:-100}"
  --image_max "${IMAGE_MAX:-20}"
  --fold_size "${FOLD_SIZE:-10}"
  --test_config_base_dir "${TEST_CONFIG_BASE_DIR:-evaluation_examples}"
  --examples_subdir "${EXAMPLES_SUBDIR:-examples}"
  --num_envs "${NUM_ENVS:-8}"
  --result_dir "${RESULT_DIR:-./results_qwen35vl_remote}"
  --test_all_meta_path "${TEST_META_PATH:-evaluation_examples/test_nogdrive.json}"
  --domain "${DOMAIN:-all}"
  --client_password "${CLIENT_PASSWORD:-${OSWORLD_CLIENT_PASSWORD:-password}}"
  --region "${REGION:-${AWS_REGION:-us-east-1}}"
  --screen_width "${SCREEN_WIDTH:-1920}"
  --screen_height "${SCREEN_HEIGHT:-1080}"
  --sleep_after_execution "${SLEEP_AFTER_EXECUTION:-3}"
  --base_url "${OPENAI_BASE_URL}"
  --api_key "${OPENAI_API_KEY}"
  --log_level "${LOG_LEVEL:-INFO}"
  "${ADD_THOUGHT_PREFIX_ARGS[@]}"
  "${SIMPLE_PATH_ARGS[@]}"
  "$@"
)

"${CMD[@]}" 2>&1 | "${TEE_CMD[@]}"
