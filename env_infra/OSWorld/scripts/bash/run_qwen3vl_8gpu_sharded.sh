#!/usr/bin/env bash
set -euo pipefail

_script_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "${_script_dir}/../../server.py" ]]; then
  cd "${_script_dir}/../.."
fi

# ----------------------------------------------------------------------------
# Run OSWorld evaluation with Qwen3-VL-8B-Instruct via OSWorldRemoteClient
# cluster environments and multiple OpenAI-compatible vLLM endpoints.
#
# Expected default endpoints:
#   Set OPENAI_HOST or OPENAI_BASE_URLS to select model endpoints.
#
# Start one vLLM server per GPU before running this script, for example with
# different GPU_ID/PORT values via scripts/bash/start_vllm_qwen3vl.sh.
#
# Override endpoint sharding explicitly:
#   OPENAI_BASE_URLS=http://host0:8000/v1,http://host1:8000/v1 \
#     NUM_ENVS=16 DOMAIN=libreoffice_calc \
#     bash scripts/bash/run_qwen3vl_8gpu_sharded.sh
# ----------------------------------------------------------------------------

mkdir -p logs

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-logs/run_qwen3vl_8gpu_sharded_${TIMESTAMP}.log}"
export PROXY_CONFIG_FILE="${PROXY_CONFIG_FILE:-$(pwd)/evaluation_examples/settings/proxy/local.json}"

# Remote OSWorld environment cluster master.
export GUI_ENV_SERVER_URL="${GUI_ENV_SERVER_URL:-http://127.0.0.1:19000/}"
export OSWORLD_USER_ID="${OSWORLD_USER_ID:-chentongbo}"
export OSWORLD_TASK_TYPE="${OSWORLD_TASK_TYPE:-evaluation}"

# Host-side proxy for task asset downloads. Set HOST_PROXY_URL= to disable.
HOST_PROXY_URL="${HOST_PROXY_URL-http://127.0.0.1:3128}"
if [[ -n "${HOST_PROXY_URL}" ]]; then
  export http_proxy="${HOST_PROXY_URL}"
  export https_proxy="${HOST_PROXY_URL}"
  export HTTP_PROXY="${HOST_PROXY_URL}"
  export HTTPS_PROXY="${HOST_PROXY_URL}"
  export all_proxy="${HOST_PROXY_URL}"
  export ALL_PROXY="${HOST_PROXY_URL}"
else
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
fi

# Build a default eight-endpoint shard list when OPENAI_BASE_URLS is not set.
if [[ -z "${OPENAI_BASE_URLS:-}" ]]; then
  OPENAI_HOST="${OPENAI_HOST:-}"
  if [[ -z "$OPENAI_HOST" ]]; then echo "Set OPENAI_HOST or OPENAI_BASE_URLS before running" >&2; exit 2; fi
  OPENAI_PORT_START="${OPENAI_PORT_START:-8000}"
  OPENAI_NUM_SHARDS="${OPENAI_NUM_SHARDS:-8}"
  _urls=()
  for ((i = 0; i < OPENAI_NUM_SHARDS; i++)); do
    _urls+=("http://${OPENAI_HOST}:$((OPENAI_PORT_START + i))/v1")
  done
  OPENAI_BASE_URLS="$(IFS=,; echo "${_urls[*]}")"
fi
export OPENAI_BASE_URLS
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${OPENAI_BASE_URLS%%,*}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-local}"

# Keep local, environment-server, and vLLM endpoints off the proxy path.
_no_proxy_add="localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
_gui_host="${GUI_ENV_SERVER_URL#*://}"
_gui_host="${_gui_host%%[:/]*}"
if [[ -n "${_gui_host}" ]]; then
  _no_proxy_add="${_no_proxy_add},${_gui_host}"
fi
IFS=',' read -r -a _base_url_array <<< "${OPENAI_BASE_URLS}"
for _url in "${_base_url_array[@]}"; do
  _trimmed_url="${_url//[[:space:]]/}"
  _host="${_trimmed_url#*://}"
  _host="${_host%%/*}"
  _host="${_host%%:*}"
  if [[ -n "${_host}" ]]; then
    _no_proxy_add="${_no_proxy_add},${_host}"
  fi
done
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${_no_proxy_add}"
export no_proxy="${no_proxy:+${no_proxy},}${_no_proxy_add}"

# Default: detach to background via nohup. Pass BACKGROUND=0 to run in foreground.
if [[ "${BACKGROUND:-1}" == "1" && -z "${_QWEN3VL_SHARDED_DETACHED:-}" ]]; then
  export _QWEN3VL_SHARDED_DETACHED=1
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
echo "PROXY_CONFIG_FILE=${PROXY_CONFIG_FILE}"
echo "HOST_PROXY_URL=${HOST_PROXY_URL}"
echo "NO_PROXY=${NO_PROXY}"

if [[ -n "${_QWEN3VL_SHARDED_DETACHED:-}" ]]; then
  # Already detached by nohup; stdout/stderr go to LOG_FILE already.
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

PYTHON_BIN="${PYTHON_BIN:-python}"
CMD=(
  "${PYTHON_BIN}" scripts/python/run_multienv_qwen3vl_8gpu_sharded.py
  --cluster_url "${GUI_ENV_SERVER_URL}"
  --provider_name "${PROVIDER_NAME:-docker_server}"
  "${HEADLESS_ARGS[@]}"
  "${PATH_TO_VM_ARGS[@]}"
  --action_space "${ACTION_SPACE:-pyautogui}"
  --observation_type "${OBSERVATION_TYPE:-screenshot}"
  --model "${MODEL:-Qwen3-VL-8B-Instruct}"
  --coord "${COORD:-relative}"
  --max_tokens "${MAX_TOKENS:-2048}"
  --temperature "${TEMPERATURE:-0}"
  --top_p "${TOP_P:-0.9}"
  --max_steps "${MAX_STEPS:-50}"
  --max_trajectory_length "${MAX_TRAJ_LEN:-3}"
  --test_config_base_dir "${TEST_CONFIG_BASE_DIR:-evaluation_examples}"
  --num_envs "${NUM_ENVS:-128}"
  --result_dir "${RESULT_DIR:-./results_qwen3vl_8gpu_sharded}"
  --test_all_meta_path "${TEST_META_PATH:-evaluation_examples/test_nogdrive.json}"
  --domain "${DOMAIN:-all}"
  --client_password "${CLIENT_PASSWORD:-${OSWORLD_CLIENT_PASSWORD:-password}}"
  --region "${REGION:-${AWS_REGION:-us-east-1}}"
  --screen_width "${SCREEN_WIDTH:-1920}"
  --screen_height "${SCREEN_HEIGHT:-1080}"
  --sleep_after_execution "${SLEEP_AFTER_EXECUTION:-3}"
  --log_level "${LOG_LEVEL:-INFO}"
  --openai_base_urls "${OPENAI_BASE_URLS}"
  "${ADD_THOUGHT_PREFIX_ARGS[@]}"
  "$@"
)

"${CMD[@]}" 2>&1 | "${TEE_CMD[@]}"
