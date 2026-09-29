#!/usr/bin/env bash
set -euo pipefail

if [ -z "${OSWORLD_JOB_ID:-}" ]; then
  cd "$(dirname "$0")/../../.."
fi

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}" 2>/dev/null || LOG_DIR="/tmp/env_infra_logs" && mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run_qwen38_c_gui_cua_gym_${TIMESTAMP}.log}"

export GUI_ENV_SERVER_URL="${GUI_ENV_SERVER_URL:-http://127.0.0.1:19000}"
export ENABLE_PROXY="${ENABLE_PROXY:-1}"

if [[ -z "${OPENAI_BASE_URLS:-}" ]]; then
  _DEFAULT_HOSTS="28.81.129.185,28.82.240.212"
  OPENAI_HOSTS="${OPENAI_HOSTS:-${OPENAI_HOST:-$_DEFAULT_HOSTS}}"
  OPENAI_PORT_START="${OPENAI_PORT_START:-8000}"
  OPENAI_NUM_SHARDS="${OPENAI_NUM_SHARDS:-8}"
  IFS=',' read -r -a _hosts <<< "${OPENAI_HOSTS}"
  _urls=()
  for _h in "${_hosts[@]}"; do
    _h="${_h// /}"; [[ -z "${_h}" ]] && continue
    for ((i = 0; i < OPENAI_NUM_SHARDS; i++)); do
      _urls+=("http://${_h}:$((OPENAI_PORT_START + i))/v1")
    done
  done
  OPENAI_BASE_URLS="$(IFS=,; echo "${_urls[*]}")"
fi
export OPENAI_BASE_URLS
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-local}"

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

if [[ "${BACKGROUND:-1}" == "1" && -z "${_QWEN38_CGUI_DETACHED:-}" ]]; then
  export _QWEN38_CGUI_DETACHED=1
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
echo "MODEL=${MODEL:-Qwen3.5-9B}"
echo "NUM_ENVS=${NUM_ENVS:-64}  MAX_STEPS=${MAX_STEPS:-50}  HISTORY_N=${HISTORY_N:-50}  IMAGE_MAX=${IMAGE_MAX:-5}"
echo "CLI_SKILLS=${CLI_SKILLS:-0}"

if [[ -n "${_QWEN38_CGUI_DETACHED:-}" ]]; then
  TEE_CMD=(cat)
else
  TEE_CMD=(tee "${LOG_FILE}")
fi

meta_arg=()
TASKS_META="${TASKS_META-./cua_gym_data/rlvr_curriculum_1000_meta.json}"
if [ -n "${TASKS_META}" ]; then
  if [ ! -f "${TASKS_META}" ]; then
    echo "[!] TASKS_META not found: ${TASKS_META}" >&2
    exit 1
  fi
  meta_arg=(--tasks_meta "${TASKS_META}")
fi
echo "TASKS_META=${TASKS_META:-<none: all bundles>}"

_DEF_PYTHON_BIN="python"
PYTHON_BIN="${PYTHON_BIN:-$_DEF_PYTHON_BIN}"
if ! "${PYTHON_BIN}" -c 'import wrapt_timeout_decorator, openai, PIL' 2>/dev/null; then
  echo "[!] ${PYTHON_BIN} missing deps; use: export PYTHON_BIN=$_DEF_PYTHON_BIN" >&2
  exit 1
fi

CMD=(
  "${PYTHON_BIN}" scripts/python/cua_gym/run_qwen38_c_gui.py
  --cluster_url "${GUI_ENV_SERVER_URL}"
  --tasks_root "${TASKS_ROOT:-./cua_gym_data/rlvr}"
  "${meta_arg[@]}"
  --domain "${DOMAIN:-all}"
  --action_space "${ACTION_SPACE:-pyautogui}"
  --observation_type "${OBSERVATION_TYPE:-screenshot}"
  --model "${MODEL:-Qwen3.5-9B}"
  --temperature "${TEMPERATURE:-0}"
  --top_p "${TOP_P:-0.9}"
  --max_tokens "${MAX_TOKENS:-4096}"
  --history_n "${HISTORY_N:-50}"
  --image_max "${IMAGE_MAX:-5}"
  --fold_size "${FOLD_SIZE:-1}"
  --coord "${COORD:-relative}"
  --max_steps "${MAX_STEPS:-50}"
  --num_envs "${NUM_ENVS:-64}"
  --result_dir "${RESULT_DIR:-./results_rlvr_c_gui_qwen35_9b}"
  --client_password "${CLIENT_PASSWORD:-password}"
  --password "${PASSWORD:-password}"
  --screen_width "${SCREEN_WIDTH:-1920}"
  --screen_height "${SCREEN_HEIGHT:-1080}"
  --sleep_after_execution "${SLEEP_AFTER_EXECUTION:-3}"
  --wait_after_reset "${WAIT_AFTER_RESET:-60}"
  --log_level "${LOG_LEVEL:-INFO}"
  --openai_base_urls "${OPENAI_BASE_URLS}"
  --api_key "${OPENAI_API_KEY}"
  "$@"
)

if [[ -n "${VM_PROXY+x}" ]]; then CMD+=(--vm_proxy "${VM_PROXY}"); fi
if [[ "${HEADLESS:-1}" == "1" ]]; then CMD+=(--headless); fi
if [[ "${CLI_SKILLS:-0}" == "0" ]]; then CMD+=(--no_cli_skills); fi
if [[ "${ADD_THOUGHT_PREFIX:-0}" == "1" ]]; then CMD+=(--add_thought_prefix); fi
if [[ "${ENABLE_THINKING:-0}" == "1" ]]; then CMD+=(--enable_thinking); fi

"${CMD[@]}" 2>&1 | "${TEE_CMD[@]}"
