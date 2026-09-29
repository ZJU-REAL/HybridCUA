#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------------
# Run OSWorld evaluation with the Hybrid GUI+CLI agent against Qwen3.5-35B-A3B,
# sharded across multiple vLLM endpoints via the platform cluster.
#
# SELF-CONTAINED copy of run_hybrid.sh with the 35B-A3B layout baked in
# (4 servers x TP=2 on ports 8000..8003, model name Qwen3.5-35B-A3B; see
# start_vllm_qwen35_35b_a3b_8gpu.sh). It is a standalone runner — NOT a wrapper —
# because the dashboard job runner copies only the selected script to /tmp, so a
# script that exec'd a sibling would break under OSWORLD_JOB_ID. Every knob is a
# ${VAR:-default} so it stays visible and overridable.
#
# Usage:
#   bash scripts/bash/osworld/run_hybrid_qwen35_35b_a3b.sh
# Override anything via env, e.g.:
#   OPENAI_HOST=127.0.0.1 NUM_ENVS=16 DOMAIN=os \
#     bash scripts/bash/osworld/run_hybrid_qwen35_35b_a3b.sh
# ----------------------------------------------------------------------------

# --- Model + sharding (35B-A3B specific) ------------------------------------
MODEL="${MODEL:-Qwen3.5-35B-A3B}"             # must match vLLM --served-model-name
OPENAI_NUM_SHARDS="${OPENAI_NUM_SHARDS:-4}"   # 4 servers x TP=2
NUM_ENVS="${NUM_ENVS:-32}"                    # parallel OSWorld sessions
IMAGE_MAX="${IMAGE_MAX:-5}"                   # max screenshots kept in prompt
FOLD_SIZE="${FOLD_SIZE:-1}"                   # history fold granularity
HISTORY_N="${HISTORY_N:-50}"                  # max history steps

# --- vLLM endpoint location -------------------------------------------------
OPENAI_HOST="${OPENAI_HOST:-127.0.0.1}"       # host running vLLM
OPENAI_PORT_START="${OPENAI_PORT_START:-8000}" # first shard port
# Set OPENAI_BASE_URLS directly to bypass host/port/shards autobuild entirely.

# --- Task selection / decoding (see run_hybrid.sh for the full set) ---------
DOMAIN="${DOMAIN:-all}"
MAX_STEPS="${MAX_STEPS:-50}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0}"
RESULT_DIR="${RESULT_DIR:-./results_hybrid}"

# ============================================================================
# Below here mirrors run_hybrid.sh verbatim (self-contained; no sibling calls).
# ============================================================================

# When run via cluster master, cwd is already repo root.
# When run manually, cd to repo root relative to script location.
if [[ -z "${OSWORLD_JOB_ID:-}" ]]; then
  _script_dir="$(cd "$(dirname "$0")" && pwd)"
  cd "${_script_dir}/../../.."
fi

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}" 2>/dev/null || LOG_DIR="/tmp/env_infra_logs" && mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run_hybrid_qwen35_35b_a3b_${TIMESTAMP}.log}"
export PROXY_CONFIG_FILE="${PROXY_CONFIG_FILE:-$(pwd)/OSWorld/evaluation_examples/settings/proxy/star_proxy.json}"

export GUI_ENV_SERVER_URL="${GUI_ENV_SERVER_URL:-http://127.0.0.1:19000/}"
export OSWORLD_USER_ID="${OSWORLD_USER_ID:-${USER:-anonymous}}"
export OSWORLD_TASK_TYPE="${OSWORLD_TASK_TYPE:-evaluation}"

# Build shard list if OPENAI_BASE_URLS not set.
if [[ -z "${OPENAI_BASE_URLS:-}" ]]; then
  _urls=()
  for ((i = 0; i < OPENAI_NUM_SHARDS; i++)); do
    _urls+=("http://${OPENAI_HOST}:$((OPENAI_PORT_START + i))/v1")
  done
  OPENAI_BASE_URLS="$(IFS=,; echo "${_urls[*]}")"
fi
export OPENAI_BASE_URLS
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-local}"

# Host-side proxy for task asset downloads.
HOST_PROXY_URL="${HOST_PROXY_URL-http://127.0.0.1:3128}"
if [[ -n "${HOST_PROXY_URL}" ]]; then
  export http_proxy="${HOST_PROXY_URL}"
  export https_proxy="${HOST_PROXY_URL}"
  export HTTP_PROXY="${HOST_PROXY_URL}"
  export HTTPS_PROXY="${HOST_PROXY_URL}"
fi

# Keep cluster, vLLM endpoints off the proxy path.
_no_proxy_add="localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,28.0.0.0/8"
IFS=',' read -r -a _base_url_array <<< "${OPENAI_BASE_URLS}"
for _url in "${_base_url_array[@]}"; do
  _host="${_url#*://}"; _host="${_host%%[:/]*}"
  [[ -n "${_host}" ]] && _no_proxy_add="${_no_proxy_add},${_host}"
done
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${_no_proxy_add}"
export no_proxy="${no_proxy:+${no_proxy},}${_no_proxy_add}"

# Background mode.
if [[ "${BACKGROUND:-1}" == "1" && -z "${_HYBRID_SHARDED_DETACHED:-}" ]]; then
  export _HYBRID_SHARDED_DETACHED=1
  nohup bash "$0" "$@" > "${LOG_FILE}" 2>&1 < /dev/null &
  CHILD_PID=$!
  echo "Started in background, PID=${CHILD_PID}"
  echo "Log:  ${LOG_FILE}"
  echo "Tail: tail -f ${LOG_FILE}"
  echo "Stop: kill ${CHILD_PID}"
  exit 0
fi

echo "Logging to ${LOG_FILE}"
echo "MODEL=${MODEL}"
echo "OPENAI_BASE_URLS=${OPENAI_BASE_URLS}"
echo "GUI_ENV_SERVER_URL=${GUI_ENV_SERVER_URL}"

if [[ -n "${_HYBRID_SHARDED_DETACHED:-}" ]]; then
  TEE_CMD=(cat)
else
  TEE_CMD=(tee "${LOG_FILE}")
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CMD=(
  "${PYTHON_BIN}" scripts/python/osworld/run_hybrid_sharded.py
  --cluster_url "${GUI_ENV_SERVER_URL}"
  --action_space "${ACTION_SPACE:-pyautogui}"
  --observation_type "${OBSERVATION_TYPE:-screenshot}"
  --model "${MODEL}"
  --coord "${COORD:-relative}"
  --max_tokens "${MAX_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P:-0.9}"
  --history_n "${HISTORY_N}"
  --image_max "${IMAGE_MAX}"
  --fold_size "${FOLD_SIZE}"
  --max_steps "${MAX_STEPS}"
  --test_config_base_dir "${TEST_CONFIG_BASE_DIR:-OSWorld/evaluation_examples}"
  --num_envs "${NUM_ENVS}"
  --result_dir "${RESULT_DIR}"
  --test_all_meta_path "${TEST_META_PATH:-OSWorld/evaluation_examples/test_nogdrive.json}"
  --domain "${DOMAIN}"
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
