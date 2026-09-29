#!/usr/bin/env bash
set -euo pipefail

# When run via cluster master, cwd is already repo root.
# When run manually, cd to repo root relative to script location.
if [[ -z "${OSWORLD_JOB_ID:-}" ]]; then
  _script_dir="$(cd "$(dirname "$0")" && pwd)"
  cd "${_script_dir}/../../.."
fi

# ----------------------------------------------------------------------------
# Run OSWorld evaluation with the c-gui agent (mm_agents.c_gui), sharded across
# vLLM endpoints via the platform cluster.
#
# Same shape as run_hybrid.sh: drives the cluster EvalRunner + OSWorldEvalSource
# (remote OSWorldSessionClient, NOT a local DesktopEnv), sharding requests across
# MULTIPLE vLLM endpoints (one per worker, round-robin). The agent exposes ONE
# tool (single bash action); GUI is pyautogui in a heredoc (coords 0-999, scaled
# by the VM shim). TOOL_NAME selects the A/B naming variable (computer_use|cli).
#
#   OPENAI_BASE_URLS=http://host:8000/v1,...,http://host:8007/v1 \
#   NUM_ENVS=16 TOOL_NAME=computer_use \
#   bash scripts/bash/osworld/run_c_gui.sh
# ----------------------------------------------------------------------------

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}" 2>/dev/null || LOG_DIR="/tmp/env_infra_logs" && mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run_c_gui_${TIMESTAMP}.log}"
export PROXY_CONFIG_FILE="${PROXY_CONFIG_FILE:-$(pwd)/OSWorld/evaluation_examples/settings/proxy/star_proxy.json}"

export GUI_ENV_SERVER_URL="${GUI_ENV_SERVER_URL:-http://127.0.0.1:19000/}"
export OSWORLD_USER_ID="${OSWORLD_USER_ID:-${USER:-anonymous}}"
export OSWORLD_TASK_TYPE="${OSWORLD_TASK_TYPE:-evaluation}"

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
if [[ "${BACKGROUND:-1}" == "1" && -z "${_C_GUI_SHARDED_DETACHED:-}" ]]; then
  export _C_GUI_SHARDED_DETACHED=1
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
echo "TOOL_NAME=${TOOL_NAME:-computer_use}"

if [[ -n "${_C_GUI_SHARDED_DETACHED:-}" ]]; then
  TEE_CMD=(cat)
else
  TEE_CMD=(tee "${LOG_FILE}")
fi

# 解释器: 必须用 env_infra 的 venv, 不能用裸 `python`。
# 裸 python 在本机解析到 /opt/conda/bin/python (3.13), 那里没装 osworld 的依赖
# (wrapt_timeout_decorator 等), 也没有 osworld 的 editable 安装 -> 32 个 worker
# 会齐刷刷 ModuleNotFoundError。集群 master/node 跑的就是下面这个 venv, client 要同源。
# 可用 PYTHON_BIN=... 覆盖。
_DEF_PYTHON_BIN="python"
PYTHON_BIN="${PYTHON_BIN:-$_DEF_PYTHON_BIN}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[!] PYTHON_BIN 不可执行: ${PYTHON_BIN}" >&2
  exit 1
fi
if ! "${PYTHON_BIN}" -c 'import wrapt_timeout_decorator' 2>/dev/null; then
  echo "[!] ${PYTHON_BIN} 缺 osworld 依赖 (import wrapt_timeout_decorator 失败)。" >&2
  echo "    别往这个解释器补装单个包 —— 换成装好 editable osworld 的 venv:" >&2
  echo "    export PYTHON_BIN=$_DEF_PYTHON_BIN" >&2
  exit 1
fi
echo "PYTHON_BIN=${PYTHON_BIN}"

CMD=(
  "${PYTHON_BIN}" scripts/python/osworld/run_c_gui_sharded.py
  --cluster_url "${GUI_ENV_SERVER_URL}"
  --action_space "${ACTION_SPACE:-pyautogui}"
  --observation_type "${OBSERVATION_TYPE:-screenshot}"
  --model "${MODEL:-Qwen3.5-9B}"
  --coord "${COORD:-relative}"
  --max_tokens "${MAX_TOKENS:-8192}"
  --temperature "${TEMPERATURE:-0}"
  --top_p "${TOP_P:-0.9}"
  --history_n "${HISTORY_N:-50}"
  --image_max "${IMAGE_MAX:-5}"
  --fold_size "${FOLD_SIZE:-1}"
  --tool_name "${TOOL_NAME:-computer_use}"
  --max_steps "${MAX_STEPS:-50}"
  --test_config_base_dir "${TEST_CONFIG_BASE_DIR:-OSWorld/evaluation_examples}"
  --num_envs "${NUM_ENVS:-64}"
  --result_dir "${RESULT_DIR:-./results-0909-40gpu-roll119}"
  --test_all_meta_path "${TEST_META_PATH:-OSWorld/evaluation_examples/test_nogdrive.json}"
  --domain "${DOMAIN:-all}"
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
