#!/usr/bin/env bash
set -euo pipefail

# When run via cluster master, cwd is already repo root.
# When run manually, cd to repo root relative to script location.
if [[ -z "${OSWORLD_JOB_ID:-}" ]]; then
  _script_dir="$(cd "$(dirname "$0")" && pwd)"
  cd "${_script_dir}/../../.."
fi

# ----------------------------------------------------------------------------
# Run OSWorld-MCP evaluation with the c-gui agent + MCP tools, sharded across
# 8 vLLM endpoints. Derived from run_qwen_qwen35_9b_sharded.sh; identical
# plumbing (endpoint sharding, proxy handling, background mode, venv preflight)
# plus the A/B variable:
#
#   MCP_MODE=off|bash|action
#     off     plain c-gui -- prompt/schema BYTE-IDENTICAL to run_c_gui_sharded.sh
#     bash    tools via the EXISTING action=bash channel (no action-space change)
#     action  adds action=mcp with name/params
#   MCP_TOOL_BUDGET=N        cap app tools per episode (0 = no cap)
#   MCP_NO_DISTRACTORS=1     drop the 28 filesystem_*/git_* bait tools
#                            (WARNING: makes TIR incomparable to the paper)
#
# Tasks default to OSWorld-MCP/evaluation_examples/test_all.json -- the same 361
# IDs as upstream test_nogdrive.json, verified identical.
#
# Start vLLM first (8 shards, one per GPU):
#   MODEL_PATH=$(pwd)/../models/Qwen3.5-9B SERVED_MODEL_NAME=Qwen3.5-9B \
#     bash scripts/bash/start_vllm_qwen35_9b_8gpu.sh
#
# Then the three arms:
#   for M in off bash action; do
#     MCP_MODE=$M bash scripts/bash/osworld/run_c_gui_mcp_sharded.sh
#   done
#   python scripts/python/osworld/rollup_mcp_usage.py results-c-gui-mcp-* --paired
#
# Override anything via env, e.g. one domain to validate the wiring first:
#   MCP_MODE=off DOMAIN=libreoffice_calc NUM_ENVS=2 MAX_STEPS=10 \
#     bash scripts/bash/osworld/run_c_gui_mcp_sharded.sh
# ----------------------------------------------------------------------------

MCP_MODE="${MCP_MODE:-bash}"
case "${MCP_MODE}" in
  off|bash|action) ;;
  *) echo "ERROR: MCP_MODE must be off|bash|action, got '${MCP_MODE}'" >&2; exit 1 ;;
esac

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}" 2>/dev/null || LOG_DIR="/tmp/env_infra_logs" && mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run_c_gui_mcp_${MCP_MODE}_${TIMESTAMP}.log}"
# image   = the qcow2 already serves the stack (OSWorld-MCP/bake_mcp_image.sh);
#           an episode only probes two ports (~2s vs ~5min) and cannot disturb the
#           task's own document. Requires PATH_TO_VM pointed at the baked image.
# runcode = install per episode (works anywhere, but bring-up races task setup).
# auto    = probe, fall back to runcode. Default.
export MCP_BRINGUP="${MCP_BRINGUP:-auto}"

export PROXY_CONFIG_FILE="${PROXY_CONFIG_FILE:-$(pwd)/OSWorld/evaluation_examples/settings/proxy/star_proxy.json}"

export GUI_ENV_SERVER_URL="${GUI_ENV_SERVER_URL:-http://127.0.0.1:19000/}"
export OSWORLD_USER_ID="${OSWORLD_USER_ID:-${USER:-anonymous}}"
export OSWORLD_TASK_TYPE="${OSWORLD_TASK_TYPE:-evaluation}"

# Build default 8-endpoint shard list if OPENAI_BASE_URLS not set.
# 8 shards = one vLLM server per GPU (TP=1), matching start_vllm_qwen35_9b_8gpu.sh.
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

# Host-side proxy for task asset downloads. (The GUEST reaches the network through
# its own proxy config; the MCP bring-up handles npm/uv separately.)
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
# cluster master 同样要绕过代理。上面的网段只覆盖内网, 而 GUI_ENV_SERVER_URL
# 可以是公网 IP —— 那时 session acquire 会被 star-proxy 拦成 403, 所有 worker
# 全部在 build_env() 阶段挂掉。
_cluster_host="${GUI_ENV_SERVER_URL#*://}"; _cluster_host="${_cluster_host%%[:/]*}"
[[ -n "${_cluster_host}" ]] && _no_proxy_add="${_no_proxy_add},${_cluster_host}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${_no_proxy_add}"
export no_proxy="${no_proxy:+${no_proxy},}${_no_proxy_add}"

# Background mode.
if [[ "${BACKGROUND:-1}" == "1" && -z "${_C_GUI_MCP_DETACHED:-}" ]]; then
  export _C_GUI_MCP_DETACHED=1
  nohup bash "$0" "$@" > "${LOG_FILE}" 2>&1 < /dev/null &
  CHILD_PID=$!
  echo "Started in background, PID=${CHILD_PID}"
  echo "Mode: MCP_MODE=${MCP_MODE}"
  echo "Log:  ${LOG_FILE}"
  echo "Tail: tail -f ${LOG_FILE}"
  echo "Stop: kill ${CHILD_PID}"
  exit 0
fi

echo "Logging to ${LOG_FILE}"
echo "MCP_MODE=${MCP_MODE}  BRINGUP=${MCP_BRINGUP}  TOOL_BUDGET=${MCP_TOOL_BUDGET:-0}  NO_DISTRACTORS=${MCP_NO_DISTRACTORS:-0}"
echo "OPENAI_BASE_URLS=${OPENAI_BASE_URLS}"
echo "GUI_ENV_SERVER_URL=${GUI_ENV_SERVER_URL}"

if [[ -n "${_C_GUI_MCP_DETACHED:-}" ]]; then
  TEE_CMD=(cat)
else
  TEE_CMD=(tee "${LOG_FILE}")
fi

# 🔴 必须用 env_infra 的 venv, 不能用默认的 `python`。
#    默认 `which python` = /opt/conda/envs/torch-base/bin/python, 那个环境【没有 openai】
#    也没有 editable 安装的 osworld —— 用它会在 import 时直接崩, 而且是在容器都申请完
#    之后才崩, 白烧一轮。默认值由 repo 位置推导 (../venvs/env_infra), 不写死盘路径:
#    这套代码同时存在于两个盘, 各自带自己的 venv。
PYTHON_BIN="${PYTHON_BIN:-$(cd .. && pwd)/venvs/env_infra/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: PYTHON_BIN 不可执行: ${PYTHON_BIN}" >&2
  echo "       期望 <repo>/../venvs/env_infra/bin/python (install_env.sh 建的)" >&2
  exit 1
fi
if ! "${PYTHON_BIN}" -c 'import openai' 2>/dev/null; then
  echo "ERROR: ${PYTHON_BIN} 里没有 openai —— CGuiMcpAgent 起不来。" >&2
  echo "       这个 venv 可能还没装完 (bash install_env.sh)。" >&2
  exit 1
fi
echo "PYTHON_BIN=${PYTHON_BIN}"

CMD=(
  "${PYTHON_BIN}" scripts/python/osworld/run_c_gui_mcp_sharded.py
  --cluster_url "${GUI_ENV_SERVER_URL}"
  --action_space "${ACTION_SPACE:-pyautogui}"
  --observation_type "${OBSERVATION_TYPE:-screenshot}"
  # Must match vLLM --served-model-name, else every request 404s on model lookup.
  --model "${MODEL:-Qwen3.5-9B}"
  --coord "${COORD:-relative}"
  --max_tokens "${MAX_TOKENS:-4096}"
  --temperature "${TEMPERATURE:-0}"
  --top_p "${TOP_P:-0.9}"
  --history_n "${HISTORY_N:-50}"
  --image_max "${IMAGE_MAX:-5}"
  --fold_size "${FOLD_SIZE:-1}"
  --tool_name "${TOOL_NAME:-computer_use}"
  --max_steps "${MAX_STEPS:-50}"
  --test_config_base_dir "${TEST_CONFIG_BASE_DIR:-OSWorld/evaluation_examples}"
  --num_envs "${NUM_ENVS:-64}"
  --result_dir "${RESULT_DIR:-./results-c-gui-mcp-${MCP_MODE}}"
  --domain "${DOMAIN:-all}"
  --client_password "${CLIENT_PASSWORD:-password}"
  --screen_width "${SCREEN_WIDTH:-1920}"
  --screen_height "${SCREEN_HEIGHT:-1080}"
  --sleep_after_execution "${SLEEP_AFTER_EXECUTION:-3}"
  --log_level "${LOG_LEVEL:-INFO}"
  --openai_base_urls "${OPENAI_BASE_URLS}"
  --api_key "${OPENAI_API_KEY}"
  --mcp_mode "${MCP_MODE}"
  --mcp_tool_budget "${MCP_TOOL_BUDGET:-0}"
)
# Only pass --test_all_meta_path when overridden; the runner defaults to
# OSWorld-MCP's 361-task list.
if [[ -n "${TEST_META_PATH:-}" ]]; then
  CMD+=(--test_all_meta_path "${TEST_META_PATH}")
fi
CMD+=("$@")

if [[ "${HEADLESS:-1}" == "1" ]]; then CMD+=(--headless); fi
if [[ "${ADD_THOUGHT_PREFIX:-0}" == "1" ]]; then CMD+=(--add_thought_prefix); fi
if [[ "${ENABLE_THINKING:-0}" == "1" ]]; then CMD+=(--enable_thinking); fi
if [[ "${MCP_NO_DISTRACTORS:-0}" == "1" ]]; then CMD+=(--mcp_no_distractors); fi

"${CMD[@]}" 2>&1 | "${TEE_CMD[@]}"
