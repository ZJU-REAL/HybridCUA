#!/usr/bin/env bash
set -euo pipefail

# When run via cluster master, cwd is already repo root.
# When run manually, cd to repo root relative to script location.
if [[ -z "${OSWORLD_JOB_ID:-}" ]]; then
  _script_dir="$(cd "$(dirname "$0")" && pwd)"
  cd "${_script_dir}/../../.."
fi

# ----------------------------------------------------------------------------
# Run OSWorld evaluation with EvoCUA-32B (meituan/EvoCUA-32B-20260105) via
# mm_agents.evocua.EvoCUAAgent -- the PURE-GUI baseline.
#
# EvoCUA is trained for native computer-use: JSON tool-calls inside <tool_call>
# tags, 14 GUI actions (left_click / type / key / scroll / ...). It does NOT
# speak the c-gui XML `action=bash` + pyautogui-heredoc surface, so
# run_c_gui.sh cannot drive it (every step would parse to zero actions).
#
# Start vLLM first, then run:
#   bash scripts/bash/start_vllm_evocua_32b_8gpu.sh
#   bash scripts/bash/osworld/run_evocua.sh
#
# Why 4 shards and not 8: the 32B (63G on disk) is served as 4 servers x TP=2
# (start_vllm_evocua_32b_8gpu.sh default GPU_GROUPS='0,1;2,3;4,5;6,7'), which
# yields ports 8000..8003 only. Pointing at 8 would send half the workers to
# dead ports -- the failure mode this repo has already hit once.
#
# Override anything via env, e.g. a smoke test on one domain:
#   NUM_ENVS=2 DOMAIN=os MAX_STEPS=10 RESULT_DIR=./results-evocua-smoke \
#     bash scripts/bash/osworld/run_evocua.sh
# ----------------------------------------------------------------------------

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}" 2>/dev/null || LOG_DIR="/tmp/env_infra_logs" && mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run_evocua_${TIMESTAMP}.log}"
export PROXY_CONFIG_FILE="${PROXY_CONFIG_FILE:-$(pwd)/OSWorld/evaluation_examples/settings/proxy/star_proxy.json}"

export GUI_ENV_SERVER_URL="${GUI_ENV_SERVER_URL:-http://127.0.0.1:19000/}"
export OSWORLD_USER_ID="${OSWORLD_USER_ID:-${USER:-anonymous}}"
export OSWORLD_TASK_TYPE="${OSWORLD_TASK_TYPE:-evaluation}"

# Build default 4-endpoint shard list if OPENAI_BASE_URLS not set.
# 4 shards = 4 servers x TP=2 on ports 8000..8003 (see header).
if [[ -z "${OPENAI_BASE_URLS:-}" ]]; then
  OPENAI_HOST="${OPENAI_HOST:-127.0.0.1}"
  OPENAI_PORT_START="${OPENAI_PORT_START:-8000}"
  OPENAI_NUM_SHARDS="${OPENAI_NUM_SHARDS:-4}"
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
if [[ "${BACKGROUND:-1}" == "1" && -z "${_EVOCUA_DETACHED:-}" ]]; then
  export _EVOCUA_DETACHED=1
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
echo "MODEL=${MODEL:-EvoCUA}"

if [[ -n "${_EVOCUA_DETACHED:-}" ]]; then
  TEE_CMD=(cat)
else
  TEE_CMD=(tee "${LOG_FILE}")
fi

# 解释器: 必须用 env_infra 的 venv, 不能用裸 `python`。
# 裸 python 在本机解析到 /opt/conda/envs/torch-base/bin/python (3.12/3.13), 那里没有
# osworld 的依赖, 也没有 editable 安装 -> 所有 worker 会齐刷刷 ModuleNotFoundError。
# 集群 master/node 跑的就是下面这个 venv, client 要同源。可用 PYTHON_BIN=... 覆盖。
#
# 默认值由【repo 位置】推导 (../venvs/env_infra), 不写死某个盘的绝对路径: 这套代码
# 同时存在于多个盘, 各自带自己的 venv, 写死会指向错误的那个。脚本开头已 cd 到 repo
# 根, 所以 $(pwd) 就是 repo 根。
PYTHON_BIN="${PYTHON_BIN:-$(cd "$(pwd)/../venvs/env_infra/bin" 2>/dev/null && pwd)/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[!] PYTHON_BIN 不可执行: ${PYTHON_BIN}" >&2
  echo "    期望 <repo>/../venvs/env_infra/bin/python (install_env.sh 建的)" >&2
  echo "    或用 PYTHON_BIN=/path/to/venv/bin/python 显式指定" >&2
  exit 1
fi
# 早失败: EvoCUAAgent 顶层 import openai (evocua_agent.py:6), 缺了就报清楚原因。
if ! "${PYTHON_BIN}" -c 'import openai' 2>/dev/null; then
  echo "[!] ${PYTHON_BIN} 里没有 openai —— EvoCUAAgent 起不来。" >&2
  echo "    这个 venv 可能还没装完 (bash install_env.sh)。" >&2
  exit 1
fi
echo "PYTHON_BIN=${PYTHON_BIN}"

CMD=(
  "${PYTHON_BIN}" scripts/python/osworld/run_evocua_sharded.py
  --cluster_url "${GUI_ENV_SERVER_URL}"
  --action_space "${ACTION_SPACE:-pyautogui}"
  --observation_type "${OBSERVATION_TYPE:-screenshot}"
  # Must match vLLM --served-model-name, else every request 404s on model lookup.
  --model "${MODEL:-EvoCUA}"
  --coord "${COORD:-relative}"
  --max_tokens "${MAX_TOKENS:-8192}"
  --temperature "${TEMPERATURE:-0}"
  --top_p "${TOP_P:-0.9}"
  --history_n "${HISTORY_N:-4}"
  --prompt_style "${PROMPT_STYLE:-S2}"
  --max_steps "${MAX_STEPS:-50}"
  --test_config_base_dir "${TEST_CONFIG_BASE_DIR:-OSWorld/evaluation_examples}"
  --num_envs "${NUM_ENVS:-32}"
  --result_dir "${RESULT_DIR:-./results-evocua-32b-gui}"
  --test_all_meta_path "${TEST_META_PATH:-OSWorld/evaluation_examples/test_nogdrive.json}"
  --domain "${DOMAIN:-all}"
  --client_password "${CLIENT_PASSWORD:-password}"
  --password "${CLIENT_PASSWORD:-password}"
  --screen_width "${SCREEN_WIDTH:-1920}"
  --screen_height "${SCREEN_HEIGHT:-1080}"
  --sleep_after_execution "${SLEEP_AFTER_EXECUTION:-3}"
  --log_level "${LOG_LEVEL:-INFO}"
  --openai_base_urls "${OPENAI_BASE_URLS}"
  --api_key "${OPENAI_API_KEY}"
  "$@"
)

if [[ "${HEADLESS:-1}" == "1" ]]; then CMD+=(--headless); fi

"${CMD[@]}" 2>&1 | "${TEE_CMD[@]}"
