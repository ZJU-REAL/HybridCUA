#!/usr/bin/env bash
set -euo pipefail

_script_dir="$(cd "$(dirname "$0")" && pwd)"
# Repo root is two levels up (cluster/scripts/ -> repo root). Anchor on
# pyproject.toml, which stays at the root after the OSWorld vendor-move.
if [[ -f "${_script_dir}/../../pyproject.toml" ]]; then
  cd "${_script_dir}/../.."
fi

# ----------------------------------------------------------------------------
# Start one cluster node in the background.
#
# World-neutral node: serves /v1/sessions backed by a RuntimePool per world.
# NODE_WORLDS empty (default) = host every world under cluster/worlds/.
#
# Examples:
#   # host ALL worlds (default):
#   NODE_MASTER_URL=http://127.0.0.1:19000 \
#     bash cluster/scripts/start_cluster_node.sh
#
#   # restrict to OSWorld only:
#   NODE_MASTER_URL=http://127.0.0.1:19000 NODE_WORLDS=osworld \
#     bash cluster/scripts/start_cluster_node.sh
#
#   # prewarm 8 slots at startup with 8 concurrent workers:
#   NODE_PREWARM=8 NODE_PREWARM_CONCURRENCY=8 \
#     bash cluster/scripts/start_cluster_node.sh
# ----------------------------------------------------------------------------


mkdir -p logs

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-logs/cluster_node_${TIMESTAMP}.log}"

# 空 = 用 cluster/worlds/osworld/world.yaml 的 config_schema 默认值。
# 不要在这里给默认值: 这个变量只通过下面的 NODE_WORLD_CONFIG 合并生效。
PATH_TO_VM="${PATH_TO_VM:-}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18080}"
_pick_public_host() {
  ip route get 1.1.1.1 2>/dev/null | sed -nE 's/.* src ([0-9.]+).*/\1/p' | head -1
}
PUBLIC_HOST="${PUBLIC_HOST:-$(_pick_public_host || true)}"
if [[ -z "${PUBLIC_HOST}" ]]; then
  echo "ERROR: 无法自动确定本机对外 IP。" >&2
  echo "       ip route get 1.1.1.1 => $(ip route get 1.1.1.1 2>&1)" >&2
  echo "       请显式指定: PUBLIC_HOST=<本机可被 master 回连的 IP> bash $0" >&2
  exit 1
fi
NODE_MASTER_URL="${NODE_MASTER_URL:-${MASTER_URL:-http://127.0.0.1:19000}}"
# Normalize: collapse an accidental double scheme (http://http://...) and drop a
# trailing slash, so a mistyped NODE_MASTER_URL still resolves.
NODE_MASTER_URL="$(printf '%s' "${NODE_MASTER_URL}" | sed -E 's#^(https?://)+#\1#; s#/+$##')"
NODE_URL="${NODE_URL:-http://${PUBLIC_HOST}:${PORT}}"
NODE_URL="$(printf '%s' "${NODE_URL}" | sed -E 's#^(https?://)+#\1#; s#/+$##')"
NODE_ID="${NODE_ID:-$(hostname)-${PORT}}"
NODE_SECRET="${NODE_SECRET:-${MASTER_NODE_SECRET:-}}"
NODE_HEARTBEAT_INTERVAL="${NODE_HEARTBEAT_INTERVAL:-10}"
NODE_LABELS="${NODE_LABELS:-}"
# world-neutral node: which worlds to host + per-world slot count / config.
# Empty (default) = host EVERY world under cluster/worlds/ (osworld+mobileworld+...).
# Set to a comma list (e.g. NODE_WORLDS=osworld) to restrict this node.
NODE_WORLDS="${NODE_WORLDS:-}"
NODE_MAX_SLOTS="${NODE_MAX_SLOTS:-${MAX_ENVS:-32}}"
NODE_IDLE_TTL="${NODE_IDLE_TTL:-0}"
NODE_PREWARM="${NODE_PREWARM:-0}"
NODE_PREWARM_CONCURRENCY="${NODE_PREWARM_CONCURRENCY:-8}"
NODE_SCALE_BUFFER="${NODE_SCALE_BUFFER:-0}"
NODE_SCALE_INTERVAL="${NODE_SCALE_INTERVAL:-5}"
NODE_WORLD_CONFIG="${NODE_WORLD_CONFIG:-}"
GUI_PROVIDER_NAME="${GUI_PROVIDER_NAME:-docker_fast}"
CLUSTER_DEBUG_EVENTS_WRITER_ID="${CLUSTER_DEBUG_EVENTS_WRITER_ID:-${NODE_ID}}"

MAX_ENVS="${MAX_ENVS:-24}"
PREWARM_ENVS="${PREWARM_ENVS:-0}"
PREWARM_CONCURRENCY="${PREWARM_CONCURRENCY:-8}"
ACTION_SPACE="${ACTION_SPACE:-pyautogui}"
OBSERVATION_TYPE="${OBSERVATION_TYPE:-screenshot}"
SCREEN_WIDTH="${SCREEN_WIDTH:-1920}"
SCREEN_HEIGHT="${SCREEN_HEIGHT:-1080}"
CLIENT_PASSWORD="${CLIENT_PASSWORD:-${OSWORLD_CLIENT_PASSWORD:-password}}"
RESET_ON_CLOSE="${RESET_ON_CLOSE:-1}"
IDLE_TTL_SECONDS="${IDLE_TTL_SECONDS:-600}"
OSWORLD_DOCKER_PS_TIMEOUT="${OSWORLD_DOCKER_PS_TIMEOUT:-15}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
CACHE_DIR="${CACHE_DIR:-OSWorld/cache}"
PROXY_CONFIG_FILE="${PROXY_CONFIG_FILE:-$(pwd)/OSWorld/evaluation_examples/settings/proxy/star_proxy.json}"
ENABLE_PROXY="${ENABLE_PROXY:-${GUI_ENABLE_PROXY:-1}}"
HOST_PROXY_URL="${HOST_PROXY_URL:-http://star-proxy.oa.com:3128}"
# 默认走 install_env.sh 建的 venv, 而不是裸 `python`: 实测 `which python` =
# /opt/conda/envs/torch-base/bin/python (3.13), 那个环境【没有 flask】, 于是
# `python -m cluster.node.world_server` 在 import flask 时直接崩
# (ModuleNotFoundError: No module named 'flask')。
#
# 默认值由【repo 位置】推导 (../venvs/env_infra), 不写死某个盘的绝对路径:
# 这套代码同时存在于多个盘, 各自带自己的 venv, 写死会指向错误的那个。
# 与 scripts/bash/osworld/run_qwen_qwen35_9b_sharded.sh:109 同一套写法。
# 脚本开头已 cd 到 repo 根, 所以 $(pwd) 就是 repo 根。
PYTHON_BIN="${PYTHON_BIN:-$(cd "$(pwd)/../venvs/env_infra/bin" 2>/dev/null && pwd)/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: PYTHON_BIN 不可执行: ${PYTHON_BIN}" >&2
  echo "       期望 <repo>/../venvs/env_infra/bin/python (install_env.sh 建的)" >&2
  echo "       或用 PYTHON_BIN=/path/to/venv/bin/python 显式指定" >&2
  exit 1
fi
# 早失败: node 起容器要 flask(HTTP 层) + docker(容器生命周期), 缺了就报清楚原因。
for _m in flask docker; do
  if ! "${PYTHON_BIN}" -c "import ${_m}" 2>/dev/null; then
    echo "ERROR: ${PYTHON_BIN} 里没有 ${_m} —— node 起不来。" >&2
    echo "       这个 venv 可能还没装完 (bash install_env.sh)。" >&2
    exit 1
  fi
done

# PATH_TO_VM 合并进 world config (cluster/node/server.py:52 的 merged.update)。
# 用 python 而不是拼字符串: 路径可能含需要 JSON 转义的字符; setdefault 保证显式
# NODE_WORLD_CONFIG 优先。放在 PYTHON_BIN 校验之后。
if [[ -n "${PATH_TO_VM}" ]]; then
  NODE_WORLD_CONFIG="$("${PYTHON_BIN}" -c '
import json, sys
cfg = json.loads(sys.argv[1]) if sys.argv[1].strip() else {}
cfg.setdefault("osworld", {}).setdefault("path_to_vm", sys.argv[2])
print(json.dumps(cfg))' "${NODE_WORLD_CONFIG}" "${PATH_TO_VM}")" || {
    echo "ERROR: 无法把 PATH_TO_VM 合并进 NODE_WORLD_CONFIG。" >&2
    echo "       NODE_WORLD_CONFIG 必须是合法 JSON, 实际: ${NODE_WORLD_CONFIG}" >&2
    exit 1
  }
fi

_master_host="${NODE_MASTER_URL#*://}"
_master_host="${_master_host%%[:/]*}"
_node_host="${NODE_URL#*://}"
_node_host="${_node_host%%[:/]*}"
NO_PROXY_DEFAULT="127.0.0.1,127.0.0.0/8,localhost,::1,host.docker.internal,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,${_master_host},${_node_host},${PUBLIC_HOST}"
NO_PROXY_COMBINED="${NO_PROXY:-${no_proxy:-}}"
if [[ -n "${NO_PROXY_COMBINED}" ]]; then
  NO_PROXY_COMBINED="${NO_PROXY_COMBINED},${NO_PROXY_DEFAULT}"
else
  NO_PROXY_COMBINED="${NO_PROXY_DEFAULT}"
fi

export NODE_MASTER_URL
export NODE_URL
export NODE_ID
export NODE_SECRET
export NODE_HEARTBEAT_INTERVAL
export NODE_LABELS
export GUI_PROVIDER_NAME
export CLUSTER_DEBUG_EVENTS_WRITER_ID
export PROXY_CONFIG_FILE
export HOST_PROXY_URL
export http_proxy="${HOST_PROXY_URL}"
export https_proxy="${HOST_PROXY_URL}"
export HTTP_PROXY="${HOST_PROXY_URL}"
export HTTPS_PROXY="${HOST_PROXY_URL}"
export all_proxy="${HOST_PROXY_URL}"
export ALL_PROXY="${HOST_PROXY_URL}"
export NO_PROXY="${NO_PROXY_COMBINED}"
export no_proxy="${NO_PROXY_COMBINED}"
export OSWORLD_DOCKER_PS_TIMEOUT

if [[ "${BACKGROUND:-1}" == "1" && -z "${_CLUSTER_NODE_DETACHED:-}" ]]; then
  export _CLUSTER_NODE_DETACHED=1
  # 子进程重跑整个脚本, 这两个必须先 export 才传得过去 (export 在下面, 分叉之后)。
  export NODE_WORLD_CONFIG PATH_TO_VM
  nohup bash "$0" "$@" > "${LOG_FILE}" 2>&1 < /dev/null &
  CHILD_PID=$!
  echo "Started cluster node in background, PID=${CHILD_PID}"
  echo "Log:  ${LOG_FILE}"
  echo "Tail: tail -f ${LOG_FILE}"
  echo "Stop: kill ${CHILD_PID}"
  exit 0
fi

export NODE_WORLDS NODE_MAX_SLOTS NODE_IDLE_TTL NODE_PREWARM NODE_PREWARM_CONCURRENCY NODE_SCALE_BUFFER NODE_SCALE_INTERVAL NODE_WORLD_CONFIG NODE_PORT="${PORT}" NODE_LOG_LEVEL="${LOG_LEVEL}"
echo "Starting cluster node..."
echo "NODE_ID=${NODE_ID}"
echo "NODE_MASTER_URL=${NODE_MASTER_URL}"
echo "NODE_URL=${NODE_URL}"
echo "NODE_WORLDS=${NODE_WORLDS:-(all)}"
echo "NODE_MAX_SLOTS=${NODE_MAX_SLOTS}"
echo "NODE_WORLD_CONFIG=${NODE_WORLD_CONFIG:-(world.yaml defaults)}"
echo "HOST=${HOST}  PORT=${PORT}"

exec "${PYTHON_BIN}" -m cluster.node.world_server \
  --host "${HOST}" \
  --port "${PORT}" \
  --node-worlds "${NODE_WORLDS}" \
  --max-slots-per-world "${NODE_MAX_SLOTS}" \
  --idle-ttl-seconds "${NODE_IDLE_TTL}" \
  --prewarm "${NODE_PREWARM}" \
  --prewarm-concurrency "${NODE_PREWARM_CONCURRENCY}" \
  --scale-buffer "${NODE_SCALE_BUFFER}" \
  --scale-interval "${NODE_SCALE_INTERVAL}" \
  --world-config "${NODE_WORLD_CONFIG}" \
  --master-url "${NODE_MASTER_URL}" \
  --node-url "${NODE_URL}" \
  --node-id "${NODE_ID}" \
  --node-secret "${NODE_SECRET}" \
  --heartbeat-interval "${NODE_HEARTBEAT_INTERVAL}" \
  --node-labels "${NODE_LABELS}" \
  --log-level "${LOG_LEVEL}"
