#!/usr/bin/env bash
set -euo pipefail

_script_dir="$(cd "$(dirname "$0")" && pwd)"
# Repo root is two levels up (cluster/scripts/ -> repo root). Anchor on
# pyproject.toml, which stays at the root after the OSWorld vendor-move.
if [[ -f "${_script_dir}/../../pyproject.toml" ]]; then
  cd "${_script_dir}/../.."
fi

# ----------------------------------------------------------------------------
# Start the OSWorld cluster master in the background.
#
# The master does not create Docker GUI environments itself. It accepts node
# registrations, schedules /v1/sessions requests, and forwards session
# operations to the node that owns each session.
#
# Override anything via env vars, e.g.:
#   MASTER_PORT=19000 MASTER_SCHEDULER=round-robin bash scripts/bash/start_cluster_master.sh
# ----------------------------------------------------------------------------

mkdir -p logs

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-logs/cluster_master_${TIMESTAMP}.log}"

MASTER_HOST="${MASTER_HOST:-0.0.0.0}"
MASTER_PORT="${MASTER_PORT:-19000}"
MASTER_SCHEDULER="${MASTER_SCHEDULER:-least-loaded}"
MASTER_NODE_SECRET="${MASTER_NODE_SECRET:-}"
MASTER_UNHEALTHY_TIMEOUT="${MASTER_UNHEALTHY_TIMEOUT:-30}"
MASTER_DEAD_TIMEOUT="${MASTER_DEAD_TIMEOUT:-60}"
MASTER_LOG_LEVEL="${MASTER_LOG_LEVEL:-INFO}"
# 默认走 install_env.sh 建的 venv, 而不是裸 `python`: 实测 `which python` =
# /opt/conda/envs/torch-base/bin/python (3.13), 那个环境【没有 flask】, 于是
# `python -m cluster.master.server` 在 master/server.py:21 的
# `from flask import ...` 直接崩 (ModuleNotFoundError: No module named 'flask')。
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
# 早失败: flask 缺了就报清楚原因, 而不是等 python 抛 ModuleNotFoundError。
if ! "${PYTHON_BIN}" -c 'import flask' 2>/dev/null; then
  echo "ERROR: ${PYTHON_BIN} 里没有 flask —— master 起不来。" >&2
  echo "       这个 venv 可能还没装完 (bash install_env.sh)。" >&2
  exit 1
fi
CLUSTER_DEBUG_EVENTS_WRITER_ID="${CLUSTER_DEBUG_EVENTS_WRITER_ID:-master-$(hostname)-${MASTER_PORT}}"
export CLUSTER_DEBUG_EVENTS_WRITER_ID

if [[ "${BACKGROUND:-1}" == "1" && -z "${_CLUSTER_MASTER_DETACHED:-}" ]]; then
  export _CLUSTER_MASTER_DETACHED=1
  nohup bash "$0" "$@" > "${LOG_FILE}" 2>&1 < /dev/null &
  CHILD_PID=$!
  echo "Started cluster master in background, PID=${CHILD_PID}"
  echo "Log:  ${LOG_FILE}"
  echo "Tail: tail -f ${LOG_FILE}"
  echo "Stop: kill ${CHILD_PID}"
  exit 0
fi

echo "Starting OSWorld cluster master..."
echo "MASTER_HOST=${MASTER_HOST}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "MASTER_SCHEDULER=${MASTER_SCHEDULER}"
echo "MASTER_UNHEALTHY_TIMEOUT=${MASTER_UNHEALTHY_TIMEOUT}"
echo "MASTER_DEAD_TIMEOUT=${MASTER_DEAD_TIMEOUT}"
echo "MASTER_LOG_LEVEL=${MASTER_LOG_LEVEL}"
echo "CLUSTER_DEBUG_EVENTS_WRITER_ID=${CLUSTER_DEBUG_EVENTS_WRITER_ID}"

"${PYTHON_BIN}" -m cluster.master.server \
  --host "${MASTER_HOST}" \
  --port "${MASTER_PORT}" \
  --scheduler "${MASTER_SCHEDULER}" \
  --node-secret "${MASTER_NODE_SECRET}" \
  --unhealthy-timeout "${MASTER_UNHEALTHY_TIMEOUT}" \
  --dead-timeout "${MASTER_DEAD_TIMEOUT}" \
  --log-level "${MASTER_LOG_LEVEL}"
