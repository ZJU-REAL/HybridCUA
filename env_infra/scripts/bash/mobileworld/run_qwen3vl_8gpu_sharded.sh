#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${OSWORLD_JOB_ID:-}" ]]; then
  _script_dir="$(cd "$(dirname "$0")" && pwd)"
  cd "${_script_dir}/../../.."
fi

# ----------------------------------------------------------------------------
# Run MobileWorld evaluation with Qwen3-VL-8B-Instruct, sharded across multiple
# vLLM endpoints (sync, process-per-env via EvalRunner).
#
# Each worker process pins one endpoint round-robin (worker_idx % n_urls).
# Total concurrent sessions = --num_envs (= cluster slots demanded).
#
# Usage:
#   GUI_ENV_SERVER_URL=http://127.0.0.1:19000 \
#   OPENAI_BASE_URLS=http://h:8000/v1,...,http://h:8007/v1 \
#   bash scripts/bash/mobileworld/run_qwen3vl_8gpu_sharded.sh
# ----------------------------------------------------------------------------

export GUI_ENV_SERVER_URL="${GUI_ENV_SERVER_URL:-http://127.0.0.1:19000}"
MODEL_NAME="${QWEN3VL_8B_MODEL_NAME:-${MODEL_NAME:-Qwen3-VL-8B-Instruct}}"
AGENT_TYPE="${AGENT_TYPE:-qwen3vl}"
API_KEY="${QWEN3VL_8B_API_KEY:-${API_KEY:-sk-local}}"
NUM_ENVS="${NUM_ENVS:-8}"
MAX_ROUND="${MAX_ROUND:-50}"
STEP_WAIT_TIME="${STEP_WAIT_TIME:-3}"
TASK="${TASK:-ALL}"

# Build the endpoint list: explicit OPENAI_BASE_URLS, else synthesize 8 shards
# on OPENAI_HOST:OPENAI_PORT_START..(START+NUM_SHARDS-1).
OPENAI_HOST="${OPENAI_HOST:-127.0.0.1}"
OPENAI_PORT_START="${OPENAI_PORT_START:-8000}"
OPENAI_NUM_SHARDS="${OPENAI_NUM_SHARDS:-8}"
if [[ -z "${OPENAI_BASE_URLS:-}" ]]; then
  OPENAI_BASE_URLS=""
  for i in $(seq 0 $((OPENAI_NUM_SHARDS - 1))); do
    port=$((OPENAI_PORT_START + i))
    if [[ -z "$OPENAI_BASE_URLS" ]]; then
      OPENAI_BASE_URLS="http://${OPENAI_HOST}:${port}/v1"
    else
      OPENAI_BASE_URLS="${OPENAI_BASE_URLS},http://${OPENAI_HOST}:${port}/v1"
    fi
  done
fi
export OPENAI_BASE_URLS

# no_proxy: cluster host + all vLLM hosts must bypass any corporate proxy.
NO_PROXY_EXTRA="${GUI_ENV_SERVER_URL#http://}"
NO_PROXY_EXTRA="${NO_PROXY_EXTRA%%:*}"
for u in ${OPENAI_BASE_URLS//,/ }; do
  h="${u#http://}"; h="${h%%/*}"; NO_PROXY_EXTRA="${NO_PROXY_EXTRA},${h}"
done
export NO_PROXY="${NO_PROXY:-localhost},${NO_PROXY_EXTRA}"
export no_proxy="$NO_PROXY"

# Trajectory root: per-task traj files + result.txt; drives retry/scan logic.
TRAJ_DIR="${TRAJ_DIR:-traj_logs/mobileworld_sharded}"
mkdir -p "$TRAJ_DIR"
TRAJ_DIR_ABS="$(cd "$TRAJ_DIR" && pwd)"

# Run log: this script's stdout/stderr -> repo-root logs/.
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR" 2>/dev/null || { LOG_DIR="/tmp/env_infra_logs"; mkdir -p "$LOG_DIR"; }
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/$(basename "$0" .sh)_${TIMESTAMP}.log}"

echo "Cluster: $GUI_ENV_SERVER_URL"
echo "Model: $MODEL_NAME"
echo "Endpoints: $OPENAI_BASE_URLS"
echo "Envs: $NUM_ENVS"
echo "Run log: $LOG_FILE"
echo "Traj dir: $TRAJ_DIR_ABS"

# Background mode
if [[ "${BACKGROUND:-1}" == "1" && -z "${_MW_SHARDED_DETACHED:-}" ]]; then
  export _MW_SHARDED_DETACHED=1
  nohup bash "$0" "$@" > "$LOG_FILE" 2>&1 < /dev/null &
  PID=$!
  echo "Started in background, PID=${PID}"
  echo "Log: $LOG_FILE"
  echo "Tail: tail -f $LOG_FILE"
  echo "Stop: kill $PID"
  exit 0
fi

EXTRA_ARGS=()
if [[ "${ENABLE_MCP:-0}" == "1" ]]; then EXTRA_ARGS+=(--enable_mcp); fi
if [[ "${ENABLE_USER_INTERACTION:-0}" == "1" ]]; then EXTRA_ARGS+=(--enable_user_interaction); fi
if [[ "${SHUFFLE:-0}" == "1" ]]; then EXTRA_ARGS+=(--shuffle_tasks); fi

if [[ -n "${_MW_SHARDED_DETACHED:-}" ]]; then
  TEE_CMD=(cat)
else
  TEE_CMD=(tee "${LOG_FILE}")
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
"${PYTHON_BIN}" scripts/python/mobileworld/run_mobileworld_sharded.py \
  --cluster_url "$GUI_ENV_SERVER_URL" \
  --agent_type "$AGENT_TYPE" \
  --model_name "$MODEL_NAME" \
  --openai_base_urls "$OPENAI_BASE_URLS" \
  --api_key "$API_KEY" \
  --num_envs "$NUM_ENVS" \
  --max_round "$MAX_ROUND" \
  --step_wait_time "$STEP_WAIT_TIME" \
  --task "$TASK" \
  --log_file_root "$TRAJ_DIR_ABS" \
  "${EXTRA_ARGS[@]}" \
  "$@" 2>&1 | "${TEE_CMD[@]}"
