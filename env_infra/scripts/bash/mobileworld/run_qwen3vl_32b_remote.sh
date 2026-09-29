#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${OSWORLD_JOB_ID:-}" ]]; then
  _script_dir="$(cd "$(dirname "$0")" && pwd)"
  cd "${_script_dir}/../../.."
fi

# ----------------------------------------------------------------------------
# Run MobileWorld evaluation with Qwen3-VL-32B-Thinking via cluster session API.
#
# This uses the unified cluster session protocol — environments are managed
# by the cluster (acquire/reset/step/evaluate/release), not directly connected.
# Points at a self-hosted (vLLM) OpenAI-compatible endpoint; no API key needed.
#
# Usage:
#   GUI_ENV_SERVER_URL=http://127.0.0.1:19000 \
#   LLM_BASE_URL=http://127.0.0.1:8000/v1 \
#   bash scripts/bash/mobileworld/run_qwen3vl_32b_remote.sh
# ----------------------------------------------------------------------------

export GUI_ENV_SERVER_URL="${GUI_ENV_SERVER_URL:-http://127.0.0.1:19000}"
MODEL_NAME="${QWEN3VL_32B_MODEL_NAME:-${MODEL_NAME:-qwen3-vl-32b-thinking}}"
LLM_BASE_URL="${QWEN3VL_32B_BASE_URL:-${LLM_BASE_URL:-}}"
if [[ -z "$LLM_BASE_URL" ]]; then echo "Set QWEN3VL_32B_BASE_URL or LLM_BASE_URL before running" >&2; exit 2; fi
API_KEY="${QWEN3VL_32B_API_KEY:-${API_KEY:-}}"
NUM_ENVS="${NUM_ENVS:-8}"
MAX_ROUND="${MAX_ROUND:-50}"
STEP_WAIT_TIME="${STEP_WAIT_TIME:-3}"
AGENT_TYPE="${AGENT_TYPE:-qwen3vl}"
TASK="${TASK:-ALL}"

# MAX_CONCURRENCY defaults to NUM_ENVS if not explicitly set
if [[ -z "${MAX_CONCURRENCY:-}" ]]; then
  MAX_CONCURRENCY="$NUM_ENVS"
fi

# Trajectory root: per-task traj files + result.txt; drives retry/scan logic.
TRAJ_DIR="${TRAJ_DIR:-traj_logs/qwen3vl_32b_remote}"
mkdir -p "$TRAJ_DIR"
TRAJ_DIR_ABS="$(cd "$TRAJ_DIR" && pwd)"

# Run log: this script's stdout/stderr -> repo-root logs/.
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR" 2>/dev/null || { LOG_DIR="/tmp/env_infra_logs"; mkdir -p "$LOG_DIR"; }
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/$(basename "$0" .sh)_${TIMESTAMP}.log}"

echo "Cluster: $GUI_ENV_SERVER_URL"
echo "Model: $MODEL_NAME"
echo "LLM URL: $LLM_BASE_URL"
echo "Envs: $NUM_ENVS"
echo "Run log: $LOG_FILE"
echo "Traj dir: $TRAJ_DIR_ABS"

# Background mode
if [[ "${BACKGROUND:-1}" == "1" && -z "${_MW_SESSION_DETACHED:-}" ]]; then
  export _MW_SESSION_DETACHED=1
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

# Foreground/job mode tees stdout to logs/; the backgrounded child (parent
# already redirected to LOG_FILE via nohup) just passes through with cat.
if [[ -n "${_MW_SESSION_DETACHED:-}" ]]; then
  TEE_CMD=(cat)
else
  TEE_CMD=(tee "${LOG_FILE}")
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
"${PYTHON_BIN}" scripts/python/mobileworld/run_mobileworld_sharded.py \
  --cluster_url "$GUI_ENV_SERVER_URL" \
  --agent_type "$AGENT_TYPE" \
  --model_name "$MODEL_NAME" \
  --llm_base_url "$LLM_BASE_URL" \
  --api_key "$API_KEY" \
  --num_envs "$NUM_ENVS" \
  --max_round "$MAX_ROUND" \
  --step_wait_time "$STEP_WAIT_TIME" \
  --task "$TASK" \
  --log_file_root "$TRAJ_DIR_ABS" \
  --max_concurrency "$MAX_CONCURRENCY" \
  "${EXTRA_ARGS[@]}" \
  "$@" 2>&1 | "${TEE_CMD[@]}"
