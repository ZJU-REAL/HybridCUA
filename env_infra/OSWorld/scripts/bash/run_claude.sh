#!/usr/bin/env bash
set -euo pipefail

_script_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "${_script_dir}/../../server.py" ]]; then
  cd "${_script_dir}/../.."
fi

mkdir -p logs

# Cluster master server (port 19000), NOT the node directly
export GUI_ENV_SERVER_URL="${GUI_ENV_SERVER_URL:-http://127.0.0.1:19000/}"

# API 中转配置
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
if [[ -z "$ANTHROPIC_BASE_URL" || -z "$ANTHROPIC_API_KEY" ]]; then echo "Set ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY before running" >&2; exit 2; fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-logs/run_claude_${TIMESTAMP}.log}"

echo "Logging to ${LOG_FILE}"

python scripts/python/run_multienv_claude.py \
  --provider_name "${PROVIDER_NAME:-docker_server}" \
  --headless \
  --action_space "${ACTION_SPACE:-claude_computer_use}" \
  --observation_type "${OBSERVATION_TYPE:-screenshot}" \
  --model "${MODEL:-claude-opus-4-6-20260205}" \
  --result_dir "${RESULT_DIR:-./results_claude}" \
  --test_all_meta_path "${TEST_META_PATH:-evaluation_examples/test_nogdrive.json}" \
  --max_steps "${MAX_STEPS:-15}" \
  --max_trajectory_length "${MAX_TRAJECTORY_LENGTH:-3}" \
  --max_tokens "${MAX_TOKENS:-2048}" \
  --num_envs "${NUM_ENVS:-5}" \
  --client_password "${CLIENT_PASSWORD:-${OSWORLD_CLIENT_PASSWORD:-}}" \
  --region "${REGION:-local}" \
  --sleep_after_execution "${SLEEP_AFTER_EXECUTION:-0}" \
  --screen_width "${SCREEN_WIDTH:-1920}" \
  --screen_height "${SCREEN_HEIGHT:-1080}" \
  --log_level "${LOG_LEVEL:-INFO}" \
  ${THINKING_MODE:+${THINKING_MODE}} \
  ${DOMAIN:+--domain "${DOMAIN}"} \
  ${SPECIFIC_TASK_ID:+--specific_task_id "${SPECIFIC_TASK_ID}"} \
  2>&1 | tee "${LOG_FILE}"
