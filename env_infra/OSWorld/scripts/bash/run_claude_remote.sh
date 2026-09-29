#!/usr/bin/env bash
set -euo pipefail

_script_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "${_script_dir}/../../server.py" ]]; then
  cd "${_script_dir}/../.."
fi

# ----------------------------------------------------------------------------
# Run OSWorld evaluation with Claude through OSWorldRemoteClient and Cluster.
#
# Start the cluster first:
#   bash scripts/bash/start_cluster_master.sh
#   NODE_MASTER_URL=http://master-ip:19000 bash scripts/bash/start_cluster_node.sh
#
# Run evaluation:
#   GUI_ENV_SERVER_URL=http://127.0.0.1:19000/ \
#   ANTHROPIC_BASE_URL=http://anthropic-proxy:8010 \
#   ANTHROPIC_API_KEY=... \
#   NUM_ENVS=8 \
#   bash scripts/bash/run_claude_remote.sh
# ----------------------------------------------------------------------------

mkdir -p logs

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-logs/run_claude_remote_${TIMESTAMP}.log}"

export GUI_ENV_SERVER_URL="${GUI_ENV_SERVER_URL:-http://127.0.0.1:19000/}"
export OSWORLD_USER_ID="${OSWORLD_USER_ID:-${USER:-anonymous}}"
export OSWORLD_TASK_TYPE="${OSWORLD_TASK_TYPE:-evaluation}"

export API_PROVIDER="${API_PROVIDER:-${ANTHROPIC_API_PROVIDER:-anthropic}}"
export ANTHROPIC_API_PROVIDER="${API_PROVIDER}"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
if [[ -z "$ANTHROPIC_BASE_URL" ]]; then echo "Set ANTHROPIC_BASE_URL before running" >&2; exit 2; fi

if [[ "${API_PROVIDER}" == "anthropic" && -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ANTHROPIC_API_KEY must be set when API_PROVIDER=anthropic" >&2
  exit 2
fi

_gui_host="${GUI_ENV_SERVER_URL#*://}"
_gui_host="${_gui_host%%[:/]*}"
_no_proxy_add="localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,${_gui_host}"
if [[ -n "${ANTHROPIC_BASE_URL:-}" ]]; then
  _claude_host="${ANTHROPIC_BASE_URL#*://}"
  _claude_host="${_claude_host%%[:/]*}"
  _no_proxy_add="${_no_proxy_add},${_claude_host}"
fi
export no_proxy="${no_proxy:+${no_proxy},}${_no_proxy_add}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${_no_proxy_add}"

if [[ "${BACKGROUND:-1}" == "1" && -z "${_CLAUDE_REMOTE_DETACHED:-}" ]]; then
  export _CLAUDE_REMOTE_DETACHED=1
  nohup bash "$0" "$@" > "${LOG_FILE}" 2>&1 < /dev/null &
  CHILD_PID=$!
  echo "Started in background, PID=${CHILD_PID}"
  echo "Log:  ${LOG_FILE}"
  echo "Tail: tail -f ${LOG_FILE}"
  echo "Stop: kill ${CHILD_PID}"
  exit 0
fi

echo "Logging to ${LOG_FILE}"
echo "GUI_ENV_SERVER_URL=${GUI_ENV_SERVER_URL}"
echo "ANTHROPIC_BASE_URL=${ANTHROPIC_BASE_URL}"
echo "API_PROVIDER=${API_PROVIDER}"
echo "OSWORLD_USER_ID=${OSWORLD_USER_ID}"
echo "OSWORLD_TASK_TYPE=${OSWORLD_TASK_TYPE}"

if [[ -n "${_CLAUDE_REMOTE_DETACHED:-}" ]]; then
  TEE_CMD=(cat)
else
  TEE_CMD=(tee "${LOG_FILE}")
fi

HEADLESS_ARGS=()
if [[ "${HEADLESS:-1}" == "1" ]]; then
  HEADLESS_ARGS=(--headless)
fi

PATH_TO_VM_ARGS=()
if [[ -n "${PATH_TO_VM:-}" ]]; then
  PATH_TO_VM_ARGS=(--path_to_vm "${PATH_TO_VM}")
fi

SPECIFIC_TASK_ARGS=()
if [[ -n "${SPECIFIC_TASK_ID:-}" ]]; then
  SPECIFIC_TASK_ARGS=(--specific_task_id "${SPECIFIC_TASK_ID}")
fi

TEMPERATURE_ARGS=()
if [[ -n "${TEMPERATURE:-}" ]]; then
  TEMPERATURE_ARGS=(--temperature "${TEMPERATURE}")
fi

TOP_P_ARGS=()
if [[ -n "${TOP_P:-}" ]]; then
  TOP_P_ARGS=(--top_p "${TOP_P}")
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CMD=(
  "${PYTHON_BIN}" scripts/python/run_multienv_claude_remote.py
  --cluster_url "${GUI_ENV_SERVER_URL}"
  --provider_name "${PROVIDER_NAME:-docker_server}"
  "${HEADLESS_ARGS[@]}"
  "${PATH_TO_VM_ARGS[@]}"
  --action_space "${ACTION_SPACE:-claude_computer_use}"
  --observation_type "${OBSERVATION_TYPE:-screenshot}"
  --model "${MODEL:-claude-opus-4-6-20260205}"
  --api_provider "${API_PROVIDER}"
  --anthropic_base_url "${ANTHROPIC_BASE_URL}"
  --max_tokens "${MAX_TOKENS:-2048}"
  --effort "${EFFORT:-max}"
  "${TEMPERATURE_ARGS[@]}"
  "${TOP_P_ARGS[@]}"
  --max_steps "${MAX_STEPS:-15}"
  --max_trajectory_length "${MAX_TRAJ_LEN:-3}"
  --test_config_base_dir "${TEST_CONFIG_BASE_DIR:-evaluation_examples}"
  --num_envs "${NUM_ENVS:-5}"
  --result_dir "${RESULT_DIR:-./results_claude_remote}"
  --test_all_meta_path "${TEST_META_PATH:-evaluation_examples/test_nogdrive.json}"
  --domain "${DOMAIN:-all}"
  "${SPECIFIC_TASK_ARGS[@]}"
  --client_password "${CLIENT_PASSWORD:-${OSWORLD_CLIENT_PASSWORD:-password}}"
  --region "${REGION:-local}"
  --screen_width "${SCREEN_WIDTH:-1920}"
  --screen_height "${SCREEN_HEIGHT:-1080}"
  --sleep_after_execution "${SLEEP_AFTER_EXECUTION:-0}"
  --log_level "${LOG_LEVEL:-INFO}"
  "$@"
)

"${CMD[@]}" 2>&1 | "${TEE_CMD[@]}"
