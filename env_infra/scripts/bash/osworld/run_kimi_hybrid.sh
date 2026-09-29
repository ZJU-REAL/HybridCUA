#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------------
# Run OSWorld evaluation with the Kimi single-bash-surface (GUI+CLI) agent via the
# platform cluster.
#
# Agent = mm_agents.kimi_hybrid.KimiHybridAgent — Kimi driving ONE bash surface: a GUI
# interaction is a pyautogui heredoc, a CLI/file operation is plain shell, both executed
# through env.run_code(lang="bash"). Coordinates are REAL SCREEN PIXELS (the screenshot
# is sent at full resolution and nothing rescales it), so SCREEN_WIDTH/SCREEN_HEIGHT must
# match the VM's actual screen.
#
# Episode loop = the shared run_single_example_kimi_hybrid, which provisions the VM
# pyautogui shim after reset and threads each command's stdout/stderr into the next
# predict so the model reads real terminal output.
#
# Model endpoint is supplied through KIMI_BASE_URL.
#
#   # verify the wiring first (~3 calls) -- do this after ANY change:
#   SELFTEST_ONLY=1 bash scripts/bash/osworld/run_kimi_hybrid.sh
#
#   KIMI_API_KEY=sk-... MODEL=moonshot/kimi-k2.6 NUM_ENVS=16 THINKING=1 \
#   bash scripts/bash/osworld/run_kimi_hybrid.sh
# ----------------------------------------------------------------------------

# When run by the master job system, cwd is already the repo root.
# When run manually, cd to repo root relative to this script.
if [[ -z "${OSWORLD_JOB_ID:-}" ]]; then
  _script_dir="$(cd "$(dirname "$0")" && pwd)"
  cd "${_script_dir}/../../.."
fi

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}" 2>/dev/null || LOG_DIR="/tmp/env_infra_logs" && mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run_kimi_hybrid_${TIMESTAMP}.log}"

export KIMI_API_KEY="${KIMI_API_KEY:-}"
export KIMI_BASE_URL="${KIMI_BASE_URL:-}"
if [[ -z "$KIMI_BASE_URL" ]]; then echo "Set KIMI_BASE_URL before running" >&2; exit 2; fi
export GUI_ENV_SERVER_URL="${GUI_ENV_SERVER_URL:-http://127.0.0.1:19000/}"
export OSWORLD_USER_ID="${OSWORLD_USER_ID:-${USER:-chentongbo}}"
export OSWORLD_TASK_TYPE="${OSWORLD_TASK_TYPE:-evaluation}"

export PROXY_CONFIG_FILE="${PROXY_CONFIG_FILE:-$(pwd)/OSWorld/evaluation_examples/settings/proxy/star_proxy.json}"

if [[ -z "${KIMI_API_KEY}" ]]; then
  echo "ERROR: KIMI_API_KEY is not set. Export it before running this script." >&2
  exit 1
fi

# Keep the configured model gateway and cluster master off-proxy.
# Clear inherited proxy env and set no_proxy accordingly.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true
_gui_host="${GUI_ENV_SERVER_URL#*://}"; _gui_host="${_gui_host%%[:/]*}"
_gw_host="${KIMI_BASE_URL#*://}"; _gw_host="${_gw_host%%[:/]*}"
_no_proxy_add="localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,${_gui_host},${_gw_host}"
export no_proxy="${no_proxy:+${no_proxy},}${_no_proxy_add}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${_no_proxy_add}"

# Interpreter: MUST be the env_infra venv. A bare `python` resolves to /opt/conda/bin/python
# (3.13), which lacks the osworld deps and the editable install -- every worker would die
# with ModuleNotFoundError and the log would show nothing but that. This is the same venv
# the cluster master/node runs; the client must match.
_DEF_PYTHON_BIN="python"
PYTHON_BIN="${PYTHON_BIN:-$_DEF_PYTHON_BIN}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[!] PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if ! "${PYTHON_BIN}" -c 'import httpx, loguru, backoff, PIL' 2>/dev/null; then
  echo "[!] ${PYTHON_BIN} is missing deps (httpx / loguru / backoff / PIL)." >&2
  echo "    Don't patch single packages into it -- use the venv with editable osworld:" >&2
  echo "    export PYTHON_BIN=$_DEF_PYTHON_BIN" >&2
  exit 1
fi

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd):$(pwd)/OSWorld"

HEADLESS_ARGS=(); [[ "${HEADLESS:-1}" == "1" ]] && HEADLESS_ARGS=(--headless)
THINKING_ARGS=(); [[ "${THINKING:-1}" == "1" ]] && THINKING_ARGS=(--thinking)

CMD=(
  "${PYTHON_BIN}" scripts/python/osworld/run_kimi_hybrid.py
  --cluster_url "${GUI_ENV_SERVER_URL}"
  "${HEADLESS_ARGS[@]}"
  --action_space "${ACTION_SPACE:-pyautogui}"
  --observation_type "${OBSERVATION_TYPE:-screenshot}"
  --model "${MODEL:-moonshot/kimi-k2.6}"
  --max_tokens "${MAX_TOKENS:-2048}"
  --temperature "${TEMPERATURE:-1}"
  --top_p "${TOP_P:-0.95}"
  --max_steps "${MAX_STEPS:-50}"
  --max_image_history_length "${MAX_TRAJ_LEN:-3}"
  --max_output_chars "${MAX_OUTPUT_CHARS:-2000}"
  --test_config_base_dir "${TEST_CONFIG_BASE_DIR:-OSWorld/evaluation_examples}"
  --num_envs "${NUM_ENVS:-8}"
  --result_dir "${RESULT_DIR:-./results_kimi_hybrid}"
  --test_all_meta_path "${TEST_META_PATH:-OSWorld/evaluation_examples/test_nogdrive.json}"
  --domain "${DOMAIN:-all}"
  --client_password "${CLIENT_PASSWORD:-${OSWORLD_CLIENT_PASSWORD:-password}}"
  --password "${PASSWORD:-osworld-public-evaluation}"
  --screen_width "${SCREEN_WIDTH:-1920}"
  --screen_height "${SCREEN_HEIGHT:-1080}"
  --sleep_after_execution "${SLEEP_AFTER_EXECUTION:-3}"
  --wait_after_reset "${WAIT_AFTER_RESET:-60}"
  --base_url "${KIMI_BASE_URL}"
  --api_key "${KIMI_API_KEY}"
  --log_level "${LOG_LEVEL:-INFO}"
  "${THINKING_ARGS[@]}"
)
# VM_PROXY="" disables the VM-side egress proxy for the bash channel.
if [[ -n "${VM_PROXY+x}" ]]; then CMD+=(--vm_proxy "${VM_PROXY}"); fi

# Preflight runs in the FOREGROUND and never detaches: you want to read the verdict, and
# a failed selftest means the run must not start at all. A model that cannot see its
# screenshots does not crash -- it invents UI detail and every trajectory is worthless.
if [[ "${SELFTEST_ONLY:-0}" == "1" ]]; then
  echo "gateway=${KIMI_BASE_URL}  model=${MODEL:-moonshot/kimi-k2.6}"
  exec "${CMD[@]}" --selftest_only "$@"
fi

if [[ "${BACKGROUND:-1}" == "1" && -z "${_KIMI_HYBRID_DETACHED:-}" ]]; then
  export _KIMI_HYBRID_DETACHED=1
  nohup bash "$0" "$@" > "${LOG_FILE}" 2>&1 < /dev/null &
  CHILD_PID=$!
  echo "Started in background, PID=${CHILD_PID}"
  echo "Log:  ${LOG_FILE}"
  echo "Tail: tail -f ${LOG_FILE}"
  echo "Stop: kill ${CHILD_PID}"
  exit 0
fi

echo "Logging to ${LOG_FILE}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "GUI_ENV_SERVER_URL=${GUI_ENV_SERVER_URL}"
echo "MODEL=${MODEL:-moonshot/kimi-k2.6}  KIMI_BASE_URL=${KIMI_BASE_URL}  THINKING=${THINKING:-1}"
echo "NUM_ENVS=${NUM_ENVS:-8}  MAX_STEPS=${MAX_STEPS:-50}  SCREEN=${SCREEN_WIDTH:-1920}x${SCREEN_HEIGHT:-1080}"
echo "VM_PROXY=${VM_PROXY-http://127.0.0.1:3128 (default)}"

# When detached, nohup already redirects stdout to LOG_FILE, so tee would duplicate.
if [[ -n "${_KIMI_HYBRID_DETACHED:-}" ]]; then
  TEE_CMD=(cat)
else
  TEE_CMD=(tee "${LOG_FILE}")
fi

"${CMD[@]}" "$@" 2>&1 | "${TEE_CMD[@]}"
