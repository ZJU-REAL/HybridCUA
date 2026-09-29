#!/usr/bin/env bash
set -euo pipefail

_script_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "${_script_dir}/../../server.py" ]]; then
  cd "${_script_dir}/../.."
fi

# ----------------------------------------------------------------------------
# Run OSWorld evaluation with a locally hosted Qwen3-VL-8B-Instruct served
# by vLLM at http://127.0.0.1:9000/v1.
#
# Matching vLLM launch command (run separately, before this script):
#   nohup env CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
#     stdbuf -oL -eL \
#     vllm serve /home/shenyl/hf/model/Qwen/Qwen3-VL-8B-Instruct \
#       --host 0.0.0.0 --port 9000 \
#       --served-model-name Qwen3-VL-8B-Instruct \
#       --trust-remote-code --tensor-parallel-size 1 \
#       --dtype bfloat16 --gpu-memory-utilization 0.9 \
#       --max-model-len 32768 \
#       --limit-mm-per-prompt '{"image": 6, "video": 0}' \
#     > ~/qwen3_vl_9000.log 2>&1 &
#
# This wrapper monkey-patches Qwen3VLAgent's default api_backend to "openai"
# at process start, so it talks to the local vLLM server instead of DashScope.
# No .py file in the repo is modified.
#
# Override anything via env vars, e.g.:
#   NUM_ENVS=4 DOMAIN=libreoffice_calc bash scripts/bash/run_qwen3vl.sh
# ----------------------------------------------------------------------------

mkdir -p logs

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-logs/run_qwen3vl_${TIMESTAMP}.log}"
export PROXY_CONFIG_FILE="${PROXY_CONFIG_FILE:-$(pwd)/evaluation_examples/settings/proxy/local.json}"

# Host-side proxy for task asset downloads. This overrides any shell-level
# localhost proxy so requests.get() in setup.py uses the LAN proxy directly.
export HOST_PROXY_URL="${HOST_PROXY_URL-http://127.0.0.1:7897}"
export http_proxy="${HOST_PROXY_URL}"
export https_proxy="${HOST_PROXY_URL}"
export HTTP_PROXY="${HOST_PROXY_URL}"
export HTTPS_PROXY="${HOST_PROXY_URL}"
export all_proxy="${HOST_PROXY_URL}"
export ALL_PROXY="${HOST_PROXY_URL}"

# Keep local services such as the vLLM server off the proxy path.
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

# Point the OpenAI-compatible client at the local vLLM server.
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:9000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-local}"

# Default: detach to background via nohup. Pass BACKGROUND=0 to run in foreground.
if [[ "${BACKGROUND:-1}" == "1" && -z "${_QWEN3VL_DETACHED:-}" ]]; then
  export _QWEN3VL_DETACHED=1
  nohup bash "$0" "$@" > "${LOG_FILE}" 2>&1 < /dev/null &
  CHILD_PID=$!
  echo "Started in background, PID=${CHILD_PID}"
  echo "Log:  ${LOG_FILE}"
  echo "Tail: tail -f ${LOG_FILE}"
  echo "Stop: kill ${CHILD_PID}"
  exit 0
fi

echo "Logging to ${LOG_FILE}"
echo "OPENAI_BASE_URL=${OPENAI_BASE_URL}"
echo "PROXY_CONFIG_FILE=${PROXY_CONFIG_FILE}"
echo "HOST_PROXY_URL=${HOST_PROXY_URL}"
echo "NO_PROXY=${NO_PROXY}"

# Inline launcher: patch Qwen3VLAgent.__init__ so api_backend defaults to "openai",
# then run the existing multienv script via runpy without editing any .py file.
PYCMD=$(cat <<'PYEOF'
import sys, os, runpy
sys.path.insert(0, os.getcwd())

import mm_agents.qwen3vl_agent as _q
_orig_init = _q.Qwen3VLAgent.__init__

def _patched_init(self, *args, **kwargs):
    kwargs.setdefault("api_backend", "openai")
    return _orig_init(self, *args, **kwargs)

_q.Qwen3VLAgent.__init__ = _patched_init

target = "scripts/python/run_multienv_qwen3vl.py"
sys.argv = [target] + sys.argv[1:]
runpy.run_path(target, run_name="__main__")
PYEOF
)

if [[ -n "${_QWEN3VL_DETACHED:-}" ]]; then
  # Already detached by nohup; stdout/stderr go to LOG_FILE already.
  TEE_CMD=(cat)
else
  TEE_CMD=(tee "${LOG_FILE}")
fi

python -c "$PYCMD" \
  --provider_name "${PROVIDER_NAME:-docker}" \
  --headless \
  --observation_type "${OBSERVATION_TYPE:-screenshot}" \
  --model "${MODEL:-Qwen3-VL-8B-Instruct}" \
  --coord "${COORD:-relative}" \
  --max_tokens "${MAX_TOKENS:-2048}" \
  --temperature "${TEMPERATURE:-0}" \
  --top_p "${TOP_P:-0.9}" \
  --max_steps "${MAX_STEPS:-50}" \
  --max_trajectory_length "${MAX_TRAJ_LEN:-3}" \
  --num_envs "${NUM_ENVS:-8}" \
  --result_dir "${RESULT_DIR:-./results_qwen3vl}" \
  --test_all_meta_path "${TEST_META_PATH:-evaluation_examples/test_nogdrive.json}" \
  --domain "${DOMAIN:-all}" \
  --client_password "${CLIENT_PASSWORD:-${OSWORLD_CLIENT_PASSWORD:-password}}" \
  --region "${REGION:-${AWS_REGION:-us-east-1}}" \
  --screen_width "${SCREEN_WIDTH:-1920}" \
  --screen_height "${SCREEN_HEIGHT:-1080}" \
  --sleep_after_execution "${SLEEP_AFTER_EXECUTION:-3}" \
  --log_level "${LOG_LEVEL:-INFO}" \
  2>&1 | "${TEE_CMD[@]}"
