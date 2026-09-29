#!/usr/bin/env bash
set -euo pipefail

# When run via cluster master, cwd is already repo root.
# When run manually, cd to repo root relative to script location.
if [[ -z "${OSWORLD_JOB_ID:-}" ]]; then
  _script_dir="$(cd "$(dirname "$0")" && pwd)"
  cd "${_script_dir}/../../.."
fi

# ----------------------------------------------------------------------------
# Run OSWorld evaluation with Qwen3.5-27B via the official mm_agents.qwen.QwenAgent,
# sharded across 4 vLLM endpoints. Each worker process pins one endpoint
# (round-robin by worker index), passed directly to QwenAgent(base_url=...).
#
# Derived from run_qwen_qwen35vl_sharded.sh, with this run's tuning baked in:
#   MODEL=Qwen3.5-27B  HISTORY_N=50  IMAGE_MAX=5  FOLD_SIZE=1  NUM_ENVS=32
#   OPENAI_NUM_SHARDS=4
#
# Why 4 shards and not 8: the 27B (52G on disk) does not fit one 97G card with
# a useful KV cache, so it is deployed as 4 servers x TP=2 -- the default
# GPU_GROUPS='0,1;2,3;4,5;6,7' of start_vllm_qwen35_27b_8gpu.sh. That yields
# ports 8000..8003, so OPENAI_NUM_SHARDS must be 4, not 8; pointing at 8 would
# send half the traffic to dead ports.
#
# Start vLLM first, then run:
#   MODEL_PATH=$(pwd)/../models/Qwen3.5-27B SERVED_MODEL_NAME=Qwen3.5-27B \
#     PYTHON_BIN=/opt/conda/bin/python3.13 \
#     bash scripts/bash/start_vllm_qwen35_27b_8gpu.sh
#   bash scripts/bash/osworld/run_qwen_qwen35_27b_sharded.sh
#
# Override anything via env, e.g. fewer envs / a single domain:
#   NUM_ENVS=16 DOMAIN=os bash scripts/bash/osworld/run_qwen_qwen35_27b_sharded.sh
# ----------------------------------------------------------------------------

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}" 2>/dev/null || LOG_DIR="/tmp/env_infra_logs" && mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run_qwen_qwen35_27b_sharded_${TIMESTAMP}.log}"
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
if [[ "${BACKGROUND:-1}" == "1" && -z "${_QWEN_SHARDED_DETACHED:-}" ]]; then
  export _QWEN_SHARDED_DETACHED=1
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

if [[ -n "${_QWEN_SHARDED_DETACHED:-}" ]]; then
  TEE_CMD=(cat)
else
  TEE_CMD=(tee "${LOG_FILE}")
fi

# 🔴 必须用 env_infra 的 venv,不能用默认的 `python`。
#    实测 `which python` = /opt/conda/envs/torch-base/bin/python,那个环境【没有 openai】
#    (ModuleNotFoundError: No module named 'openai'),而 mm_agents/qwen/client.py 顶层就
#    import openai —— 用默认 python 会在 import QwenAgent 时直接崩。
#
#    这里默认值由【repo 位置】推导 (../venvs/env_infra),而不是写死某个盘的绝对路径:
#    这套代码同时存在于两个盘 (源盘 dop-fuse / 目标盘 ceph-fuse),各自带自己的 venv。
#    写死一个绝对路径,在另一个盘上就会指向错误的 venv。
#    _script_dir 在文件开头算过,但那时可能已 cd,所以这里用 REPO_ROOT=$(pwd) —— 脚本
#    开头保证了 cwd 就是 repo 根 (OSWORLD_JOB_ID 分支也一样)。
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: PYTHON_BIN 不可执行: ${PYTHON_BIN}" >&2
  echo "       期望 <repo>/../venvs/env_infra/bin/python (install_env.sh 建的)" >&2
  echo "       或用 PYTHON_BIN=/path/to/venv/bin/python 显式指定" >&2
  exit 1
fi
# 早失败:openai 缺了的话,报清楚原因,而不是等 python 抛 ModuleNotFoundError。
if ! "${PYTHON_BIN}" -c 'import openai' 2>/dev/null; then
  echo "ERROR: ${PYTHON_BIN} 里没有 openai —— QwenAgent 起不来。" >&2
  echo "       这个 venv 可能还没装完 (bash install_env.sh)。" >&2
  exit 1
fi
CMD=(
  "${PYTHON_BIN}" scripts/python/osworld/run_qwen_qwen35vl_sharded.py
  --cluster_url "${GUI_ENV_SERVER_URL}"
  --action_space "${ACTION_SPACE:-pyautogui}"
  --observation_type "${OBSERVATION_TYPE:-screenshot}"
  # Must match vLLM --served-model-name, else every request 404s on model lookup.
  --model "${MODEL:-Qwen3.5-27B}"
  --coord "${COORD:-relative}"
  --max_tokens "${MAX_TOKENS:-2048}"
  --temperature "${TEMPERATURE:-0}"
  --top_p "${TOP_P:-0.9}"
  --history_n "${HISTORY_N:-50}"
  --image_max "${IMAGE_MAX:-5}"
  --fold_size "${FOLD_SIZE:-1}"
  --max_steps "${MAX_STEPS:-50}"
  --test_config_base_dir "${TEST_CONFIG_BASE_DIR:-OSWorld/evaluation_examples}"
  --num_envs "${NUM_ENVS:-32}"
  --result_dir "${RESULT_DIR:-./results_qwen_qwen35_27b_sharded}"
  --test_all_meta_path "${TEST_META_PATH:-OSWorld/evaluation_examples/test_nogdrive.json}"
  --domain "${DOMAIN:-all}"
  --client_password "${CLIENT_PASSWORD:-password}"
  --screen_width "${SCREEN_WIDTH:-1920}"
  --screen_height "${SCREEN_HEIGHT:-1080}"
  --sleep_after_execution "${SLEEP_AFTER_EXECUTION:-3}"
  --log_level "${LOG_LEVEL:-INFO}"
  --openai_base_urls "${OPENAI_BASE_URLS}"
  --api_key "${OPENAI_API_KEY}"
  "$@"
)

if [[ "${HEADLESS:-1}" == "1" ]]; then CMD+=(--headless); fi
if [[ "${ADD_THOUGHT_PREFIX:-0}" == "1" ]]; then CMD+=(--add_thought_prefix); fi
if [[ "${ENABLE_THINKING:-0}" == "1" ]]; then CMD+=(--enable_thinking); fi

"${CMD[@]}" 2>&1 | "${TEE_CMD[@]}"
