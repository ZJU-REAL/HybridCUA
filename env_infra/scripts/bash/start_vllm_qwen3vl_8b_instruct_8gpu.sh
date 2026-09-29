#!/usr/bin/env bash
set -euo pipefail

_script_dir="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${_script_dir}/../.." && pwd)"

# ----------------------------------------------------------------------------
# Start Qwen3-VL-8B-Instruct with vLLM on 8 GPUs.
#
# Self-contained launcher (does NOT delegate to start_vllm_qwen35_8gpu.sh):
# one OpenAI-compatible server per GPU, TP=1, on ports 8000..8007. For an 8B
# model a full replica fits on a single GPU, so per-GPU serving gives the best
# throughput.
#
# Usage:
#   source /mnt/share/chentongbo/venvs/env_infra/bin/activate
#   bash scripts/bash/start_vllm_qwen3vl_8b_instruct_8gpu.sh
#
# Override anything via env, e.g. a different checkpoint / ports / GPUs:
#   MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct PORT_START=8100 GPU_IDS=0,1,2,3 \
#     bash scripts/bash/start_vllm_qwen3vl_8b_instruct_8gpu.sh
#
# NOTE: does NOT inject a sitepatch. If you hit the vLLM-0.17/prometheus
# "every request returns 500" bug, copy gui-env's sitepatch/ here and set
# SITEPATCH_DIR to its directory:
#   SITEPATCH_DIR=/mnt/llmshared/chentongbo/gui-env/sitepatch \
#     bash scripts/bash/start_vllm_qwen3vl_8b_instruct_8gpu.sh
# ----------------------------------------------------------------------------

MODEL_PATH="${MODEL_PATH:-/apdcephfs_zwfy6/share_303098609/hunyuan/jasperniu/dev/huggingface/models/Qwen3-VL-8B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-VL-8B-Instruct}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
PORT_START="${PORT_START:-8000}"
HOST="${HOST:-0.0.0.0}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 5, \"video\": 0}}"
PYTHON_BIN="${PYTHON_BIN:-/apdcephfs_zwfy6/share_303098609/hunyuan/jasperniu/envs/vllm017/bin/python}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"

# Optional vLLM-0.17/prometheus 500 workaround. Disabled by default: only
# injected when SITEPATCH_DIR is set and actually contains sitecustomize.py.
SITEPATCH_DIR="${SITEPATCH_DIR:-}"
if [[ -n "${SITEPATCH_DIR}" ]]; then
  if [[ -f "${SITEPATCH_DIR}/sitecustomize.py" ]]; then
    export PYTHONPATH="${SITEPATCH_DIR}:${PYTHONPATH:-}"
  else
    echo "Warning: SITEPATCH_DIR set but ${SITEPATCH_DIR}/sitecustomize.py not found; skipping sitepatch." >&2
  fi
fi

mkdir -p "${LOG_DIR}"
timestamp="$(date +%Y%m%d_%H%M%S)"

if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "Warning: MODEL_PATH does not exist on this machine: ${MODEL_PATH}"
fi

extra_vllm_args=()
if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
  read -r -a extra_vllm_args <<< "${EXTRA_VLLM_ARGS}"
fi

IFS=',' read -r -a gpu_id_array <<< "${GPU_IDS}"

echo "Starting ${SERVED_MODEL_NAME} vLLM servers (one per GPU, TP=1)..."
echo "  Model:                  ${MODEL_PATH}"
echo "  Served name:            ${SERVED_MODEL_NAME}"
echo "  GPUs:                   ${GPU_IDS}"
echo "  Port start:             ${PORT_START}"
echo "  Max model len:          ${MAX_MODEL_LEN}"
echo "  Max num seqs/server:    ${MAX_NUM_SEQS}"
echo "  GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo "  Sitepatch (PYTHONPATH): ${SITEPATCH_DIR:-<disabled>}"
echo "  Log dir:                ${LOG_DIR}"
echo

started_pids=()
openai_base_urls=()

for index in "${!gpu_id_array[@]}"; do
  gpu_id="${gpu_id_array[$index]}"
  gpu_id="${gpu_id//[[:space:]]/}"
  if [[ -z "${gpu_id}" ]]; then
    continue
  fi

  port=$((PORT_START + index))
  log_file="${LOG_DIR}/vllm_${SERVED_MODEL_NAME//\//_}_gpu${gpu_id}_port${port}_${timestamp}.log"

  echo "Starting shard ${index}: GPU ${gpu_id}, TP=1, port ${port}"

  nohup env CUDA_VISIBLE_DEVICES="${gpu_id}" \
    PYTHONPATH="${PYTHONPATH:-}" \
    "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
      --model "${MODEL_PATH}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --host "${HOST}" \
      --port "${port}" \
      --trust-remote-code \
      --tensor-parallel-size 1 \
      --dtype "${DTYPE}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}" \
      "${extra_vllm_args[@]}" \
    > "${log_file}" 2>&1 &

  pid=$!
  started_pids+=("${pid}")
  openai_base_urls+=("http://127.0.0.1:${port}/v1")

  echo "  PID: ${pid}"
  echo "  Log: ${log_file}"
done

echo
echo "Started ${#started_pids[@]} vLLM server(s)."
echo "PIDs: ${started_pids[*]}"
echo "OPENAI_BASE_URLS=$(IFS=,; echo "${openai_base_urls[*]}")"
echo
echo "Check readiness (wait for 'Application startup complete' in logs, ~1 min):"
echo "  curl http://127.0.0.1:${PORT_START}/v1/models"
echo
echo "Stop all shards:"
echo "  kill ${started_pids[*]}"
