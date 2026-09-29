#!/usr/bin/env bash
set -euo pipefail

_script_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "${_script_dir}/../../server.py" ]]; then
  cd "${_script_dir}/../.."
fi

# ----------------------------------------------------------------------------
# Start one Qwen3-VL vLLM OpenAI-compatible server per GPU.
#
# Defaults:
#   GPUs:  0,1,2,3,4,5,6,7
#   Ports: 8000,8001,8002,8003,8004,8005,8006,8007
#
# Example:
#   source /mnt/llmshared/chentongbo/venvs/vllm/bin/activate
#   bash scripts/bash/start_vllm_qwen3vl_8gpu.sh
#
# Override:
#   MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct PORT_START=8100 GPU_IDS=0,1,2,3 \
#     bash scripts/bash/start_vllm_qwen3vl_8gpu.sh
# ----------------------------------------------------------------------------

MODEL_PATH="${MODEL_PATH:-/mnt/llmshared-ssd-hd/models/Qwen3-VL-8B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-VL-8B-Instruct}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
PORT_START="${PORT_START:-8000}"
HOST="${HOST:-0.0.0.0}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 5, \"video\": 0}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-$(pwd)/logs}"

mkdir -p "${LOG_DIR}"

timestamp="$(date +%Y%m%d_%H%M%S)"

IFS=',' read -r -a gpu_id_array <<< "${GPU_IDS}"

echo "Starting Qwen3-VL vLLM shard servers..."
echo "  Model:       ${MODEL_PATH}"
echo "  Served name: ${SERVED_MODEL_NAME}"
echo "  GPUs:        ${GPU_IDS}"
echo "  Port start:  ${PORT_START}"
echo "  Log dir:     ${LOG_DIR}"
echo

started_pids=()

for index in "${!gpu_id_array[@]}"; do
  gpu_id="${gpu_id_array[$index]}"
  gpu_id="${gpu_id//[[:space:]]/}"
  if [[ -z "${gpu_id}" ]]; then
    continue
  fi

  port=$((PORT_START + index))
  log_file="${LOG_DIR}/vllm_qwen3vl_gpu${gpu_id}_port${port}_${timestamp}.log"

  echo "Starting shard ${index}: GPU ${gpu_id}, port ${port}"
  nohup env CUDA_VISIBLE_DEVICES="${gpu_id}" \
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
      --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}" \
    > "${log_file}" 2>&1 &

  pid=$!
  started_pids+=("${pid}")
  echo "  PID:  ${pid}"
  echo "  Log:  ${log_file}"
done

openai_base_urls=()
for index in "${!gpu_id_array[@]}"; do
  gpu_id="${gpu_id_array[$index]}"
  gpu_id="${gpu_id//[[:space:]]/}"
  if [[ -n "${gpu_id}" ]]; then
    openai_base_urls+=("http://127.0.0.1:$((PORT_START + index))/v1")
  fi
done

echo
echo "Started ${#started_pids[@]} vLLM shard server(s)."
echo "PIDs: ${started_pids[*]}"
echo "OPENAI_BASE_URLS=$(IFS=,; echo "${openai_base_urls[*]}")"
echo
echo "Check readiness:"
echo "  curl http://127.0.0.1:${PORT_START}/v1/models"
echo
echo "Stop all shards:"
echo "  kill ${started_pids[*]}"
