#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/llmshared-ssd-hd/models/Qwen3-VL-8B-Instruct}"
PORT="${PORT:-9000}"
GPU_ID="${GPU_ID:-0}"

_script_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "${_script_dir}/../../server.py" ]]; then
  LOG_DIR="$(cd "${_script_dir}/../.." && pwd)/logs"
else
  LOG_DIR="$(pwd)/logs"
fi
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/vllm_qwen3vl_$(date +%Y%m%d_%H%M%S).log"

echo "Starting vLLM server..."
echo "  Model:    ${MODEL_PATH}"
echo "  GPU:      ${GPU_ID}"
echo "  Port:     ${PORT}"
echo "  Log:      ${LOG_FILE}"

nohup env CUDA_VISIBLE_DEVICES=${GPU_ID} \
  python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name Qwen3-VL-8B-Instruct \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --limit-mm-per-prompt '{"image": 5, "video": 0}' \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Started with PID=${PID}"
echo "Tail:  tail -f ${LOG_FILE}"
echo "Stop:  kill ${PID}"
