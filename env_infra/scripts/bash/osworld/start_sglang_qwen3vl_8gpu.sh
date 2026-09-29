#!/usr/bin/env bash
set -euo pipefail

_script_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "${_script_dir}/../../../server.py" ]]; then
  cd "${_script_dir}/../../.."
fi

# ----------------------------------------------------------------------------
# Start one Qwen3-VL SGLang OpenAI-compatible server per GPU (single-card each,
# NO tensor parallel). SGLang counterpart of start_vllm_qwen3vl_8gpu.sh.
#
# Defaults:
#   GPUs:  0,1,2,3,4,5,6,7   (one server per GPU, tp=1)
#   Ports: 8000,8001,8002,8003,8004,8005,8006,8007
#
# Example (run in an env that has sglang installed, e.g. the online-rl venv):
#   bash scripts/bash/osworld/start_sglang_qwen3vl_8gpu.sh
#
# Override:
#   MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct PORT_START=8100 GPU_IDS=0,1,2,3 \
#     bash scripts/bash/osworld/start_sglang_qwen3vl_8gpu.sh
#
# The resulting endpoints feed the eval driver's OPENAI_BASE_URLS.
# ----------------------------------------------------------------------------

MODEL_PATH="${MODEL_PATH:-/apdcephfs_zwfy6/share_303098609/hunyuan/jasperniu/dev/dev_ccc/projects/models/Qwen3-VL-8B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-VL-8B-Instruct}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
PORT_START="${PORT_START:-8000}"
HOST="${HOST:-0.0.0.0}"
DTYPE="${DTYPE:-bfloat16}"
# SGLang memory knob (fraction of GPU mem reserved for the static KV/weights).
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.9}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-$(pwd)/logs}"

mkdir -p "${LOG_DIR}"

timestamp="$(date +%Y%m%d_%H%M%S)"

IFS=',' read -r -a gpu_id_array <<< "${GPU_IDS}"

echo "Starting Qwen3-VL SGLang shard servers (single-card, tp=1)..."
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
  log_file="${LOG_DIR}/sglang_qwen3vl_gpu${gpu_id}_port${port}_${timestamp}.log"

  echo "Starting shard ${index}: GPU ${gpu_id}, port ${port}"
  nohup env CUDA_VISIBLE_DEVICES="${gpu_id}" \
    "${PYTHON_BIN}" -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --host "${HOST}" \
      --port "${port}" \
      --trust-remote-code \
      --tp 1 \
      --dtype "${DTYPE}" \
      --mem-fraction-static "${MEM_FRACTION_STATIC}" \
      --context-length "${CONTEXT_LENGTH}" \
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
echo "Started ${#started_pids[@]} SGLang shard server(s)."
echo "PIDs: ${started_pids[*]}"
echo "OPENAI_BASE_URLS=$(IFS=,; echo "${openai_base_urls[*]}")"
echo
echo "Check readiness (look for 'The server is fired up and ready to roll!' in the log):"
echo "  curl http://127.0.0.1:${PORT_START}/v1/models"
echo
echo "Stop all shards:"
echo "  kill ${started_pids[*]}"
