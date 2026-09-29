#!/usr/bin/env bash
set -euo pipefail

_script_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "${_script_dir}/../../server.py" ]]; then
  cd "${_script_dir}/../.."
fi

# ----------------------------------------------------------------------------
# Start Qwen3-VL-32B-Thinking with vLLM on 8 GPUs.
#
# Default topology:
#   4 OpenAI-compatible servers
#   each server uses 2 GPUs with tensor parallel size 2
#   ports: 8000,8001,8002,8003
#
# Typical usage:
#   source /mnt/share/chentongbo/venvs/osworld/bin/activate
#   bash scripts/bash/start_vllm_qwen3vl_32b_thinking_8gpu.sh
#
# Override example:
#   MODEL_PATH=/path/to/Qwen3-VL-32B-Thinking \
#   GPU_GROUPS='0,1;2,3;4,5;6,7' \
#   PORT_START=9100 \
#   bash scripts/bash/start_vllm_qwen3vl_32b_thinking_8gpu.sh
# ----------------------------------------------------------------------------

MODEL_PATH="${MODEL_PATH:-/mnt/xiaoai-vision-share/GUI-MOPD/checkpoints/Qwen3-vl-32b-thinking/stage1-osworld-final/global_step_200/huggingface/}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-vl-32b-thinking}"
GPU_GROUPS="${GPU_GROUPS:-0,1;2,3;4,5;6,7}"
PORT_START="${PORT_START:-8000}"
HOST="${HOST:-0.0.0.0}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 5, \"video\": 0}}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-$(pwd)/logs}"

mkdir -p "${LOG_DIR}"

timestamp="$(date +%Y%m%d_%H%M%S)"

if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "Warning: MODEL_PATH does not exist on this machine: ${MODEL_PATH}"
fi

reasoning_args=()
if [[ -n "${REASONING_PARSER}" ]]; then
  reasoning_args+=(--reasoning-parser "${REASONING_PARSER}")
fi

extra_vllm_args=()
if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
  read -r -a extra_vllm_args <<< "${EXTRA_VLLM_ARGS}"
fi

IFS=';' read -r -a gpu_groups_array <<< "${GPU_GROUPS}"

echo "Starting Qwen3-VL-32B-Thinking vLLM servers..."
echo "  Model:                  ${MODEL_PATH}"
echo "  Served name:            ${SERVED_MODEL_NAME}"
echo "  GPU groups:             ${GPU_GROUPS}"
echo "  Port start:             ${PORT_START}"
echo "  Max model len:          ${MAX_MODEL_LEN}"
echo "  Max num seqs/server:    ${MAX_NUM_SEQS}"
echo "  GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo "  Reasoning parser:       ${REASONING_PARSER:-disabled}"
echo "  Log dir:                ${LOG_DIR}"
echo

started_pids=()
openai_base_urls=()
for index in "${!gpu_groups_array[@]}"; do
  gpu_group="${gpu_groups_array[$index]}"
  gpu_group="${gpu_group//[[:space:]]/}"
  if [[ -z "${gpu_group}" ]]; then
    continue
  fi

  IFS=',' read -r -a group_gpus <<< "${gpu_group}"
  tensor_parallel_size="${#group_gpus[@]}"
  port=$((PORT_START + index))
  log_file="${LOG_DIR}/vllm_qwen3vl_32b_thinking_gpus${gpu_group//,/}_port${port}_${timestamp}.log"

  echo "Starting shard ${index}: GPUs ${gpu_group}, TP=${tensor_parallel_size}, port ${port}"
  nohup env CUDA_VISIBLE_DEVICES="${gpu_group}" \
    "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
      --model "${MODEL_PATH}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --host "${HOST}" \
      --port "${port}" \
      --trust-remote-code \
      --tensor-parallel-size "${tensor_parallel_size}" \
      --dtype "${DTYPE}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}" \
      "${reasoning_args[@]}" \
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
echo "Check readiness:"
echo "  curl http://127.0.0.1:${PORT_START}/v1/models"
echo
echo "Stop all shards:"
echo "  kill ${started_pids[*]}"
