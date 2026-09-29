#!/usr/bin/env bash
set -euo pipefail

_script_dir="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${_script_dir}/../.." && pwd)"

# ----------------------------------------------------------------------------
# Generic Qwen3.5 (VL / MoE-VL) vLLM launcher on 8 GPUs.
#
# Starts N OpenAI-compatible servers (one per GPU group), each with
# tensor-parallel = number of GPUs in that group. With the default
# GPU_GROUPS='0,1;2,3;4,5;6,7' that's 4 servers x TP=2 on ports 8000..8003.
#
# This is the shared core. Prefer the model-specific wrappers, which set the
# right MODEL_PATH / SERVED_MODEL_NAME / MAX_MODEL_LEN for you:
#   bash scripts/bash/start_vllm_qwen35_35b_a3b_8gpu.sh   # Qwen3.5-35B-A3B (MoE)
#   bash scripts/bash/start_vllm_qwen35_27b_8gpu.sh       # Qwen3.5-27B (dense)
#
# Direct use requires MODEL_PATH and SERVED_MODEL_NAME:
#   MODEL_PATH=/path/to/model SERVED_MODEL_NAME=my-model \
#     bash scripts/bash/start_vllm_qwen35_8gpu.sh
#
# Override anything via env, e.g. different topology / ports:
#   GPU_GROUPS='0,1,2,3;4,5,6,7' PORT_START=8100 \
#     MODEL_PATH=... SERVED_MODEL_NAME=... \
#     bash scripts/bash/start_vllm_qwen35_8gpu.sh
#
# NOTE: This script does NOT inject the vLLM-0.17/prometheus sitepatch. If you
# hit "every HTTP request returns 500", copy gui-env's sitepatch/ here and set
# SITEPATCH_DIR to its directory (it is otherwise left empty / disabled):
#   SITEPATCH_DIR=/mnt/llmshared/chentongbo/gui-env/sitepatch \
#     bash scripts/bash/start_vllm_qwen35_35b_a3b_8gpu.sh
# ----------------------------------------------------------------------------

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH (or use a model-specific wrapper script)}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:?set SERVED_MODEL_NAME (or use a model-specific wrapper script)}"
GPU_GROUPS="${GPU_GROUPS:-0,1;2,3;4,5;6,7}"
PORT_START="${PORT_START:-8000}"
HOST="${HOST:-0.0.0.0}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
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

# Fail fast: vLLM does not validate this either, so a bad path means all N shards
# spawn and then die one by one deep inside their own log files -- which reads
# like a GPU/env problem rather than a stale path.
if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH does not exist on this machine: ${MODEL_PATH}" >&2
  echo "       (mount missing on this host, or the wrapper's default is stale)" >&2
  exit 1
fi

extra_vllm_args=()
if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
  read -r -a extra_vllm_args <<< "${EXTRA_VLLM_ARGS}"
fi

IFS=';' read -r -a gpu_groups_array <<< "${GPU_GROUPS}"

echo "Starting ${SERVED_MODEL_NAME} vLLM servers..."
echo "  Model:                  ${MODEL_PATH}"
echo "  Served name:            ${SERVED_MODEL_NAME}"
echo "  GPU groups:             ${GPU_GROUPS}"
echo "  Port start:             ${PORT_START}"
echo "  Max model len:          ${MAX_MODEL_LEN}"
echo "  Max num seqs/server:    ${MAX_NUM_SEQS}"
echo "  GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo "  Sitepatch (PYTHONPATH): ${SITEPATCH_DIR:-<disabled>}"
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
  log_file="${LOG_DIR}/vllm_${SERVED_MODEL_NAME//\//_}_gpus${gpu_group//,/}_port${port}_${timestamp}.log"

  echo "Starting shard ${index}: GPUs ${gpu_group}, TP=${tensor_parallel_size}, port ${port}"

  nohup env CUDA_VISIBLE_DEVICES="${gpu_group}" \
    PYTHONPATH="${PYTHONPATH:-}" \
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
