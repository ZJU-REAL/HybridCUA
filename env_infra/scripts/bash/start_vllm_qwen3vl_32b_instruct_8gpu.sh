#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------------
# Start Qwen3-VL-32B-Instruct with vLLM on 8 GPUs.
# Thin wrapper around start_vllm_qwen35_8gpu.sh: 4 OpenAI-compatible servers,
# each TP=2, on ports 8000..8003.
#
# Usage:
#   bash scripts/bash/start_vllm_qwen3vl_32b_instruct_8gpu.sh
#
# Override anything via env, e.g.:
#   GPU_GROUPS='0,1,2,3;4,5,6,7' PORT_START=8100 \
#     bash scripts/bash/start_vllm_qwen3vl_32b_instruct_8gpu.sh
#
# NOTE: does NOT inject a sitepatch. If you hit the vLLM-0.17/prometheus
# "every request returns 500" bug, set SITEPATCH_DIR (see
# start_vllm_qwen35_8gpu.sh header).
# ----------------------------------------------------------------------------

export MODEL_PATH="${MODEL_PATH:-/mnt/llmshared-ssd-hd/models/Qwen3-VL-32B-Instruct/}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-vl-32b-instruct}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

exec bash "$(cd "$(dirname "$0")" && pwd)/start_vllm_qwen35_8gpu.sh" "$@"
