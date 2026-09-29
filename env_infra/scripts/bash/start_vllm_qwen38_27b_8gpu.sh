#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------------
# Start Qwen3.8-27B (dense + VL) with vLLM on 8 GPUs.
# Thin wrapper around start_vllm_qwen35_8gpu.sh: 4 OpenAI-compatible servers,
# each TP=2, on ports 8000..8003.
#
# Usage:
#   bash scripts/bash/start_vllm_qwen38_27b_8gpu.sh
#
# Override anything via env, e.g.:
#   GPU_GROUPS='0,1,2,3;4,5,6,7' PORT_START=8100 \
#     bash scripts/bash/start_vllm_qwen38_27b_8gpu.sh
#
# Paired sampler: scripts/bash/cua_gym/run_qwen38_c_gui.sh
#
# LIMIT_MM_PER_PROMPT must stay >= the sampler's --image_max (IMAGE_MAX), or vLLM
# rejects any prompt once folding has not yet collapsed the oldest screenshots.
# The shared launcher defaults to {"image": 5}, matching IMAGE_MAX=5.
#
# NOTE: does NOT inject a sitepatch. If you hit the vLLM-0.17/prometheus
# "every request returns 500" bug, set SITEPATCH_DIR (see
# start_vllm_qwen35_8gpu.sh header).
# ----------------------------------------------------------------------------

export MODEL_PATH="${MODEL_PATH:-/apdcephfs_zwfy6/share_303098609/hunyuan/jasperniu/dev/dev_ccc/projects/models/Qwen3.8-27B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.8-27B}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"

exec bash "$(cd "$(dirname "$0")" && pwd)/start_vllm_qwen35_8gpu.sh" "$@"
