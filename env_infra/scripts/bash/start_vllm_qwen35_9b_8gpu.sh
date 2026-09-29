#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------------
# Start Qwen3.5-9B (VL) with vLLM on 8 GPUs.
# Thin wrapper around start_vllm_qwen35_8gpu.sh: 8 OpenAI-compatible servers,
# each single-GPU (TP=1), on ports 8000..8007. The 9B fits on one GPU, so
# single-card gives more independent servers (higher throughput/concurrency)
# than the shared core's default 4x TP=2.
#
# Usage:
#   bash scripts/bash/start_vllm_qwen35_9b_8gpu.sh
#
# Override anything via env, e.g. dual-card (4x TP=2) instead:
#   GPU_GROUPS='0,1;2,3;4,5;6,7' PORT_START=8100 \
#     bash scripts/bash/start_vllm_qwen35_9b_8gpu.sh
#
# NOTE: does NOT inject a sitepatch. If you hit the vLLM-0.17/prometheus
# "every request returns 500" bug, set SITEPATCH_DIR (see
# start_vllm_qwen35_8gpu.sh header).
# ----------------------------------------------------------------------------

export MODEL_PATH="${MODEL_PATH:-/apdcephfs_zwfy6/share_303098609/hunyuan/jasperniu/dev/dev_ccc/projects/online-rl/ckpt/gui-hybridcua-9b-40gpu_20260909_165613_hf/rollout_119}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-9B}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
# Single-GPU deployment: one server per GPU (TP=1), 8 servers on 8000..8007.
export GPU_GROUPS="${GPU_GROUPS:-0;1;2;3;4;5;6;7}"

exec bash "$(cd "$(dirname "$0")" && pwd)/start_vllm_qwen35_8gpu.sh" "$@"
