#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------------
# Start Qwen3.5-35B-A3B (MoE + VL) with vLLM on 8 GPUs.
# Thin wrapper around start_vllm_qwen35_8gpu.sh: 4 OpenAI-compatible servers,
# each TP=2, on ports 8000..8003.
#
# Usage:
#   bash scripts/bash/start_vllm_qwen35_35b_a3b_8gpu.sh
#
# Override anything via env, e.g.:
#   GPU_GROUPS='0,1,2,3;4,5,6,7' PORT_START=8100 \
#     bash scripts/bash/start_vllm_qwen35_35b_a3b_8gpu.sh
#
# NOTE: unlike the original one-off command, this does NOT inject a sitepatch.
# If you hit the vLLM-0.17/prometheus "every request returns 500" bug, set
# SITEPATCH_DIR (see start_vllm_qwen35_8gpu.sh header).
# ----------------------------------------------------------------------------

# Base weights on cephfs (all four hosts see this path). Downloaded from
# Qwen/Qwen3.5-35B-A3B, 14 shards / ~72GB, arch Qwen3_5MoeForConditionalGeneration
# (registered in vLLM 0.17.0 -- no upgrade needed).
#
# The two previous defaults both pointed at /mnt/llmshared*, which is NOT mounted
# on these hosts, so every launch died four times over in the shard logs:
#   /mnt/llmshared-ssd-hd/wuchangqiao/data/models/Qwen/Qwen3.5-35B-A3B  (base)
#   /mnt/llmshared/chentongbo/cua-h/checkpoints/moe-hy-e4-gt50/final/huggingface
# The second is a FINETUNE, not base -- to compare against it, pass MODEL_PATH
# explicitly from a host where that mount exists.
export MODEL_PATH="${MODEL_PATH:-/apdcephfs_zwfy6/share_303098609/hunyuan/jasperniu/dev/huggingface/models/Qwen3.5-35B-A3B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-35B-A3B}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"

exec bash "$(cd "$(dirname "$0")" && pwd)/start_vllm_qwen35_8gpu.sh" "$@"
