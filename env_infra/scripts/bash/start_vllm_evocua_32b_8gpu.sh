#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------------
# Start EvoCUA-32B (meituan/EvoCUA-32B-20260105) with vLLM on 8 GPUs.
# Thin wrapper around start_vllm_qwen35_8gpu.sh: 4 OpenAI-compatible servers,
# each TP=2, on ports 8000..8003. EvoCUA is a Qwen3-VL derivative
# (architectures = Qwen3VLForConditionalGeneration), so the generic Qwen3.5-VL
# launcher serves it unchanged.
#
# Usage:
#   bash scripts/bash/start_vllm_evocua_32b_8gpu.sh
#
# Override anything via env, e.g.:
#   GPU_GROUPS='0,1,2,3;4,5,6,7' PORT_START=8100 \
#     bash scripts/bash/start_vllm_evocua_32b_8gpu.sh
#
# MAX_MODEL_LEN: the checkpoint advertises max_position_embeddings=262144.
# Serving at that length would size the KV cache far past what 2x97G leaves
# after a 63G model, so we cap at 32768 -- ample for OSWorld episodes (the
# agent folds history to `max_history_turns` screenshots per request).
#
# The served name (EvoCUA) is what scripts/bash/osworld/run_evocua.sh sends as
# `--model`; the two MUST match or every request 404s on model lookup.
# ----------------------------------------------------------------------------

_script_dir="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${_script_dir}/../.." && pwd)"

# Derive from repo location rather than hard-coding a disk: this tree exists on
# several mounts, each with its own models/ sibling.
export MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/../models/EvoCUA-32B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-EvoCUA}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

exec bash "${_script_dir}/start_vllm_qwen35_8gpu.sh" "$@"
