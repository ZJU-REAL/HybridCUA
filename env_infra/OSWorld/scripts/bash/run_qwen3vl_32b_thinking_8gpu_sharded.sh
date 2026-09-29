#!/usr/bin/env bash
set -euo pipefail

_script_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "${_script_dir}/../../server.py" ]]; then
  cd "${_script_dir}/../.."
fi

# ----------------------------------------------------------------------------
# Evaluate Qwen3-VL-32B-Thinking through OSWorldRemoteClient + OSWorld Cluster.
#
# This is a thin, model-specific wrapper around run_qwen3vl_8gpu_sharded.sh.
# It expects the default vLLM topology from:
#   scripts/bash/start_vllm_qwen3vl_32b_thinking_8gpu.sh
#
# Default endpoints:
#   Set OPENAI_HOST or OPENAI_BASE_URLS to select model endpoints.
#
# Typical smoke:
#   BACKGROUND=0 NUM_ENVS=8 DOMAIN=chrome MAX_STEPS=1 \
#     bash scripts/bash/run_qwen3vl_32b_thinking_8gpu_sharded.sh
#
# Remote vLLM host:
#   OPENAI_HOST=10.xx.xx.xx NUM_ENVS=128 \
#     bash scripts/bash/run_qwen3vl_32b_thinking_8gpu_sharded.sh
# ----------------------------------------------------------------------------

export MODEL="${MODEL:-qwen3-vl-32b-thinking}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-local}"
export OPENAI_HOST="${OPENAI_HOST:-${VLLM_HOST:-}}"
export OPENAI_PORT_START="${OPENAI_PORT_START:-8000}"
export OPENAI_NUM_SHARDS="${OPENAI_NUM_SHARDS:-4}"
export MAX_TOKENS="${MAX_TOKENS:-2048}"
export NUM_ENVS="${NUM_ENVS:-128}"
export ADD_THOUGHT_PREFIX="${ADD_THOUGHT_PREFIX:-1}"
export RESULT_DIR="${RESULT_DIR:-./results_qwen3vl_32b_thinking_8gpu_sharded}"

exec bash scripts/bash/run_qwen3vl_8gpu_sharded.sh "$@"
