#!/usr/bin/env bash
set -euo pipefail

_script_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "${_script_dir}/../../server.py" ]]; then
  cd "${_script_dir}/../.."
fi

# ----------------------------------------------------------------------------
# Evaluate Qwen3.5-27B through OSWorldRemoteClient + OSWorld Cluster.
#
# This is a thin, model-specific wrapper around run_qwen35vl_8gpu_sharded.sh.
# It matches a 4-endpoint vLLM topology such as:
#   GPU_GROUPS='0,1;2,3;4,5;6,7'
#   PORT_START=8000
#
# Typical smoke:
#   BACKGROUND=0 NUM_ENVS=8 DOMAIN=chrome MAX_STEPS=1 \
#     bash scripts/bash/run_qwen35vl_27b_8gpu_sharded.sh
#
# Remote vLLM host:
#   OPENAI_HOST=10.xx.xx.xx NUM_ENVS=32 \
#     bash scripts/bash/run_qwen35vl_27b_8gpu_sharded.sh
# ----------------------------------------------------------------------------

export MODEL="${MODEL:-qwen3.5-27b}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-local}"
export OPENAI_HOST="${OPENAI_HOST:-${VLLM_HOST:-127.0.0.1}}"
export OPENAI_PORT_START="${OPENAI_PORT_START:-8000}"
export OPENAI_NUM_SHARDS="${OPENAI_NUM_SHARDS:-4}"
export MAX_TOKENS="${MAX_TOKENS:-2048}"
export NUM_ENVS="${NUM_ENVS:-32}"
export HISTORY_N="${HISTORY_N:-100}"
export IMAGE_MAX="${IMAGE_MAX:-20}"
export FOLD_SIZE="${FOLD_SIZE:-10}"
export RESULT_DIR="${RESULT_DIR:-./results_qwen35vl_27b_8gpu_sharded}"

exec bash scripts/bash/run_qwen35vl_8gpu_sharded.sh "$@"
