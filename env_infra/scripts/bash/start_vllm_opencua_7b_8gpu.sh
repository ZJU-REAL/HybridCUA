#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------------
# Start OpenCUA-7B (xlangai/OpenCUA-7B) with vLLM on 8 GPUs.
# Thin wrapper around start_vllm_qwen35_8gpu.sh: 8 OpenAI-compatible servers,
# each TP=1, on ports 8000..8007.
#
# Qwen2.5-VL-7B shape (28 layers, hidden 3584, 28 heads / 4 KV heads, vocab 152064)
# and ~16G on disk, so a single H20 holds the weights with room to spare -- no need
# to spend two cards on TP=2 the way the 32B does. 8 independent replicas instead of
# 4 is twice the concurrency for the same 8 GPUs, which is why the small-model
# wrappers in this repo all use GPU_GROUPS='0;1;2;3;4;5;6;7' (see
# start_vllm_qwen35_9b_8gpu.sh). Pair it with OPENAI_NUM_SHARDS=8 -- see
# scripts/bash/osworld/run_opencua_7b.sh, which sets that for you.
#
# The vLLM model file's own docstring reads "Inference-only OpenCUA-7B model
# compatible with HuggingFace weights" -- 7B was the original target of that
# integration, and 32B was added to it later, so this path is the better-trodden one.
#
# Same as the 32B wrapper otherwise: trust-remote-code is required (TikTokenV3
# tokenizer reached via auto_map) and comes from the shared launcher, and the
# AllReduce fusion pass is disabled for the host-singleton workspace race (see the
# long comment in start_vllm_opencua_32b_8gpu.sh -- the reasoning is model-independent).
#
# MAX_MODEL_LEN: the checkpoint advertises max_position_embeddings=128000. Same cap
# as the 32B: OSWorld episodes keep only 3 screenshots of history, so 32768 is ample
# and avoids sizing a KV cache nobody will use.
#
# Usage:
#   bash scripts/bash/start_vllm_opencua_7b_8gpu.sh
#
# Override anything via env, e.g. the 4-shard TP=2 layout instead:
#   GPU_GROUPS='0,1;2,3;4,5;6,7' OPENAI_NUM_SHARDS=4 \
#     bash scripts/bash/start_vllm_opencua_7b_8gpu.sh
# ----------------------------------------------------------------------------

_script_dir="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${_script_dir}/../.." && pwd)"

# See start_vllm_opencua_32b_8gpu.sh for the full explanation. Short version: the
# FlashInfer allreduce workspace is a per-host singleton, so with several vLLM
# instances per box all but one disable the AllReduce fusion pass; the shared
# torch_compile_cache then hands the losers a graph containing the fused op, and
# CUDA-graph capture asserts. Forcing the pass off makes every shard compile
# identically. No spaces in the JSON: the shared launcher re-splits it via `read -a`.
_fusion_off='--compilation-config {"pass_config":{"fuse_allreduce_rms":false}}'
export EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:+${EXTRA_VLLM_ARGS} }${_fusion_off}"

# Derive from repo location rather than hard-coding a disk: this tree exists on
# several mounts, each with its own models/ sibling.
export MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/../models/OpenCUA-7B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-OpenCUA-7B}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export GPU_GROUPS="${GPU_GROUPS:-0;1;2;3;4;5;6;7}"

exec bash "${_script_dir}/start_vllm_qwen35_8gpu.sh" "$@"
