#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------------
# Start OpenCUA-32B (xlangai/OpenCUA-32B) with vLLM on 8 GPUs.
# Thin wrapper around start_vllm_qwen35_8gpu.sh: 4 OpenAI-compatible servers,
# each TP=2, on ports 8000..8003.
#
# OpenCUA is a Qwen2.5-VL derivative with a custom `OpenCUAForConditionalGeneration`
# arch, but vLLM >= 0.12 ships native support for it
# (vllm/model_executor/models/opencua.py, registered in models/registry.py), and its
# WeightsMapper maps `vision_tower.` -> `visual.` -- exactly this checkpoint's layout.
# No --hf-overrides and no arch rewrite are needed. Verified against the local
# vllm017 env (0.17.0).
#
# --trust-remote-code comes from the shared launcher (:114) and IS required here:
# the tokenizer is `TikTokenV3` (tokenizer_config.json) reached via auto_map, so
# transformers cannot load it without it.
#
# Usage:
#   bash scripts/bash/start_vllm_opencua_32b_8gpu.sh
#
# Override anything via env, e.g.:
#   GPU_GROUPS='0,1,2,3;4,5,6,7' PORT_START=8100 \
#     bash scripts/bash/start_vllm_opencua_32b_8gpu.sh
#
# MAX_MODEL_LEN: the checkpoint advertises max_position_embeddings=262144.
# Serving at that length would size the KV cache far past what 2x97G leaves after a
# 63G model, so we cap at 32768 -- ample for OSWorld episodes, where the agent keeps
# only 3 screenshots (--max_image_history_length) of history per request.
#
# The served name is what scripts/bash/osworld/run_opencua.sh sends as `--model`;
# the two MUST match or every request 404s on model lookup.
#
# OpenCUA-7B: same generic launcher, but it is 28 layers / hidden 3584 (~16G) so it
# fits TP=1. Override both name and topology, and keep them consistent:
#   MODEL_PATH=.../models/OpenCUA-7B SERVED_MODEL_NAME=OpenCUA-7B \
#     GPU_GROUPS='0;1;2;3;4;5;6;7' OPENAI_NUM_SHARDS=8 OPENAI_PORT_START=8000 \
#     bash scripts/bash/start_vllm_opencua_32b_8gpu.sh
# ----------------------------------------------------------------------------

_script_dir="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${_script_dir}/../.." && pwd)"

# ----------------------------------------------------------------------------
# Disable vLLM's AllReduce+RMSNorm fusion pass.
#
# That pass (vllm/compilation/passes/fusion/allreduce_rms_fusion.py) creates a
# FlashInfer allreduce workspace which is a HOST-LEVEL singleton. With four vLLM
# instances on one box only the first acquires it; the rest log
#
#   WARNING [allreduce_rms_fusion.py:779] Failed to initialize FlashInfer All
#   Reduce workspace: [Errno 98] Address already in use. AllReduce fusion pass
#   will be disabled.
#
# and run unfused. That alone is harmless -- but all four shards share ONE
# torch_compile_cache directory (same model, same compilation-config hash), so the
# shard that won the race writes a graph CONTAINING the fused op, and the losers read
# that artifact back and hit
#
#   AssertionError: Flashinfer workspace must be initialized when using flashinfer
#
# inside `capture_model()` at CUDA-graph capture, which kills the engine
# ("Engine core initialization failed"). Observed 2026-09-23: 2 of the 4 shards died
# this way while the other 2 came up fine -- same launcher, same flags, pure race.
#
# Disabling the pass explicitly makes every shard compile identically, so the shared
# cache can never disagree with a shard's own pass state. Nothing is given up: in this
# topology the pass is already disabled on every shard regardless.
#
# Kept here (not in the shared launcher) because it changes the compilation-config
# hash and therefore the cache directory -- it would invalidate the warm cache for
# every other model. No spaces in the JSON: the shared launcher reads this back with
# `read -r -a`, which would split on whitespace.
# ----------------------------------------------------------------------------
_fusion_off='--compilation-config {"pass_config":{"fuse_allreduce_rms":false}}'
export EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:+${EXTRA_VLLM_ARGS} }${_fusion_off}"

# Derive from repo location rather than hard-coding a disk: this tree exists on
# several mounts, each with its own models/ sibling.
export MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/../models/OpenCUA-32B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-OpenCUA-32B}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

exec bash "${_script_dir}/start_vllm_qwen35_8gpu.sh" "$@"
