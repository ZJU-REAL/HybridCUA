#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------------
# Evaluate OpenCUA-7B through scripts/bash/osworld/run_opencua.sh.
#
# Thin, model-specific wrapper in the style of run_qwen35vl_27b_8gpu_sharded.sh:
# it only pins the values that differ from the 32B defaults and then defers to the
# shared runner. The Python side is model-agnostic, so nothing under
# scripts/python/osworld/ needs a 7B variant.
#
# Two values differ from run_opencua.sh's 32B defaults, and they must stay in step
# with each other:
#
#   MODEL              OpenCUA-7B      must equal vLLM's --served-model-name or
#                                      every request 404s on model lookup
#   OPENAI_NUM_SHARDS  8              because start_vllm_opencua_7b_8gpu.sh serves
#                                      8 instances x TP=1 on ports 8000..8007.
#                                      Leaving this at 4 would strand half the
#                                      endpoints idle; setting it to 8 against the
#                                      32B's 4 ports would send half the workers to
#                                      dead ports -- the failure this repo has
#                                      already hit once.
#
# Everything else (coord=qwen25, cot_level=l2, history_n=3, max_tokens=4096,
# use_old_sys_prompt=1) already matches what upstream prescribes for OpenCUA-7B:
# its docstring puts `--use_old_sys_prompt` on the 7B and 32B commands alike and
# omits it only for 72B.
#
# Usage:
#   bash scripts/bash/start_vllm_opencua_7b_8gpu.sh
#   bash scripts/bash/osworld/run_opencua_7b.sh
#
# Smoke:
#   NUM_ENVS=2 DOMAIN=os MAX_STEPS=3 RESULT_DIR=./smoke-opencua-7b \
#     bash scripts/bash/osworld/run_opencua_7b.sh
# ----------------------------------------------------------------------------

export MODEL="${MODEL:-OpenCUA-7B}"
export OPENAI_NUM_SHARDS="${OPENAI_NUM_SHARDS:-8}"
export RESULT_DIR="${RESULT_DIR:-./results-opencua-7b-gui}"

exec bash "$(cd "$(dirname "$0")" && pwd)/run_opencua.sh" "$@"
