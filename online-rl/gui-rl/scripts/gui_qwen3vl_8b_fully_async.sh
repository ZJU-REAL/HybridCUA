#!/bin/bash
# FULLY-ASYNC GUI-RL with a REMOTE OSWorld env server.
# Single node, 8x GPU: actor 4 (TP=4) + rollout 4 (4 SGLang engines x 1 GPU).
#
# This is the fully-async counterpart of scripts/gui_qwen3vl_8b.sh.
# It stacks TWO orthogonal forms of asynchrony:
#
#   Layer 1 (process pool, GIL/CPU): rollout/ runs each GUI trajectory in
#     its own worker process, so the per-trajectory synchronous CPU work
#     (env step, image processing, build_train_data) never blocks the rollout
#     event loop. This layer is INHERITED unchanged from gui_qwen3vl_8b.sh:
#       --custom-generate-function-path rollout.partial_async_gui_rollout.generate
#       GUI_FAST_ROLLOUT_PROCS = N worker processes.
#
#   Layer 2 (in-flight pool, long-tail): slime's fully-async worker keeps a
#     FIXED pool of in-flight trajectories across rollout boundaries, so the
#     next training step does not wait for the slowest in-flight episode.
#     This is the ONLY thing added on top of the fast.sh baseline:
#       --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
#
# Three concurrency knobs are aligned to the SAME number (default 64):
#   in-flight pool = sglang_server_concurrency x num_engines
#   process pool   = GUI_FAST_ROLLOUT_PROCS
#   env sessions   = GUI_TRAJECTORY_CONCURRENCY  (remote OSWorld cluster capacity)
# The real ceiling is min() of the three; keeping them equal removes hidden
# bottlenecks. With 4 engines: sglang_server_concurrency = 64 / 4 = 16.
#
# Eval is OFF by default (GUI_EVAL_INTERVAL=0). fully-async training cannot eval
# (generate_rollout_fully_async raises on evaluation=True) and a periodic eval
# contends with the in-flight pool for worker/env slots. Opt in with
# GUI_EVAL_INTERVAL=<n>; it then uses its own --eval-function-path.
#
# Usage:  bash scripts/gui_qwen3vl_8b_fully_async.sh
# Requires a reachable remote env server at GUI_ENV_SERVER_URL.

pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3
pkill -9 ray || true
pkill -9 python || true

set -ex

: "${WANDB_API_KEY:?ERROR: WANDB_API_KEY must be set. Get your key from https://wandb.ai/authorize}"
export WANDB_API_KEY
export WANDB_BASE_URL=${WANDB_BASE_URL:-"https://api.wandb.ai"}

# SCRIPT_DIR = gui-rl/ (scripts/.. resolves to the package root).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../slime" &>/dev/null && pwd)"
MODEL_ARGS_ROTARY_BASE=5000000 source "${SLIME_DIR}/scripts/models/qwen3-8B.sh"
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-"${SCRIPT_DIR}/../Megatron-LM"}
CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-"${SCRIPT_DIR}/scripts/gui_partial_async.yaml"}

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

export RAY_health_check_failure_threshold=${RAY_health_check_failure_threshold:-20}
export RAY_health_check_period_ms=${RAY_health_check_period_ms:-5000}
export RAY_health_check_timeout_ms=${RAY_health_check_timeout_ms:-30000}
export RAY_num_heartbeats_timeout=${RAY_num_heartbeats_timeout:-60}

NUM_GPUS=${NUM_GPUS:-8}
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}

if (( ACTOR_GPUS + ROLLOUT_GPUS > NUM_GPUS )); then
  echo "ACTOR_GPUS + ROLLOUT_GPUS must be <= NUM_GPUS"
  echo "ACTOR_GPUS=${ACTOR_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, NUM_GPUS=${NUM_GPUS}"
  exit 1
fi

# Remote env server (OSWorld cluster, /v1/sessions protocol). Override as needed.
export GUI_ENV_SERVER_URL=${GUI_ENV_SERVER_URL:-"http://127.0.0.1:18000"}
# GUI_ENV_CLIENT is intentionally NOT set here: aligned with gui_qwen3vl_8b.sh,
# which leaves it unset so the rollout worker falls back to the legacy
# lease-HTTP GuiEnvClient (matches the /allocate protocol on port 18000). Set
# GUI_ENV_CLIENT=session only if pointing at a /v1/sessions server (e.g. :19000).
# Concurrent GUI env sessions (independent from sglang concurrency). Under
# fully-async the worker keeps a fixed in-flight pool, so this cap is what
# actually protects the remote env cluster. Aligned to the in-flight pool below.
export GUI_POOL_MAX_ENVS=${GUI_POOL_MAX_ENVS:-64}
export GUI_TRAJECTORY_CONCURRENCY=${GUI_TRAJECTORY_CONCURRENCY:-64}
# Fast multiprocess rollout (rollout/): pool size = N worker processes,
# one trajectory per process. Disable the legacy Ray TrajectoryDispatcher.
export GUI_ROLLOUT_WORKERS=1
export GUI_FAST_ROLLOUT_PROCS=${GUI_FAST_ROLLOUT_PROCS:-64}
export GUI_ACTION_SPACE=${GUI_ACTION_SPACE:-"pyautogui"}
export GUI_OBSERVATION_TYPE=${GUI_OBSERVATION_TYPE:-"screenshot"}
export GUI_COORDINATE_TYPE=${GUI_COORDINATE_TYPE:-"relative"}
export GUI_AGENT_CLASS_PATH=${GUI_AGENT_CLASS_PATH:-"agents.qwen3vl_agent.Qwen3VLAgentLocal"}
MULTIMODAL_KEYS=${MULTIMODAL_KEYS:-'{"image":"images"}'}
# GUI rollout/eval step counts etc. come from CUSTOM_CONFIG_PATH (see yaml).

WANDB_PROJECT=${WANDB_PROJECT:-slime_gui}
WANDB_GROUP=${WANDB_GROUP:-qwen3-8b-fully-async-fast}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
GUI_PROJECT_NAME=${GUI_PROJECT_NAME:-slime_gui_8b_fully_async_fast_${RUN_TIMESTAMP}}
export OSWORLD_PROJECT="${GUI_PROJECT_NAME}"
export GUI_RESULT_DIR=${GUI_RESULT_DIR:-"${SCRIPT_DIR}/results"}
export GUI_RESULT_DIR="${GUI_RESULT_DIR}/${GUI_PROJECT_NAME}"
export GUI_TEST_CONFIG_BASE_DIR=${GUI_TEST_CONFIG_BASE_DIR:-"${SCRIPT_DIR}/evaluation_examples"}
export GUI_TRAIN_META_PATH=${GUI_TRAIN_META_PATH:-"${GUI_TEST_CONFIG_BASE_DIR}/train_nochrome.json"}
export GUI_EVAL_META_PATH=${GUI_EVAL_META_PATH:-"${GUI_TEST_CONFIG_BASE_DIR}/test_nochrome.json"}
# RLVR (CUA-Gym) task data. GUI_CUA_GYM_TASKS_META is an OSWorld-shaped
# {app_type: [bundle_uuid, ...]} map; each uuid resolves to a bundle dir under
# GUI_CUA_GYM_BUNDLES carrying task.json/config.json + reward.py.
CUA_GYM_DATA=${CUA_GYM_DATA:-"${SCRIPT_DIR}/../../env_infra/cua_gym_data"}
export GUI_CUA_GYM_BUNDLES=${GUI_CUA_GYM_BUNDLES:-"${CUA_GYM_DATA}/rlvr"}
export GUI_ENV_RUNTIME=${GUI_ENV_RUNTIME:-"cua_gym"}
export GUI_CUA_GYM_TASKS_META=${GUI_CUA_GYM_TASKS_META:-"${CUA_GYM_DATA}/rlvr_curriculum_1000_meta.json"}

export GUI_DATA_SOURCE_PATH=${GUI_DATA_SOURCE_PATH:-"data.gui_data_source.CuaGymDataSource"}

if [[ -n "${GUI_RESULT_DIR}" && "${GUI_RESULT_DIR}" != "/" ]]; then
  rm -rf "${GUI_RESULT_DIR}"
fi
mkdir -p "${GUI_RESULT_DIR}"

export download_proxy=${download_proxy:-}

HF_CKPT=${HF_CKPT:-/mnt/llmshared-ssd-hd/models/Qwen3-VL-8B-Instruct}
REF_LOAD=${REF_LOAD:-${HF_CKPT}}

if [[ -z "${HF_CKPT}" ]]; then
  echo "Set HF_CKPT to your Qwen3-VL-8B checkpoint path"
  exit 1
fi
if [[ ! -e "${HF_CKPT}" ]]; then
  echo "HF_CKPT does not exist: ${HF_CKPT}"
  exit 1
fi

CKPT_ROOT=${CKPT_ROOT:-"${SCRIPT_DIR}/../ckpt"}
CKPT_NAME=${CKPT_NAME:-"gui-qwen3vl-8b-fully-async-fast"}
SAVE_CKPT=${SAVE_CKPT:-"${CKPT_ROOT}/${CKPT_NAME}_${RUN_TIMESTAMP}"}
SAVE_HF_CKPT=${SAVE_HF_CKPT:-"${CKPT_ROOT}/${CKPT_NAME}_${RUN_TIMESTAMP}_hf/rollout_{rollout_id}"}
echo "Megatron checkpoint dir: ${SAVE_CKPT}"
echo "HuggingFace checkpoint template: ${SAVE_HF_CKPT}"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --save "${SAVE_CKPT}"
  --save-hf "${SAVE_HF_CKPT}"
  --save-interval 20
)

ENABLE_RESUME_LOAD=${ENABLE_RESUME_LOAD:-0}
if [[ "${ENABLE_RESUME_LOAD}" == "1" ]]; then
  if [[ -z "${RESUME_LOAD:-}" ]]; then
    echo "Set RESUME_LOAD to an existing Megatron checkpoint dir when ENABLE_RESUME_LOAD=1"
    exit 1
  fi
  CKPT_ARGS+=(--load "${RESUME_LOAD}")
fi

ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}

# ===========================================================================
# FULLY-ASYNC CHANGE #1: switch the train rollout scheduler to slime's
# fully-async worker. This is the only flag that distinguishes this script
# from the (already process-pooled) gui_qwen3vl_8b.sh baseline.
# ===========================================================================
ROLLOUT_ARGS=(
  --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
  --data-source-path ${GUI_DATA_SOURCE_PATH:-data.gui_data_source.CuaGymDataSource}
  --reward-key score
  --num-rollout 1000
  --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
  --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
  --rollout-max-response-len 1024
  --rollout-temperature 1.0
  --num-steps-per-rollout 1
)

# ===========================================================================
# FULLY-ASYNC CHANGE #2: pin the in-flight pool so it equals the process pool
# and env-session cap. in-flight = sglang_server_concurrency x num_engines.
# Target 64 with 4 engines -> sglang_server_concurrency = 16.
# (Without this, fully-async uses the sglang default 512 -> 512x4 = 2048 in
#  flight, which would flood both the process pool queue and the env cluster.)
# ===========================================================================
NUM_ENGINES=$(( ROLLOUT_GPUS / ROLLOUT_NUM_GPUS_PER_ENGINE ))
if (( NUM_ENGINES < 1 )); then NUM_ENGINES=1; fi
TARGET_IN_FLIGHT=${TARGET_IN_FLIGHT:-64}
SGLANG_SERVER_CONCURRENCY=${SGLANG_SERVER_CONCURRENCY:-$(( (TARGET_IN_FLIGHT + NUM_ENGINES - 1) / NUM_ENGINES ))}
if (( SGLANG_SERVER_CONCURRENCY < 1 )); then SGLANG_SERVER_CONCURRENCY=1; fi

IN_FLIGHT_SAMPLES_ESTIMATE=$(( ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT ))
echo "Configured rollout-batch-size x n-samples-per-prompt = ${IN_FLIGHT_SAMPLES_ESTIMATE}"
echo "fully-async in-flight pool = sglang_server_concurrency(${SGLANG_SERVER_CONCURRENCY}) x num_engines(${NUM_ENGINES}) = $(( SGLANG_SERVER_CONCURRENCY * NUM_ENGINES ))"
echo "process pool GUI_FAST_ROLLOUT_PROCS=${GUI_FAST_ROLLOUT_PROCS}, env cap GUI_TRAJECTORY_CONCURRENCY=${GUI_TRAJECTORY_CONCURRENCY} (aim: all three equal)"
echo "Using remote GUI env server: ${GUI_ENV_SERVER_URL}"
echo "Injecting custom config: ${CUSTOM_CONFIG_PATH}"

# --gui-eval-* come from CUSTOM_CONFIG_PATH, not here.
# online-rl/slime requires a non-empty args.eval_datasets whenever --eval-interval
# is set. GUI eval does NOT consume these datasets (the real tasks come from
# GUI_EVAL_META_PATH via our custom --eval-function-path); --eval-config only
# satisfies slime's validation so the periodic eval hook fires.
# Eval is OFF by default under fully-async: the in-flight pool keeps trajectories
# alive across rollout boundaries, so a periodic eval contends with in-flight
# training samples for the same worker/env slots. Save ckpts and eval offline.
# Opt in with GUI_EVAL_INTERVAL=<n>; that routes eval through its own entrypoint
# (fully-async training itself raises on evaluation=True).
GUI_EVAL_CONFIG=${GUI_EVAL_CONFIG:-"${SCRIPT_DIR}/scripts/gui_eval_dataset.yaml"}
GUI_EVAL_INTERVAL=${GUI_EVAL_INTERVAL:-0}
if (( GUI_EVAL_INTERVAL > 0 )); then
  EVAL_ARGS=(
    --eval-temperature 0.0
    --n-samples-per-eval-prompt 3
    --eval-interval "${GUI_EVAL_INTERVAL}"
    --eval-config "${GUI_EVAL_CONFIG}"
    --eval-reward-key acc
    --eval-function-path rollout.fully_async_rollout.eval_rollout_fully_async
  )
  echo "[EVAL] enabled: interval=${GUI_EVAL_INTERVAL}"
else
  echo "Eval DISABLED (GUI_EVAL_INTERVAL=0). Eval ckpts offline instead."
  EVAL_ARGS=()
fi

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

PERF_ARGS=(
  --tensor-model-parallel-size 4
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --megatron-to-hf-mode bridge
  --use-dynamic-batch-size
  --max-tokens-per-gpu 1024
)

# --dynamic_history is injected via CUSTOM_CONFIG_PATH (not a valid upstream flag).
GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-type low_var_kl
  --kl-loss-coef 0.01
  --loss-mask-type qwen3
)

# FULLY-ASYNC CHANGE #2 (cont.): the pinned --sglang-server-concurrency.
SGLANG_ARGS=(
  --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}
  --sglang-mem-fraction-static 0.72
  --sglang-server-concurrency ${SGLANG_SERVER_CONCURRENCY}
)

# Refactored entrypoints. No --custom-rollout-log-function-path: the online-rl
# tree has no gui_rollout_logging module.
# --custom-generate-function-path STAYS on rollout (the process pool) — this
# is what fully-async dispatches each trajectory through. Do NOT point it at the
# single-process rollout.partial_async_rollout_gui.generate.
CUSTOM_ARGS=(
  --custom-generate-function-path rollout.partial_async_gui_rollout.generate
  --custom-rm-path reward.reward_func.reward_func
  --custom-config-path "${CUSTOM_CONFIG_PATH}"
)

WANDB_ARGS=(
  --use-wandb
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
)
WANDB_KEY_VALUE=${WANDB_KEY:-${WANDB_API_KEY:-}}
if [[ -n "${WANDB_KEY_VALUE}" ]]; then
  WANDB_ARGS+=(--wandb-key "${WANDB_KEY_VALUE}")
fi
if [[ -n "${WANDB_BASE_URL:-}" ]]; then
  WANDB_ARGS+=(--wandb-host "${WANDB_BASE_URL}")
fi

for i in {1..60}; do
  if curl -fsS "${GUI_ENV_SERVER_URL}/healthz" >/dev/null 2>&1; then
    echo "Remote GUI env server is ready: ${GUI_ENV_SERVER_URL}"
    break
  fi
  sleep 2
  if (( i == 60 )); then
    echo "Timed out waiting for remote GUI env server: ${GUI_ENV_SERVER_URL}"
    exit 1
  fi
done

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:2048
export RAY_object_spilling_threshold=0.80
export RAY_local_fs_capacity_threshold=0.99

RAY_TEMP_DIR=${RAY_TEMP_DIR:-"/mnt/llmshared-ssd-hd/chentongbo/ray"}
mkdir -p "${RAY_TEMP_DIR}"
ray start --head --num-gpus "${NUM_GPUS}" --temp-dir "${RAY_TEMP_DIR}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

echo "Verifying Ray cluster GPUs..."
ray status

RAY_JOB_SUBMISSION_ID=${RAY_JOB_SUBMISSION_ID:-"gui_qwen3vl_8b_fully_async_fast_$(date +%Y%m%d_%H%M%S)"}

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}:${SCRIPT_DIR}:${SLIME_DIR}\",
    \"PYTHONUNBUFFERED\": \"${PYTHONUNBUFFERED}\",
    \"PYTHONFAULTHANDLER\": \"${PYTHONFAULTHANDLER}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\",
    \"GUI_ENV_SERVER_URL\": \"${GUI_ENV_SERVER_URL}\",
    \"GUI_ENV_CLIENT\": \"${GUI_ENV_CLIENT}\",
    \"GUI_POOL_MAX_ENVS\": \"${GUI_POOL_MAX_ENVS}\",
    \"GUI_TRAJECTORY_CONCURRENCY\": \"${GUI_TRAJECTORY_CONCURRENCY}\",
    \"GUI_ROLLOUT_WORKERS\": \"${GUI_ROLLOUT_WORKERS}\",
    \"GUI_FAST_ROLLOUT_PROCS\": \"${GUI_FAST_ROLLOUT_PROCS}\",
    \"GUI_RESULT_DIR\": \"${GUI_RESULT_DIR}\",
    \"GUI_COORDINATE_TYPE\": \"${GUI_COORDINATE_TYPE}\",
    \"GUI_ACTION_SPACE\": \"${GUI_ACTION_SPACE}\",
    \"GUI_OBSERVATION_TYPE\": \"${GUI_OBSERVATION_TYPE}\",
    \"GUI_TEST_CONFIG_BASE_DIR\": \"${GUI_TEST_CONFIG_BASE_DIR}\",
    \"GUI_TRAIN_META_PATH\": \"${GUI_TRAIN_META_PATH}\",
    \"GUI_ENV_RUNTIME\": \"${GUI_ENV_RUNTIME}\",
    \"GUI_DATA_SOURCE_PATH\": \"${GUI_DATA_SOURCE_PATH}\",
    \"GUI_CUA_GYM_BUNDLES\": \"${GUI_CUA_GYM_BUNDLES}\",
    \"GUI_CUA_GYM_TASKS_META\": \"${GUI_CUA_GYM_TASKS_META}\",
    \"GUI_EVAL_META_PATH\": \"${GUI_EVAL_META_PATH}\",
    \"OSWORLD_PROJECT\": \"${OSWORLD_PROJECT}\",
    \"download_proxy\": \"${download_proxy}\",
    \"GUI_AGENT_CLASS_PATH\": \"${GUI_AGENT_CLASS_PATH}\",
    \"HF_CKPT\": \"${HF_CKPT}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\",
    \"WANDB_BASE_URL\": \"${WANDB_BASE_URL:-}\",
    \"WANDB_PROJECT\": \"${WANDB_PROJECT}\",
    \"GUI_USER_ID\": \"${GUI_USER_ID:-fully_async_fast}\",
    \"GUI_JOB_ID\": \"${RAY_JOB_SUBMISSION_ID}\"
  }
}"

TRAIN_ENTRY=${TRAIN_ENTRY:-"${SLIME_DIR}/train_async.py"}

ray job submit --address="http://127.0.0.1:8265" \
  --submission-id "${RAY_JOB_SUBMISSION_ID}" \
  --no-wait \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 -u "${TRAIN_ENTRY}" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node ${ACTOR_GPUS} \
  --rollout-num-gpus ${ROLLOUT_GPUS} \
  --multimodal-keys "${MULTIMODAL_KEYS}" \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${EVAL_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${GRPO_ARGS[@]} \
  ${ROUTER_ARGS[@]} \
  ${SGLANG_ARGS[@]} \
  ${WANDB_ARGS[@]} \
  ${CUSTOM_ARGS[@]}

echo "Following live Ray logs for ${RAY_JOB_SUBMISSION_ID}"
set +e
ray job logs --address="http://127.0.0.1:8265" "${RAY_JOB_SUBMISSION_ID}" -f --log-style=record
RAY_LOG_EXIT=$?
RAY_STATUS_OUTPUT=$(ray job status --address="http://127.0.0.1:8265" "${RAY_JOB_SUBMISSION_ID}" --log-style=record 2>&1)
echo "${RAY_STATUS_OUTPUT}"
set -e

if [[ "${RAY_STATUS_OUTPUT}" == *"SUCCEEDED"* ]]; then
  exit 0
fi

echo "Ray job failed (submission id: ${RAY_JOB_SUBMISSION_ID}, logs exit: ${RAY_LOG_EXIT})"
exit 1
