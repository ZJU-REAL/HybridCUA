#!/bin/bash
# HybridCUA-9B fully-async GRPO, 40 GPUs = 5 nodes x 8.
#   Actor:   node0..node3 = 32 GPU (TP=4, DP=8)
#   Rollout: node4        =  8 GPU (8 sglang engines x 1 GPU)
#
# Config kept in step with HybridCUA-9B_24gpu_fully_async.sh; only the SCALE
# differs. Aligned: OSWorld env runtime, GuiMetaDataSource, and the reward
# post-process that consumes lambda_cli/lambda_exec from
# gui_partial_async_osworld.yaml.
#
# 用法:
#   bash scripts/HybridCUA-9B_40gpu_fully_async.sh                                    # head (actor node0)
#   RAY_HEAD_ADDR=<head_ip> WORKER_NUM_GPUS=8 bash scripts/gpu_worker_join_ray.sh      # 其余 4 个节点各跑一次

if [[ "${BACKGROUND:-1}" == "1" && -z "${_HYBRIDCUA40_DETACHED:-}" ]]; then
  _RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
  mkdir -p "${_RUN_DIR}/logs"
  LOG_FILE="${LOG_FILE:-${_RUN_DIR}/logs/hybridcua40_$(date +%Y%m%d_%H%M%S).log}"
  _HYBRIDCUA40_DETACHED=1 setsid bash "$0" "$@" > "${LOG_FILE}" 2>&1 < /dev/null &
  echo "Started HybridCUA-9B 40gpu in background: PID=$!"
  echo "Log:  ${LOG_FILE}"
  echo "Tail: tail -f ${LOG_FILE}"
  echo "Stop: ray job stop <submission_id> --address=http://127.0.0.1:8265   (or ray stop --force)"
  exit 0
fi

GUI_ENV_SERVER_PROC_PATTERN=${GUI_ENV_SERVER_PROC_PATTERN:-cluster.master.server}
kill_stale_python() {
  local pid
  for pid in $(pgrep -f python 2>/dev/null); do
    if ! tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null | grep -q "${GUI_ENV_SERVER_PROC_PATTERN}"; then
      kill -9 "${pid}" 2>/dev/null || true
    fi
  done
}

pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
kill_stale_python
sleep 3
pkill -9 ray || true
kill_stale_python

set -ex

: "${WANDB_API_KEY:?ERROR: WANDB_API_KEY must be set. Get your key from https://wandb.ai/authorize}"
export WANDB_API_KEY
export WANDB_BASE_URL=${WANDB_BASE_URL:-"https://api.wandb.ai"}

USE_STAR_PROXY=${USE_STAR_PROXY:-0}
if [[ "${USE_STAR_PROXY}" == "1" ]]; then
  STAR_PROXY_URL=${STAR_PROXY_URL:-""}
  [[ -z "${STAR_PROXY_URL}" ]] && { echo "ERROR: USE_STAR_PROXY=1 requires STAR_PROXY_URL (e.g., http://your-proxy.example.com:3128)"; exit 1; }
  LOCAL_IPS=$(hostname -I 2>/dev/null | tr ' ' ',' | sed 's/,\+/,/g; s/^,//; s/,$//')
  STAR_NO_PROXY=${STAR_NO_PROXY:-"localhost,127.0.0.1,::1,${LOCAL_IPS},<node-ip>/8,<node-ip>/12,<node-ip>/16"}
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
  GUI_JOB_HTTP_PROXY="${STAR_PROXY_URL}"
  GUI_JOB_NO_PROXY="${STAR_NO_PROXY}"
  echo "star-proxy for TRAINING JOB ONLY (wandb egress): ${STAR_PROXY_URL}; ray/shell stay proxy-free"
  echo "job no_proxy: ${GUI_JOB_NO_PROXY}"
else
  GUI_JOB_HTTP_PROXY=""
  GUI_JOB_NO_PROXY=""
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

NUMA_LIB_DIR=""
VENDORED_NUMA="${SCRIPT_DIR}/../vendor_libs"
if ! ldconfig -p | grep -q libnuma; then
  if [[ -f "${VENDORED_NUMA}/libnuma.so.1" ]]; then
    NUMA_LIB_DIR="${VENDORED_NUMA}"
    export LD_LIBRARY_PATH="${NUMA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
    echo "libnuma: using vendored copy at ${NUMA_LIB_DIR}"
  else
    if command -v apt-get &>/dev/null; then
      apt-get update -qq && apt-get install -y -qq libnuma1 libnuma-dev 2>/dev/null || true
    elif command -v yum &>/dev/null; then
      yum install -y numactl-libs 2>/dev/null || true
    fi
    if ! ldconfig -p | grep -q libnuma; then
      NUMA_PATH=$(find /usr /opt /mnt -name "libnuma.so.1" 2>/dev/null | head -1)
      if [[ -n "${NUMA_PATH}" ]]; then
        NUMA_LIB_DIR="$(dirname "${NUMA_PATH}")"
        export LD_LIBRARY_PATH="${NUMA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
      fi
    fi
  fi
fi

SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../slime" &>/dev/null && pwd)"
source "${SLIME_DIR}/scripts/models/qwen3.5-9B.sh"
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-"${SCRIPT_DIR}/../Megatron-LM"}
CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-"${SCRIPT_DIR}/scripts/gui_partial_async_osworld.yaml"}

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

export RAY_health_check_failure_threshold=${RAY_health_check_failure_threshold:-20}
export RAY_health_check_period_ms=${RAY_health_check_period_ms:-5000}
export RAY_health_check_timeout_ms=${RAY_health_check_timeout_ms:-30000}
export RAY_num_heartbeats_timeout=${RAY_num_heartbeats_timeout:-60}

NUM_GPUS=${NUM_GPUS:-40}
ACTOR_GPUS=${ACTOR_GPUS:-32}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-8}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-4}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}

if (( ACTOR_GPUS + ROLLOUT_GPUS > NUM_GPUS )); then
  echo "ACTOR_GPUS + ROLLOUT_GPUS must be <= NUM_GPUS"
  echo "ACTOR_GPUS=${ACTOR_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, NUM_GPUS=${NUM_GPUS}"
  exit 1
fi

export GUI_ENV_SERVER_URL=${GUI_ENV_SERVER_URL:-"http://127.0.0.1:19000"}
export GUI_ENV_CLIENT=${GUI_ENV_CLIENT:-session}
export GUI_POOL_MAX_ENVS=${GUI_POOL_MAX_ENVS:-64}
export GUI_TRAJECTORY_CONCURRENCY=${GUI_TRAJECTORY_CONCURRENCY:-64}
export GUI_ROLLOUT_WORKERS=1
export GUI_FAST_ROLLOUT_PROCS=${GUI_FAST_ROLLOUT_PROCS:-64}
export GUI_LOG_LEVEL=${GUI_LOG_LEVEL:-INFO}
export GUI_ACTION_SPACE=${GUI_ACTION_SPACE:-"pyautogui"}
export GUI_OBSERVATION_TYPE=${GUI_OBSERVATION_TYPE:-"screenshot"}
export GUI_COORDINATE_TYPE=${GUI_COORDINATE_TYPE:-"relative"}
export GUI_AGENT_CLASS_PATH=${GUI_AGENT_CLASS_PATH:-"agents.qwen35_hybrid_cua.Qwen35HybridCuaAgentLocal"}
export GUI_COORD_SHIM=${GUI_COORD_SHIM:-1}
export GUI_ENV_RUNTIME=${GUI_ENV_RUNTIME:-"cua_gym"}
export ENABLE_THINKING=${ENABLE_THINKING:-False}
export GUI_DISABLE_MTP=${GUI_DISABLE_MTP:-1}
MULTIMODAL_KEYS=${MULTIMODAL_KEYS:-'{"image":"images"}'}

WANDB_PROJECT=${WANDB_PROJECT:-slime_gui}
WANDB_GROUP=${WANDB_GROUP:-hybridcua-9b-40gpu}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
GUI_PROJECT_NAME=${GUI_PROJECT_NAME:-slime_gui_hybridcua_9b_40gpu_${RUN_TIMESTAMP}}
export GUI_USER_ID="${GUI_USER_ID:-hybridcua_9b_40gpu}_${RUN_TIMESTAMP}"
export OSWORLD_PROJECT="${GUI_PROJECT_NAME}"
export GUI_RESULT_DIR=${GUI_RESULT_DIR:-"${SCRIPT_DIR}/results"}
export GUI_RESULT_DIR="${GUI_RESULT_DIR}/${GUI_PROJECT_NAME}"
export GUI_TEST_CONFIG_BASE_DIR=${GUI_TEST_CONFIG_BASE_DIR:-"${SCRIPT_DIR}/evaluation_examples"}
export GUI_TRAIN_META_PATH=${GUI_TRAIN_META_PATH:-"${GUI_TEST_CONFIG_BASE_DIR}/train_nochrome.json"}
export GUI_EVAL_META_PATH=${GUI_EVAL_META_PATH:-"${GUI_TEST_CONFIG_BASE_DIR}/test_nochrome.json"}

export GUI_DATA_SOURCE_PATH=${GUI_DATA_SOURCE_PATH:-"data.gui_data_source.CuaGymDataSource"}
CUA_GYM_DATA=${CUA_GYM_DATA:-"${SCRIPT_DIR}/../../env_infra/cua_gym_data"}
export GUI_CUA_GYM_BUNDLES=${GUI_CUA_GYM_BUNDLES:-"${CUA_GYM_DATA}/rlvr"}
export GUI_CUA_GYM_TASKS_META=${GUI_CUA_GYM_TASKS_META:-"${CUA_GYM_DATA}/rlvr_curriculum_1000_meta.json"}

if [[ -n "${GUI_RESULT_DIR}" && "${GUI_RESULT_DIR}" != "/" ]]; then
  rm -rf "${GUI_RESULT_DIR}"
fi
mkdir -p "${GUI_RESULT_DIR}"

export download_proxy=${download_proxy:-}

: "${HF_CKPT:?ERROR: HF_CKPT must be set to your Qwen3.5-9B checkpoint path}"
REF_LOAD=${REF_LOAD:-${HF_CKPT}}

if [[ -z "${HF_CKPT}" ]]; then
  echo "Set HF_CKPT to your Qwen3.5-9B checkpoint path"
  exit 1
fi
if [[ ! -e "${HF_CKPT}" ]]; then
  echo "HF_CKPT does not exist: ${HF_CKPT}"
  exit 1
fi

CKPT_ROOT=${CKPT_ROOT:-"${SCRIPT_DIR}/../ckpt"}
CKPT_NAME=${CKPT_NAME:-"gui-hybridcua-9b-40gpu"}
SAVE_CKPT=${SAVE_CKPT:-"${CKPT_ROOT}/${CKPT_NAME}_${RUN_TIMESTAMP}"}
SAVE_HF_CKPT=${SAVE_HF_CKPT:-"${CKPT_ROOT}/${CKPT_NAME}_${RUN_TIMESTAMP}_hf/rollout_{rollout_id}"}
echo "Megatron checkpoint dir: ${SAVE_CKPT}"
echo "HuggingFace checkpoint template: ${SAVE_HF_CKPT}"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --save "${SAVE_CKPT}"
  --save-hf "${SAVE_HF_CKPT}"
  --save-interval ${SAVE_INTERVAL:-10}
)
if [[ "${NO_SAVE_OPTIM:-0}" == "1" ]]; then
  CKPT_ARGS+=(--no-save-optim)
fi

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

NUM_ROLLOUT=${NUM_ROLLOUT:-1000}
ROLLOUT_ARGS=(
  --rollout-function-path rollout.fully_async_rollout.generate_rollout_fully_async
  --data-source-path ${GUI_DATA_SOURCE_PATH:-data.gui_data_source.CuaGymDataSource}
  --reward-key score
  --num-rollout ${NUM_ROLLOUT}
  --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
  --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
  --rollout-max-response-len 4096
  --rollout-temperature 1.0
  --num-steps-per-rollout 1
)

ROLLOUT_MAX_STALENESS=${ROLLOUT_MAX_STALENESS:-2}
if [[ -n "${ROLLOUT_MAX_STALENESS}" && "${ROLLOUT_MAX_STALENESS}" != "0" ]]; then
  ROLLOUT_ARGS+=(--rollout-max-staleness ${ROLLOUT_MAX_STALENESS})
  echo "Staleness filter ENABLED: --rollout-max-staleness ${ROLLOUT_MAX_STALENESS} (>=999 = observe-only, never drops)"
else
  echo "Staleness filter DISABLED (ROLLOUT_MAX_STALENESS empty/0 -> no curl, no staleness.jsonl)."
fi

NUM_ENGINES=$(( ROLLOUT_GPUS / ROLLOUT_NUM_GPUS_PER_ENGINE ))
if (( NUM_ENGINES < 1 )); then NUM_ENGINES=1; fi
TARGET_IN_FLIGHT=${TARGET_IN_FLIGHT:-64}
SGLANG_SERVER_CONCURRENCY=${SGLANG_SERVER_CONCURRENCY:-$(( (TARGET_IN_FLIGHT + NUM_ENGINES - 1) / NUM_ENGINES ))}
if (( SGLANG_SERVER_CONCURRENCY < 1 )); then SGLANG_SERVER_CONCURRENCY=1; fi

IN_FLIGHT_SAMPLES_ESTIMATE=$(( ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT ))
echo "Configured rollout-batch-size x n-samples-per-prompt = ${IN_FLIGHT_SAMPLES_ESTIMATE}"
echo "fully-async in-flight pool = sglang_server_concurrency(${SGLANG_SERVER_CONCURRENCY}) x num_engines(${NUM_ENGINES}) = $(( SGLANG_SERVER_CONCURRENCY * NUM_ENGINES ))"
echo "process pool GUI_FAST_ROLLOUT_PROCS=${GUI_FAST_ROLLOUT_PROCS}, env cap GUI_TRAJECTORY_CONCURRENCY=${GUI_TRAJECTORY_CONCURRENCY} (aim: all three equal)"
echo "Using GUI env server: ${GUI_ENV_SERVER_URL}"
echo "Injecting custom config: ${CUSTOM_CONFIG_PATH}"

GUI_EVAL_CONFIG=${GUI_EVAL_CONFIG:-"${SCRIPT_DIR}/scripts/gui_eval_dataset.yaml"}
GUI_EVAL_INTERVAL=${GUI_EVAL_INTERVAL:-0}
if (( GUI_EVAL_INTERVAL > 0 )); then
  EVAL_ARGS=(
    --eval-temperature 0.0
    --n-samples-per-eval-prompt 1
    --eval-interval "${GUI_EVAL_INTERVAL}"
    --eval-config "${GUI_EVAL_CONFIG}"
    --eval-reward-key acc
    --eval-function-path rollout.fully_async_rollout.eval_rollout_fully_async
  )
else
  echo "Eval DISABLED (GUI_EVAL_INTERVAL=0)."
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
if [[ "${NO_PIN_CPU_PARAMS:-0}" == "1" ]]; then
  OPTIMIZER_ARGS+=(--no-pin-cpu-params)
fi
if [[ "${NO_PIN_CPU_GRADS:-0}" == "1" ]]; then
  OPTIMIZER_ARGS+=(--no-pin-cpu-grads)
fi

PERF_ARGS=(
  --tensor-model-parallel-size ${TRAIN_TP:-4}
  --log-probs-chunk-size ${LOG_PROBS_CHUNK_SIZE:-2048}
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size ${TRAIN_CP:-1}
  --megatron-to-hf-mode bridge
  --qkv-format bshd
  --micro-batch-size 1
$(if [[ "${RECOMPUTE:-full}" == "none" ]]; then echo ""; else echo "  --recompute-granularity ${RECOMPUTE:-full} --recompute-method ${RECOMPUTE_METHOD:-uniform} --recompute-num-layers ${RECOMPUTE_NUM_LAYERS:-1}"; fi)
  --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-16384}
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-type low_var_kl
  --kl-loss-coef 0.01
  --loss-mask-type qwen3_5
)

SGLANG_CHUNKED_PREFILL_SIZE=${SGLANG_CHUNKED_PREFILL_SIZE:-4096}
SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-fa3}
SGLANG_ARGS=(
  --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}
  --sglang-mem-fraction-static 0.7
  --sglang-attention-backend ${SGLANG_ATTENTION_BACKEND}
  --sglang-server-concurrency ${SGLANG_SERVER_CONCURRENCY}
  --sglang-max-running-requests ${GUI_POOL_MAX_ENVS}
  --sglang-chunked-prefill-size ${SGLANG_CHUNKED_PREFILL_SIZE}
  --use-distributed-post
  --sglang-enable-metrics
)

CUSTOM_ARGS=(
  --custom-generate-function-path rollout.partial_async_gui_rollout.generate
  --custom-rm-path reward.reward_func.reward_func
  --custom-reward-post-process-path reward.reward_post_process.reward_post_process
  --custom-config-path "${CUSTOM_CONFIG_PATH}"
)

read -r -a EXTRA_ARGS <<< "${SLIME_EXTRA_ARGS:-}"

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

NCCL_NVLS_ENABLE_VALUE=${GUI_NCCL_NVLS_ENABLE:-${HAS_NVLINK}}
echo "NCCL_NVLS_ENABLE = ${NCCL_NVLS_ENABLE_VALUE}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:2048
export RAY_object_spilling_threshold=0.80
export RAY_local_fs_capacity_threshold=0.99

VENV_CUDNN_DIR="$(python3 - <<'PY' 2>/dev/null || true
import os, nvidia.cudnn
print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), "lib"))
PY
)"
if [[ -z "${VENV_CUDNN_DIR}" || ! -d "${VENV_CUDNN_DIR}" ]]; then
  PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
  VENV_CUDNN_DIR="${PROJECT_ROOT}/venvs/online-rl/lib/python3.12/site-packages/nvidia/cudnn/lib"
fi
if [[ ! -d "${VENV_CUDNN_DIR}" ]]; then
  echo "WARNING: could not locate venv cuDNN dir (${VENV_CUDNN_DIR}); cuDNN ABI fix not applied"
fi
ACTOR_NUMA_DIR="${NUMA_LIB_DIR}"
export ACTOR_LD_LIBRARY_PATH="${VENV_CUDNN_DIR}${ACTOR_NUMA_DIR:+:${ACTOR_NUMA_DIR}}:${LD_LIBRARY_PATH:-}"
echo "Pinning actor LD_LIBRARY_PATH cuDNN dir: ${VENV_CUDNN_DIR}; libnuma dir: ${ACTOR_NUMA_DIR:-<system ldconfig>}"

RAY_TEMP_DIR=${RAY_TEMP_DIR:-"/mnt/llmshared-ssd-hd/chentongbo/ray"}
mkdir -p "${RAY_TEMP_DIR}"
OBJECT_STORE_GB=${OBJECT_STORE_GB:-600}
SHM_GB=$(df -BG /dev/shm 2>/dev/null | awk 'NR==2{gsub(/G/,"",$2); print $2}')
if [[ -n "${SHM_GB}" ]] && (( OBJECT_STORE_GB > SHM_GB * 9 / 10 )); then
  echo "OBJECT_STORE_GB=${OBJECT_STORE_GB} exceeds 90% of /dev/shm (${SHM_GB}Gi); clamping to $(( SHM_GB * 9 / 10 ))"
  OBJECT_STORE_GB=$(( SHM_GB * 9 / 10 ))
fi
OBJECT_STORE_BYTES=$(( OBJECT_STORE_GB * 1024 * 1024 * 1024 ))
echo "Ray object store (plasma) = ${OBJECT_STORE_GB} GB"
RAY_NUM_CPUS=${RAY_NUM_CPUS:-64}
HEAD_NUM_GPUS=${HEAD_NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
if (( HEAD_NUM_GPUS < 1 )); then HEAD_NUM_GPUS=8; fi
echo "Head contributes ${HEAD_NUM_GPUS} GPUs to the Ray cluster"
ray start --head --num-gpus "${HEAD_NUM_GPUS}" --num-cpus "${RAY_NUM_CPUS}" --object-store-memory ${OBJECT_STORE_BYTES} --temp-dir "${RAY_TEMP_DIR}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

EXPECTED_NODES=${EXPECTED_NODES:-5}
echo "Waiting for ${EXPECTED_NODES} nodes to join Ray cluster (run gpu_worker_join_ray.sh on each of the other 4 nodes)..."
for i in $(seq 1 120); do
  ACTIVE_NODES=$(ray status 2>/dev/null | grep -c "node_" || echo 0)
  if (( ACTIVE_NODES >= EXPECTED_NODES )); then
    echo "All ${EXPECTED_NODES} nodes joined."
    break
  fi
  echo "  ... ${ACTIVE_NODES}/${EXPECTED_NODES} nodes (attempt ${i}/120)"
  sleep 5
  if (( i == 120 )); then
    echo "WARNING: Only ${ACTIVE_NODES}/${EXPECTED_NODES} nodes joined after 600s."
    ray status
  fi
done

echo "Verifying Ray cluster GPUs..."
ray status

RAY_JOB_SUBMISSION_ID=${RAY_JOB_SUBMISSION_ID:-"hybridcua_9b_40gpu_fully_async_$(date +%Y%m%d_%H%M%S)"}

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}:${SCRIPT_DIR}:${SLIME_DIR}\",
    \"PYTHONUNBUFFERED\": \"${PYTHONUNBUFFERED}\",
    \"PYTHONFAULTHANDLER\": \"${PYTHONFAULTHANDLER}\",
    \"http_proxy\": \"${GUI_JOB_HTTP_PROXY:-}\",
    \"https_proxy\": \"${GUI_JOB_HTTP_PROXY:-}\",
    \"HTTP_PROXY\": \"${GUI_JOB_HTTP_PROXY:-}\",
    \"HTTPS_PROXY\": \"${GUI_JOB_HTTP_PROXY:-}\",
    \"no_proxy\": \"${GUI_JOB_NO_PROXY:-}\",
    \"NO_PROXY\": \"${GUI_JOB_NO_PROXY:-}\",
    \"LD_LIBRARY_PATH\": \"${ACTOR_LD_LIBRARY_PATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${NCCL_NVLS_ENABLE_VALUE}\",
    \"SGLANG_VLM_CACHE_SIZE_MB\": \"${SGLANG_VLM_CACHE_SIZE_MB:-4096}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\",
    \"GUI_ENV_SERVER_URL\": \"${GUI_ENV_SERVER_URL}\",
    \"GUI_ENV_CLIENT\": \"${GUI_ENV_CLIENT}\",
    \"GUI_ENV_RUNTIME\": \"${GUI_ENV_RUNTIME}\",
    \"GUI_POOL_MAX_ENVS\": \"${GUI_POOL_MAX_ENVS}\",
    \"GUI_TRAJECTORY_CONCURRENCY\": \"${GUI_TRAJECTORY_CONCURRENCY}\",
    \"GUI_ROLLOUT_WORKERS\": \"${GUI_ROLLOUT_WORKERS}\",
    \"GUI_FAST_ROLLOUT_PROCS\": \"${GUI_FAST_ROLLOUT_PROCS}\",
    \"GUI_COORD_SHIM\": \"${GUI_COORD_SHIM}\",
    \"GUI_RAY_ACTOR_CPUS\": \"${GUI_RAY_ACTOR_CPUS:-1}\",
    \"GUI_LOG_LEVEL\": \"${GUI_LOG_LEVEL}\",
    \"GUI_RESULT_DIR\": \"${GUI_RESULT_DIR}\",
    \"GUI_COORDINATE_TYPE\": \"${GUI_COORDINATE_TYPE}\",
    \"GUI_ACTION_SPACE\": \"${GUI_ACTION_SPACE}\",
    \"GUI_OBSERVATION_TYPE\": \"${GUI_OBSERVATION_TYPE}\",
    \"GUI_TEST_CONFIG_BASE_DIR\": \"${GUI_TEST_CONFIG_BASE_DIR}\",
    \"GUI_TRAIN_META_PATH\": \"${GUI_TRAIN_META_PATH}\",
    \"GUI_DATA_SOURCE_PATH\": \"${GUI_DATA_SOURCE_PATH}\",
    \"GUI_CUA_GYM_BUNDLES\": \"${GUI_CUA_GYM_BUNDLES}\",
    \"GUI_CUA_GYM_TASKS_META\": \"${GUI_CUA_GYM_TASKS_META}\",
    \"GUI_EVAL_META_PATH\": \"${GUI_EVAL_META_PATH}\",
    \"OSWORLD_PROJECT\": \"${OSWORLD_PROJECT}\",
    \"download_proxy\": \"${download_proxy}\",
    \"GUI_AGENT_CLASS_PATH\": \"${GUI_AGENT_CLASS_PATH}\",
    \"ENABLE_THINKING\": \"${ENABLE_THINKING}\",
    \"GUI_DISABLE_MTP\": \"${GUI_DISABLE_MTP}\",
    \"HF_CKPT\": \"${HF_CKPT}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\",
    \"WANDB_BASE_URL\": \"${WANDB_BASE_URL:-}\",
    \"WANDB_PROJECT\": \"${WANDB_PROJECT}\",
    \"GUI_USER_ID\": \"${GUI_USER_ID}\",
    \"SLIME_GRAD_PROBE\": \"${SLIME_GRAD_PROBE:-0}\",
    \"SLIME_GRAD_PROBE_DIR\": \"${SLIME_GRAD_PROBE_DIR:-/tmp/grad_probe}\",
    \"SLIME_ACT_PROBE\": \"${SLIME_ACT_PROBE:-0}\",
    \"SLIME_ACT_PROBE_DIR\": \"${SLIME_ACT_PROBE_DIR:-/tmp/act_probe}\",
    \"GUI_JOB_ID\": \"${RAY_JOB_SUBMISSION_ID}\"
  }
}"

TRAIN_ENTRY=${TRAIN_ENTRY:-"${SCRIPT_DIR}/train_fully_async.py"}

ray job submit --address="http://127.0.0.1:8265" \
  --submission-id "${RAY_JOB_SUBMISSION_ID}" \
  --no-wait \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 -u "${TRAIN_ENTRY}" \
  --actor-num-nodes ${ACTOR_NUM_NODES} \
  --actor-num-gpus-per-node ${ACTOR_NUM_GPUS_PER_NODE} \
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
  ${CUSTOM_ARGS[@]} \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

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
