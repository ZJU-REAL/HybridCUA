#!/bin/bash
# Semi-async ("partial-async") GUI-RL with a REMOTE OSWorld env server.
# 2 nodes x 8 GPU = 16 GPU: actor 8 (2 nodes x 4, TP=4 DP=2) + rollout 8 (8 SGLang engines x 1 GPU).
#
# This is the 16-GPU two-node counterpart of gui_qwen3vl_8b_partial_async.sh.
# ONLY the GPU/node counts differ; everything else (env server, concurrency,
# eval, PRM-off, custom-config, entrypoints) is identical.
#   Head node:   bash scripts/gui_qwen3vl_16gpu_partial_async.sh
#   Worker node: bash gpu_worker_join_ray.sh   (join the head before/after)
# The head waits for all nodes to join before submitting the job.
#
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

# 16 GPUs across 2 nodes: actor 8 (2x4, TP=4 DP=2) + rollout 8 (8 engines x 1 GPU).
NUM_GPUS=${NUM_GPUS:-16}
ACTOR_GPUS=${ACTOR_GPUS:-8}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-8}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-2}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-4}

if (( ACTOR_GPUS + ROLLOUT_GPUS > NUM_GPUS )); then
  echo "ACTOR_GPUS + ROLLOUT_GPUS must be <= NUM_GPUS"
  echo "ACTOR_GPUS=${ACTOR_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, NUM_GPUS=${NUM_GPUS}"
  exit 1
fi

# Remote env server (OSWorld cluster, /allocate lease protocol). Override as needed.
export GUI_ENV_SERVER_URL=${GUI_ENV_SERVER_URL:-"http://127.0.0.1:19000"}
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
# env client 实现：session = 自带的 /v1/sessions 适配器(clients/SessionGuiEnvClient)；
# legacy = 老的 lease-HTTP GuiEnvClient。默认 session；回退用 GUI_ENV_CLIENT=legacy bash ...
export GUI_ENV_CLIENT=${GUI_ENV_CLIENT:-"session"}
MULTIMODAL_KEYS=${MULTIMODAL_KEYS:-'{"image":"images"}'}
# sglang tokenizer 进程的图像 base64 解码线程池大小(io_executor)。sglang 默认 4。
# A/B 测 image worker 用:SGLANG_IO_WORKERS=32 bash 本脚本。改了需重启 engine 才生效。
export SGLANG_IO_WORKERS=${SGLANG_IO_WORKERS:-4}
# GUI rollout/eval step counts etc. come from CUSTOM_CONFIG_PATH (see yaml).

WANDB_PROJECT=${WANDB_PROJECT:-slime_gui}
WANDB_GROUP=${WANDB_GROUP:-qwen3-8b-partial-async-16gpu}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
GUI_PROJECT_NAME=${GUI_PROJECT_NAME:-slime_gui_8b_partial_async_16gpu_${RUN_TIMESTAMP}}
export GUI_USER_ID="${GUI_USER_ID:-partial_async}_${RUN_TIMESTAMP}"
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
CKPT_NAME=${CKPT_NAME:-"gui-qwen3vl-8b-partial-async-16gpu"}
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

# NOTE: --gui-* flags are intentionally NOT passed here (upstream slime would
# drop them). They are injected via CUSTOM_CONFIG_PATH instead.
# Debug rollout dumps: slime writes one .pt per rollout to this dir. Timestamped
# so each run goes to its own folder ({rollout_id} expands per rollout).
DEBUG_ROLLOUT_DIR="/mnt/llmshared-ssd-hd/chentongbo/online-rl/gui-rl/results/debug/${RUN_TIMESTAMP}"
mkdir -p "${DEBUG_ROLLOUT_DIR}"

ROLLOUT_ARGS=(
  --data-source-path ${GUI_DATA_SOURCE_PATH:-data.gui_data_source.CuaGymDataSource}
  --reward-key score
  --num-rollout 1000
  --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
  --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
  --rollout-max-response-len 1024
  --rollout-temperature 1.0
  --num-steps-per-rollout 1
  # --save-debug-rollout-data "${DEBUG_ROLLOUT_DIR}/rollout_{rollout_id}.pt"
)

# 压测模式:SLEEP_ROLLOUT=1 时 rollout 进程初始化后死睡(engine 就绪但不跑训练流量),
# 用于干净地对 sglang 做 A/B 压测(如 SGLANG_IO_WORKERS)。默认关,不影响正常训练。
# 与多平台互斥:压测优先,多平台被忽略(压测本就不跑训练流量)。
if [[ "${SLEEP_ROLLOUT:-0}" == "1" ]]; then
  ROLLOUT_ARGS+=(--rollout-function-path slime.rollout.sleep_rollout.sleep)
  echo "[SLEEP_ROLLOUT] rollout 将死睡,engine 起来后用 scripts/bench_io_workers.py 压测"
elif [[ -n "${GUI_MULTI_PLATFORM_MODE:-}" ]]; then
  # 多平台 MRPO(半异步):thin rollout 函数把 rollout_id 传给 MultiPlatformDataSource,
  # 由其 get_samples 按 mode(mix/interleaved/single)组批。GUI_DATA_SOURCE_PATH 应指向
  # data.gui_data_source.MultiPlatformDataSource(由 run_multiplatform_*.sh 设置)。
  ROLLOUT_ARGS+=(--rollout-function-path ${GUI_ROLLOUT_FUNCTION_PATH:-data.gui_data_source.generate_rollout_multiplatform_semi})
  ROLLOUT_ARGS+=(--multi-platform-mode "${GUI_MULTI_PLATFORM_MODE}")
  ROLLOUT_ARGS+=(--alternating-switch-interval "${GUI_ALTERNATING_SWITCH_INTERVAL:-10}")
  echo "[MULTI_PLATFORM] mode=${GUI_MULTI_PLATFORM_MODE} platforms=${GUI_MULTI_PLATFORM_LIST:-mobileworld,osworld} K=${GUI_ALTERNATING_SWITCH_INTERVAL:-10}"
fi

IN_FLIGHT_SAMPLES_ESTIMATE=$(( ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT ))
echo "Configured rollout-batch-size x n-samples-per-prompt = ${IN_FLIGHT_SAMPLES_ESTIMATE}"
echo "Using remote GUI env server: ${GUI_ENV_SERVER_URL}"
echo "Injecting custom config: ${CUSTOM_CONFIG_PATH}"

# 周期性 eval，默认关（GUI_ENABLE_EVAL=1 打开）。不传 --eval-interval 时 slime 既不
# 触发 eval 也不校验 eval_datasets（train_async.py 的 should_run_periodic_action(None)
# 返回 False / arguments.py 仅在 eval_interval is not None 时 assert）。ckpt 每
# SAVE_INTERVAL step 已存，离线单独 eval 即可。
# --gui-eval-* come from CUSTOM_CONFIG_PATH, not here. --eval-config only satisfies
# slime's validation so the periodic eval hook fires (GUI eval reads GUI_EVAL_META_PATH
# via our custom --eval-function-path, not these datasets).
EVAL_ARGS=()
if [[ "${GUI_ENABLE_EVAL:-0}" == "1" ]]; then
  GUI_EVAL_CONFIG=${GUI_EVAL_CONFIG:-"${SCRIPT_DIR}/scripts/gui_eval_dataset.yaml"}
  GUI_EVAL_INTERVAL=${GUI_EVAL_INTERVAL:-10}
  EVAL_ARGS=(
    --eval-temperature 0.0
    --n-samples-per-eval-prompt 1
    --eval-interval "${GUI_EVAL_INTERVAL}"
    # --eval-at-start
    --eval-config "${GUI_EVAL_CONFIG}"
    --eval-reward-key acc
    --eval-function-path rollout.partial_async_gui_rollout.fast_eval_rollout
  )
  echo "[EVAL] enabled: interval=${GUI_EVAL_INTERVAL}"
else
  echo "[EVAL] disabled (GUI_ENABLE_EVAL=1 to turn on). Eval ckpts offline instead."
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

# chunked_prefill_size: GUI prompts are ~7640 token, which under the default
# 8192 produces ONE half-full chunk (low GPU util, ~11% full-chunk rate vs eval's
# 64%). Splitting at 4096 makes a 7640 prompt -> 4096+3544 (two fuller chunks),
# raising prefill GPU utilization. Tune via SGLANG_CHUNKED_PREFILL_SIZE; 8192 = sglang default.
# SGLANG_CHUNKED_PREFILL_SIZE=${SGLANG_CHUNKED_PREFILL_SIZE:-8192}
SGLANG_ARGS=(
  --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}
  --sglang-mem-fraction-static 0.72
  # --sglang-chunked-prefill-size ${SGLANG_CHUNKED_PREFILL_SIZE}
  --use-distributed-post
  --sglang-enable-metrics
)

# Refactored entrypoints. No --custom-rollout-log-function-path: the online-rl
# tree has no gui_rollout_logging module.
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

# libnuma.so.1: sgl_kernel 的 common_ops.abi3.so 依赖它;worker 节点常缺 → SGLangEngine
# 起不来("libnuma.so.1: cannot open shared object file")。首选共享盘 vendor_libs 副本
# (两节点可见),apt/yum 次选。解析出的目录后面塞进 actor 的 LD_LIBRARY_PATH。
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

# cuDNN ABI fix: the system cuDNN (/usr/lib .so.9.10.2) is missing the
# OperationGraph vtable symbol that the venv's transformer_engine expects, so the
# dynamic linker resolves libcudnn_cnn.so.9 against a mismatched libcudnn_graph.so.9
# -> "undefined symbol ... libcudnn_graph.so.9" and every actor dies at import.
# Force the venv's self-consistent cuDNN set to the FRONT of LD_LIBRARY_PATH for
# every Ray actor (both nodes share this venv over /mnt/llmshared-ssd-hd).
VENV_CUDNN_DIR="$(python3 - <<'PY' 2>/dev/null || true
import os, nvidia.cudnn
print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), "lib"))
PY
)"
if [[ -z "${VENV_CUDNN_DIR}" || ! -d "${VENV_CUDNN_DIR}" ]]; then
  PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
  VENV_CUDNN_DIR="${PROJECT_ROOT}/venvs/online-rl/lib/python3.12/site-packages/nvidia/cudnn/lib"
fi
if [[ ! -d "${VENV_CUDNN_DIR}" ]]; then
  echo "WARNING: could not locate venv cuDNN dir (${VENV_CUDNN_DIR}); cuDNN ABI fix not applied"
fi
# Actor LD_LIBRARY_PATH = venv cuDNN + vendored libnuma(无条件,worker 缺) + inherited。
ACTOR_NUMA_DIR=""
if [[ -f "${SCRIPT_DIR}/../vendor_libs/libnuma.so.1" ]]; then
  ACTOR_NUMA_DIR="$(cd -- "${SCRIPT_DIR}/../vendor_libs" &>/dev/null && pwd)"
elif [[ -n "${NUMA_LIB_DIR}" ]]; then
  ACTOR_NUMA_DIR="${NUMA_LIB_DIR}"
fi
export ACTOR_LD_LIBRARY_PATH="${VENV_CUDNN_DIR}${ACTOR_NUMA_DIR:+:${ACTOR_NUMA_DIR}}:${LD_LIBRARY_PATH:-}"
echo "Pinning actor LD_LIBRARY_PATH cuDNN dir: ${VENV_CUDNN_DIR}; libnuma dir: ${ACTOR_NUMA_DIR:-<system ldconfig>}"

RAY_TEMP_DIR=${RAY_TEMP_DIR:-"/mnt/llmshared-ssd-hd/chentongbo/ray"}
mkdir -p "${RAY_TEMP_DIR}"

# Ray plasma object store size. Default (30% of RAM = ~200GB) gets pegged to 99.7%
# by the fan-out'd rollout batch (dynamic_history -> 700~960 image-bearing samples),
# triggering multi-TB spill to JuiceFS that stalls training until NCCL times out.
# 2TB RAM node -> give plasma 600GB so the batch stays in-memory and never spills.
# The worker node must use the SAME value (pass OBJECT_STORE_GB to gpu_worker_join_ray.sh).
OBJECT_STORE_GB=${OBJECT_STORE_GB:-600}
OBJECT_STORE_BYTES=$(( OBJECT_STORE_GB * 1024 * 1024 * 1024 ))
echo "Ray object store (plasma) = ${OBJECT_STORE_GB} GB"

# Head node starts Ray. Worker nodes join via gpu_worker_join_ray.sh separately.
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
EXPECTED_NODES=${EXPECTED_NODES:-2}
ray start --head --num-gpus "${NUM_GPUS_PER_NODE}" --object-store-memory ${OBJECT_STORE_BYTES} --temp-dir "${RAY_TEMP_DIR}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
echo "Head node started. Run gpu_worker_join_ray.sh on worker nodes to join."

# Wait for all nodes to join before submitting the job.
echo "Waiting for ${EXPECTED_NODES} nodes to join Ray cluster..."
for i in $(seq 1 120); do
  ACTIVE_NODES=$(ray status 2>/dev/null | grep -c "node_" || echo 0)
  if (( ACTIVE_NODES >= EXPECTED_NODES )); then
    echo "All ${EXPECTED_NODES} nodes joined. Total GPUs: $((EXPECTED_NODES * NUM_GPUS_PER_NODE))"
    break
  fi
  echo "  ... ${ACTIVE_NODES}/${EXPECTED_NODES} nodes (attempt ${i}/120)"
  sleep 5
  if (( i == 120 )); then
    echo "WARNING: Only ${ACTIVE_NODES}/${EXPECTED_NODES} nodes joined after 600s."
    ray status
    exit 1
  fi
done

ray status

RAY_JOB_SUBMISSION_ID=${RAY_JOB_SUBMISSION_ID:-"gui_qwen3vl_16gpu_partial_async_$(date +%Y%m%d_%H%M%S)"}

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}:${SCRIPT_DIR}:${SLIME_DIR}\",
    \"PYTHONUNBUFFERED\": \"${PYTHONUNBUFFERED}\",
    \"PYTHONFAULTHANDLER\": \"${PYTHONFAULTHANDLER}\",
    \"LD_LIBRARY_PATH\": \"${ACTOR_LD_LIBRARY_PATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\",
    \"SGLANG_VLM_CACHE_SIZE_MB\": \"4096\",
    \"SGLANG_IO_WORKERS\": \"${SGLANG_IO_WORKERS}\",
    \"GUI_ENV_SERVER_URL\": \"${GUI_ENV_SERVER_URL}\",
    \"GUI_ENV_CLIENT\": \"${GUI_ENV_CLIENT}\",
    \"GUI_ENV_RUNTIME\": \"${GUI_ENV_RUNTIME:-cua_gym}\",
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
    \"GUI_USER_ID\": \"${GUI_USER_ID}\",
    \"GUI_JOB_ID\": \"${RAY_JOB_SUBMISSION_ID}\"
  }
}"

TRAIN_ENTRY=${TRAIN_ENTRY:-"${SLIME_DIR}/train_async.py"}

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
