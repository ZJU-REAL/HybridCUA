#!/bin/bash
# Qwen3.5-9B (DENSE multimodal VLM) GRPO training on OSWorld GUI tasks, SEMI-async.
# SINGLE NODE, 8 GPUs.
#
# Derived from gui_qwen3.5_9B_16gpu.sh (2-node/16-GPU, manual Ray bring-up),
# collapsed to a single 8-GPU node using the topology/Ray layout of
# gui_qwen3vl_8b.sh. ALL Qwen3.5-9B model specifics are preserved verbatim.
#
# 9B model facts (from HF config.json + slime/scripts/models/qwen3.5-9B.sh):
#   - DENSE: num_experts absent -> no --expert-*/--moe-*/DeepEP flags.
#   - VLM:   config has vision_config/image_token_id (真多模态，自带 ViT)。
#   - GDN:   --use-gated-attention -> Gated Delta Net 线性注意力层。GDN 不支持 packed
#            sequence -> 必须 --qkv-format bshd + 关 dynamic-batch。
#   - MTP:   mtp_num_hidden_layers=1 -> 需 GUI_DISABLE_MTP=1(否则 bridge 映射崩)。
#   - heads: num_attention_heads=16, num_query_groups=4。
#
# ⚠️ TP 必须 <= num_query_groups(4)，否则触发 Megatron GQA output-gate 二次切分补丁。
#    本脚本 actor 用 TP=4 (num_query_groups(4)==world_size(4))，不触发补丁。
#
# 单机 8 卡布局 (8 GPUs = 4 actor + 4 rollout)：
#   - Actor:   1 节点 × 4 GPU (TP=4, DP=1)。dense 无 MoE/EP。4 个 train actor 单节点
#              host RAM 压力小，pinning 保持 ON (16gpu 版拆 2 节点是为了 8 actor 的
#              host-RAM，这里只有 4 个，不需要)。
#   - Rollout: 4 GPUs as 4 sglang engines of 1 GPU each(与 actor 同节点共存)。
#
# 半异步(semi-async)：rollout 走 slime 默认 generate_rollout(不设 --rollout-function-path),
#   主循环由 train_async.py 驱动(预取下一轮/ray.get 上一轮/训练)。无 staleness, 不钉死池。
#
# 用法:
#   HF_CKPT=... GUI_ENV_SERVER_URL=... bash scripts/gui_qwen3.5_9B_8gpu.sh
#   可选 eval:  GUI_ENABLE_EVAL=1 开周期 eval。
# ====================================================================

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

# 确保 libnuma.so.1 存在 (sgl_kernel 依赖;缺了 SGLangEngine 起不来)。
# 首选共享盘上的 vendor_libs/libnuma.so.1，apt/yum 次选。
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
# Qwen3.5-9B is a DENSE multimodal model. Its model def sets --rotary-base 10000000
# itself, so we do NOT override MODEL_ARGS_ROTARY_BASE.
source "${SLIME_DIR}/scripts/models/qwen3.5-9B.sh"
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-"${SCRIPT_DIR}/../Megatron-LM"}
CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-"${SCRIPT_DIR}/scripts/gui_partial_async.yaml"}

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

export RAY_health_check_failure_threshold=${RAY_health_check_failure_threshold:-20}
export RAY_health_check_period_ms=${RAY_health_check_period_ms:-5000}
export RAY_health_check_timeout_ms=${RAY_health_check_timeout_ms:-30000}
export RAY_num_heartbeats_timeout=${RAY_num_heartbeats_timeout:-60}

# 8 GPUs single node: actor 4 (TP=4, DENSE 无 EP) + rollout 4 (4 engines x 1 GPU).
NUM_GPUS=${NUM_GPUS:-8}
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
# 单节点：actor 全部 4 GPU 在本机。
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-4}

if (( ACTOR_GPUS + ROLLOUT_GPUS > NUM_GPUS )); then
  echo "ACTOR_GPUS + ROLLOUT_GPUS must be <= NUM_GPUS"
  echo "ACTOR_GPUS=${ACTOR_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, NUM_GPUS=${NUM_GPUS}"
  exit 1
fi

# Remote env server (OSWorld cluster, /v1/sessions session protocol on :19000).
export GUI_ENV_SERVER_URL=${GUI_ENV_SERVER_URL:-"http://127.0.0.1:19000"}
export GUI_ENV_CLIENT=${GUI_ENV_CLIENT:-session}
export GUI_POOL_MAX_ENVS=${GUI_POOL_MAX_ENVS:-64}
export GUI_TRAJECTORY_CONCURRENCY=${GUI_TRAJECTORY_CONCURRENCY:-64}
export GUI_ROLLOUT_WORKERS=1
export GUI_FAST_ROLLOUT_PROCS=${GUI_FAST_ROLLOUT_PROCS:-64}
# Rollout 后端: ray = 每条轨迹结果(含图 Sample)由常驻 Ray actor 写 plasma、RolloutManager
# 零拷贝读回，绕开 process 池单管道 unpickle 大 blob 的瓶颈(dispatch_wait 卡顿)。
export GUI_LOG_LEVEL=${GUI_LOG_LEVEL:-INFO}
export GUI_ACTION_SPACE=${GUI_ACTION_SPACE:-"pyautogui"}
export GUI_OBSERVATION_TYPE=${GUI_OBSERVATION_TYPE:-"screenshot"}
export GUI_COORDINATE_TYPE=${GUI_COORDINATE_TYPE:-"relative"}
# Hybrid CUA agent: c_gui 的 bash 动作空间(bash/wait/terminate/answer + pyautogui heredoc)。
# 与 env_infra/OSWorld/mm_agents/c_gui 评测侧对齐(system prompt/tool schema 逐字复刻)。
export GUI_AGENT_CLASS_PATH=${GUI_AGENT_CLASS_PATH:-"agents.qwen35_hybrid_cua.Qwen35HybridCuaAgentLocal"}
# GUI_COORD_SHIM=1: 每 episode reset 后往 VM 装 usercustomize.py 坐标 shim, 把 bash heredoc
# 里的 0-999 坐标缩放到真实像素(由 clients/coord_shim.py 提供, 消费方 osworld_remote_async)。
export GUI_COORD_SHIM=${GUI_COORD_SHIM:-1}
export GUI_ENV_RUNTIME=${GUI_ENV_RUNTIME:-"cua_gym"}
export ENABLE_THINKING=${ENABLE_THINKING:-False}
# GUI_DISABLE_MTP=1: config 有 mtp_num_hidden_layers=1，但 GRPO 不用 MTP，且 bridge
# 映射对不上 -> 加载崩溃。由 slime/backends/megatron_utils/model_provider.py 消费。
export GUI_DISABLE_MTP=${GUI_DISABLE_MTP:-1}
MULTIMODAL_KEYS=${MULTIMODAL_KEYS:-'{"image":"images"}'}

WANDB_PROJECT=${WANDB_PROJECT:-slime_gui}
WANDB_GROUP=${WANDB_GROUP:-qwen3.5-9b-8gpu}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
GUI_PROJECT_NAME=${GUI_PROJECT_NAME:-slime_gui_9b_fast_8gpu_${RUN_TIMESTAMP}}
export GUI_USER_ID="${GUI_USER_ID:-fast_9b}_${RUN_TIMESTAMP}"
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
export GUI_CUA_GYM_TASKS_META=${GUI_CUA_GYM_TASKS_META:-"${CUA_GYM_DATA}/rlvr_curriculum_1000_meta.json"}

export GUI_DATA_SOURCE_PATH=${GUI_DATA_SOURCE_PATH:-"data.gui_data_source.CuaGymDataSource"}

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
CKPT_NAME=${CKPT_NAME:-"gui-qwen3.5-9b-fast-8gpu"}
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

# ===========================================================================
# SEMI-ASYNC: rollout 走 slime 默认 generate_rollout(不设 --rollout-function-path),
# 主循环由 train_async.py 驱动(预取下一轮/ray.get 上一轮/训练)。无 staleness 过滤,
# 不钉死 in-flight 池(无 --sglang-server-concurrency)。
# ===========================================================================
NUM_ROLLOUT=${NUM_ROLLOUT:-1000}
ROLLOUT_ARGS=(
  --data-source-path ${GUI_DATA_SOURCE_PATH:-data.gui_data_source.CuaGymDataSource}
  --reward-key score
  --num-rollout ${NUM_ROLLOUT}
  --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
  --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
  --rollout-max-response-len 1024
  --rollout-temperature 1.0
  --num-steps-per-rollout 1
)

IN_FLIGHT_SAMPLES_ESTIMATE=$(( ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT ))
echo "Configured rollout-batch-size x n-samples-per-prompt = ${IN_FLIGHT_SAMPLES_ESTIMATE}"
echo "process pool GUI_FAST_ROLLOUT_PROCS=${GUI_FAST_ROLLOUT_PROCS}, env cap GUI_TRAJECTORY_CONCURRENCY=${GUI_TRAJECTORY_CONCURRENCY} (aim: all three equal)"
echo "Using remote GUI env server: ${GUI_ENV_SERVER_URL}"
echo "Injecting custom config: ${CUSTOM_CONFIG_PATH}"

# Eval: 半异步用 partial-async 专用 eval 入口。默认关(GUI_ENABLE_EVAL=1 开)。
EVAL_ARGS=()
if [[ "${GUI_ENABLE_EVAL:-0}" == "1" ]]; then
  GUI_EVAL_CONFIG=${GUI_EVAL_CONFIG:-"${SCRIPT_DIR}/scripts/gui_eval_dataset.yaml"}
  GUI_EVAL_INTERVAL=${GUI_EVAL_INTERVAL:-10}
  EVAL_ARGS=(
    --eval-temperature 0.0
    --n-samples-per-eval-prompt 1
    --eval-interval "${GUI_EVAL_INTERVAL}"
    --eval-config "${GUI_EVAL_CONFIG}"
    --eval-reward-key acc
    --eval-function-path rollout.partial_async_gui_rollout.fast_eval_rollout
  )
  echo "[EVAL] enabled: interval=${GUI_EVAL_INTERVAL} (OSWorld test_nochrome only)"
else
  echo "[EVAL] disabled (GUI_ENABLE_EVAL=1 to turn on)."
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
# 单节点仅 4 个 train actor，pinned host copies 小，pinning 保持 ON (Megatron 默认)。
# 仅在 host-RAM OOM 时才翻 NO_PIN_CPU_*=1。
if [[ "${NO_PIN_CPU_PARAMS:-0}" == "1" ]]; then
  OPTIMIZER_ARGS+=(--no-pin-cpu-params)
fi
if [[ "${NO_PIN_CPU_GRADS:-0}" == "1" ]]; then
  OPTIMIZER_ARGS+=(--no-pin-cpu-grads)
fi

# DENSE parallelism: actor 4 GPU 单节点，attention runs TP=4 (DP=1). NO experts.
# TP=4 时 num_query_groups(4)==world_size(4) 不触发 attention.py 的 GQA gate 二次切分(TP=8 才触发)。
PERF_ARGS=(
  --tensor-model-parallel-size ${TRAIN_TP:-4}
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size ${TRAIN_CP:-1}
  --megatron-to-hf-mode bridge
  # Qwen3.5 有 GDN(Gated Delta Net)线性注意力层，不支持 packed sequence。
  # 必须关 --use-dynamic-batch-size + 用 --qkv-format bshd，否则 compute_log_prob 到
  # GDN 层报 "NotImplementedError: GDN does not support packed sequence for now."。
  --qkv-format bshd
  --micro-batch-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --max-tokens-per-gpu 1024
)

# --dynamic_history is injected via CUSTOM_CONFIG_PATH (not a valid upstream flag).
GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-type low_var_kl
  --kl-loss-coef 0.01
  --loss-mask-type qwen3_5
)

SGLANG_CHUNKED_PREFILL_SIZE=${SGLANG_CHUNKED_PREFILL_SIZE:-4096}
SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-fa3}
# 半异步不钉死 in-flight 池(无 --sglang-server-concurrency)，但仍需 --sglang-max-running-requests
# 限住 sglang engine 内部并发，避免 GUI 大截图图像预处理峰值打爆运行时显存 -> CUDA OOM。
SGLANG_ARGS=(
  --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}
  # mem-fraction 0.7(非 0.8): 9B 是 GDN 混合模型，mamba state cache 与 KV cache 是两套
  # 独立显存。0.8 时长 GUI 截图(单请求可达 11390 token)的 prefill 激活 + mamba state 峰值
  # 会把某个 TP=1 engine 剩余运行时显存打爆 -> CUDA OOM。降到 0.7 给运行时留更多余量。
  --sglang-mem-fraction-static 0.7
  --sglang-attention-backend ${SGLANG_ATTENTION_BACKEND}
  --sglang-max-running-requests ${GUI_POOL_MAX_ENVS}
  --sglang-chunked-prefill-size ${SGLANG_CHUNKED_PREFILL_SIZE}
  --use-distributed-post
  --sglang-enable-metrics
)

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

NCCL_NVLS_ENABLE_VALUE=${GUI_NCCL_NVLS_ENABLE:-${HAS_NVLINK}}
echo "NCCL_NVLS_ENABLE = ${NCCL_NVLS_ENABLE_VALUE}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:2048
export RAY_object_spilling_threshold=0.80
export RAY_local_fs_capacity_threshold=0.99

# cuDNN ABI fix: pin the venv's self-consistent cuDNN set to the FRONT of
# LD_LIBRARY_PATH for every Ray actor.
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
ACTOR_NUMA_DIR="${NUMA_LIB_DIR}"
export ACTOR_LD_LIBRARY_PATH="${VENV_CUDNN_DIR}${ACTOR_NUMA_DIR:+:${ACTOR_NUMA_DIR}}:${LD_LIBRARY_PATH:-}"
echo "Pinning actor LD_LIBRARY_PATH cuDNN dir: ${VENV_CUDNN_DIR}; libnuma dir: ${ACTOR_NUMA_DIR:-<system ldconfig>}"

# ===========================================================================
# 单节点 Ray: 本节点起 Head, 用满 8 GPU, 直接提交 job(无 worker 节点等待)。
# ===========================================================================
RAY_TEMP_DIR=${RAY_TEMP_DIR:-"/mnt/llmshared-ssd-hd/chentongbo/ray"}
mkdir -p "${RAY_TEMP_DIR}"
ray start --head --num-gpus "${NUM_GPUS}" --temp-dir "${RAY_TEMP_DIR}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

echo "Verifying Ray cluster GPUs..."
ray status

RAY_JOB_SUBMISSION_ID=${RAY_JOB_SUBMISSION_ID:-"gui_qwen3.5_9b_8gpu_fast_$(date +%Y%m%d_%H%M%S)"}

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}:${SCRIPT_DIR}:${SLIME_DIR}\",
    \"PYTHONUNBUFFERED\": \"${PYTHONUNBUFFERED}\",
    \"PYTHONFAULTHANDLER\": \"${PYTHONFAULTHANDLER}\",
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
    \"GUI_JOB_ID\": \"${RAY_JOB_SUBMISSION_ID}\"
  }
}"

# Entry: slime 上游半异步训练主循环(train_async.py)。全异步版才用 gui-rl 私有 train_fully_async.py。
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
