#!/bin/bash
# Qwen3.5-9B (DENSE multimodal VLM) GRPO training on OSWorld GUI tasks, SEMI-async.
#
# Derived from gui_qwen3.5_9B_16gpu_fast_platform.sh. 9B is a DENSE model (no
# experts), so ALL MoE/EP/DeepEP machinery is stripped.
#
# 9B model facts (from HF config.json + slime/scripts/models/qwen3.5-9B.sh):
#   - DENSE: num_experts absent -> no --expert-*/--moe-*/DeepEP flags.
#   - VLM:   config has vision_config/image_token_id (真多模态，自带 ViT)。
#   - GDN:   --use-gated-attention -> Gated Delta Net 线性注意力层。GDN 不支持 packed
#            sequence -> 必须 --qkv-format bshd + 关 dynamic-batch。
#   - MTP:   mtp_num_hidden_layers=1 -> 需 GUI_DISABLE_MTP=1(否则 bridge 映射崩)。
#   - heads: num_attention_heads=16, num_query_groups=4。
#
# ⚠️ TP=8 触发 Megatron GQA output-gate 补丁(attention.py)：num_query_groups(4)<world_size(8)。
#    补丁已在 attention.py，会自动生效。
#
# 半异步 vs 全异步的唯一区别 = 训练主循环调度:
#   - 全异步(fully_async): TRAIN_ENTRY=train_fully_async.py + --rollout-function-path
#             rollout.fully_async_rollout(后台超采 + staleness + 钉死 in-flight 池)。
#   - 本脚本(半异步):      TRAIN_ENTRY=slime/train_async.py, rollout 走 slime 默认
#             generate_rollout(预取下一轮/ray.get 上一轮/训练)。无 staleness, 不钉死池。
#
# ====================================================================
# 手动启动版 (manual Ray bring-up)。
# 与 gui_qwen3.5_9B_16gpu_fast_platform.sh 的唯一区别 = Ray 启动方式:
#   - platform.sh: 平台注入 RANK/WORLD_SIZE, 所有节点跑同一脚本, RANK guard 分流
#   - 本脚本:      本节点 ray start --head 起 Head 并提交 job;
#                  其余 GPU 节点各自手动跑 gpu_worker_join_ray.sh 加入集群。
# 模型/算法/rollout body 完全相同。
#
# 2-node layout (16 GPUs = 2 x 8)，每节点 = 4 actor + 4 rollout(对齐 qwen3-vl-16gpu):
#   - Actor:   2 节点 × 4 GPU (TP=4, DP=2)。dense 无 MoE/EP 约束，actor 拆 2 节点使每节点
#              host RAM 减半(8 个 train actor 各~120GB 挤单节点会 Ray host-RAM OOM)。
#   - Rollout: 8 GPUs as 8 sglang engines of 1 GPU each(与 actor 同节点共存)。
#   TP=4 时 num_query_groups=4 不触发 attention.py 的 GQA gate 二次切分(TP=8 才触发)。
#
# 用法:
#   # Head 节点(有 actor GPU 的那台):
#   bash scripts/gui_qwen3.5_9B_16gpu.sh
#   # 每个 Worker 节点(另起终端/另一台机器)：
#   RAY_HEAD_ADDR=<head_ip> WORKER_NUM_GPUS=8 bash scripts/gpu_worker_join_ray.sh
#
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

# 16 GPUs across 2 nodes: actor 8 (2节点×4, TP=4, DENSE 无 EP) + rollout 8 (8 engines x 1 GPU).
NUM_GPUS=${NUM_GPUS:-16}
ACTOR_GPUS=${ACTOR_GPUS:-8}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-8}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
# actor 拆 2 节点 × 4 GPU(对齐 qwen3-vl-16gpu)：dense 无 EP 约束，分散后每节点只有 4 个 train
# actor，host RAM 减半，避免 8 actor 挤单节点的 Ray host-RAM OOM。
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-2}
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
export GUI_AGENT_CLASS_PATH=${GUI_AGENT_CLASS_PATH:-"agents.qwen35_agent.Qwen35VLAgentLocal"}
export GUI_ENV_RUNTIME=${GUI_ENV_RUNTIME:-"cua_gym"}
export ENABLE_THINKING=${ENABLE_THINKING:-False}
# GUI_DISABLE_MTP=1: config 有 mtp_num_hidden_layers=1，但 GRPO 不用 MTP，且 bridge
# 映射对不上 -> 加载崩溃。由 slime/backends/megatron_utils/model_provider.py 消费。
export GUI_DISABLE_MTP=${GUI_DISABLE_MTP:-1}
MULTIMODAL_KEYS=${MULTIMODAL_KEYS:-'{"image":"images"}'}

WANDB_PROJECT=${WANDB_PROJECT:-slime_gui}
WANDB_GROUP=${WANDB_GROUP:-qwen3.5-9b-16gpu}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
GUI_PROJECT_NAME=${GUI_PROJECT_NAME:-slime_gui_9b_fast_16gpu_${RUN_TIMESTAMP}}
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

HF_CKPT=${HF_CKPT:-/mnt/llmshared-ssd-hd/wuchangqiao/data/models/Qwen/Qwen3.5-9B}
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
CKPT_NAME=${CKPT_NAME:-"gui-qwen3.5-9b-fast-16gpu"}
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
# On a single-node 8-actor DENSE 9B layout the pinned host copies are small, so
# pinning is left ON (Megatron default). Flip NO_PIN_CPU_*=1 only on host-RAM OOM.
if [[ "${NO_PIN_CPU_PARAMS:-0}" == "1" ]]; then
  OPTIMIZER_ARGS+=(--no-pin-cpu-params)
fi
if [[ "${NO_PIN_CPU_GRADS:-0}" == "1" ]]; then
  OPTIMIZER_ARGS+=(--no-pin-cpu-grads)
fi

# DENSE parallelism: actor 8 GPU 拆 2 节点，attention runs TP=4 (DP=2). NO experts.
# TP=4(而非 8)：dense 无 EP 约束可跨节点，拆 2 节点让每节点 host RAM 减半(避免 host OOM)。
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
  # 会把某个 TP=1 engine 剩余运行时显存打爆 -> CUDA OOM 死亡，采样期不报、update_weights
  # 调 pause_generation 时才暴露成 Connection refused。降到 0.7 给运行时留更多余量。
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
# 手动 Ray 启动: 本节点起 Head。其余 GPU 节点各自跑 gpu_worker_join_ray.sh 加入。
# ===========================================================================
RAY_TEMP_DIR=${RAY_TEMP_DIR:-"/mnt/llmshared-ssd-hd/chentongbo/ray"}
mkdir -p "${RAY_TEMP_DIR}"
# Plasma object store — MUST match gpu_worker_join_ray.sh (600GB)。含图 rollout 批很大。
OBJECT_STORE_GB=${OBJECT_STORE_GB:-600}
OBJECT_STORE_BYTES=$(( OBJECT_STORE_GB * 1024 * 1024 * 1024 ))
echo "Ray object store (plasma) = ${OBJECT_STORE_GB} GB"
ray start --head --num-gpus "${ACTOR_GPUS}" --object-store-memory ${OBJECT_STORE_BYTES} --temp-dir "${RAY_TEMP_DIR}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Head: 等待 worker 节点加入(rollout 8 GPU 在另一台)。EXPECTED_NODES 默认 2。
EXPECTED_NODES=${EXPECTED_NODES:-2}
echo "Waiting for ${EXPECTED_NODES} nodes to join Ray cluster (run gpu_worker_join_ray.sh on the other node)..."
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

RAY_JOB_SUBMISSION_ID=${RAY_JOB_SUBMISSION_ID:-"gui_qwen3.5_9b_16gpu_fast_$(date +%Y%m%d_%H%M%S)"}

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
