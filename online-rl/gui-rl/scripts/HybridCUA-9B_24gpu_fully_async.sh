#!/bin/bash
# Qwen3.5-9B (DENSE multimodal VLM) GRPO training on OSWorld GUI tasks, fully-async.
#
# Derived from gui_qwen3.5_9B_16gpu_fully_async_fast_platform.sh. 9B is a DENSE
# model (no experts), so ALL MoE/EP/DeepEP machinery is stripped. The fully-async
# rollout body (staleness filter, cuDNN ABI fix, plasma sizing, eval drain) is
# identical to the platform script.
#
# 9B model facts (from HF config.json + slime/scripts/models/qwen3.5-9B.sh):
#   - DENSE: num_experts absent -> no --expert-*/--moe-*/DeepEP flags.
#   - VLM:   config has vision_config/image_token_id (真多模态，自带 ViT)。
#   - GDN:   --use-gated-attention -> Gated Delta Net 线性注意力层。GDN 不支持 packed
#            sequence，所以必须 bshd + --micro-batch-size(见 PERF_ARGS 注释)。
#   - MTP:   mtp_num_hidden_layers=1 -> 需 GUI_DISABLE_MTP=1(否则 bridge 映射崩)。
#   - heads: num_attention_heads=16, num_query_groups=4。
#
# ⚠️ TP=8 触发 Megatron GQA output-gate 补丁(attention.py)：num_query_groups(4)<world_size(8)。
#    补丁已在 attention.py，会自动生效。本脚本 TP=4，不触发。
#
# ====================================================================
# 手动启动版 (manual Ray bring-up)。
#   - 本节点 ray start --head 起 Head 并提交 job;
#   - 其余 GPU 节点各自手动跑 gpu_worker_join_ray.sh 加入集群。
#
# 3-node layout (24 GPUs = 3 x 8) — 训推分离:
#   - Actor:   node0 + node1，每节点 8 GPU 全给训练 = 16 GPU (TP=4, DP=4)。
#   - Rollout: node2 独占 8 GPU = 8 个 sglang engine × 1 GPU。
#   与 16gpu 版的区别: 16gpu 是每节点 4 actor + 4 rollout 混布(共 2 节点);
#   这里 actor 和 rollout 完全分节点，互不抢 host RAM / PCIe。
#   DP 从 2 -> 4，每个训练 step 的样本数需要是 4 的倍数(见 PERF_ARGS 注释)。
#
# 用法:
#   # Head 节点(actor node0):
#   bash scripts/HybridCUA-9B_24gpu_fully_async.sh
#   # 另外 2 个节点各跑一次:
#   RAY_HEAD_ADDR=<head_ip> WORKER_NUM_GPUS=8 bash scripts/gpu_worker_join_ray.sh
#
#   staleness: 默认 =2 真过滤; ROLLOUT_MAX_STALENESS=999 仅观察 / =0 关闭。
# ====================================================================

# --- Self-background -------------------------------------------------------------------
# Re-exec detached (setsid) with all output to a log file, then return the shell to you.
# Training itself is a Ray job that already survives this script, so detaching here just
# frees the terminal + captures everything to the log. Set BACKGROUND=0 to run in foreground.
if [[ "${BACKGROUND:-1}" == "1" && -z "${_HYBRIDCUA24_DETACHED:-}" ]]; then
  _RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
  mkdir -p "${_RUN_DIR}/logs"
  LOG_FILE="${LOG_FILE:-${_RUN_DIR}/logs/hybridcua24_$(date +%Y%m%d_%H%M%S).log}"
  _HYBRIDCUA24_DETACHED=1 setsid bash "$0" "$@" > "${LOG_FILE}" 2>&1 < /dev/null &
  echo "Started HybridCUA-9B 24gpu in background: PID=$!"
  echo "Log:  ${LOG_FILE}"
  echo "Tail: tail -f ${LOG_FILE}"
  echo "Stop: ray job stop <submission_id> --address=http://127.0.0.1:8265   (or ray stop --force)"
  exit 0
fi

# Kill leftover python from a prior run, but SPARE the co-located OSWorld cluster
# env server (`python -m cluster.master.server`, :19000). A blanket `pkill -9 python`
# would kill it too. Match by cmdline so ONLY the env server is spared; everything else
# python (train/rollout pool/etc.) is still nuked. Override via GUI_ENV_SERVER_PROC_PATTERN.
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

# Official wandb.ai (uploaded via star-proxy, configured below). Provide YOUR OWN key:
#   export WANDB_API_KEY=<key from https://wandb.ai/authorize>
# To use the self-hosted instance instead: set WANDB_BASE_URL (that host) + its key + USE_STAR_PROXY=0.
: "${WANDB_API_KEY:?ERROR: WANDB_API_KEY must be set. Get your key from https://wandb.ai/authorize}"
export WANDB_API_KEY
# Official wandb cloud. MUST be a valid URL, NOT "" — wandb's pydantic Settings parses
# WANDB_BASE_URL from the env and rejects an empty string (url_parsing error) in wandb.login.
export WANDB_BASE_URL=${WANDB_BASE_URL:-"https://api.wandb.ai"}

# --- Outbound proxy (star-proxy) so the ray training job can reach wandb.ai ---------
# Taiji intranet node: no direct public egress. wandb.init runs INSIDE the ray job, so the
# proxy is injected ONLY into RUNTIME_ENV_JSON below (NOT this shell / Ray) — a proxy in
# Ray's env breaks job submission. no_proxy MUST cover localhost + ALL local node IPs so the
# job's sglang / env-server / cross-node traffic stays direct; only wandb goes out.
# 2-node note: this shell (head) and gpu_worker_join_ray.sh (worker) BOTH stay proxy-free.
# Set USE_STAR_PROXY=0 to disable (offline / self-hosted wandb).
USE_STAR_PROXY=${USE_STAR_PROXY:-0}
if [[ "${USE_STAR_PROXY}" == "1" ]]; then
  STAR_PROXY_URL=${STAR_PROXY_URL:-""}
  [[ -z "${STAR_PROXY_URL}" ]] && { echo "ERROR: USE_STAR_PROXY=1 requires STAR_PROXY_URL (e.g., http://your-proxy.example.com:3128)"; exit 1; }
  # requests/urllib3 no_proxy does NOT understand CIDR, so list EXACT local IPs (hostname -I)
  # plus <node-ip>/8 (for curl). For 2-node runs, append the OTHER node's IPs via STAR_NO_PROXY.
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
CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH:-"${SCRIPT_DIR}/scripts/gui_partial_async_osworld.yaml"}

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

export RAY_health_check_failure_threshold=${RAY_health_check_failure_threshold:-20}
export RAY_health_check_period_ms=${RAY_health_check_period_ms:-5000}
export RAY_health_check_timeout_ms=${RAY_health_check_timeout_ms:-30000}
export RAY_num_heartbeats_timeout=${RAY_num_heartbeats_timeout:-60}

# 24 GPUs across 3 nodes: actor 16 (4节点×4? 否 — 2节点×8, TP=4, DENSE 无 EP) + rollout 8。
NUM_GPUS=${NUM_GPUS:-24}
ACTOR_GPUS=${ACTOR_GPUS:-16}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-8}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
# actor 16 GPU = 2 节点 × 8。TP=4 -> DP=4。
# 注意与 16gpu 版的区别: 那里 actor 拆成 2×4 是为了和 rollout 同节点共存(每节点 4+4)。
# 这里 actor 独占 2 个节点(每节点 8 GPU 全给 actor), rollout 独占第 3 个节点, 互不挤 host RAM。
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-2}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}

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
WANDB_GROUP=${WANDB_GROUP:-hybridcua-9b-24gpu}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
GUI_PROJECT_NAME=${GUI_PROJECT_NAME:-slime_gui_hybridcua_9b_24gpu_${RUN_TIMESTAMP}}
export GUI_USER_ID="${GUI_USER_ID:-hybridcua_9b_24gpu}_${RUN_TIMESTAMP}"
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
CKPT_NAME=${CKPT_NAME:-"gui-hybridcua-9b-24gpu"}
SAVE_CKPT=${SAVE_CKPT:-"${CKPT_ROOT}/${CKPT_NAME}_${RUN_TIMESTAMP}"}
SAVE_HF_CKPT=${SAVE_HF_CKPT:-"${CKPT_ROOT}/${CKPT_NAME}_${RUN_TIMESTAMP}_hf/rollout_{rollout_id}"}
echo "Megatron checkpoint dir: ${SAVE_CKPT}"
echo "HuggingFace checkpoint template: ${SAVE_HF_CKPT}"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --save "${SAVE_CKPT}"
  --save-hf "${SAVE_HF_CKPT}"
  # 一个 rollout ≈ 35-45min(step_time 2100-2600s)，20 太稀疏：上一次 run 跑了 7h20m
  # 崩在 rollout 13，第一次保存要等到 rollout 19，全部进度丢失。10 ≈ 6-7h 一存。
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

# ===========================================================================
# FULLY-ASYNC: train rollout scheduler = slime's fully-async worker.
# ===========================================================================
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

# Consumer-side staleness filter. Default 2 = real filter (drop samples generated more
# than 2 rollouts behind the current policy) — the fully-async loop over-samples in the
# background, so without a cap the trainer can consume badly off-policy data.
# Override: 999 = observe-only (log staleness.jsonl, never drop); "" / 0 = fully disabled.
ROLLOUT_MAX_STALENESS=${ROLLOUT_MAX_STALENESS:-2}
if [[ -n "${ROLLOUT_MAX_STALENESS}" && "${ROLLOUT_MAX_STALENESS}" != "0" ]]; then
  ROLLOUT_ARGS+=(--rollout-max-staleness ${ROLLOUT_MAX_STALENESS})
  echo "Staleness filter ENABLED: --rollout-max-staleness ${ROLLOUT_MAX_STALENESS} (>=999 = observe-only, never drops)"
else
  echo "Staleness filter DISABLED (ROLLOUT_MAX_STALENESS empty/0 -> no curl, no staleness.jsonl)."
fi

# in-flight pool = sglang_server_concurrency x num_engines. Target 64 with
# 8 engines (1 GPU each) -> sglang_server_concurrency = 8.
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
# On a single-node 8-actor DENSE 9B layout the pinned host copies are small, so
# pinning is left ON (Megatron default). Flip NO_PIN_CPU_*=1 only on host-RAM OOM.
if [[ "${NO_PIN_CPU_PARAMS:-0}" == "1" ]]; then
  OPTIMIZER_ARGS+=(--no-pin-cpu-params)
fi
if [[ "${NO_PIN_CPU_GRADS:-0}" == "1" ]]; then
  OPTIMIZER_ARGS+=(--no-pin-cpu-grads)
fi

# DENSE parallelism: actor 16 GPU, TP=4 (DP=4)。OOM 靠 --log-probs-chunk-size 分块算
# log_probs 解决(不再一次性 materialize [seq x 248320] logits),不必牺牲 DP。
PERF_ARGS=(
  --tensor-model-parallel-size ${TRAIN_TP:-4}
  --log-probs-chunk-size ${LOG_PROBS_CHUNK_SIZE:-2048}
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size ${TRAIN_CP:-1}
  --megatron-to-hf-mode bridge
  # Qwen3.5 有 GDN(Gated Delta Net)线性注意力层，不支持 packed sequence。
  # 必须关 --use-dynamic-batch-size + 用 --qkv-format bshd，否则 compute_log_prob 到
  # GDN 层报 "NotImplementedError: GDN does not support packed sequence for now."。
  #
  # ⚠️ mbs 只能是 1。bshd 走 dp_schedule.py 的静态打包路径，num_mbs 必须是 dp_size 的
  # 倍数，否则直接 raise AssertionError("static path: num_mbs is not a multiple of ...")。
  # 这里 DP=4，mbs=1 时要求每步样本数 n % 4 == 0；mbs=2 会要求 n % 8 == 0。
  # dynamic_history 下 n 由轨迹步数决定(不定)，rollout.py:841 的
  # `dynamic_gbs = (per_step_target // dp_size) * dp_size` 只保证 gbs 对齐 dp_size，
  # 所以 mbs>1 会在训练中途随机崩。要调大 mbs 得先让 dp_size=1(TP=8)。
  --qkv-format bshd
  --micro-batch-size 1
$(if [[ "${RECOMPUTE:-full}" == "none" ]]; then echo ""; else echo "  --recompute-granularity ${RECOMPUTE:-full} --recompute-method ${RECOMPUTE_METHOD:-uniform} --recompute-num-layers ${RECOMPUTE_NUM_LAYERS:-1}"; fi)
  # bshd 下这个参数没有作用(dp_schedule.py 只在 use_dynamic_batch_size 时读它)，留着备用。
  --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-16384}
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
SGLANG_ARGS=(
  --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}
  # mem-fraction 0.7(非 0.8): 9B 是 GDN 混合模型，mamba state cache 与 KV cache 是两套
  # 独立显存。0.8 时长 GUI 截图(单请求可达 11390 token)的 prefill 激活 + mamba state 峰值
  # 会把某个 TP=1 engine 剩余运行时显存打爆 -> CUDA OOM 死亡，采样期不报、update_weights
  # 调 pause_generation 时才暴露成 Connection refused。降到 0.7 给运行时留更多余量。
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

# 追加任意 slime 参数, 空格分隔。用于 debug 对照跑（如 --save-debug-rollout-data）。
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
# Clamp to /dev/shm. Ray's plasma store lives in /dev/shm and REFUSES to start if the
# requested object-store size exceeds it (ValueError). Cap to ~90% of /dev/shm so this
# works regardless of the box's shm size (here shm≈494Gi, so 600 gets clamped to ~444).
SHM_GB=$(df -BG /dev/shm 2>/dev/null | awk 'NR==2{gsub(/G/,"",$2); print $2}')
if [[ -n "${SHM_GB}" ]] && (( OBJECT_STORE_GB > SHM_GB * 9 / 10 )); then
  echo "OBJECT_STORE_GB=${OBJECT_STORE_GB} exceeds 90% of /dev/shm (${SHM_GB}Gi); clamping to $(( SHM_GB * 9 / 10 ))"
  OBJECT_STORE_GB=$(( SHM_GB * 9 / 10 ))
fi
OBJECT_STORE_BYTES=$(( OBJECT_STORE_GB * 1024 * 1024 * 1024 ))
echo "Ray object store (plasma) = ${OBJECT_STORE_GB} GB"
# ⚠️ Cap --num-cpus. Ray prestarts `num_cpus` python workers at startup
# (services.py: --num_prestart_python_workers=int(num_cpus)). On this box nproc≈376,
# so it forks ~376 workers that each import the heavy venv (torch/sglang/TE) and race to
# register with the single-threaded raylet. That registration storm wedges the raylet:
# CoreWorker::RegisterWorkerToRaylet blocks in recv() forever, which also freezes the
# dashboard agent's ray.init() -> JobAgent(:52365) never answers -> `ray job submit`
# hangs 300s -> 504. Capping to ~64 kills the storm (submit works instantly) while still
# leaving far more CPU slots than the job's ~20-30 Ray actors need.
RAY_NUM_CPUS=${RAY_NUM_CPUS:-64}
# Head 只贡献本机的 GPU。16gpu 版这里用 ACTOR_GPUS(=8)恰好等于单机 GPU 数，纯属巧合;
# 24gpu 下 ACTOR_GPUS=16 跨 2 台，head 只能报自己的 8 张，否则 Ray 会以为本机有 16 张
# -> placement group 全 PACK 到 head -> 8 个 bundle 拿不到 GPU 永久 pending。
HEAD_NUM_GPUS=${HEAD_NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
if (( HEAD_NUM_GPUS < 1 )); then HEAD_NUM_GPUS=8; fi
echo "Head contributes ${HEAD_NUM_GPUS} GPUs to the Ray cluster"
ray start --head --num-gpus "${HEAD_NUM_GPUS}" --num-cpus "${RAY_NUM_CPUS}" --object-store-memory ${OBJECT_STORE_BYTES} --temp-dir "${RAY_TEMP_DIR}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Head: 等待另外 2 个节点加入(actor 第 2 节点 8 GPU + rollout 节点 8 GPU)。
EXPECTED_NODES=${EXPECTED_NODES:-3}
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

RAY_JOB_SUBMISSION_ID=${RAY_JOB_SUBMISSION_ID:-"hybridcua_9b_24gpu_fully_async_$(date +%Y%m%d_%H%M%S)"}

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

# Entry: private GUI-RL copy of slime/train_async.py under gui-rl/.
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
