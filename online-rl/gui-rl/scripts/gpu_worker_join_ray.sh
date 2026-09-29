#!/bin/bash
#
# GPU Worker 脚本 — 只加入 Ray 集群，不运行训练逻辑
# ====================================================
# 用法(在【非 head】的 GPU 节点上跑):
#   RAY_HEAD_ADDR=<head_ip> WORKER_NUM_GPUS=8 bash scripts/gpu_worker_join_ray.sh
#
# 脚本会自己 activate <projects>/venvs/online-rl，无需调用者先激活环境。
# 覆盖开关: VENV_PATH=<path> / SKIP_VENV_ACTIVATE=1 / EXPECT_RAY_VERSION=<ver>

# --- Self-background -------------------------------------------------------------------
# Re-exec detached (setsid) with output to a log file, then free the terminal. The worker
# holds the Ray join alive via `sleep infinity`, so detaching keeps it running in the
# background. Set BACKGROUND=0 to run in foreground.
if [[ "${BACKGROUND:-1}" == "1" && -z "${_GPUWORKER_DETACHED:-}" ]]; then
  _RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
  mkdir -p "${_RUN_DIR}/logs"
  LOG_FILE="${LOG_FILE:-${_RUN_DIR}/logs/gpu_worker_$(date +%Y%m%d_%H%M%S).log}"
  _GPUWORKER_DETACHED=1 setsid bash "$0" "$@" > "${LOG_FILE}" 2>&1 < /dev/null &
  echo "Started GPU worker (Ray join) in background: PID=$!"
  echo "Log:  ${LOG_FILE}"
  echo "Tail: tail -f ${LOG_FILE}"
  echo "Stop: ray stop --force   (or kill this PID)"
  exit 0
fi

set -ex

# --- venv ---------------------------------------------------------------------------
# 自己激活 projects/venvs/online-rl，不依赖调用者的 shell 环境。
# 为什么必须做: head 是从这个 venv 起的 (ray 2.56.1)。如果 worker 用了 conda
# torch-base 的 ray 2.46.0，TCP 能连上 6379(下面的探活会通过)，但 GCS 的
# internal-KV 协议不兼容 -> `ConnectionError: Could not read 'temp_dir' from GCS`,
# 报错看起来像 GCS 没起来，其实 GCS 好得很，纯粹是版本错配。
# 路径: 本脚本在 <projects>/online-rl/gui-rl/scripts/ -> 上溯三级到 <projects>/。
_SCRIPTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECTS_ROOT="$(cd -- "${_SCRIPTS_DIR}/../../.." &>/dev/null && pwd)"
VENV_PATH=${VENV_PATH:-"${PROJECTS_ROOT}/venvs/online-rl"}
if [[ "${SKIP_VENV_ACTIVATE:-0}" != "1" ]]; then
  if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    # activate 里有未定义变量引用，set -u 下会炸;本脚本没开 -u，但保持防御。
    source "${VENV_PATH}/bin/activate"
    echo "venv activated: ${VENV_PATH} (python=$(which python3), ray=$(which ray))"
  else
    echo "ERROR: venv not found at ${VENV_PATH}/bin/activate"
    echo "  用 VENV_PATH=<path> 覆盖，或 SKIP_VENV_ACTIVATE=1 用当前环境(需自行保证 ray 版本一致)。"
    exit 1
  fi
fi

# ray 版本必须和 head 完全一致(见上)。不一致就 fail fast，别拖到 ray start 才抛
# 那个看不懂的 GCS ConnectionError。
EXPECT_RAY_VERSION=${EXPECT_RAY_VERSION:-2.56.1}
ACTUAL_RAY_VERSION=$(ray --version 2>&1 | awk '{print $NF}')
if [[ "${ACTUAL_RAY_VERSION}" != "${EXPECT_RAY_VERSION}" ]]; then
  echo "ERROR: ray version mismatch — head 需要 ${EXPECT_RAY_VERSION}, 本节点是 ${ACTUAL_RAY_VERSION}"
  echo "  ray binary: $(which ray)"
  echo "  版本不一致会导致 'Could not read temp_dir from GCS'(端口通但协议不兼容)。"
  echo "  用 EXPECT_RAY_VERSION=<ver> 覆盖此检查。"
  exit 1
fi
echo "ray version OK: ${ACTUAL_RAY_VERSION} ($(which ray))"

# 确保 libnuma.so.1 存在 (sgl_kernel 依赖)
if ! ldconfig -p | grep -q libnuma; then
  if command -v apt-get &>/dev/null; then
    apt-get update -qq && apt-get install -y -qq libnuma1 libnuma-dev 2>/dev/null || true
  elif command -v yum &>/dev/null; then
    yum install -y numactl-libs 2>/dev/null || true
  fi
  if ! ldconfig -p | grep -q libnuma; then
    NUMA_PATH=$(find /usr /opt /mnt -name "libnuma.so*" 2>/dev/null | head -1)
    if [[ -n "${NUMA_PATH}" ]]; then
      export LD_LIBRARY_PATH="$(dirname ${NUMA_PATH}):${LD_LIBRARY_PATH}"
    fi
  fi
fi

WORKER_NUM_GPUS=${WORKER_NUM_GPUS:-8}
RAY_HEAD_ADDR="${RAY_HEAD_ADDR:-${HEAD_IP:-${MASTER_ADDR:-127.0.0.1}}}"
RAY_HEAD_PORT=${RAY_HEAD_PORT:-6379}

echo "GPU Worker: joining Ray head at ${RAY_HEAD_ADDR}:${RAY_HEAD_PORT} with ${WORKER_NUM_GPUS} GPUs"

# ⚠️ 拒绝在 head 节点上跑本脚本。下面的 `ray stop --force` 是无条件的，在 head 上
# 执行会直接把整个集群的 GCS 干掉(head 和 worker 版本一致时必然发生)。
if hostname -I 2>/dev/null | tr ' ' '\n' | grep -qx "${RAY_HEAD_ADDR}"; then
  echo "ERROR: RAY_HEAD_ADDR=${RAY_HEAD_ADDR} 是本机 IP —— 这里是 head 节点。"
  echo "  worker 脚本只能在【其他】节点上跑; 在 head 上跑会被下面的 ray stop --force 杀掉集群。"
  echo "  本机 IP: $(hostname -I)"
  exit 1
fi

# 清理旧进程
pkill -9 sglang || true
ray stop --force || true
pkill -9 ray || true
sleep 2

# 等待 Head 端口可达
for attempt in $(seq 1 90); do
  if python3 -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('${RAY_HEAD_ADDR}', ${RAY_HEAD_PORT})); s.close()" 2>/dev/null; then
    echo "Ray head reachable at ${RAY_HEAD_ADDR}:${RAY_HEAD_PORT}"
    break
  fi
  if (( attempt == 90 )); then
    echo "ERROR: Cannot reach Ray head after 180s"
    exit 1
  fi
  sleep 2
done

# 加入 Ray 集群
RAY_TEMP_DIR=${RAY_TEMP_DIR:-"/mnt/llmshared-ssd-hd/chentongbo/ray"}
mkdir -p "${RAY_TEMP_DIR}"
# Plasma object store size — must match the head (fast.sh 默认 600GB)。默认 30% RAM
# (~200GB) 会被 fan-out 的含图 rollout 批打满 -> 刷 spill 拖死训练 -> NCCL 超时。
OBJECT_STORE_GB=${OBJECT_STORE_GB:-600}
# Clamp to /dev/shm (see head script) — plasma refuses to start if it exceeds /dev/shm.
SHM_GB=$(df -BG /dev/shm 2>/dev/null | awk 'NR==2{gsub(/G/,"",$2); print $2}')
if [[ -n "${SHM_GB}" ]] && (( OBJECT_STORE_GB > SHM_GB * 9 / 10 )); then
  echo "OBJECT_STORE_GB=${OBJECT_STORE_GB} exceeds 90% of /dev/shm (${SHM_GB}Gi); clamping to $(( SHM_GB * 9 / 10 ))"
  OBJECT_STORE_GB=$(( SHM_GB * 9 / 10 ))
fi
OBJECT_STORE_BYTES=$(( OBJECT_STORE_GB * 1024 * 1024 * 1024 ))
echo "Ray object store (plasma) = ${OBJECT_STORE_GB} GB"
# ⚠️ Cap --num-cpus (see head script). This node is also ~376 CPUs, so without a cap the
# raylet here prestarts ~376 python workers that storm registration and hang (workers
# never register -> rollout actors on this node never come up). Match the head's cap.
RAY_NUM_CPUS=${RAY_NUM_CPUS:-64}
# temp-dir: the comment below used to say "don't pass --temp-dir, ray ignores it
# on workers and inherits the head's value". Observed 2026-09-13 on this ray
# version: the inherited value arrives EMPTY, so `ray start` dies in
# _init_temp -> try_to_create_directory("") with
#   FileNotFoundError: [Errno 2] No such file or directory: ''
# Passing it explicitly costs at most an "option will be ignored" warning and
# makes the worker robust to that inheritance bug. Must match the head's value.
ray start --address="${RAY_HEAD_ADDR}:${RAY_HEAD_PORT}" --num-gpus ${WORKER_NUM_GPUS} --num-cpus "${RAY_NUM_CPUS}" --object-store-memory ${OBJECT_STORE_BYTES} --temp-dir "${RAY_TEMP_DIR}"

echo "GPU Worker RANK=${RANK:-?} joined. Sleeping until cluster shuts down..."
sleep infinity
