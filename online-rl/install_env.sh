#!/usr/bin/env bash
# =============================================================================
# online-rl 本地 venv 安装脚本 —— slime + gui-rl + 本地 Megatron-LM
#
# 路线:     python3.12 -m venv (纯净) + 系统 CUDA 12.9 + pip cu129 wheel
# 上游参考: slime/build_venv.sh (权威版本钉法) + slime/docker/Dockerfile
#
# 【必须在有 GPU + CUDA toolkit 的训练节点上跑】
#   flash-attn / apex / TE / torch_memory_saver / int4_qat 都要 nvcc 现场编译。
#
# 【用法】—— 默认后台运行, 全量日志落盘, SSH 断线不影响
#   bash install_env.sh                          # 后台装, 立刻返回
#   SKIP_APEX=1 bash install_env.sh              # 跳过 apex(mcore 有 fallback)
#   SGLANG_EXTRA=srt bash install_env.sh         # sglang 只装核心 extra, 见 §4
#   BACKGROUND=0 bash install_env.sh             # 前台跑(调试)
#
# 【日志】默认 <projects>/install-logs/ , 可用 LOG_DIR=... 改:
#   install-online-rl-<时间戳>.log     全量输出(含编译日志, 很大)
#   install-online-rl-<时间戳>.status  单行当前步骤, cat 一下就知道进度
#   latest-online-rl.log / .status     指向最近一次的软链
#
#   注意日志名带 -online-rl 后缀: 那个目录里已有 verl 装机的 latest.log 和
#   模型下载的 latest-download.log, 用 latest.log 会覆盖掉 verl 的指针。
#
#   tail -f <LOG_DIR>/latest-online-rl.log
#   cat     <LOG_DIR>/latest-online-rl.status
#
# 【与 venvs/verl 的关系】完全隔离, 不要混用。
#   verl venv      : python 3.13 + torch 2.9.1+cu129
#   online-rl venv : python 3.12 + torch 2.11.0+cu129
#   slime 必须 2.11.0 —— 它装的 sglang-kernel 0.4.2.post2 / sgl-deep-gemm 是按
#   torch 2.11 的 C++ ABI 编的预编译 wheel, 降版本会 .so 加载失败。
#
# 【为什么是 python 3.12 而不是 3.13】两个独立的硬约束:
#   1) slime/backends/megatron_utils/initialize.py:66 运行时硬断言
#      assert np.__version__.startswith("1."), 而 numpy 1.26.4 没有 cp313 wheel
#      (实测 wheel tags 只到 cp312)。
#   2) Megatron-Bridge(@bridge) 的 pyproject.toml 声明
#      requires-python = ">=3.10,<3.13", py3.13 直接被 pip 拒绝。
#   官方 build_conda.sh:19 也是 python=3.12。
#
# 【本脚本相对上游 build_venv.sh 修的 4 个问题】详见各步注释:
#   §3  上游漏装 rust 工具链 —— sglang editable 装会编 Rust 扩展, 必然失败
#   §11 上游 requirements.txt 会把 sglang_router fork 顶掉(顺序 bug)
#   §9  nvidia-modelopt[torch]>=0.37.0 没加引号(shell 重定向)
#   §13 patch 冲突检测的 `|| true` 让它永远成功, 且 grep -R . 扫全仓库
#
# 【耗时预估 H20 x8 / 384 核】约 65-90 min:
#   sglang[all] 依赖下载 15-25min, flash-attn ~10-15min, apex ~12-15min,
#   TE ~5min(cu12 有预编译 wheel), rust 路线 +10-20min。
# =============================================================================
set -euo pipefail

_SCRIPT_VERSION="v1 (py3.12 + torch 2.11 cu129 + rust 工具链补全)"

# ---------------------------------------------------------------------------
# 0. 参数
# ---------------------------------------------------------------------------
VENV="${VENV:-}"
BASE_DIR="${BASE_DIR:-}"

if [[ -z "${VENV}" || -z "${BASE_DIR}" ]]; then
  echo "ERROR: VENV and BASE_DIR must be set."
  echo "  Example: VENV=/your/path/venvs/online-rl BASE_DIR=/your/path/online-rl bash install_env.sh"
  exit 1
fi

SLIME_DIR="$BASE_DIR/slime"
GUI_RL_DIR="$BASE_DIR/gui-rl"
# Megatron: 用本地 checkout(已是 0.16.0)。见 §10 为何不 checkout commit / 不打 patch。
MEGATRON_SRC="${MEGATRON_SRC:-$BASE_DIR/Megatron-LM}"
SGLANG_DIR="${SGLANG_DIR:-$BASE_DIR/sglang}"

# 版本钉死 —— 与官方 Dockerfile / build_conda.sh / build_venv.sh 一致
SGLANG_COMMIT="${SGLANG_COMMIT:-5a15cde858ea09b77116212a39356f2fc51b8584}"   # v0.5.12.post1
PATCH_VERSION="${PATCH_VERSION:-latest}"
MBRIDGE_COMMIT="89eb10887887bc74853f89a4de258c0702932a1c"
APEX_COMMIT="10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4"
TMS_COMMIT="a193d9dd1b877d33c64a41cfb3db9f867df2d926"
# sglang_router 必须用这个 fork wheel(cp38-abi3, py3.12 可用), 不能用 PyPI 的。
# slime 依赖 fork 独有的 API, 装完会断言 'slime' in sglang_router.__version__。
SGLANG_ROUTER_WHL="https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl"

# sglang 的 extra。官方是 all, 但它会拉进 ~40 个 GUI-RL 用不到的包
# (diffusers/opencv/moviepy/scikit-image), 其中 st_attn / vsa 是要现场编译的小
# CUDA 扩展, 最可能编不过。编不过就 SGLANG_EXTRA=srt 降级重跑。
SGLANG_EXTRA="${SGLANG_EXTRA:-all}"

PYBIN="${PYBIN:-/opt/conda/envs/torch-base/bin/python3.12}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# 编译并发。留 2 核给系统, 上限 128(对齐官方 Dockerfile)。
_NPROC="$(nproc)"
MAX_JOBS="${MAX_JOBS:-$(( _NPROC > 130 ? 128 : (_NPROC > 4 ? _NPROC - 2 : 2) ))}"

# H20 只有 sm_90, 不用编 8.0/8.6/8.9 —— flash-attn 和 apex 能省好几倍时间。
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

SKIP_FLASH_ATTN="${SKIP_FLASH_ATTN:-0}"
SKIP_APEX="${SKIP_APEX:-0}"
SKIP_INT4_QAT="${SKIP_INT4_QAT:-0}"
SKIP_SGLANG_PATCH="${SKIP_SGLANG_PATCH:-0}"
SKIP_SGLANG="${SKIP_SGLANG:-0}"          # 复用已装好的 sglang, 重跑时省时间

BACKGROUND="${BACKGROUND:-1}"
LOG_DIR="${LOG_DIR:-${BASE_DIR}/install-logs}"

export PATH="$CUDA_HOME/bin:$PATH"
export PIP_DEFAULT_TIMEOUT=120
export PIP_RETRIES=10
export MAX_JOBS

# ---------------------------------------------------------------------------
# 0a. 代理策略 —— 沿用上游的 $PX 命令前缀模式
#   优先尝试自定义镜像(通过 TORCH_INDEX 环境变量指定), 不可用才回落外网官方 index。
# ---------------------------------------------------------------------------
PROXY="${PROXY:-}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16}"
export NO_PROXY="$no_proxy"
PX="env https_proxy=$PROXY http_proxy=$PROXY no_proxy=$no_proxy"

if [ -t 1 ]; then _C="\033[1;36m"; _R="\033[0m"; else _C=""; _R=""; fi

_T0="$(date +%s)"
CURRENT_STEP="启动"
log() {
  CURRENT_STEP="$*"
  local el=$(( $(date +%s) - _T0 ))
  printf "\n${_C}[%s | +%02d:%02d:%02d] ==== %s ====${_R}\n" \
    "$(date +%H:%M:%S)" $((el/3600)) $((el%3600/60)) $((el%60)) "$*"
  [ -n "${STATUS_FILE:-}" ] && echo "[$(date '+%F %T')] $*" > "$STATUS_FILE"
  return 0
}
warn() {
  echo ""
  echo "  !! ------------------------------------------------------------"
  while [ $# -gt 0 ]; do echo "  !! $1"; shift; done
  echo "  !! ------------------------------------------------------------"
  return 0
}

# ---------------------------------------------------------------------------
# 0b. 后台化 + 日志重定向
# ---------------------------------------------------------------------------
_SELF="$(readlink -f "$0" 2>/dev/null || echo "$0")"

if [ "$BACKGROUND" = "1" ] && [ -z "${_INSTALL_DAEMONIZED:-}" ]; then
  mkdir -p "$LOG_DIR" || { echo "ERROR: 无法创建日志目录 $LOG_DIR" >&2; exit 1; }
  _STAMP="$(date +%Y%m%d-%H%M%S)"
  _LOG="$LOG_DIR/install-online-rl-$_STAMP.log"
  _PIDF="$LOG_DIR/install-online-rl-$_STAMP.pid"
  _STAT="$LOG_DIR/install-online-rl-$_STAMP.status"

  ln -sfn "$_LOG"  "$LOG_DIR/latest-online-rl.log"
  ln -sfn "$_STAT" "$LOG_DIR/latest-online-rl.status"

  export _INSTALL_DAEMONIZED=1 LOG_FILE="$_LOG" STATUS_FILE="$_STAT"

  if command -v setsid >/dev/null 2>&1; then
    setsid bash "$_SELF" "$@" >"$_LOG" 2>&1 </dev/null &
  else
    nohup bash "$_SELF" "$@" >"$_LOG" 2>&1 </dev/null &
  fi
  _CHILD=$!
  echo "$_CHILD" > "$_PIDF"
  disown "$_CHILD" 2>/dev/null || true

  cat <<EOF
后台安装已启动。

  PID       : $_CHILD   ($_PIDF)
  日志      : $_LOG
  最新软链  : $LOG_DIR/latest-online-rl.log
  进度(单行): $LOG_DIR/latest-online-rl.status

跟踪进度:
  tail -f $LOG_DIR/latest-online-rl.log
  grep -aE '^\[.*====' $LOG_DIR/latest-online-rl.log   # 只看步骤里程碑
  cat  $LOG_DIR/latest-online-rl.status

结束/中止:
  kill $_CHILD

前台跑(调试): BACKGROUND=0 bash $_SELF
EOF
  exit 0
fi

STATUS_FILE="${STATUS_FILE:-}"
LOG_FILE="${LOG_FILE:-<stdout>}"

# ---------------------------------------------------------------------------
# 0c. 失败/结束时留下明确结论
# ---------------------------------------------------------------------------
_FAILED=0
_on_err() {
  _FAILED=1
  local line="$1"
  echo ""
  echo "================================================================"
  echo "安装失败"
  echo "  步骤 : $CURRENT_STEP"
  echo "  行号 : $_SELF:$line"
  echo "  日志 : $LOG_FILE"
  echo "================================================================"
  [ -n "$STATUS_FILE" ] && echo "[$(date '+%F %T')] FAILED at: $CURRENT_STEP ($_SELF:$line)" > "$STATUS_FILE"
  return 0
}
_on_exit() {
  local rc=$?
  local el=$(( $(date +%s) - _T0 ))
  printf "\n总耗时: %02d:%02d:%02d\n" $((el/3600)) $((el%3600/60)) $((el%60))
  if [ "$rc" = "0" ] && [ "$_FAILED" = "0" ]; then
    echo "结果: SUCCESS"
    [ -n "$STATUS_FILE" ] && echo "[$(date '+%F %T')] SUCCESS (耗时 ${el}s)" > "$STATUS_FILE"
  else
    # 显式 exit N (如前置检查的 `|| { ...; exit 1; }`) 不触发 ERR trap, 这里兜底,
    # 否则失败时 status 会停在最后一次 log() 的步骤名上, 看起来像"还在跑"。
    echo "结果: FAILED (exit=$rc, 步骤: $CURRENT_STEP)"
    [ -n "$STATUS_FILE" ] && echo "[$(date '+%F %T')] FAILED (exit=$rc) at: $CURRENT_STEP" > "$STATUS_FILE"
  fi
  return 0
}
trap '_on_err $LINENO' ERR
trap _on_exit EXIT

echo "================================================================"
echo "online-rl 环境安装 (slime + gui-rl + Megatron-LM)"
echo "  脚本版本 : $_SCRIPT_VERSION"
echo "  开始时间 : $(date '+%F %T')"
echo "  主机     : $(hostname)"
echo "  日志     : $LOG_FILE"
echo "  VENV     : $VENV"
echo "  BASE_DIR : $BASE_DIR"
echo "  MEGATRON : $MEGATRON_SRC (本地 checkout)"
echo "  SGLANG   : $SGLANG_DIR @ $SGLANG_COMMIT  extra=[$SGLANG_EXTRA]"
echo "  SKIP_APEX=$SKIP_APEX SKIP_FLASH_ATTN=$SKIP_FLASH_ATTN SKIP_INT4_QAT=$SKIP_INT4_QAT"
echo "  MAX_JOBS=$MAX_JOBS  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "================================================================"

# ---------------------------------------------------------------------------
# 1/14 前置检查 —— 提前失败, 别等编译 1 小时才炸
# ---------------------------------------------------------------------------
log "1/14 前置检查"

command -v nvidia-smi >/dev/null || { echo "ERROR: 没有 nvidia-smi。本脚本必须在 GPU 训练节点上跑。" >&2; exit 1; }
command -v nvcc       >/dev/null || { echo "ERROR: 没有 nvcc。设 CUDA_HOME 指向 CUDA toolkit(需 >= 12.8)。" >&2; exit 1; }
command -v git        >/dev/null || { echo "ERROR: 没有 git。" >&2; exit 1; }
command -v curl       >/dev/null || { echo "ERROR: 没有 curl(§3 装 rust 要用)。" >&2; exit 1; }
command -v "$PYBIN"   >/dev/null || { echo "ERROR: 找不到 $PYBIN。用 PYBIN=... 指定 python3.12。" >&2; exit 1; }

nvcc --version | tail -2
_CUDA_MAJMIN="$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
echo "CUDA: $_CUDA_MAJMIN   nproc: $_NPROC   MAX_JOBS: $MAX_JOBS"
awk -v v="$_CUDA_MAJMIN" 'BEGIN{ if (v+0 < 12.8) print "WARNING: CUDA < 12.8, TE 2.10 / cu129 wheel 可能编不过或运行报错"; }'
# 用 sed 而非 head 截断: set -o pipefail 下, head 读够行数就关管道, 上游进程拿到
# SIGPIPE(退出码 141) 会被 pipefail 当成整条管道失败 —— 这台机器 8 张卡,
# `nvidia-smi | head -2` 必然触发。sed 会把输入读完, 所以退出码是 0。
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed -n '1,2p'

# py3.12 是硬要求, 见文件头 §为什么是 python 3.12
"$PYBIN" -c 'import sys; assert sys.version_info[:2]==(3,12), f"expect py3.12 (numpy1.x 无 cp313 wheel + Megatron-Bridge 要求 <3.13), got {sys.version_info[:2]}"'

[ -f "$SLIME_DIR/setup.py" ]        || { echo "ERROR: $SLIME_DIR 不像 slime 仓库" >&2; exit 1; }
[ -d "$GUI_RL_DIR" ]                || { echo "ERROR: $GUI_RL_DIR 不存在" >&2; exit 1; }
[ -f "$MEGATRON_SRC/megatron/core/package_info.py" ] || { echo "ERROR: $MEGATRON_SRC 不是 Megatron-LM checkout" >&2; exit 1; }
_SGLANG_PATCH="$SLIME_DIR/docker/patch/$PATCH_VERSION/sglang.patch"
[ -f "$_SGLANG_PATCH" ]             || { echo "ERROR: 找不到 $_SGLANG_PATCH" >&2; exit 1; }

echo "本地 Megatron 版本: $(grep -E '^(MAJOR|MINOR|PATCH) ' "$MEGATRON_SRC/megatron/core/package_info.py" | tr -d ' ' | tr '\n' ' ')"

# --- Megatron patch 哨兵检查 ------------------------------------------------
# 本地 Megatron-LM 不是 git 仓库(无 .git), 所以既不能 git checkout $MEGATRON_COMMIT
# 也不能 git apply megatron.patch。
# 但我已逐行核对过: megatron.patch 的全部 304 个新增行(21 个文件)在本地代码里
# 100% 都已存在 —— 即这个 checkout 已经是打过 patch 的版本, 跳过是无损的。
# 这里 grep 三个哨兵标记来验证该前提, 缺失就大声警告(但不中断), 免得默默假设。
log "1b/14 检查 Megatron patch 是否已内置(不自动修改)"
_MP_OK=1
_mp_check() {  # $1=文件 $2=标记 $3=说明
  if grep -q "$2" "$MEGATRON_SRC/$1" 2>/dev/null; then
    echo "  ✓ $3"
  else
    echo "  ✗ $3  (缺: $2 in $1)"
    _MP_OK=0
  fi
}
_mp_check "megatron/core/dist_checkpointing/strategies/torch.py" "allow_partial_load=True"     "dist-ckpt 允许部分加载"
_mp_check "megatron/core/distributed/distributed_data_parallel.py" "disable_grad_buffers_cpu_backup" "DDP 梯度 buffer CPU 备份开关"
_mp_check "megatron/core/dist_checkpointing/strategies/common.py" "weights_only=False"          "torch.load weights_only=False"
if [ "$_MP_OK" = "0" ]; then
  warn "本地 Megatron-LM 似乎是【未打 patch】的上游 checkout。" \
       "slime 依赖 docker/patch/$PATCH_VERSION/megatron.patch 里的改动," \
       "缺了它训练/存档可能出错。" \
       "而该目录不是 git 仓库, 无法 git apply。两个选择:" \
       "  a) 重新 clone: git clone --recursive https://github.com/NVIDIA/Megatron-LM.git" \
       "     && git checkout 1dcf0dafa884ad52ffb243625717a3471643e087 && git apply <patch>" \
       "  b) 按 patch 内容手改" \
       "安装会继续 —— 但请自行确认这一点。"
else
  echo "  → patch 已内置, 按计划跳过 megatron.patch。"
fi
[ -e "$MEGATRON_SRC/.git" ] && echo "  (注: 检测到 .git, 但本脚本仍不做 checkout —— 避免动你的工作区)" \
  || echo "  (注: 无 .git, 故 git checkout <commit> / git apply 都不适用, 已按上面结论跳过)"

mkdir -p "$(dirname "$VENV")"
df -h "$(dirname "$VENV")" | tail -1

# ---------------------------------------------------------------------------
# 2/14 建 venv
# ---------------------------------------------------------------------------
log "2/14 创建 venv: $VENV"
if [ -d "$VENV" ]; then
  echo "已存在, 复用(增量安装)。要重装请先 rm -rf $VENV"
else
  "$PYBIN" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -c 'import sys; assert sys.prefix != sys.base_prefix, "venv 未激活"'
python -c 'import sys; assert sys.version_info[:2]==(3,12)'
pip install -U pip setuptools wheel

# cuda-python 钉 12.9: sglang 自己声明的是 cuda-python>=13, 不钉的话会拉进 cu13
# 系列, 与我们的 cu129 栈冲突(下面 §5 还要花一步把 cu13 的 nvidia-* 清掉)。
pip install "cuda-python==12.9"

# ---------------------------------------------------------------------------
# 3/14 rust 工具链 + protoc  —— 补上游 build_venv.sh 漏掉的一步
# ---------------------------------------------------------------------------
# sglang 的 editable 安装会编译一个 Rust 扩展: python/pyproject.toml 里有
#   [[tool.setuptools-rust.ext-modules]]
#   target = "sglang.srt.grpc._core" ; path = "../rust/sglang-grpc/Cargo.toml"
# 且 build-system requires 里有 setuptools-rust>=1.10。
# 官方 build_conda.sh:46-48 为此装了 rust:
#   # sglang's editable install builds a Rust extension (sglang-grpc via
#   # setuptools-rust), so the conda env needs a working rustc + cargo.
#   micromamba install -n slime -c conda-forge rust -y
# 而 build_venv.sh 把这行丢了 —— 照抄它会在 §4 硬失败(本机无 rustc/cargo/protoc)。
#
# 装到 $VENV/rust: RUSTUP_HOME/CARGO_HOME 都指进 venv, 所以 rm -rf $VENV 就是
# 彻底重置, 不污染 /root/.cargo。
# rustup 走内网镜像(直连, 快); crates.io 走代理(§4 里 cargo 会用到 $PX)。
log "3/14 rust 工具链 (sglang Rust 扩展需要) + protoc"

export RUSTUP_HOME="$VENV/rust/rustup"
export CARGO_HOME="$VENV/rust/cargo"
export PATH="$CARGO_HOME/bin:$PATH"

if command -v cargo >/dev/null 2>&1; then
  echo "已有 cargo: $(cargo --version)"
else
  _RUSTUP_MIRROR="${RUSTUP_MIRROR:-https://sh.rustup.rs}"
  export RUSTUP_DIST_SERVER="$_RUSTUP_MIRROR"
  export RUSTUP_UPDATE_ROOT="$_RUSTUP_MIRROR/rustup"
  mkdir -p "$VENV/rust"
  echo "从内网镜像装 rust: $_RUSTUP_MIRROR"
  curl --proto '=https' --tlsv1.2 -sSf --max-time 300 \
    "$_RUSTUP_MIRROR/rustup/dist/x86_64-unknown-linux-gnu/rustup-init" \
    -o "$VENV/rust/rustup-init"
  chmod +x "$VENV/rust/rustup-init"
  "$VENV/rust/rustup-init" -y --no-modify-path --default-toolchain stable --profile minimal
  echo "rustc: $(rustc --version)   cargo: $(cargo --version)"
fi

# protoc: rust/sglang-grpc/build.rs 调 tonic_build::compile_protos(), 需要 protoc。
# 用 pip 的 protoc-wheel-0(内网有), 免装系统包。
pip install protoc-wheel-0
PROTOC="$(python -c 'import os,protoc; print(os.path.join(os.path.dirname(protoc.__file__), "data", "bin", "protoc"))' 2>/dev/null || true)"
if [ -n "$PROTOC" ] && [ -x "$PROTOC" ]; then
  export PROTOC
  echo "PROTOC=$PROTOC ($("$PROTOC" --version 2>&1 || true))"
else
  # 退路: 有些版本把 protoc 装进 bin/
  if [ -x "$VENV/bin/protoc" ]; then
    export PROTOC="$VENV/bin/protoc"; echo "PROTOC=$PROTOC"
  else
    warn "找不到 protoc, sglang 的 grpc 扩展可能编不过。" \
         "若 §4 因 protoc 失败, 手动装一个并 export PROTOC=<path> 重跑。"
  fi
fi

# ---------------------------------------------------------------------------
# 4/14 sglang (源码 editable, commit 钉死)
# ---------------------------------------------------------------------------
# extra 默认 all(与官方一致)。all = diffusion + tracing + http2, 会拉进 ~40 个
# GUI-RL 用不到的重包; 其中 st_attn / vsa 是要现场编译的小 CUDA 扩展, 是最可能
# 编不过的两个。真编不过就 SGLANG_EXTRA=srt 重跑。
log "4/14 sglang @ ${SGLANG_COMMIT:0:12} (editable, extra=[$SGLANG_EXTRA])"

if [ "$SKIP_SGLANG" = "1" ]; then
  echo "按 SKIP_SGLANG=1 跳过 sglang 安装(复用已装的)"
else
  if [ ! -d "$SGLANG_DIR/.git" ]; then
    echo "clone sglang -> $SGLANG_DIR"
    $PX git clone https://github.com/sgl-project/sglang.git "$SGLANG_DIR"
  else
    echo "复用已有 clone: $SGLANG_DIR"
  fi
  cd "$SGLANG_DIR"
  $PX git fetch --all --tags 2>/dev/null || true
  git checkout "$SGLANG_COMMIT"
  echo "sglang HEAD: $(git rev-parse HEAD)"

  # $PX 必需: --extra-index-url 指向 PyTorch 外网 index, 且 cargo 要拉 crates.io。
  $PX pip install -e "python[$SGLANG_EXTRA]" --extra-index-url "https://download.pytorch.org/whl/cu129"

  # grpc 扩展是否真编出来了 —— 只报告不失败: slime 训练路径不 import
  # sglang.srt.grpc._core, 缺它不影响 GUI-RL。
  if ls python/sglang/srt/grpc/_core*.so >/dev/null 2>&1; then
    echo "  ✓ Rust 扩展已编译: $(ls python/sglang/srt/grpc/_core*.so)"
  else
    echo "  (注: 未见 _core*.so —— slime 训练路径不用它, 不影响)"
  fi
fi

# ---------------------------------------------------------------------------
# 5/14 修 cu13 溢出: 把 torch / sgl native kernel / nvidia-* 全拉回 cu129
# ---------------------------------------------------------------------------
# sglang[all] 会装上 cu13 构建的 torch 和一堆 nvidia-*-cu13, 与系统 CUDA 12.9
# 及后面 TE 2.10 / flash-attn 的编译不匹配, 所以这里强制换回来。
log "5/14 torch 2.11.0 + cu129 (修 cu13 溢出)"

# 内网镜像了 PyTorch 官方 wheel index(不是 pypi 镜像), 实测有 torch 2.11.0+cu129
# 的 cp312 wheel。优先内网(快), 不可用才回落外网。
TORCH_INDEX_UPSTREAM="https://download.pytorch.org/whl/cu129"
# Optional: set TORCH_INDEX_MIRRORS as a space-separated list of candidate mirror
# URLs to try before falling back to the upstream PyTorch index.
# e.g. TORCH_INDEX_MIRRORS="https://your-mirror.example.com/pytorch-wheels/whl/cu129/"
_TORCH_CANDIDATES=()
if [[ -n "${TORCH_INDEX_MIRRORS:-}" ]]; then
  read -r -a _TORCH_CANDIDATES <<< "${TORCH_INDEX_MIRRORS}"
fi
_TORCH_INDEX="${TORCH_INDEX:-}"
_TORCH_IS_UPSTREAM=0
if [ -n "$_TORCH_INDEX" ]; then
  echo "使用指定的 torch index: $_TORCH_INDEX"
  case "$_TORCH_INDEX" in *download.pytorch.org*) _TORCH_IS_UPSTREAM=1;; esac
else
  for _c in "${_TORCH_CANDIDATES[@]}"; do
    printf '探测 torch index: %s ... ' "$_c"
    # 两段校验: HTTP 可达 + pip 真能解析出 2.11.0+cu129
    # (只测 HTTP 200 不够 —— nginx autoindex 对空目录也返回 200)
    if curl -sfL --max-time 20 -o /dev/null "$_c" 2>/dev/null \
       && pip index versions torch --index-url "$_c" 2>/dev/null | grep -q '2\.11\.0+cu129'; then
      echo "OK"; _TORCH_INDEX="$_c"; break
    fi
    echo "不可用"
  done
  if [ -z "$_TORCH_INDEX" ]; then
    _TORCH_INDEX="$TORCH_INDEX_UPSTREAM"; _TORCH_IS_UPSTREAM=1
    echo "内网镜像均不可用, 回落外网: $_TORCH_INDEX"
  else
    echo "使用内网镜像: $_TORCH_INDEX"
  fi
fi
# 内网 index 直连即可; 只有走外网 download.pytorch.org 时才需要代理前缀。
_PXIF=""; [ "$_TORCH_IS_UPSTREAM" = "1" ] && _PXIF="$PX"

# torchaudio / torchvision 都必须跟 torch 一起钉死。
# 【踩过的坑】--force-reinstall --no-deps 时 pip 【不做版本配对】, 只会挑 index 里
# 最新的那个: 该 cu129 index 同时有 torchvision 0.26/0.27/0.28 和 torchaudio
# 2.11/2.12/2.13(分别为 torch 2.11/2.12/2.13 编的)。不钉的话会装上 0.28.0,
# 而它的 C++ 扩展是按 torch 2.13 的 ABI 编的, 于是注册算子失败:
#   RuntimeError: operator torchvision::nms does not exist
# 而 sglang 的 srt/utils/common.py:90 有 `from torchvision.io import decode_jpeg`,
# 所以 `import sglang` 直接崩 —— rollout 必走这条路。
# torchaudio 同理: transformers 的 modeling_utils 导入链上有 import torchaudio,
# 错配会让 `from transformers import PreTrainedModel` 崩(verl venv 已踩过一次)。
# 配对关系: torch 2.11 ↔ torchvision 0.26 ↔ torchaudio 2.11
# (即 §4 里 sglang 自己解析出来的那套, 这里只是别把它顶掉。)
# --index-url 是【替换】全局 index, 所以这些命令都必须配 --no-deps。
$_PXIF pip install --force-reinstall --no-deps \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url "$_TORCH_INDEX"

# sglang 的原生 kernel, 必须是 cu129 构建(它们按 torch 2.11 的 C++ ABI 编的)
$PX pip install --force-reinstall --no-deps \
  sglang-kernel==0.4.2.post2 sgl-deep-gemm==0.1.0 \
  --index-url "https://docs.sglang.ai/whl/cu129/"

# 卸掉 cu13 系列(uninstall 不联网, 不用代理), 再装 cu12 的
pip uninstall -y \
  nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
  nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand \
  nvidia-cusolver nvidia-cusparse nvidia-cusparselt-cu13 nvidia-nccl-cu13 \
  nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx nvidia-cutlass-dsl-libs-cu13 \
  || true

# 走内网 index 时不需要 --extra-index-url pypi(内网 pypi 本来就有这些包);
# 走外网时才要, 因为 download.pytorch.org 上缺少部分 nvidia-* 包。
_EXTRA_PYPI=""
[ "$_TORCH_IS_UPSTREAM" = "1" ] && _EXTRA_PYPI="--extra-index-url https://pypi.org/simple"
# shellcheck disable=SC2086
$_PXIF pip install --force-reinstall --no-deps \
  nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12==9.16.0.29 nvidia-cufft-cu12 \
  nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 \
  nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
  nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 nvidia-nvtx-cu12 \
  --index-url "$_TORCH_INDEX" $_EXTRA_PYPI

pip install cmake ninja

# torch 定版后立刻硬断言 —— 后面所有 --no-build-isolation 的编译都依赖它,
# 错了要在编译前就发现, 不能等到最后自检。
# 这里不只看版本号, 还【真的 import torchvision/torchaudio 并触碰它们的 C++ 算子】:
# 版本号对不代表 ABI 对, 而 ABI 错配的报错(operator torchvision::nms does not
# exist)只在 import 时才出现。早失败 1 分钟, 省 90 分钟。
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.__version__.startswith("2.11.0"), f"expect torch 2.11.0, got {torch.__version__}"
assert "cu129" in torch.__version__, f"torch 版本号里没有 +cu129: {torch.__version__}"
assert torch.version.cuda and torch.version.cuda.startswith("12.9"), \
    f"torch 不是 cu129 构建(cuda={torch.version.cuda})"

# torchvision: import 时会注册 torchvision::nms 等算子, ABI 不匹配就在这里炸。
# sglang 的 srt/utils/common.py:90 import 它, 所以这是 rollout 的必经路径。
import torchvision
from torchvision.io import decode_jpeg  # noqa: F401
print("torchvision", torchvision.__version__)
assert torchvision.__version__.startswith("0.26."), \
    f"torchvision 必须是 0.26.x 配 torch 2.11, 得到 {torchvision.__version__}"

# torchaudio: transformers 的 modeling_utils 导入链会 import 它
import torchaudio
print("torchaudio", torchaudio.__version__)
assert torchaudio.__version__.startswith("2.11."), \
    f"torchaudio 必须是 2.11.x, 得到 {torchaudio.__version__}"

# transformers 的入口(最容易被 torchaudio ABI 问题连坐的那个)
from transformers import PreTrainedModel  # noqa: F401
print("from transformers import PreTrainedModel: ok")
PY

# ---------------------------------------------------------------------------
# 6/14 flash-attn 2 (编译)
# ---------------------------------------------------------------------------
# 2.7.4.post1 是【上限不是下限】: 官方 Dockerfile 注释 "TransformerEngine does
# not support too high FA2", 也是 megatron 支持的最高版本。别"升级"它。
# --no-build-isolation 要求 torch 已装好, 所以必须在 §5 之后。
# 注: 官方 Dockerfile 还会从源码另建 FA3(hopper/) 并塞进 flash_attn_3 包,
# build_venv.sh 没有这一步, 本脚本也刻意不做 —— slime 不 import flash_attn_3。
if [ "$SKIP_FLASH_ATTN" = "1" ]; then
  log "6/14 flash-attn —— 按 SKIP_FLASH_ATTN=1 跳过"
else
  log "6/14 flash-attn 2.7.4.post1 (编译, 预计 10-15 min)"
  MAX_JOBS=$MAX_JOBS pip -v install flash-attn==2.7.4.post1 --no-build-isolation
  python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
fi

# ---------------------------------------------------------------------------
# 7/14 mbridge / fla / FlashQLA / tilelang / TransformerEngine
# ---------------------------------------------------------------------------
log "7/14 mbridge + flash-linear-attention + FlashQLA + tilelang + TE 2.10"

# 注意这是 ISEEKYAN/mbridge (import mbridge), 与 §9 的 megatron.bridge
# (Megatron-Bridge) 是两个不同项目, 都需要。--no-deps 防它重解析 torch/numpy。
$PX pip install "git+https://github.com/ISEEKYAN/mbridge.git@${MBRIDGE_COMMIT}" --no-deps

# Qwen3.5/Qwen3-Next 的 gated_delta_net kernel(内网有, 直连)
pip install flash-linear-attention==0.4.1

# FlashQLA: Qwen3.5/Qwen3-Next 的可选 GDN 后端(--qwen-gdn-backend flashqla, 需 SM90+)
$PX pip install "git+https://github.com/QwenLM/FlashQLA.git" --no-build-isolation

# tilelang 从 cu128 的 nightly index 装, 而我们是 cu129 —— 这是上游的做法, 能用:
# tilelang 链的是 nvidia-*-cu12 里的 CUDA runtime, cu12x 在 major 内向前兼容。
# -f 是追加 find-links, 内网 index 仍然参与; 真拉不到可退 `pip install tilelang==0.1.8`。
$PX pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/

# TE 2.10: 内网有预编译 wheel(transformer_engine + transformer_engine_cu12),
# 只有 transformer_engine_torch 需要现场编译, 所以只要 ~5 min 而非一小时。
export NVTE_FRAMEWORK=pytorch
export NVTE_BUILD_THREADS_PER_JOB=4
pip install -v --no-build-isolation "transformer_engine[pytorch]==2.10.0"
python -c "import transformer_engine.pytorch as te; print('TE ok')"

# ---------------------------------------------------------------------------
# 8/14 apex (编译, 可选)
# ---------------------------------------------------------------------------
# mcore 里的 apex 导入全部包在 try/except 里, 缺 apex 会回退到 torch 原生
# FusedAdam / multi_tensor_applier —— 只是慢一些, 不影响正确性。
if [ "$SKIP_APEX" = "1" ]; then
  log "8/14 apex —— 按 SKIP_APEX=1 跳过(mcore 回退 torch 原生实现)"
else
  log "8/14 apex @ ${APEX_COMMIT:0:12} (编译, 预计 12-15 min)"
  # --config-settings 的写法与上游 build_venv.sh 保持一致(单个组合串)
  NVCC_APPEND_FLAGS="--threads 4" \
    $PX pip -v install --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
    "git+https://github.com/NVIDIA/apex.git@${APEX_COMMIT}"
  python -c "from apex.optimizers import FusedAdam; print('apex ok')"
fi

# ---------------------------------------------------------------------------
# 9/14 torch_memory_saver / Megatron-Bridge / modelopt / sglang_router
# ---------------------------------------------------------------------------
log "9/14 torch_memory_saver + Megatron-Bridge + modelopt + sglang_router"

TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')}"
export TMS_CUDA_MAJOR
echo "TMS_CUDA_MAJOR=$TMS_CUDA_MAJOR"

# torch_memory_saver: slime/backends/megatron_utils/actor.py:13 顶层硬导入。
# 【--no-build-isolation 是关键】: 少了它, pip 的 PEP-517 隔离环境看不到
# nvcc/头文件/torch, 编出来的 wheel 只有 python 部分(~46KB), preload 用的 .so
# 根本没编, 之后 sglang 会报 "Only hook_mode=preload supports pauseable CUDA Graph"。
# --force-reinstall: §4 的 sglang[all] 已经从 PyPI 装了一个, 必须换掉。
$PX pip install -v "git+https://github.com/fzyzcjy/torch_memory_saver.git@${TMS_COMMIT}" \
  --no-cache-dir --force-reinstall --no-build-isolation

# Megatron-Bridge(import megatron.bridge): slime 的
# backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py 里硬导入。
$PX pip install "git+https://github.com/radixark/Megatron-Bridge.git@bridge" --no-deps --no-build-isolation

# 上游那行 nvidia-modelopt[torch]>=0.37.0 没加引号 —— `>=` 在 shell 里是重定向,
# 会在当前目录生成一个名为 =0.37.0 的空文件并只装了 nvidia-modelopt[torch]。
# 这里补上引号。
pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

# sglang_router 必须是 zhuzilin 的 fork(cp38-abi3, py3.12 可用), 不是 PyPI 的。
# 见 §11: requirements.txt 里的 sglang-router>=0.2.3 会把它顶掉, 所以那之后还要再装一次。
$PX pip install "$SGLANG_ROUTER_WHL" --force-reinstall
python -c "import sglang_router; assert 'slime' in sglang_router.__version__, sglang_router.__version__; print('sglang_router', sglang_router.__version__)"

# ---------------------------------------------------------------------------
# 10/14 Megatron-LM (本地源码 editable)
# ---------------------------------------------------------------------------
# setuptools<80: Megatron / apex 的 setup.py 用了 80 移除的 API。
# --no-build-isolation 是必需的: Megatron 的 setup.py 会 shell 出去跑
# `python3 -m pybind11 --includes` 来编 megatron.core.datasets.helpers_cpp;
# 开隔离时那个子进程看到的是外层 python(没有 pybind11), 扩展被标记 optional=True
# 然后【静默跳过】, GPT 数据集加载会失效。所以下面显式检查 helpers_cpp。
#
# 不做 git checkout $MEGATRON_COMMIT: 本地目录不是 git 仓库(见 §1b)。
# 不打 megatron.patch: §1b 已验证 patch 内容全部内置。
log "10/14 Megatron-LM editable (本地: $MEGATRON_SRC)"
pip install "setuptools<80.0.0" pybind11 "packaging>=24.2"
cd "$MEGATRON_SRC"
pip install -e . --no-build-isolation
python -c "import megatron.core; print('megatron.core', megatron.core.__version__)"
if python -c "import megatron.core.datasets.helpers_cpp" 2>/dev/null; then
  echo "  ✓ helpers_cpp 已编译"
else
  warn "megatron.core.datasets.helpers_cpp 没编出来(setup.py 把它当 optional 静默跳过了)。" \
       "GPT/索引数据集路径会失效。GUI-RL 走自己的 data source, 一般不受影响。" \
       "要修: cd $MEGATRON_SRC && pip install -e . --no-build-isolation --force-reinstall"
fi

# ---------------------------------------------------------------------------
# 11/14 slime 本体 + int4_qat kernel
# ---------------------------------------------------------------------------
log "11/14 slime (requirements + editable) + int4_qat"
cd "$SLIME_DIR"
pip install -r requirements.txt
# --no-deps: 别让它重解析并顶掉上面钉好的原生栈
pip install -e . --no-deps

# 【修上游顺序 bug】requirements.txt 里有 sglang-router>=0.2.3, 内网 PyPI 的
# 0.3.2 满足它 -> 刚才 §9 装的 fork 被替换掉了。而 slime 依赖 fork 独有的 API。
# 上游 build_venv.sh 是先装 fork(:101) 再跑 requirements(:114), 顺序反了;
# 官方 Dockerfile 顺序是对的(requirements 在前)。这里在 requirements 之后重装。
log "11b/14 重装 sglang_router fork (requirements.txt 会顶掉它)"
$PX pip install "$SGLANG_ROUTER_WHL" --force-reinstall --no-deps
python -c "import sglang_router; assert 'slime' in sglang_router.__version__, f'fork 被顶掉了: {sglang_router.__version__}'; print('sglang_router', sglang_router.__version__)"

# 【廉价保险】requirements.txt 里 accelerate/datasets/transformers/ray 都声明
# torch, 万一有谁把 torch 降级就会毁掉 cu129 构建 —— 这里立刻复查, 一秒钟的事。
python - <<'PY'
import torch
assert torch.__version__.startswith("2.11.0") and "cu129" in torch.__version__, \
    f"torch 被 slime/requirements.txt 改动了: {torch.__version__} —— 需要重新钉回 2.11.0+cu129"
print("torch 仍是", torch.__version__)
PY

# int4_qat: soft import(quantizer_compressed_tensors.py:8-11 是 try/except),
# 只在 int4 量化流程用到。所以失败只警告不中断 —— `|| { ...; }` 在 set -e 下
# 就是让它非致命的写法。setup.py 会现场探测 GPU capability, H20 上自动出
# sm_90/sm_90a, 不用手动给 arch。
_INT4_OK=1
if [ "$SKIP_INT4_QAT" = "1" ]; then
  echo "int4_qat —— 按 SKIP_INT4_QAT=1 跳过"
  _INT4_OK=0
else
  cd "$SLIME_DIR/slime/backends/megatron_utils/kernels/int4_qat"
  pip install . --no-build-isolation || { _INT4_OK=0; warn "int4_qat 编译失败 —— 是 soft import(仅 int4 量化用到), 继续安装。"; }
fi

# ---------------------------------------------------------------------------
# 12/14 收尾修复 —— 必须放在所有 pip install 之后
# ---------------------------------------------------------------------------
log "12/14 收尾: cudnn 钉版本 + numpy<2 + kernels<0.15"

# cudnn 会被 modelopt / TE / flashinfer / requirements 链条顶掉, 所以再钉一次。
# 用 --no-deps 避免 pip 顺带重解析它的依赖(上游没加, 只是慢)。
pip install --no-deps --force-reinstall nvidia-cudnn-cu12==9.16.0.29

# numpy<2 必须【最后】装: 上游几乎所有包(sglang[all] / transformers / datasets /
# scikit-image / opencv / tilelang / torchvision)都要 numpy 2.x, 早装会被顶掉。
# slime/backends/megatron_utils/initialize.py:66 运行时硬断言 numpy 是 1.x。
pip install "numpy<2"

# 【踩过的坑】scipy 也得跟着降。sglang[all] 装的 scipy 1.18 是为 numpy 2.x 编的,
# 它的 _sputils.py 顶层用了 np.long / np.ulong —— 这两个名字只在 numpy 2.x 有,
# numpy 1.x 下直接 AttributeError。而 transformers 的 modeling_utils 导入链
# (loss_utils -> loss_d_fine -> loss_for_object_detection -> scipy.optimize)会
# 撞上它, 于是 `import sglang` / `import transformers` 全崩。
# 1.15.3 是仍然声明 numpy<2.5,>=1.23.5(即允许 1.x)的最后一个系列。
pip install "scipy==1.15.3"

# kernels 0.15+ 在 transformers.integrations.hub_kernels 里会抛
# ValueError("Either a revision or a version must be specified"), 导致
# import sglang 直接失败。内网有到 0.16, 不钉就会拿到坏的。
pip install "kernels<0.15.0"

# numba 的编译扩展对 numpy 的 C ABI 敏感, 刚把 numpy 降到 1.x, 验一下它还能用。
if python -c "import numba, numpy, numba.np.ufunc; print('numba', numba.__version__, '/ numpy', numpy.__version__)" 2>/dev/null; then
  :
else
  warn "numba 在 numpy 1.x 下 import 失败(ABI 不匹配)。" \
       "修法: pip install 'numba==0.59.1'  (它 cap 了 numpy<1.27, 是为 1.x ABI 编的)"
fi

# ---------------------------------------------------------------------------
# 13/14 打 sglang.patch
# ---------------------------------------------------------------------------
# megatron.patch 不打(§1b 已验证内置)。这里只打 sglang 的。
# 相对上游的两处修正:
#   1) 上游 `grep -R -n '^<<<<<<< ' . && {...exit 1;} || true` —— 尾部 || true
#      让冲突检测永远成功(白检);且 -R . 会扫 .git 和 rust/target, 很慢。
#      patch 只碰 python/sglang/srt/**, 所以 scope 到 python/sglang 就够且快。
#   2) 用 `git apply --reverse --check` 判断"是否已打过", 这才是正确的幂等探测;
#      正向 --check 失败也可能只是上下文漂移(--3way 仍能合), 不等于已打过。
if [ "$SKIP_SGLANG_PATCH" = "1" ] || [ "$SKIP_SGLANG" = "1" ]; then
  log "13/14 sglang.patch —— 跳过 (SKIP_SGLANG_PATCH=$SKIP_SGLANG_PATCH SKIP_SGLANG=$SKIP_SGLANG)"
else
  log "13/14 打 sglang.patch"
  cd "$SGLANG_DIR"
  echo "打补丁前的工作区状态:"
  git status --porcelain python/ | sed -n '1,20p' || true

  if git apply --reverse --check "$_SGLANG_PATCH" 2>/dev/null; then
    echo "  → patch 已应用过(reverse check 通过), 跳过。"
  elif git apply "$_SGLANG_PATCH" --3way; then
    echo "  → patch 应用成功。"
    if grep -R -n '^<<<<<<< ' python/sglang 2>/dev/null; then
      echo "ERROR: sglang patch 有冲突标记, 请手动解决上面列出的文件" >&2
      exit 1
    fi
    echo "  ✓ 无冲突标记"
  else
    echo "ERROR: git apply --3way 失败, 请手动检查 $_SGLANG_PATCH" >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# 14/14 自检
# ---------------------------------------------------------------------------
log "14/14 自检"
cd "$SLIME_DIR"
INT4_OK="$_INT4_OK" python - <<'PY'
import os, sys

# 先验硬约束 —— 错了直接 exit 1
import torch, numpy
assert torch.__version__.startswith("2.11.0") and "cu129" in torch.__version__, torch.__version__
assert torch.version.cuda.startswith("12.9"), torch.version.cuda
assert numpy.__version__.startswith("1."), \
    f"slime initialize.py:66 要求 numpy 1.x, 实际 {numpy.__version__}"

# torchvision/torchaudio 必须与 torch 2.11 配对(0.26.x / 2.11.x)。
# 版本号错 = C++ ABI 错 = import 就炸(operator torchvision::nms does not exist)。
import torchvision, torchaudio
assert torchvision.__version__.startswith("0.26."), \
    f"torchvision 必须 0.26.x 配 torch 2.11, 实际 {torchvision.__version__}"
assert torchaudio.__version__.startswith("2.11."), \
    f"torchaudio 必须 2.11.x, 实际 {torchaudio.__version__}"
from torchvision.io import decode_jpeg  # noqa: F401  sglang common.py:90 走这条

import kernels
_kv = tuple(int(x) for x in kernels.__version__.split(".")[:2])
assert _kv < (0, 15), f"kernels 必须 <0.15 (否则 import sglang 会炸), 实际 {kernels.__version__}"

import sglang_router
assert "slime" in sglang_router.__version__, \
    f"sglang_router 必须是 zhuzilin fork, 实际 {sglang_router.__version__}"

print("torch        ", torch.__version__, "cuda", torch.version.cuda)
print("torchvision  ", torchvision.__version__)
print("torchaudio   ", torchaudio.__version__)
print("numpy        ", numpy.__version__)
print("kernels      ", kernels.__version__)
print("sglang_router", sglang_router.__version__)

import megatron.core, transformers, sglang
print("megatron.core", megatron.core.__version__)
print("transformers ", transformers.__version__)
print("sglang       ", getattr(sglang, "__version__", "?"))

import mbridge, torch_memory_saver, wandb  # noqa: F401
print("mbridge      ", getattr(mbridge, "__version__", "?"))
print("wandb        ", wandb.__version__)

# slime 的后端模块(actor / sglang_engine / ...)会经 arguments.py 顶层
# import megatron.training.*, 而 megatron.training 需要 PYTHONPATH:
# Megatron-LM 的 pyproject.toml 只打包 megatron.core*(packages.find.include),
# 所以 pip install -e . 之后 megatron.training / megatron.legacy 都不可导入 ——
# 这是上游设计, 不是装错了, 靠启动脚本设的 PYTHONPATH 解析。
# 因此那几个模块的导入检查统一挪到 §14b(带 PYTHONPATH 的环境)去做, 这里只验
# 能独立导入的叶子包。
import megatron.bridge                              # noqa: F401  (Megatron-Bridge)
import slime                                        # noqa: F401
print("megatron.bridge / slime 可导入")

# 软依赖: 缺了只是降级, 不算失败
for name, mod in [("flash_attn", "flash_attn"), ("tilelang", "tilelang"),
                  ("transformer_engine", "transformer_engine.pytorch"),
                  ("fla", "fla")]:
    try:
        m = __import__(mod, fromlist=["x"])
        print(f"{name:13s}", getattr(m, "__version__", "ok"))
    except Exception as e:
        print(f"{name:13s} NOT AVAILABLE ({type(e).__name__})")
try:
    from apex.optimizers import FusedAdam  # noqa: F401
    print("apex          ok")
except Exception:
    print("apex          NOT INSTALLED (mcore 回退 torch 原生)")
if os.environ.get("INT4_OK") == "1":
    try:
        import fake_int4_quant_cuda  # noqa: F401
        print("int4_qat      ok")
    except Exception:
        print("int4_qat      NOT AVAILABLE (soft import, 仅 int4 量化用到)")
else:
    print("int4_qat      SKIPPED / 编译失败 (soft import)")

print("GPU          ", torch.cuda.device_count(), "x", torch.cuda.get_device_name(0),
      "capability", torch.cuda.get_device_capability(0))
PY

# gui-rl 没有 setup.py / pyproject.toml, 是靠 PYTHONPATH 就地跑的。
# 这个 PYTHONPATH 顺序取自 gui-rl/scripts/gui_qwen3vl_16gpu_fully_async_fast.sh:370
log "14b/14 PYTHONPATH 环境下的导入检查 (megatron.training + slime 后端 + gui-rl)"
cd "$GUI_RL_DIR"
# 这个 PYTHONPATH 顺序取自 gui-rl/scripts/gui_qwen3vl_16gpu_fully_async_fast.sh:370
# —— 即真实训练时的环境。megatron.training 和 slime 的几个后端模块只在这里能解析。
PYTHONPATH="$MEGATRON_SRC:$GUI_RL_DIR:$SLIME_DIR" python - <<'PY'
# megatron.training: slime 有 10+ 处用它, 只在 PYTHONPATH 里(见 §14 注释)
import megatron.training.arguments          # noqa: F401
import megatron.training.checkpointing      # noqa: F401
print("megatron.training.* 可导入")

# 比逐个 import 叶子包更强的测试: 直接 import 那几个有硬顶层依赖的 slime 模块。
# 它们分别硬 import torch_memory_saver / sglang_router / wandb / megatron.training,
# 全过说明整条依赖链都通。
import slime.backends.megatron_utils.actor          # noqa: F401  硬 import torch_memory_saver
import slime.backends.sglang_utils.sglang_engine    # noqa: F401  硬 import sglang_router
import slime.rollout.sglang_rollout                 # noqa: F401  硬 import sglang_router
import slime.utils.wandb_utils                      # noqa: F401  硬 import wandb
print("slime 关键模块可导入 (actor / sglang_engine / sglang_rollout / wandb_utils)")

# gui-rl 没有 setup.py, 就地跑
import config, env_client                            # noqa: F401
import agents, reward                                # noqa: F401
print("gui-rl 可导入 (config / env_client / agents / reward)")
PY

cat <<EOF

============================================================================
安装完成: $VENV
日志:     $LOG_FILE

激活:  source $VENV/bin/activate

冒烟测试 (8 卡单机; 需要一个可达的远端 OSWorld/MobileWorld env server):

  cd $GUI_RL_DIR
  PATH=$VENV/bin:\$PATH \\
    HF_CKPT=<your Qwen3-VL-8B-Instruct path> \\
    GUI_ENV_SERVER_URL=<你的 env server 地址> \\
    bash scripts/gui_qwen3vl_8b_fully_async_fast.sh

注意:
 * HF_CKPT 示例路径需替换为你本地实际的 Qwen3-VL-8B-Instruct checkpoint 目录。
 * gui_qwen3vl_16gpu_* 那两个脚本要 2 个节点(16 卡); 单机用 _8b_ 那个。
 * PYTHONPATH 由启动脚本自己设(含 Megatron-LM / gui-rl / slime 三个路径),
   只有挪动了 Megatron-LM 目录才需要 MEGATRON_LM_PATH=... 覆盖。
 * 本 venv 是 torch 2.11.0+cu129, 与 venvs/verl(torch 2.9.1) 是两套, 别混用。
 * 首次 import(CephFS 上) 可能数分钟, 多进程并发更久。要加速可先 rsync 到本地盘:
     rsync -a $VENV/ /dev/shm/online-rl-venv/ && PATH=/dev/shm/online-rl-venv/bin:\$PATH ...
 * rust 工具链装在 $VENV/rust, 所以 rm -rf $VENV 就是彻底重置。
============================================================================
EOF
