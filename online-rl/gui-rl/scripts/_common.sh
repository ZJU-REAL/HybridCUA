#!/usr/bin/env bash
# _common.sh — shared boilerplate for gui-rl training scripts.
#
# Source this file near the top of each training script, after defining
# LOG_FILE_PREFIX (e.g. "hybridcua8") and optionally DETACH_GUARD_VAR.
#
# Provides:
#   setup_background_exec <log_prefix>  — re-exec detached with log capture
#   kill_stale_python                   — kill leftover python, spare env server
#   cleanup_ray                         — stop sglang + ray + stale python
#   setup_proxy                         — configure outbound proxy for Ray jobs
#   ensure_libnuma                      — locate / install libnuma.so.1
#
# Required env vars (must be set before sourcing or calling functions):
#   (none mandatory at source time; each function documents its own inputs)
#
# Optional env vars:
#   BACKGROUND=0              disable auto-detach (default: 1)
#   GUI_ENV_SERVER_PROC_PATTERN  pattern to spare from pkill (default: cluster.master.server)
#   USE_STAR_PROXY=1          enable outbound proxy; requires STAR_PROXY_URL
#   STAR_PROXY_URL            proxy URL, e.g. http://your-proxy.example.com:3128
#   STAR_NO_PROXY             comma-separated no-proxy list (auto-built from hostname -I)

# ---------------------------------------------------------------------------
# setup_background_exec <log_prefix>
#
# Re-execs the calling script detached (setsid) with all output captured to
# a timestamped log file under <SCRIPT_DIR>/logs/, then exits the foreground
# shell. Set BACKGROUND=0 to skip and run in the foreground.
#
# Must be called before set -ex so the foreground exit is clean.
# ---------------------------------------------------------------------------
setup_background_exec() {
  local prefix="${1:?setup_background_exec requires a log prefix argument}"
  local guard_var="_${prefix^^}_DETACHED"   # e.g. _HYBRIDCUA8_DETACHED

  if [[ "${BACKGROUND:-1}" == "1" && -z "${!guard_var:-}" ]]; then
    local run_dir
    run_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[1]}")/.." &>/dev/null && pwd)"
    mkdir -p "${run_dir}/logs"
    local log_file="${LOG_FILE:-${run_dir}/logs/${prefix}_$(date +%Y%m%d_%H%M%S).log}"
    export LOG_FILE="${log_file}"
    declare -gx "${guard_var}=1"
    setsid bash "${BASH_SOURCE[1]}" "$@" > "${log_file}" 2>&1 < /dev/null &
    local pid=$!
    echo "Started ${prefix} in background: PID=${pid}"
    echo "Log:  ${log_file}"
    echo "Tail: tail -f ${log_file}"
    echo "Stop: ray job stop <submission_id> --address=http://127.0.0.1:8265   (or ray stop --force)"
    exit 0
  fi
}

# ---------------------------------------------------------------------------
# kill_stale_python
#
# Kill all python processes EXCEPT the env server (matched by
# GUI_ENV_SERVER_PROC_PATTERN, default: cluster.master.server).
# A blanket pkill -9 python would kill the co-located OSWorld/MobileWorld
# cluster env server, breaking the healthz check on the next run.
# ---------------------------------------------------------------------------
GUI_ENV_SERVER_PROC_PATTERN=${GUI_ENV_SERVER_PROC_PATTERN:-cluster.master.server}

kill_stale_python() {
  local pid
  for pid in $(pgrep -f python 2>/dev/null); do
    if ! tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null \
         | grep -q "${GUI_ENV_SERVER_PROC_PATTERN}"; then
      kill -9 "${pid}" 2>/dev/null || true
    fi
  done
}

# ---------------------------------------------------------------------------
# cleanup_ray
#
# Stop sglang and Ray, then kill leftover python processes (twice, to catch
# children spawned during the first round).
# ---------------------------------------------------------------------------
cleanup_ray() {
  pkill -9 sglang || true
  sleep 3
  ray stop --force || true
  pkill -9 ray || true
  kill_stale_python
  sleep 3
  pkill -9 ray || true
  kill_stale_python
}

# ---------------------------------------------------------------------------
# setup_proxy
#
# Configure the outbound proxy for the Ray training job (wandb egress).
# The proxy is intentionally NOT set in this shell or in Ray itself — only
# injected into RUNTIME_ENV_JSON so Ray workers can reach wandb.ai while
# all Ray/sglang/env-server traffic stays direct.
#
# Reads:
#   USE_STAR_PROXY   1 = enable (default: 0)
#   STAR_PROXY_URL   proxy URL (required when USE_STAR_PROXY=1)
#   STAR_NO_PROXY    override the auto-built no-proxy list
#
# Writes (exported for use by the caller in RUNTIME_ENV_JSON):
#   GUI_JOB_HTTP_PROXY
#   GUI_JOB_NO_PROXY
# ---------------------------------------------------------------------------
setup_proxy() {
  USE_STAR_PROXY=${USE_STAR_PROXY:-0}
  if [[ "${USE_STAR_PROXY}" == "1" ]]; then
    if [[ -z "${STAR_PROXY_URL:-}" ]]; then
      echo "ERROR: USE_STAR_PROXY=1 requires STAR_PROXY_URL" \
           "(e.g., http://your-proxy.example.com:3128)" >&2
      exit 1
    fi
    local local_ips
    local_ips=$(hostname -I 2>/dev/null | tr ' ' ',' | sed 's/,\+/,/g; s/^,//; s/,$//')
    STAR_NO_PROXY=${STAR_NO_PROXY:-"localhost,127.0.0.1,::1,${local_ips},<node-ip>/8,<node-ip>/12,<node-ip>/16"}
    # Keep the proxy OUT of this shell and out of Ray — only inject into
    # RUNTIME_ENV_JSON below. A proxy in Ray's env breaks job submission.
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
    GUI_JOB_HTTP_PROXY="${STAR_PROXY_URL}"
    GUI_JOB_NO_PROXY="${STAR_NO_PROXY}"
    echo "Outbound proxy for training job (wandb egress): ${STAR_PROXY_URL}"
    echo "job no_proxy: ${GUI_JOB_NO_PROXY}"
  else
    GUI_JOB_HTTP_PROXY=""
    GUI_JOB_NO_PROXY=""
  fi
  export GUI_JOB_HTTP_PROXY GUI_JOB_NO_PROXY
}

# ---------------------------------------------------------------------------
# ensure_libnuma
#
# Ensure libnuma.so.1 is on LD_LIBRARY_PATH.
# sgl_kernel links against it; without it SGLangEngine fails to start.
# Search order: vendored copy → apt/yum → find fallback.
#
# Reads:
#   SCRIPT_DIR   gui-rl package root (set by caller before sourcing)
# ---------------------------------------------------------------------------
ensure_libnuma() {
  local vendored_numa="${SCRIPT_DIR}/../vendor_libs"
  if ldconfig -p 2>/dev/null | grep -q libnuma; then
    return 0
  fi
  if [[ -f "${vendored_numa}/libnuma.so.1" ]]; then
    export LD_LIBRARY_PATH="${vendored_numa}:${LD_LIBRARY_PATH:-}"
    echo "libnuma: using vendored copy at ${vendored_numa}"
    return 0
  fi
  if command -v apt-get &>/dev/null; then
    apt-get update -qq && apt-get install -y -qq libnuma1 libnuma-dev 2>/dev/null || true
  elif command -v yum &>/dev/null; then
    yum install -y numactl-libs 2>/dev/null || true
  fi
  if ! ldconfig -p 2>/dev/null | grep -q libnuma; then
    local numa_path
    numa_path=$(find /usr /opt /mnt -name "libnuma.so.1" 2>/dev/null | head -1)
    if [[ -n "${numa_path}" ]]; then
      export LD_LIBRARY_PATH="$(dirname "${numa_path}"):${LD_LIBRARY_PATH:-}"
    fi
  fi
}
