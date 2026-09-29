#!/usr/bin/env bash

setup_background_exec() {
  local prefix="${1:?setup_background_exec requires a log prefix argument}"
  local guard_var="_${prefix^^}_DETACHED"

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
