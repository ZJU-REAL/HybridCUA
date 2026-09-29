#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUI_RL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECTS_ROOT="$(cd "${GUI_RL_DIR}/../.." && pwd)"
ENV_INFRA_DIR="${PROJECTS_ROOT}/env_infra"
ONLINE_RL_VENV="${PROJECTS_ROOT}/venvs/online-rl"
ENV_INFRA_VENV="${PROJECTS_ROOT}/venvs/env_infra"

TOPOLOGY="${TOPOLOGY_ENV:-${TOPOLOGY:-2node}}"
TOPO_FILE="${SCRIPT_DIR}/topologies/${TOPOLOGY}.env"
if [[ ! -f "${TOPO_FILE}" ]]; then
  echo "ERROR: no such topology: ${TOPOLOGY} (looked for ${TOPO_FILE})"
  echo "  available: $(cd "${SCRIPT_DIR}/topologies" 2>/dev/null && ls *.env 2>/dev/null | sed 's/\.env$//' | paste -sd' ' -)"
  exit 1
fi
source "${TOPO_FILE}"

: "${HEAD_IP:?topology '${TOPOLOGY}' did not set HEAD_IP}"
: "${ENV_IP:?topology '${TOPOLOGY}' did not set ENV_IP}"
: "${WORKER_IPS:?topology '${TOPOLOGY}' did not set WORKER_IPS}"
: "${NUM_GPUS:?topology '${TOPOLOGY}' did not set NUM_GPUS}"
: "${ACTOR_GPUS:?topology '${TOPOLOGY}' did not set ACTOR_GPUS}"
: "${ROLLOUT_GPUS:?topology '${TOPOLOGY}' did not set ROLLOUT_GPUS}"
: "${EXPECTED_NODES:?topology '${TOPOLOGY}' did not set EXPECTED_NODES}"
: "${ENV_WORLDS:?topology '${TOPOLOGY}' did not set ENV_WORLDS -- it must match the train script GUI_ENV_RUNTIME}"
: "${TRAIN_SCRIPT:?topology '${TOPOLOGY}' did not set TRAIN_SCRIPT}"
WORKER_IPS=(${WORKER_IPS})

export NUM_GPUS ACTOR_GPUS ACTOR_NUM_NODES ACTOR_NUM_GPUS_PER_NODE ROLLOUT_GPUS EXPECTED_NODES
if (( ACTOR_GPUS != ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE )); then
  echo "ERROR: topology '${TOPOLOGY}': ACTOR_GPUS=${ACTOR_GPUS} != ACTOR_NUM_NODES=${ACTOR_NUM_NODES} x ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE}"
  exit 1
fi
if (( NUM_GPUS != (ACTOR_NUM_NODES + 1) * ACTOR_NUM_GPUS_PER_NODE )); then
  echo "ERROR: topology '${TOPOLOGY}': NUM_GPUS=${NUM_GPUS} != (ACTOR_NUM_NODES=${ACTOR_NUM_NODES} actor + 1 rollout) x ${ACTOR_NUM_GPUS_PER_NODE}"
  exit 1
fi
if (( ${#WORKER_IPS[@]} != ACTOR_NUM_NODES )); then
  echo "ERROR: topology '${TOPOLOGY}': ${#WORKER_IPS[@]} worker(s) (${WORKER_IPS[*]}) but ACTOR_NUM_NODES=${ACTOR_NUM_NODES}"
  exit 1
fi
echo "topology=${TOPOLOGY}: head/env=${HEAD_IP} workers=(${WORKER_IPS[*]}) worlds=${ENV_WORLDS}"
_ROLES="$(printf '%s\n' "${HEAD_IP}" "${WORKER_IPS[@]}" | sort -t. -k1,1n -k2,2n -k3,3n -k4,4n | paste -sd' ' -)"
_ROLLOUT_NODE="${_ROLES##* }"
echo "  actor ${ACTOR_GPUS} GPU = ${ACTOR_NUM_NODES} x ${ACTOR_NUM_GPUS_PER_NODE} | rollout ${ROLLOUT_GPUS} GPU on ${_ROLLOUT_NODE} (highest IP) | ${EXPECTED_NODES} nodes expected"
[[ "${_ROLLOUT_NODE}" == "${ENV_IP}" ]] \
  || echo "  note: env server is on ${ENV_IP}, not the rollout node -- screenshots cross the network"

export GUI_ENV_SERVER_PROC_PATTERN='cluster\.master\.server\|cluster\.node\.world_server'
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-128}"

ENV_MAX_SLOTS="${ENV_MAX_SLOTS:-72}"

CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${GUI_RL_DIR}/scripts/gui_partial_async.yaml}"
FREE_GPUS_SH="${GUI_RL_DIR}/scripts/free_gpus.sh"
STOP_ENV_SH="${GUI_RL_DIR}/scripts/stop_env_node.sh"
JOIN_SH="${GUI_RL_DIR}/scripts/gpu_worker_join_ray.sh"
MASTER_URL="http://${ENV_IP}:19000"
export GUI_ENV_SERVER_URL="${MASTER_URL}"

SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes"

say() { echo; echo "=== $* ==="; }

require_head_host() {
  if ! hostname -I 2>/dev/null | tr ' ' '\n' | grep -qx "${HEAD_IP}"; then
    echo "ERROR: this host is not HEAD_IP=${HEAD_IP} (local IPs: $(hostname -I))"
    echo "  run it there:  ${SSH} ${HEAD_IP} \"setsid bash ${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}") $*\""
    echo "  or override HEAD_IP=<this host> to move the head."
    exit 1
  fi
}

run_on_env_host() {
  local secs="$1" cmd="$2"
  if [[ "${ENV_IP}" == "${HEAD_IP}" ]]; then
    timeout "${secs}" bash -c "${cmd}"
  else
    timeout "${secs}" ${SSH} "${ENV_IP}" "${cmd}"
  fi
}

do_clean() {
  say "stopping ray + training (head)"
  source "${ONLINE_RL_VENV}/bin/activate" 2>/dev/null
  ray stop --force >/dev/null 2>&1
  local p
  for p in $(pgrep -f "$(basename "${TRAIN_SCRIPT}")" 2>/dev/null); do
    [ "$p" = "$$" ] && continue
    kill -9 "$p" 2>/dev/null
  done
  sleep 2
  bash "${FREE_GPUS_SH}"

  say "stopping env master + node + containers (head)"
  bash "${STOP_ENV_SH}"
  for p in $(pgrep -f 'python -m cluster[.]master[.]server' 2>/dev/null); do
    [ "$p" = "$$" ] && continue
    kill -9 "$p" 2>/dev/null
  done

  local h
  for h in "${WORKER_IPS[@]}"; do
    say "cleaning ${h}"
    timeout 90 ${SSH} "$h" \
      "source ${ONLINE_RL_VENV}/bin/activate 2>/dev/null; ray stop --force >/dev/null 2>&1; sleep 2; bash ${FREE_GPUS_SH}; bash ${STOP_ENV_SH}" \
      2>&1 | sed 's/^/  /' | tail -4
  done
  say "clean done"
}

start_env_master() {
  say "starting cluster master on ${ENV_IP}"
  run_on_env_host 120 \
    "cd ${ENV_INFRA_DIR} && bash cluster/scripts/start_cluster_master.sh" \
    2>&1 | sed 's/^/  /' | tail -3
  local i
  for i in $(seq 1 30); do
    if curl -fsS -m 5 "${MASTER_URL}/healthz" >/dev/null 2>&1; then
      echo "  master healthy after ${i} attempt(s)"; return 0
    fi
    sleep 2
  done
  echo "  ERROR: master did not come up at ${MASTER_URL}"; return 1
}

start_env_node() {
  say "starting world node on ${ENV_IP} (worlds=${ENV_WORLDS}, slots=${ENV_MAX_SLOTS})"
  run_on_env_host 300 \
    "cd ${ENV_INFRA_DIR} && MASTER_URL=${MASTER_URL} PUBLIC_HOST=${ENV_IP} NODE_WORLDS=${ENV_WORLDS} NODE_MAX_SLOTS=${ENV_MAX_SLOTS} bash scripts/bash/restart_node.sh" \
    2>&1 | sed 's/^/  /' | tail -5
  local i
  for i in $(seq 1 30); do
    sleep 3
    if env_healthy_nodes | grep -qE '^[1-9]'; then
      echo "  node registered and healthy"; env_status; return 0
    fi
  done
  echo "  WARNING: node did not report healthy; check ${ENV_INFRA_DIR}/logs/ on ${ENV_IP}"
  env_status
}

env_healthy_nodes() {
  curl -fsS -m 6 "${MASTER_URL}/status" 2>/dev/null \
    | "${ENV_INFRA_VENV}/bin/python" -c \
      "import json,sys; print(json.load(sys.stdin)['cluster']['healthy_nodes'])" 2>/dev/null
}

env_status() {
  curl -fsS -m 6 "${MASTER_URL}/status" 2>/dev/null \
    | "${ENV_INFRA_VENV}/bin/python" -c "
import json,sys
d=json.load(sys.stdin); c=d['cluster']
print('  env cluster:', {k:c[k] for k in ['healthy_nodes','max_envs','total_envs','busy_envs','training_sessions']})
for n in d.get('nodes',[]):
    print('  node:', n['node_id'], n['status'], '| free', n['free_slots'], '| hb', round(n.get('heartbeat_age_seconds',-1),1),'s')
" 2>/dev/null || echo "  (master unreachable)"
}

start_training() {
  say "launching training head (16 GPU: ${ACTOR_GPUS} actor / ${ROLLOUT_GPUS} rollout, ${EXPECTED_NODES} nodes)"
  source "${ONLINE_RL_VENV}/bin/activate"
  echo "  ray: $(ray --version 2>&1 | head -1)"
  echo "  ray num-cpus: ${RAY_NUM_CPUS} (item 3b)"
  echo "  env-spare pattern: ${GUI_ENV_SERVER_PROC_PATTERN}"
  local rw
  rw="$(grep -hE '^lambda_(cli|exec):' "${CUSTOM_CONFIG_PATH}" 2>/dev/null | paste -sd' ' -)"
  echo "  reward: ${rw:-none (outcome-only; CLI/PRM terms off)}  [$(basename "${CUSTOM_CONFIG_PATH}")]"
  ( cd "${GUI_RL_DIR}" && bash "${TRAIN_SCRIPT}" 2>&1 | tail -6 )
}

TRAIN_LOG_PREFIX="$(sed -nE 's|.*LOG_FILE=.*/logs/([A-Za-z0-9_]+)_\$\(date.*|\1|p' "${TRAIN_SCRIPT}" | head -1)"
TRAIN_LOG_GLOB="${GUI_RL_DIR}/logs/${TRAIN_LOG_PREFIX:-hybridcua24}_*.log"
latest_train_log() { ls -t ${TRAIN_LOG_GLOB} 2>/dev/null | head -1; }

wait_for_ray_head() {
  say "waiting for ray head"
  source "${ONLINE_RL_VENV}/bin/activate" 2>/dev/null
  local i
  for i in $(seq 1 40); do
    if ray status >/dev/null 2>&1; then echo "  head up after ${i} attempt(s)"; return 0; fi
    sleep 3
  done
  echo "  ERROR: ray head never came up"; return 1
}

join_as_worker() {
  say "joining Ray cluster at ${HEAD_IP}:6379 as a worker"
  echo "  this node: $(hostname -I 2>/dev/null | tr ' ' '\n' | grep -v '^$' | head -1)"
  echo "  topology:  ${TOPOLOGY} (${ACTOR_NUM_GPUS_PER_NODE} GPU expected here, actor node)"
  echo "  ray cpus:  ${RAY_NUM_CPUS} (must match the head's declared value, item 3b)"
  if hostname -I 2>/dev/null | tr ' ' '\n' | grep -qx "${HEAD_IP}"; then
    echo "  ERROR: this host IS the head (${HEAD_IP}). Run 'env' and 'train' here instead."
    exit 1
  fi
  RAY_HEAD_ADDR="${HEAD_IP}" WORKER_NUM_GPUS="${ACTOR_NUM_GPUS_PER_NODE}" \
    RAY_NUM_CPUS="${RAY_NUM_CPUS}" bash "${JOIN_SH}"
}

join_workers() {
  local h
  for h in "${WORKER_IPS[@]}"; do
    say "joining ${h}"
    timeout 120 ${SSH} "$h" "RAY_HEAD_ADDR=${HEAD_IP} WORKER_NUM_GPUS=8 RAY_NUM_CPUS=${RAY_NUM_CPUS} bash ${JOIN_SH}" \
      2>&1 | sed 's/^/  /' | tail -2
  done

  say "waiting for ${EXPECTED_NODES}-node cluster"
  source "${ONLINE_RL_VENV}/bin/activate" 2>/dev/null
  local i n
  for i in $(seq 1 40); do
    n=$(ray status 2>/dev/null | grep -c "node_")
    if [ "${n:-0}" -ge "${EXPECTED_NODES}" ]; then
      echo "  ${n}/${EXPECTED_NODES} nodes joined"
      ray status 2>&1 | grep -E "/[0-9.]+ (CPU|GPU)" | sed 's/^/  /'
      return 0
    fi
    sleep 5
  done
  echo "  WARNING: only ${n:-0}/${EXPECTED_NODES} nodes joined"
}

do_status() {
  source "${ONLINE_RL_VENV}/bin/activate" 2>/dev/null
  say "ray"
  ray status 2>&1 | grep -E "^ [0-9]+ node_|/[0-9.]+ (CPU|GPU)" | sed 's/^/  /' || echo "  (no cluster)"
  say "env"
  env_status
  say "training"
  local L; L="$(latest_train_log)"
  if [ -n "$L" ]; then
    echo "  log: $L"
    echo "  mtime: $(stat -c %y "$L" | cut -c1-19)"
    echo "  acquire-409: $(grep -c 'GUI env acquire failed' "$L" 2>/dev/null)"
    echo "  step-term hits (lambda_exec live): $(grep -c 'carry a step term' "$L" 2>/dev/null)"
    grep -oE "train/(step|loss|global_batch_size)'?: *[0-9.-]+" "$L" 2>/dev/null | tail -3 | sed 's/^/  /'
  else
    echo "  (no training log yet)"
  fi
}

case "${1:-all}" in
  clean)  require_head_host "$@"; do_clean ;;
  env)    start_env_master && start_env_node ;;
  train)  require_head_host "$@"; start_training && wait_for_ray_head && join_workers ;;
  worker) join_as_worker ;;
  status) do_status ;;
  all)
    require_head_host "$@"
    do_clean
    start_env_master || exit 1
    start_env_node
    start_training
    wait_for_ray_head || exit 1
    join_workers
    say "launched -- watch with: bash $0 status"
    echo "  train log: $(latest_train_log)"
    ;;
  *) echo "usage: TOPOLOGY=<name> $0 [all|clean|env|train|worker|status]"; exit 1 ;;
esac
