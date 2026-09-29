#!/usr/bin/env bash
# ============================================================================
# RL run on the Taiji cluster (8 GPU per node). The node count / GPU split is
# NOT baked in: it comes from a topology file under scripts/topologies/,
# selected with TOPOLOGY=<name>. Shipped topologies:
#
#   2node  16 GPU -- 8 rollout (head) + 8 actor (1 node x 8, TP=4, DP=2)
#   3node  24 GPU -- 8 rollout (head) + 16 actor (2 nodes x 8, TP=4, DP=4)
#
#   bash scripts/run_rl_24gpu.sh              # full restart: clean -> env -> train
#   bash scripts/run_rl_24gpu.sh clean        # tear everything down, stop
#   bash scripts/run_rl_24gpu.sh env          # (re)start master + world node only
#   bash scripts/run_rl_24gpu.sh train        # assume env is up, start training
#   bash scripts/run_rl_24gpu.sh worker       # on each NON-head node: join the Ray cluster
#   bash scripts/run_rl_24gpu.sh status       # cluster / env / training snapshot
#
#   TOPOLOGY=3node bash scripts/run_rl_24gpu.sh all     # or prefix any other verb
#
# Everything below is a DEFAULT baked into the selected topology file. Run it
# with no arguments and no env overrides -- that is the whole point of the file.
#
# --- manual (multi-homed) bring-up -----------------------------------------
# `worker` exists because the worker nodes are not always ssh-reachable from the
# head: Taiji injects the shared ssh key per POD, so a worker in a different pod
# answers ping and has :22 open but rejects the head's key (observed 2026-09-18
# against a previous pod's <node-ip> / <node-ip>). `all`/`train` still ssh
# in, so when that happens: run `env` + `train` on the head and run
# `TOPOLOGY=<name> ... worker` by hand on each worker. Coordinate the timing --
# the train script's ray head waits EXPECTED_NODES for only 600s and then
# submits anyway, which silently produces the wrong actor/rollout placement.
#
# ---------------------------------------------------------------------------
# Topology. Taken from the pods' OWN env vars, not guessed:
#     HOST_NUM=2  TAIJI_HOST_NUM=2  HOST_GPU_NUM=8  NODE_NUM=16
#     NODE_IP_LIST=<node-ip>:8,<node-ip>:8
#     CHIEF_IP=<node-ip>    <- launcher pod, INDEX=0
#     <node-ip>             <- worker-0 pod, INDEX=1
#   /etc/taiji/hostfile agrees: "<node-ip> slots=8" + "<node-ip> slots=8".
#
#   node0  <node-ip>  (HEAD; the launcher/chief pod -- run this script here)
#     - GUI env server: cluster master :19000 + world_server :18080 + docker VMs
#     - Rollout: 8 sglang engines, TP=1 each
#   node1  <node-ip>  (worker)
#     - Actor: 8 GPU, TP=4 -> DP=2
#
# Why the rollout lands on the head (== the env box) instead of the worker:
#   slime builds ONE placement group of actor_num_gpus + rollout_num_gpus = 16
#   bundles with strategy PACK, then reorders the bundles by (node IP, gpu id)
#   (slime/ray/placement_group.py: sort_key). <node-ip> sorts before
#   <node-ip>, so bundles 0-7 are the worker and bundles 8-15 are the head.
#   RayTrainGroup takes reordered_bundle_indices[rank] for rank < 8 -> actor on
#   the worker; rollout = indices[rollout_offset:] -> the head. That is exactly
#   what we want: the sglang engines sit next to the env server, so the
#   per-request GUI screenshots (a single request can be ~11390 tokens) never
#   cross the network. NOTE this is IP-sort luck: swap the host IPs and the two
#   roles swap with them. Functionally it still works (the env URL is an IP),
#   only the screenshot hop gets slower.
#
# ---------------------------------------------------------------------------
# Why this script exists (each item cost us a failed launch on 2026-09-13,
# re-derived for the 2-node layout on 2026-09-18):
#
# 1. GPU layout. The upstream HybridCUA-9B_32gpu script defaults to 32 GPUs
#    (24 actor + 8 rollout, EXPECTED_NODES=4). This cluster has 16: 8 actor +
#    8 rollout, so EXPECTED_NODES=2. EXPECTED_NODES is hardcoded to 4 upstream
#    and does NOT follow NUM_GPUS; leaving it wrong just burns the 600s join
#    timeout and then proceeds with the wrong topology.
#
#    16 GPUs, actor 8 -> TP=4 gives DP=2, and the global batch size
#    8 x 8 = 64 satisfies Megatron's init-time assertion
#    64 % (micro_batch_size 1 x dp_size 2) == 0. (DP=6 on a 24-GPU actor did
#    NOT: 64 % 6 == 4, which is the trap the 32gpu script's header documents.)
#
# 2. ray venv. The head MUST start from projects/venvs/online-rl (ray 2.56.1).
#    A bare `bash script.sh` inherits conda's /opt/conda/envs/torch-base ray
#    2.46.0, and every worker join then dies with "Version mismatch".
#    gpu_worker_join_ray.sh activates the venv itself; the head does not.
#
# 3. env node gets killed by the trainer. The training script's
#    kill_stale_python() spares only `cluster.master.server`, so it reaps
#    `cluster.node.world_server` on every launch -- the node goes `dead`, the
#    master reports healthy_nodes=0, and rollouts cannot acquire an env.
#    Here the env server is co-located with the head, so that reaping WOULD hit
#    us. Fixed head-on by exporting GUI_ENV_SERVER_PROC_PATTERN (consumed at
#    HybridCUA-9B_32gpu_fully_async.sh:74) to spare BOTH processes, instead of
#    the older "start the world node after the trainer's kill phase" dance.
#    The env server can therefore be fully up before training starts, and
#    rollouts never race an empty cluster.
#
# 3b. RAY_NUM_CPUS. The 64 rollout actors (_RolloutActor, GUI_FAST_ROLLOUT_PROCS
#    =64) each reserve GUI_RAY_ACTOR_CPUS=1 and are NOT in the placement group --
#    they take the node's own CPU pool, pinned by NodeAffinitySchedulingStrategy
#    to wherever the RolloutManager landed. The manager itself (num_cpus=1) and
#    its RolloutManager lock (num_cpus=1) are ordinary actors too. With the
#    upstream default of --num-cpus 64 on the head, 64 actors alone exhaust the
#    budget and the manager/lock can never schedule -> rollout hangs. Raise it
#    to 128 so all three fit with headroom. (Do NOT go near nproc=384: Ray
#    prestarts int(num_cpus) python workers, and ~376 of them importing
#    torch/sglang/TE wedge the raylet's registration path.)
#
# 4. env node slot count. Our data source is RLVR CUA-Gym (CuaGymDataSource,
#    GUI_CUA_GYM_TASKS_META -- see the train script's GUI_DATA_SOURCE_PATH), and
#    the in-flight pool is sglang_server_concurrency(8) x num_engines(8) = 64.
#    Keep NODE_WORLDS=cua_gym + NODE_MAX_SLOTS=72: enough slots for the pool with
#    a little headroom. A smaller --max-slots-per-world starves the pool and
#    produces a steady stream of "GUI env acquire failed after 10 retries:
#    409 CONFLICT", silently shrinking the GRPO batch.
#
# 5. leftover GPU holders. `ray stop` + `pkill sglang` do NOT free the GPUs left
#    by a previous *eval* run: those are vLLM (`VLLM::EngineCore`) whose children
#    outlive the api_server parent. scripts/free_gpus.sh handles both vLLM and
#    sglang remnants. Both boxes had 8 vLLM eval servers holding all 8 GPUs
#    before this run (2026-09-18).
#
# 6. inline pkill over ssh kills the ssh session itself (the pattern matches the
#    wrapper's own command line). All remote cleanup goes through the committed
#    helper scripts, never an inline `pkill -f`.
#
# 7. this script must run ON the head. do_clean, start_training and `ray start`
#    are local operations; running it elsewhere heads the cluster at the wrong
#    box AND frees GPUs belonging to another workload. require_head_host aborts.
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUI_RL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECTS_ROOT="$(cd "${GUI_RL_DIR}/../.." && pwd)"
ENV_INFRA_DIR="${PROJECTS_ROOT}/env_infra"
ONLINE_RL_VENV="${PROJECTS_ROOT}/venvs/online-rl"
ENV_INFRA_VENV="${PROJECTS_ROOT}/venvs/env_infra"

# --- topology (item 1) ------------------------------------------------------
# The node list and the actor/rollout split come from scripts/topologies/
# <TOPOLOGY>.env, not from this file. Every value in a topology file is written
# as ${VAR:-value}, so precedence is: explicit env var > topology file > error.
# That is what makes one runner serve several cluster shapes.
TOPOLOGY="${TOPOLOGY_ENV:-${TOPOLOGY:-2node}}"
TOPO_FILE="${SCRIPT_DIR}/topologies/${TOPOLOGY}.env"
if [[ ! -f "${TOPO_FILE}" ]]; then
  echo "ERROR: no such topology: ${TOPOLOGY} (looked for ${TOPO_FILE})"
  echo "  available: $(cd "${SCRIPT_DIR}/topologies" 2>/dev/null && ls *.env 2>/dev/null | sed 's/\.env$//' | paste -sd' ' -)"
  exit 1
fi
# shellcheck source=/dev/null
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
# WORKER_IPS arrives space-separated ("ip1 ip2"); the rest of the file wants an array.
WORKER_IPS=(${WORKER_IPS})

export NUM_GPUS ACTOR_GPUS ACTOR_NUM_NODES ACTOR_NUM_GPUS_PER_NODE ROLLOUT_GPUS EXPECTED_NODES
# Sanity: the actor must fit the nodes it is spread over, and the split must fit
# the GPU count -- a wrong topology file otherwise fails deep inside slime.
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
# Which node gets the rollout is slime's IP sort, not the head -- so compute it
# instead of asserting "lands on the head" (true only while the head sorts
# last, which the 2026-09-24 IPs no longer do).
_ROLES="$(printf '%s\n' "${HEAD_IP}" "${WORKER_IPS[@]}" | sort -t. -k1,1n -k2,2n -k3,3n -k4,4n | paste -sd' ' -)"
_ROLLOUT_NODE="${_ROLES##* }"
echo "  actor ${ACTOR_GPUS} GPU = ${ACTOR_NUM_NODES} x ${ACTOR_NUM_GPUS_PER_NODE} | rollout ${ROLLOUT_GPUS} GPU on ${_ROLLOUT_NODE} (highest IP) | ${EXPECTED_NODES} nodes expected"
[[ "${_ROLLOUT_NODE}" == "${ENV_IP}" ]] \
  || echo "  note: env server is on ${ENV_IP}, not the rollout node -- screenshots cross the network"

# --- items 3 / 3b: trainer must not reap the env server; actor CPUs must fit --
# BRE alternation (\|), consumed by `grep -q` in the train script's
# kill_stale_python(). Spares both cluster.master.server AND
# cluster.node.world_server; everything else python is still nuked.
export GUI_ENV_SERVER_PROC_PATTERN='cluster\.master\.server\|cluster\.node\.world_server'
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-128}"

# --- item 4: env node worlds/slots ------------------------------------------
# ENV_WORLDS comes from the topology (required above). It MUST match the train
# script's GUI_ENV_RUNTIME: every train script now defaults that to `cua_gym`,
# and a mismatch makes every POST /v1/sessions fail with "409 no healthy node
# can serve world 'cua_gym'" while the rollout spins at "collected 0/8" and the
# master log fills with 409s.
# ENV_MAX_SLOTS stays here because it is pool arithmetic, not topology: the
# in-flight pool is sglang-server-concurrency(8) x num_engines(8) = 64, so 72
# leaves a little headroom.
ENV_MAX_SLOTS="${ENV_MAX_SLOTS:-72}"

# TRAIN_SCRIPT comes from the topology file (required above and checked there).
# CUSTOM_CONFIG_PATH mirrors the train script's own default; the runner only
# reads it (for the reward line below), it does not pass it on.
CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${GUI_RL_DIR}/scripts/gui_partial_async.yaml}"
FREE_GPUS_SH="${GUI_RL_DIR}/scripts/free_gpus.sh"
STOP_ENV_SH="${GUI_RL_DIR}/scripts/stop_env_node.sh"
JOIN_SH="${GUI_RL_DIR}/scripts/gpu_worker_join_ray.sh"
MASTER_URL="http://${ENV_IP}:19000"
# The train scripts default GUI_ENV_SERVER_URL to whatever pod they were written
# on. Export it so the rollout actors talk to ENV_IP instead.
export GUI_ENV_SERVER_URL="${MASTER_URL}"

SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes"

say() { echo; echo "=== $* ==="; }

# --- item 7: this script must run ON the head -------------------------------
# do_clean / start_training / ray start are all local operations. Running it
# from any other host silently builds a cluster headed by the wrong machine and
# frees GPUs that belong to something else. Fail fast instead.
require_head_host() {
  if ! hostname -I 2>/dev/null | tr ' ' '\n' | grep -qx "${HEAD_IP}"; then
    echo "ERROR: this host is not HEAD_IP=${HEAD_IP} (local IPs: $(hostname -I))"
    echo "  run it there:  ${SSH} ${HEAD_IP} \"setsid bash ${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}") $*\""
    echo "  or override HEAD_IP=<this host> to move the head."
    exit 1
  fi
}

# The env host IS the head in this layout, so env bring-up is local. Keep the
# ssh branch so ENV_IP can be moved to the worker again without a rewrite.
# `timeout` cannot exec a shell function, so it is applied to `bash -c` /
# `${SSH}` inside here rather than around the call site.
run_on_env_host() {
  local secs="$1" cmd="$2"
  if [[ "${ENV_IP}" == "${HEAD_IP}" ]]; then
    timeout "${secs}" bash -c "${cmd}"
  else
    timeout "${secs}" ${SSH} "${ENV_IP}" "${cmd}"
  fi
}

# --- clean ------------------------------------------------------------------
do_clean() {
  say "stopping ray + training (head)"
  # shellcheck disable=SC1091
  source "${ONLINE_RL_VENV}/bin/activate" 2>/dev/null
  ray stop --force >/dev/null 2>&1
  local p
  # Match the train script this topology actually runs. This was hardcoded to
  # the 32gpu basename, so once the topology switched to the 24gpu script the
  # old training process survived `clean` and kept holding the GPUs.
  for p in $(pgrep -f "$(basename "${TRAIN_SCRIPT}")" 2>/dev/null); do
    [ "$p" = "$$" ] && continue
    kill -9 "$p" 2>/dev/null
  done
  sleep 2
  bash "${FREE_GPUS_SH}"          # item 5

  say "stopping env master + node + containers (head)"
  bash "${STOP_ENV_SH}"           # item 6
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

# --- env server (on ENV_IP) --------------------------------------------------
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

# item 4: OSWorld-only, 72 slots for the 64-slot in-flight pool.
# restart_node.sh hardcodes MASTER_URL to <node-ip> (:13), so pass ours.
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

# --- training ---------------------------------------------------------------
start_training() {
  say "launching training head (16 GPU: ${ACTOR_GPUS} actor / ${ROLLOUT_GPUS} rollout, ${EXPECTED_NODES} nodes)"
  # item 2: the head must run from the online-rl venv (ray 2.56.1).
  # shellcheck disable=SC1091
  source "${ONLINE_RL_VENV}/bin/activate"
  echo "  ray: $(ray --version 2>&1 | head -1)"
  echo "  ray num-cpus: ${RAY_NUM_CPUS} (item 3b)"
  echo "  env-spare pattern: ${GUI_ENV_SERVER_PROC_PATTERN}"
  # Read the config this run ACTUALLY uses. This used to hardcode the 32gpu yaml,
  # which reports coefficients (lambda_cli/lambda_exec) the 24gpu run never loads
  # -- printing a reward line that was simply wrong.
  local rw
  rw="$(grep -hE '^lambda_(cli|exec):' "${CUSTOM_CONFIG_PATH}" 2>/dev/null | paste -sd' ' -)"
  echo "  reward: ${rw:-none (outcome-only; CLI/PRM terms off)}  [$(basename "${CUSTOM_CONFIG_PATH}")]"
  ( cd "${GUI_RL_DIR}" && bash "${TRAIN_SCRIPT}" 2>&1 | tail -6 )
}

# The log glob is READ OUT OF TRAIN_SCRIPT rather than guessed from its name:
# each train script names its own log (hybridcua24_ / hybridcua32_ / gui9b24_),
# and the prefix does not follow from the filename -- the GUI-only 24gpu script
# writes gui9b24_*.log, not hybridcua24_*.log. This used to be hardcoded to
# hybridcua32 and then derived as hybridcua<N>, so every path it printed (and
# the log `status` reported) pointed at an unrelated earlier run.
TRAIN_LOG_PREFIX="$(sed -nE 's|.*LOG_FILE=.*/logs/([A-Za-z0-9_]+)_\$\(date.*|\1|p' "${TRAIN_SCRIPT}" | head -1)"
TRAIN_LOG_GLOB="${GUI_RL_DIR}/logs/${TRAIN_LOG_PREFIX:-hybridcua24}_*.log"
latest_train_log() { ls -t ${TRAIN_LOG_GLOB} 2>/dev/null | head -1; }

wait_for_ray_head() {
  say "waiting for ray head"
  # shellcheck disable=SC1091
  source "${ONLINE_RL_VENV}/bin/activate" 2>/dev/null
  local i
  for i in $(seq 1 40); do
    if ray status >/dev/null 2>&1; then echo "  head up after ${i} attempt(s)"; return 0; fi
    sleep 3
  done
  echo "  ERROR: ray head never came up"; return 1
}

# --- worker: run by hand on a NON-head GPU node ------------------------------
# Exists because the head cannot always ssh into its workers: Taiji injects the
# shared ssh key per POD, so a worker from another pod answers ping and has :22
# open but then rejects the head's key. In that case run `env` + `train` on the
# head and this verb on each worker.
# gpu_worker_join_ray.sh activates the venv itself, refuses to run on the head,
# and self-detaches (setsid) unless BACKGROUND=0.
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
    # gpu_worker_join_ray.sh activates the venv itself (item 2).
    # RAY_NUM_CPUS must match the head's declared value (item 3b).
    timeout 120 ${SSH} "$h" "RAY_HEAD_ADDR=${HEAD_IP} WORKER_NUM_GPUS=8 RAY_NUM_CPUS=${RAY_NUM_CPUS} bash ${JOIN_SH}" \
      2>&1 | sed 's/^/  /' | tail -2
  done

  say "waiting for ${EXPECTED_NODES}-node cluster"
  # shellcheck disable=SC1091
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
  # shellcheck disable=SC1091
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

# --- main -------------------------------------------------------------------
case "${1:-all}" in
  clean)  require_head_host "$@"; do_clean ;;
  env)    start_env_master && start_env_node ;;
  train)  require_head_host "$@"; start_training && wait_for_ray_head && join_workers ;;
  worker) join_as_worker ;;
  status) do_status ;;
  all)
    require_head_host "$@"
    do_clean
    # Env fully up BEFORE training: kill_stale_python now spares the world node
    # (item 3), so there is no longer any reason to order around it -- and this
    # way rollouts never race an empty cluster.
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
