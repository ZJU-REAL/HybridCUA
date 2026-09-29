#!/usr/bin/env bash
# Start this host's cluster node with the standard 27B-eval settings.
# Run via pssh across the node hosts; the repo lives on shared cephfs.
#
# NODE_ID is derived from the host's own IP so one command works everywhere.
set -uo pipefail

_script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "${_script_dir}/../.."

_ip="$(ip route get 1.1.1.1 2>/dev/null | sed -nE 's/.* src ([0-9.]+).*/\1/p' | head -1)"

MASTER_URL="${MASTER_URL:-}"
if [[ -z "$MASTER_URL" ]]; then echo "Set MASTER_URL before starting the node" >&2; exit 2; fi

NODE_MASTER_URL="${MASTER_URL}" \
  NODE_ID="node27b-${_ip##*.}" \
  NODE_WORLDS="${NODE_WORLDS:-cua_gym,osworld}" \
  NODE_MAX_SLOTS="${NODE_MAX_SLOTS:-64}" \
  PORT="${PORT:-18080}" \
  LOG_FILE="logs/node_27b_${_ip##*.}.log" \
  bash cluster/scripts/start_cluster_node.sh
