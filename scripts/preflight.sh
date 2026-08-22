#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
h3_load_env
h3_require_license_acknowledgement

HEAD_HOST="${HEAD_HOST:-spark-head}"
WORKER_HOST="${WORKER_HOST:-spark-peer}"
HEAD_IP="${HEAD_IP:-}"
WORKER_IP="${WORKER_IP:-}"
IFACE="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}"
HCA="${NCCL_IB_HCA:-rocep1s0f1}"
HEAD_GID="${H3_HEAD_GID_INDEX:-${NCCL_IB_GID_INDEX:-3}}"
WORKER_GID="${H3_WORKER_GID_INDEX:-${NCCL_IB_GID_INDEX:-3}}"
EXPECTED_HEAD_HOSTNAME="${EXPECTED_HEAD_HOSTNAME:-spark-head}"
EXPECTED_WORKER_HOSTNAME="${EXPECTED_WORKER_HOSTNAME:-spark-peer}"
IMAGE="${H3_2X_IMAGE:-minimax-h3-2x-dgx-spark:experimental}"
MODEL_DIR="${MINIMAX_H3_MODEL_DIR:-}"
MODEL="$MODEL_DIR/model_index.json"
API_PORT="${H3_API_PORT:-8000}"

h3_require_command ssh
h3_require_safe_value HEAD_HOST "$HEAD_HOST"
h3_require_safe_value WORKER_HOST "$WORKER_HOST"
h3_require_safe_value EXPECTED_HEAD_HOSTNAME "$EXPECTED_HEAD_HOSTNAME"
h3_require_safe_value EXPECTED_WORKER_HOSTNAME "$EXPECTED_WORKER_HOSTNAME"
h3_require_safe_value H3_2X_IMAGE "$IMAGE"
h3_require_safe_value MINIMAX_H3_MODEL_DIR "$MODEL_DIR"
h3_require_safe_value NCCL_SOCKET_IFNAME "$IFACE"
h3_require_safe_value NCCL_IB_HCA "$HCA"
h3_require_safe_value H3_HEAD_GID_INDEX "$HEAD_GID"
h3_require_safe_value H3_WORKER_GID_INDEX "$WORKER_GID"
h3_require_nonnegative_integer H3_HEAD_GID_INDEX "$HEAD_GID"
h3_require_nonnegative_integer H3_WORKER_GID_INDEX "$WORKER_GID"
h3_require_ipv4 HEAD_IP "$HEAD_IP"
h3_require_ipv4 WORKER_IP "$WORKER_IP"
h3_require_port H3_API_PORT "$API_PORT"
h3_validate_lora_profile
h3_preflight_lora_artifacts

check_node() {
  local host="$1" expected="$2" gid="$3"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "
    test \"\$(uname -m)\" = aarch64
    test \"\$(hostname)\" = '$expected'
    test -f '$MODEL'
    docker image inspect '$IMAGE' >/dev/null
    test -e /dev/infiniband/uverbs0
    test \"\$(cat '/sys/class/infiniband/$HCA/ports/1/gid_attrs/types/$gid')\" = 'RoCE v2'
    test \"\$(cat '/sys/class/infiniband/$HCA/ports/1/gid_attrs/ndevs/$gid')\" = '$IFACE'
  "
}

check_node "$HEAD_HOST" "$EXPECTED_HEAD_HOSTNAME" "$HEAD_GID"
check_node "$WORKER_HOST" "$EXPECTED_WORKER_HOSTNAME" "$WORKER_GID"

# Validated values are intentionally expanded on the client for remote checks.
# shellcheck disable=SC2029
if ssh "$HEAD_HOST" "ss -lntH 'sport = :$API_PORT' | grep -q ."; then
  ssh "$HEAD_HOST" "test \"\$(docker inspect -f '{{.State.Running}}' minimax-h3-2x-api 2>/dev/null)\" = true" || {
    echo "refusing to use occupied head port $API_PORT: it is not owned by minimax-h3-2x-api" >&2
    exit 1
  }
fi

# shellcheck disable=SC2029
ssh "$HEAD_HOST" "ping -I '$IFACE' -c 2 -W 2 '$WORKER_IP' >/dev/null"
# shellcheck disable=SC2029
ssh "$WORKER_HOST" "ping -I '$IFACE' -c 2 -W 2 '$HEAD_IP' >/dev/null"

echo "preflight passed: exact hosts, ARM64, model, image, port ownership, RoCEv2 GID, and fabric reachability"
