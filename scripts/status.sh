#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
h3_load_env

HEAD_HOST="${HEAD_HOST:-spark-head}"
WORKER_HOST="${WORKER_HOST:-spark-peer}"
HEAD_IP="${HEAD_IP:-}"
API_PORT="${H3_API_PORT:-8000}"
MODEL_DIR="${MINIMAX_H3_MODEL_DIR:-}"

h3_require_command curl
h3_require_command python3
h3_require_command ssh
h3_require_safe_value HEAD_HOST "$HEAD_HOST"
h3_require_safe_value WORKER_HOST "$WORKER_HOST"
h3_require_safe_value MINIMAX_H3_MODEL_DIR "$MODEL_DIR"
h3_require_ipv4 HEAD_IP "$HEAD_IP"
h3_require_port H3_API_PORT "$API_PORT"

inspect_container() {
  local host="$1" container="$2"
  # Both values are fixed or passed h3_require_safe_value.
  # shellcheck disable=SC2029
  ssh "$host" "docker inspect -f '{{.Name}} status={{.State.Status}} restarts={{.RestartCount}} oom={{.State.OOMKilled}}' '$container'"
}

inspect_container "$HEAD_HOST" minimax-h3-2x-ray-head
inspect_container "$HEAD_HOST" minimax-h3-2x-api
inspect_container "$WORKER_HOST" minimax-h3-2x-ray-worker

ssh "$HEAD_HOST" "docker exec minimax-h3-2x-ray-head ray status"

health_code="$(curl -sS -o /dev/null -w '%{http_code}' "http://$HEAD_IP:$API_PORT/health")"
test "$health_code" = 200

served_model="$(curl -sS "http://$HEAD_IP:$API_PORT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
test "$served_model" = "$MODEL_DIR"

echo "health=$health_code served_model=$served_model api_base=http://$HEAD_IP:$API_PORT/v1"
