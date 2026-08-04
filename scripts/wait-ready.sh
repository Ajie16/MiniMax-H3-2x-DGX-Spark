#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
h3_load_env

HEAD_IP="${HEAD_IP:-}"
API_PORT="${H3_API_PORT:-8000}"
TIMEOUT_SECONDS="${H3_READY_TIMEOUT_SECONDS:-1800}"

h3_require_command curl
h3_require_ipv4 HEAD_IP "$HEAD_IP"
h3_require_port H3_API_PORT "$API_PORT"
[[ "$TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || h3_fail "H3_READY_TIMEOUT_SECONDS must be an integer"

deadline=$((SECONDS + TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
  code="$(curl -sS --max-time 3 -o /dev/null -w '%{http_code}' "http://$HEAD_IP:$API_PORT/health" 2>/dev/null || true)"
  if [[ "$code" = 200 ]]; then
    "$SCRIPT_DIR/status.sh"
    exit 0
  fi
  sleep 10
done

echo "service did not become healthy within ${TIMEOUT_SECONDS}s" >&2
exit 1
